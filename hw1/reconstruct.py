import os, re, glob, argparse
from tqdm import tqdm
import numpy as np
import open3d as o3d

# ---------------------------
# 通用工具
# ---------------------------
def natural_key(p):
    b = os.path.basename(p)
    s = re.split(r'(\d+)', b)
    return [int(t) if t.isdigit() else t for t in s]

def get_intrinsic_from_fov(width, height, fov_deg=90.0):
    fx = fy = (width / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
    cx, cy = width / 2.0, height / 2.0
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# ---------------------------
# 影像讀取與點雲
# ---------------------------
def load_rgb_depth(rgb_path, depth_path):
    color = o3d.io.read_image(rgb_path)
    depth_raw = o3d.io.read_image(depth_path)
    d = np.asarray(depth_raw)

    # 自動處理 8-bit（0~255 對應 0~10m）與 16-bit（mm）
    if d.dtype == np.uint8:
        d_m = (d.astype(np.float32) / 255.0) * 10.0
        depth = o3d.geometry.Image(d_m)  # 直接用公尺
        depth_scale, depth_trunc = 1.0, 8.0
    elif d.dtype == np.uint16:
        depth = depth_raw                   # 單位：毫米
        depth_scale, depth_trunc = 1000.0, 8.0
    else:  # 已是 float 公尺
        depth = depth_raw
        depth_scale, depth_trunc = 1.0, 8.0

    h, w = np.asarray(color).shape[:2]
    return color, depth, (w, h), depth_scale, depth_trunc

def depth_image_to_point_cloud(color_img, depth_img, intr, depth_scale, depth_trunc):
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_img, depth_img,
        depth_scale=depth_scale, depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)

    # 重要：把 Open3D 相機座標 (x右, y下, z前) 翻到常見世界 (x右, y上, z前)
    flip = np.array([[1,0,0,0],
                     [0,-1,0,0],
                     [0,0,-1,0],
                     [0,0,0,1]], dtype=np.float64)
    pcd.transform(flip)
    return pcd

# ---------------------------
# 前處理與特徵
# ---------------------------
def preprocess_point_cloud(pcd, voxel_size):
    pcd = pcd.voxel_down_sample(voxel_size)
    # 截斷過近/過遠的點（非必要但可穩定 ICP）
    # 以 z 前向的假設，保留 0.2~8 m
    pts = np.asarray(pcd.points)
    mask = np.isfinite(pts).all(axis=1)
    pcd = pcd.select_by_index(np.where(mask)[0])

    radius_normal = voxel_size * 2.0
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    radius_feature = voxel_size * 5.0
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd, fpfh

# ---------------------------
# 配準：全域 + 局部
# ---------------------------
def execute_global_registration_t_to_s(tgt_down, src_down, tgt_fpfh, src_fpfh, voxel_size):
    """
    回傳 T_t_to_s（把 target=t 對齊到 source=s）
    注意：Open3D 的 API 是 registration(source, target, ...) 回傳 source→target
    因此這裡把 "source=tgt_down, target=src_down" 以得到 t→s。
    """
    dist = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        tgt_down, src_down, tgt_fpfh, src_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 1000),
    )
    return result

def local_icp_algorithm_t_to_s(tgt_down, src_down, trans_init, threshold):
    """
    局部 ICP（point-to-plane），求 T_t_to_s
    仍然把 source 設為 "tgt_down"，target 設為 "src_down"
    """
    result = o3d.pipelines.registration.registration_icp(
        tgt_down, src_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result

# ---------------------------
# 自寫 ICP（點到點）— 直接輸出 T_t_to_s
# ---------------------------
class SimpleResult:
    def __init__(self, T, fitness, rmse):
        self.transformation = T
        self.fitness = fitness
        self.inlier_rmse = rmse

def kabsch(A, B):
    cA, cB = A.mean(0), B.mean(0)
    A0, B0 = A - cA, B - cB
    U, S, Vt = np.linalg.svd(A0.T @ B0)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cB - R @ cA
    return R, t

def my_local_icp_algorithm_t_to_s(tgt_down, src_down, trans_init, voxel_size, max_iter=30):
    src = np.asarray(src_down.points)  # s
    tgt = np.asarray(tgt_down.points)  # t
    if len(src) < 20 or len(tgt) < 20:
        return SimpleResult(trans_init.copy(), 0.0, np.inf)

    T = trans_init.copy()
    kdt = o3d.geometry.KDTreeFlann(src_down)  # 在 source 上找近鄰（因為要 t→s）
    max_corr = voxel_size * 1.5

    for _ in range(max_iter):
        # 把 target 點雲丟進目前估計的座標：t -> s
        tgt_tf = (T[:3,:3] @ tgt.T).T + T[:3,3]
        pairs_t, pairs_s = [], []
        for i, p in enumerate(tgt_tf):
            k, idx, d2 = kdt.search_knn_vector_3d(p, 1)
            if k == 1 and d2[0] <= max_corr * max_corr:
                pairs_t.append(i); pairs_s.append(idx[0])

        if len(pairs_t) < 8: break
        A = tgt_tf[pairs_t]   # 已在 s 座標下的 target
        B = src[pairs_s]      # source
        R, t = kabsch(A, B)
        T_delta = np.eye(4); T_delta[:3,:3] = R; T_delta[:3,3] = t
        T = T_delta @ T

        # 收斂（可省略，或依旋量/平移長度判斷）
        if np.linalg.norm(t) < 1e-5 and abs(np.trace(R)-3) < 1e-4:
            break

    # 計算 inlier rmse/fitness
    tgt_tf = (T[:3,:3] @ tgt.T).T + T[:3,3]
    d2s, inl = [], 0
    for p in tgt_tf:
        k, idx, d2 = kdt.search_knn_vector_3d(p, 1)
        if k == 1 and d2[0] <= max_corr * max_corr:
            d2s.append(d2[0]); inl += 1
    rmse = float(np.sqrt(np.mean(d2s))) if d2s else np.inf
    fitness = inl / max(1, len(tgt))
    return SimpleResult(T, fitness, rmse)

# ---------------------------
# 視覺化/評估
# ---------------------------
def make_trajectory_lineset(P, color):
    pts = o3d.utility.Vector3dVector(P)
    lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(P)-1)])
    ls = o3d.geometry.LineSet(points=pts, lines=lines)
    ls.colors = o3d.utility.Vector3dVector([color]* (len(P)-1))
    return ls

def kabsch_align(A, B):
    R, t = kabsch(A, B)
    return (R @ A.T).T + t, R, t

# ---------------------------
# 主流程
# ---------------------------
def reconstruct(args):
    data_root = args.data_root
    rgb = sorted(glob.glob(os.path.join(data_root, "rgb", "*.png")), key=natural_key)
    dep = sorted(glob.glob(os.path.join(data_root, "depth", "*.png")), key=natural_key)
    assert len(rgb) >= 2 and len(rgb) == len(dep)

    color0, depth0, (w,h), ds0, dt0 = load_rgb_depth(rgb[0], dep[0])
    intr = get_intrinsic_from_fov(w, h, 90.0)

    voxel_size = 0.03
    threshold = voxel_size * 1.5

    # 初始化
    world_T_curr = np.eye(4)              # world = frame_0
    scene = o3d.geometry.PointCloud()
    pred_cam = [world_T_curr[:3,3].copy()]  # 第一幀在原點

    # 先放第 0 幀
    pcd0 = depth_image_to_point_cloud(color0, depth0, intr, ds0, dt0)
    pcd0_down, _ = preprocess_point_cloud(pcd0, voxel_size)
    scene += pcd0_down

    for i in tqdm(range(len(rgb)-1)):
        # s = i, t = i+1
        color_s, depth_s, _, ds_s, dt_s = load_rgb_depth(rgb[i],   dep[i])
        color_t, depth_t, _, ds_t, dt_t = load_rgb_depth(rgb[i+1], dep[i+1])

        pcd_s = depth_image_to_point_cloud(color_s, depth_s, intr, ds_s, dt_s)
        pcd_t = depth_image_to_point_cloud(color_t, depth_t, intr, ds_t, dt_t)

        src_down, src_fpfh = preprocess_point_cloud(pcd_s, voxel_size)  # s
        tgt_down, tgt_fpfh = preprocess_point_cloud(pcd_t, voxel_size)  # t

        # 先全域（t→s）
        g = execute_global_registration_t_to_s(tgt_down, src_down, tgt_fpfh, src_fpfh, voxel_size)
        T_t_to_s_init = g.transformation

        # 再局部（t→s）
        if args.version.lower() == 'open3d':
            l = local_icp_algorithm_t_to_s(tgt_down, src_down, T_t_to_s_init, threshold)
        else:
            l = my_local_icp_algorithm_t_to_s(tgt_down, src_down, T_t_to_s_init, voxel_size)
        T_t_to_s = l.transformation

        # 累積到世界： T_world_(i+1) = T_world_i @ T_t_to_s
        world_T_curr = world_T_curr @ T_t_to_s

        # 把第 (i+1) 幀放到世界
        t_down_world = tgt_down.voxel_down_sample(voxel_size)
        t_down_world.transform(world_T_curr)

        # 輕度去噪（避免豎直針）
        t_down_world, _ = t_down_world.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        scene += t_down_world
        pred_cam.append(world_T_curr[:3,3].copy())

    pred_cam = np.vstack(pred_cam)

    # GT 與評估
    gt = np.load(os.path.join(data_root, "GT_pose.npy"))[:, :3]  # (N,3)
    n = min(len(gt), len(pred_cam))
    pred_aligned, R_a, t_a = kabsch_align(pred_cam[:n], gt[:n])
    T_a = np.eye(4); T_a[:3,:3] = R_a; T_a[:3,3] = t_a

    scene_vis = o3d.geometry.PointCloud(scene)
    scene_vis.transform(T_a)

    # 去天花板：保留地面以上 0~1.8m 範圍（可依你的資料微調）
    # 這裡直接用 AABB（更穩做法是 RANSAC 平面找地板）
    ymax = np.percentile(np.asarray(scene_vis.points)[:,1], 95)  # 避免過度裁切
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(-np.inf, -np.inf, -np.inf),
        max_bound=( np.inf,  ymax-0.2,  np.inf)
    )
    scene_vis = scene_vis.crop(bbox)

    l2 = np.linalg.norm(pred_aligned - gt[:n], axis=1)
    mean_l2 = float(np.mean(l2))

    pred_ls = make_trajectory_lineset(pred_aligned, (1,0,0))  # 紅
    gt_ls   = make_trajectory_lineset(gt[:n],       (0,0,0))  # 黑

    return scene_vis, pred_aligned, mean_l2, pred_ls, gt_ls

# ---------------------------
# 進入點
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='my_icp', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str, default='data_collection/first_floor/')
    args = parser.parse_args()

    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"

    scene_pcd, pred_traj, mean_l2, pred_ls, gt_ls = reconstruct(args)
    print(f"Mean L2 distance: {mean_l2:.4f} m")

    o3d.visualization.draw_geometries([scene_pcd, pred_ls, gt_ls])
