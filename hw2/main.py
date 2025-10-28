#!/usr/bin/env python3
# semantic_rrt_planner.py

from __future__ import annotations
import os
import json
import argparse
import math
import random
from typing import Tuple, Dict, Optional, List
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# -------------------------------
# Section 0. Palette utilities
# -------------------------------


def _parse_rgb_tuple_str(s: str) -> Optional[Tuple[int, int, int]]:
    # "(120, 34, 90)" -> (120,34,90)
    try:
        if not isinstance(s, str):
            return None
        t = s.strip()
        if t.startswith("(") and t.endswith(")"):
            t = t[1:-1]
        parts = [p.strip() for p in t.split(",")]
        if len(parts) != 3:
            return None
        r, g, b = [int(float(p)) for p in parts]
        return (
            max(0, min(255, r)),
            max(0, min(255, g)),
            max(0, min(255, b))
        )
    except Exception:
        return None


def load_palette_from_excel(excel_path: str) -> Dict[str, Tuple[int, int, int]]:
    """
    讀 color_coding_semantic_segmentation_classes.xlsx
    期望欄位至少包含:
      - 'Color_Code (R,G,B)'
      - 'Name'
    回傳: { class_name_lower : (r,g,b) }
    """
    import pandas as pd
    df = pd.read_excel(excel_path)

    # 嘗試用不區分大小寫的欄名
    cols = {str(c).strip().lower(): c for c in df.columns}
    rgb_col_key = "color_code (r,g,b)".lower()
    name_col_key = "name".lower()
    if rgb_col_key not in cols or name_col_key not in cols:
        raise RuntimeError(
            "Excel format unexpected. Need columns 'Color_Code (R,G,B)' and 'Name'."
        )

    rgb_col = cols[rgb_col_key]
    name_col = cols[name_col_key]

    mapping: Dict[str, Tuple[int, int, int]] = {}
    for _, row in df.iterrows():
        rgb = _parse_rgb_tuple_str(row.get(rgb_col))
        n = row.get(name_col)
        if rgb is None or n is None:
            continue
        cls_name = str(n).strip().lower()
        mapping[cls_name] = rgb
    return mapping


# -------------------------------
# Section 1. Point cloud loading, filtering, 2D map projection
# -------------------------------

def load_points_and_colors(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    載入 point.npy、color0255.npy / color01.npy
    並將 point 轉成 apartment_0 座標系
    回傳:
        points_apartment: (N,3) float64
        colors_255:       (N,3) uint8
    """
    point_path = os.path.join(data_dir, "point.npy")
    c255_path = os.path.join(data_dir, "color0255.npy")
    c01_path = os.path.join(data_dir, "color01.npy")

    if not os.path.isfile(point_path):
        raise FileNotFoundError(point_path)

    points = np.load(point_path)  # (N,3) in [0,255]
    scale_factor = 10000.0 / 255.0
    points_apartment = points.astype(np.float64) * scale_factor

    if os.path.isfile(c255_path):
        colors = np.load(c255_path)
    elif os.path.isfile(c01_path):
        colors = np.load(c01_path)
        colors = np.clip(np.rint(colors * 255.0), 0, 255)
    else:
        raise FileNotFoundError(
            "Need color0255.npy or color01.npy in data_dir"
        )

    if colors.ndim != 2 or colors.shape[1] != 3 or colors.shape[0] != points.shape[0]:
        raise ValueError("color shape mismatch with points")

    if np.issubdtype(colors.dtype, np.floating):
        cmax = float(colors.max())
        if cmax <= 1.0 + 1e-6:
            colors = np.clip(np.rint(colors * 255.0), 0, 255)
        colors_255 = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
    else:
        colors_255 = np.clip(colors, 0, 255).astype(np.uint8)

    return points_apartment, colors_255


def filter_floor_ceiling(
    points_apartment: np.ndarray,
    colors_255: np.ndarray,
    floor_frac: float = 0.25,
    ceiling_frac: float = 0.4,
    floor_abs: Optional[float] = None,
    ceiling_abs: Optional[float] = None,
    keep_floor: bool = False,
    keep_ceiling: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    移除地板/天花板:
      - y 最小附近的 slice 當地板
      - y 最大附近的 slice 當天花板
    """
    y = points_apartment[:, 1]
    y_min = float(np.min(y))
    y_max = float(np.max(y))

    if floor_abs is not None:
        y_floor_thr = y_min + float(floor_abs)
    else:
        y_floor_thr = float(np.quantile(y, min(max(floor_frac, 0.0), 1.0)))

    if ceiling_abs is not None:
        y_ceil_thr = y_max - float(ceiling_abs)
    else:
        y_ceil_thr = float(np.quantile(
            y, 1.0 - min(max(ceiling_frac, 0.0), 1.0)))

    keep_mask = np.ones(points_apartment.shape[0], dtype=bool)
    if not keep_floor:
        keep_mask &= (y > y_floor_thr)
    if not keep_ceiling:
        keep_mask &= (y < y_ceil_thr)

    pts_f = points_apartment[keep_mask]
    col_f = colors_255[keep_mask]

    info = dict(
        y_min=y_min,
        y_max=y_max,
        floor_threshold=y_floor_thr,
        ceiling_threshold=y_ceil_thr,
        kept_points=int(pts_f.shape[0]),
        removed_points=int(points_apartment.shape[0]-pts_f.shape[0])
    )
    return pts_f, col_f, info


def project_points_to_image(
    points: np.ndarray,
    colors_255: np.ndarray,
    target_longer_side: int = 2000,
    margin_px: int = 20,
    swap_axes: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    把世界座標 (apartment_0 frame) 投影成 2D 像素:
    swap_axes=True:
        img_x <- world z
        img_y <- world x  (倒過來讓y往下)
    swap_axes=False:
        img_x <- world x
        img_y <- world z
    回傳:
      px_coords: (N,2) float32  [px_x, px_y]
      rgb01:     (N,3) float32  in [0,1]
      transform_info: dict (for往返轉換)
    """
    x = points[:, 0]
    y = points[:, 1]  # 用不到投影，但可給排序
    z = points[:, 2]

    x_min, x_max = float(x.min()), float(x.max())
    z_min, z_max = float(z.min()), float(z.max())
    range_x = max(x_max - x_min, 1e-6)
    range_z = max(z_max - z_min, 1e-6)

    if swap_axes:
        world_w = range_z
        world_h = range_x
    else:
        world_w = range_x
        world_h = range_z

    longer_world = max(world_w, world_h)
    scale = (float(target_longer_side) - 2.0 *
             float(margin_px)) / float(longer_world)
    img_w = int(round(2*margin_px + world_w*scale))
    img_h = int(round(2*margin_px + world_h*scale))

    if swap_axes:
        px_x = margin_px + (z - z_min) * scale
        px_y = margin_px + (x_max - x) * scale  # flip vertical
    else:
        px_x = margin_px + (x - x_min) * scale
        px_y = margin_px + (z_max - z) * scale

    rgb01 = (colors_255.astype(np.float32)/255.0).clip(0, 1)

    transform = {
        "x_min": x_min,
        "x_max": x_max,
        "z_min": z_min,
        "z_max": z_max,
        "scale_px_per_world": float(scale),
        "margin_px": int(margin_px),
        "width": int(img_w),
        "height": int(img_h),
        "swap_axes": bool(swap_axes),
        "pixel_from_world": (
            "px_x = margin + (z-z_min)*scale ; px_y = margin + (x_max-x)*scale"
            if swap_axes else
            "px_x = margin + (x-x_min)*scale ; px_y = margin + (z_max-z)*scale"
        ),
        "world_from_pixel": (
            "z = z_min + (px_x-margin)/scale ; x = x_max - (px_y-margin)/scale"
            if swap_axes else
            "x = x_min + (px_x-margin)/scale ; z = z_max - (px_y-margin)/scale"
        )
    }

    # px_coords只需要2D
    px_coords = np.stack([px_x, px_y], axis=1).astype(np.float32)

    return px_coords, rgb01, transform


def draw_and_save_map(
    out_png: str,
    px_coords: np.ndarray,
    rgb01: np.ndarray,
    img_w: int,
    img_h: int,
    point_size: float = 2.0,
    dpi: int = 100,
    chunk_size: int = 40000,
):
    """
    繪製 scatter 到乾淨的 figure 並另存 map.png
    我們用 chunk 疊加，避免一次 scatter 幾十萬點太慢
    """
    # matplotlib.use("Agg")

    # 按 y 世界座標排序可以讓高處(天花板附近)覆蓋低處，
    # 但在這裡我們已經砍掉天花板/地板，所以就按順序畫
    n = px_coords.shape[0]
    idx = np.arange(n)

    fig_w_in = img_w / float(dpi)
    fig_h_in = img_h / float(dpi)
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    start = 0
    zorder_base = 2
    while start < n:
        end = min(start + chunk_size, n)
        ax.scatter(
            px_coords[idx[start:end], 0],
            px_coords[idx[start:end], 1],
            c=rgb01[idx[start:end]],
            s=point_size,
            linewidths=0,
            marker=",",
            rasterized=True,
            zorder=zorder_base,
        )
        start = end
        zorder_base += 1

    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.axis("off")

    fig.savefig(out_png, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def save_artifacts(
    out_dir: str,
    points_f: np.ndarray,
    colors_f: np.ndarray,
    px_coords: np.ndarray,
    transform: Dict,
    filter_info: Dict[str, float],
):
    """
    存 npz/csv 以及 map_transform.json
    """
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, "map_points.npz")
    csv_path = os.path.join(out_dir, "map_points.csv")
    tf_path = os.path.join(out_dir, "map_transform.json")

    np.savez_compressed(
        npz_path,
        x=points_f[:, 0].astype(np.float32),
        y=points_f[:, 1].astype(np.float32),
        z=points_f[:, 2].astype(np.float32),
        rgb255=colors_f.astype(np.uint8),
        px=px_coords[:, 0].astype(np.float32),
        py=px_coords[:, 1].astype(np.float32),
    )

    # 只輸出 2D 地圖上的 x,z 和 rgb
    xz = points_f[:, [0, 2]]
    rgb = colors_f.astype(np.uint8)
    csv_mat = np.concatenate([xz, rgb.astype(np.float32)], axis=1)
    header = "x,z,r,g,b"
    np.savetxt(
        csv_path,
        csv_mat,
        fmt=["%.3f", "%.3f", "%d", "%d", "%d"],
        delimiter=",",
        header=header,
        comments=""
    )

    tf_all = {
        "point_to_apartment_scale": 10000.0/255.0,
        "filter": filter_info,
        "transform": transform,
    }
    with open(tf_path, "w", encoding="utf-8") as f:
        json.dump(tf_all, f, indent=2)

    return npz_path, csv_path, tf_path


# -------------------------------
# Section 2. Occupancy grid + goal detection for a class
# -------------------------------

def pick_goal_for_class(
    class_name: str,
    class_to_rgb: Dict[str, Tuple[int, int, int]],
    rgb255: np.ndarray,
    px_coords: np.ndarray,
    goal_offset_px: float = 10.0,
) -> Tuple[float, float]:
    """
    找出屬於這個 semantic 類別的所有點，把它們平均後當作目標，
    再稍微往外偏移一下(目前先不做距離法線估計，簡化處理)。
    如果沒找到該類別，raise。
    """
    key = class_name.strip().lower()
    if key not in class_to_rgb:
        raise RuntimeError(f"class {class_name} not found in palette")

    target_rgb = np.array(class_to_rgb[key], dtype=np.uint8)  # (r,g,b)
    all_rgb = rgb255.astype(np.uint8)

    mask = np.all(all_rgb == target_rgb[None, :], axis=1)
    if not np.any(mask):
        raise RuntimeError(f"No points found for class '{class_name}'")

    pts_px = px_coords[mask]  # (M,2)
    cx, cy = pts_px.mean(axis=0)

    # 簡單做個位移：往下偏移一些畫素（模擬"在物件前面")
    goal = (float(cx), float(cy + goal_offset_px))
    return goal


# -------------------------------
# Section 3. RRT implementation
# -------------------------------

class RRTNode:
    def __init__(self, pt: Tuple[float, float], parent: Optional[int]):
        self.pt = pt
        self.parent = parent


def dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0]-b[0]
    dy = a[1]-b[1]
    return dx*dx+dy*dy


def steer(from_pt: Tuple[float, float], to_pt: Tuple[float, float], step: float) -> Tuple[float, float]:
    dx = to_pt[0]-from_pt[0]
    dy = to_pt[1]-from_pt[1]
    d = math.sqrt(dx*dx+dy*dy)
    if d <= step:
        return to_pt
    if d < 1e-6:
        return from_pt
    return (from_pt[0] + step*dx/d, from_pt[1] + step*dy/d)


def make_occupancy_grid(
    px_coords: np.ndarray,
    rgb255: np.ndarray,
    img_w: int,
    img_h: int,
    occ_thresh: int = 1,
    dilate_iters: int = 2,
) -> np.ndarray:
    """
    建立 2D 佔據格:
      obstacle[y,x] = True 代表不能走 (牆/家具)
      False 代表可走

    做法：
    1. 把每個投影點累加到像素格
    2. 任何像素累積 >= occ_thresh 就是障礙
    3. 對障礙做 binary dilation 把牆加厚，補掉稀疏點雲造成的小洞
    """
    occ = np.zeros((img_h, img_w), dtype=np.int32)

    xs = np.clip(np.rint(px_coords[:, 0]).astype(np.int32), 0, img_w - 1)
    ys = np.clip(np.rint(px_coords[:, 1]).astype(np.int32), 0, img_h - 1)

    for x, y in zip(xs, ys):
        occ[y, x] += 1

    # 初步障礙
    obstacle = (occ >= occ_thresh)

    # 形態學膨脹，讓牆壁更厚實、門縫保留但小洞補起來
    try:
        from scipy.ndimage import binary_dilation
        struct = np.ones((3, 3), dtype=bool)
        obstacle = binary_dilation(
            obstacle, structure=struct, iterations=dilate_iters)
    except Exception:
        # 沒有 scipy 時用簡易 inflate 當 fallback
        inflated = obstacle.copy()
        k = 1  # 3x3
        ys_arr = np.where(obstacle)[0]
        xs_arr = np.where(obstacle)[1]
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                yy = np.clip(ys_arr + dy, 0, img_h - 1)
                xx = np.clip(xs_arr + dx, 0, img_w - 1)
                inflated[yy, xx] = True
        obstacle = inflated

    # True=blocked, False=free
    return obstacle.astype(bool)


def collision_free_segment(
    a: Tuple[float, float],
    b: Tuple[float, float],
    obstacle: np.ndarray
) -> bool:
    """
    檢查 a->b 的線段是否穿過障礙（True 的格子）
    用逐像素採樣 (Bresenham-ish / dense sampling)
    """
    dist = math.dist(a, b)
    steps = int(math.ceil(dist))
    if steps < 1:
        steps = 1

    h, w = obstacle.shape
    for i in range(steps + 1):
        t = i / steps
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t
        xi = int(round(x))
        yi = int(round(y))

        # 出界就當成撞牆
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            return False
        # True 代表阻擋，不能通過
        if obstacle[yi, xi]:
            return False

    return True


def run_rrt(
    start_px: Tuple[float, float],
    goal_px: Tuple[float, float],
    obstacle: np.ndarray,
    step_size: float = 15.0,
    goal_sample_rate: float = 0.1,
    max_iter: int = 5000,
    goal_thresh: float = 50.0,
    rng_seed: int = 0,
) -> Optional[List[Tuple[float, float]]]:
    """
    RRT:
    - 隨機sample一個點(有機率直接sample goal以加速收斂)
    - 找最近的樹節點，往該sample方向step_size距離建立新節點
    - 確認線段無碰撞再加入
    - 若新節點離goal足夠近 => 成功
    回傳: path(從start到goal)，或 None
    """

    random.seed(rng_seed)
    h, w = obstacle.shape

    nodes: List[RRTNode] = [RRTNode(start_px, parent=None)]

    for it in range(max_iter):
        # 1. sample
        if random.random() < goal_sample_rate:
            sample = goal_px
        else:
            sample = (random.uniform(0, w-1), random.uniform(0, h-1))

        # 2. 最近節點
        dists = [dist2(n.pt, sample) for n in nodes]
        nearest_id = int(np.argmin(dists))
        nearest_pt = nodes[nearest_id].pt

        # 3. steer
        new_pt = steer(nearest_pt, sample, step_size)

        # 4. collision check
        if not collision_free_segment(nearest_pt, new_pt, obstacle):
            continue

        # 5. append
        nodes.append(RRTNode(new_pt, parent=nearest_id))
        new_id = len(nodes)-1

        # 6. goal check
        if math.dist(new_pt, goal_px) < goal_thresh:
            path_pts = [goal_px]
            cur = new_id
            while cur is not None:
                path_pts.append(nodes[cur].pt)
                cur = nodes[cur].parent
            path_pts.reverse()
            return path_pts

    return None


# -------------------------------
# Section 4. Coordinate conversion (pixel -> apartment_0 world)
# -------------------------------

def pixel_to_world(
    px: float,
    py: float,
    transform: Dict,
) -> Tuple[float, float, float]:
    """
    把像素座標(px,py) 轉回 apartment_0 的 (x,y,z).
    注意: 我們無法找回原本的 y(高度)，因為 2D map 是 top-down。
    這裡回傳 y=None 等價，或是用 floor_threshold 略估地面高度。
    簡單做法: 取 filter['floor_threshold'] 略當導航高度。
    """
    scale = float(transform["scale_px_per_world"])
    margin = float(transform["margin_px"])
    swap = bool(transform["swap_axes"])

    if swap:
        z_min = float(transform["z_min"])
        x_max = float(transform["x_max"])
        # invert formulas we used in project_points_to_image
        z = z_min + (px - margin)/scale
        x = x_max - (py - margin)/scale
    else:
        x_min = float(transform["x_min"])
        z_max = float(transform["z_max"])
        x = x_min + (px - margin)/scale
        z = z_max - (py - margin)/scale

    # y 高度：用 floor_threshold 當近似行走高度
    # (可根據需要改成固定常數或robot sensor高度)
    # 這裡 caller 可以覆寫
    y_approx = None
    return (x, y_approx, z)


def convert_path_pixel_to_world(
    path_px: List[Tuple[float, float]],
    tf_all: Dict
) -> List[Tuple[float, Optional[float], float]]:
    out = []
    transform = tf_all["transform"]
    for (px, py) in path_px:
        out.append(pixel_to_world(px, py, transform))
    return out


# -------------------------------
# Section 5. Interactive workflow glue
# -------------------------------

def interactive_pick_start(map_png: str) -> Tuple[float, float]:
    """
    開一個視窗顯示 map.png，讓使用者點一下選起點，回傳像素座標 (px,py)
    """
    img = plt.imread(map_png)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.set_title("Click to choose START. Close window after click.")
    pts = plt.ginput(1, timeout=0)  # wait for 1 click
    plt.close(fig)
    if len(pts) == 0:
        raise RuntimeError("No start point selected.")
    # ginput returns [(x,y)] in image pixel coords (origin top-left implied by imshow)
    return (float(pts[0][0]), float(pts[0][1]))


# ---- 修改後的 visualize_rrt_result（替換原有函式） ----
def visualize_rrt_result(
    map_png: str,
    start_px: Tuple[float, float],
    goal_px: Tuple[float, float],
    path_px: Optional[List[Tuple[float, float]]],
    out_path: str = "rrt_result.png",
    show: bool = True,
):
    """
    在 map 上疊加 start, goal, path，存檔並依情況用 plt.show() 顯示。
    show=True 時會嘗試顯示；若 backend 為 Agg（非互動）則不呼叫 show()。
    """
    img = plt.imread(map_png)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.scatter([start_px[0]], [start_px[1]], c="lime",
               s=80, marker="o", label="start")
    ax.scatter([goal_px[0]],  [goal_px[1]],  c="red",
               s=80, marker="x", label="goal")

    if path_px is not None and len(path_px) > 1:
        xs = [p[0] for p in path_px]
        ys = [p[1] for p in path_px]
        ax.plot(xs, ys, c="red", linewidth=2)

    ax.set_title("RRT path")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)

    # 嘗試顯示（若適合）
    try:
        backend = matplotlib.get_backend().lower()
        if show and not backend.startswith("agg"):
            # interactive backend available -> show window (blocks until closed)
            plt.show()
        elif show and backend.startswith("agg"):
            # 非互動環境
            print(f"[visualize_rrt_result] matplotlib backend '{backend}' (non-interactive). Saved result to {out_path} but won't call plt.show().")
    except Exception as e:
        print(f"[visualize_rrt_result] Warning when attempting to show figure: {e}")
    finally:
        plt.close(fig)



# -------------------------------
# Section 6. Main routine
# -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build 2D semantic map and run RRT navigation"
    )
    parser.add_argument("--data-dir", default="semantic_3d_pointcloud")
    parser.add_argument(
        "--palette-xlsx", default="color_coding_semantic_segmentation_classes.xlsx")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--class-name", default="rack",
                        help="Target category, e.g. rack / cushion / sofa / stair / cooktop")
    parser.add_argument("--skip-build-map", action="store_true",
                        help="If set, assume map.png and map_transform.json already exist, just run RRT.")
    parser.add_argument("--no-interactive", action="store_true",
                        help="If set, won't pop up ginput(). We'll approximate start at middle of map.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ===== Task 1: Build map (unless skipped)
    if not args.skip_build_map:
        points_apartment, colors_255 = load_points_and_colors(args.data_dir)

        # remove floor/ceiling
        # tweak frac so that we aggressively slice out bottom and top clutter
        points_f, colors_f, filter_info = filter_floor_ceiling(
            points_apartment,
            colors_255,
            floor_frac=0.25,
            ceiling_frac=0.4,
            floor_abs=None,
            ceiling_abs=None,
            keep_floor=False,
            keep_ceiling=False,
        )

        # project to 2D pixel
        px_coords, rgb01, transform = project_points_to_image(
            points_f, colors_f,
            target_longer_side=2000,
            margin_px=20,
            swap_axes=True
        )

        # draw map.png
        map_png = os.path.join(args.out_dir, "map.png")
        draw_and_save_map(
            map_png,
            px_coords,
            rgb01,
            img_w=transform["width"],
            img_h=transform["height"],
            point_size=25.0,
            dpi=100,
            chunk_size=40000,
        )

        # save transform + data
        npz_path, csv_path, tf_path = save_artifacts(
            args.out_dir,
            points_f, colors_f,
            px_coords, transform,
            filter_info
        )

        print("[Build Map] wrote:")
        print("  map.png           :", map_png)
        print("  map_points.npz    :", npz_path)
        print("  map_points.csv    :", csv_path)
        print("  map_transform.json:", tf_path)
    else:
        print("[Skip Build Map] Using existing map.png and map_transform.json")

    # ===== Load artifacts for Task 2 / RRT
    # reload npz so we are consistent (also works when --skip-build-map)
    npz_loaded = np.load(os.path.join(args.out_dir, "map_points.npz"))
    px_coords_all = np.stack(
        [npz_loaded["px"], npz_loaded["py"]], axis=1)  # (N,2)
    rgb255_all = npz_loaded["rgb255"]

    with open(os.path.join(args.out_dir, "map_transform.json"), "r", encoding="utf-8") as f:
        tf_all = json.load(f)

    map_png_path = os.path.join(args.out_dir, "map.png")
    img_h = int(tf_all["transform"]["height"])
    img_w = int(tf_all["transform"]["width"])

    # ===== Build occupancy
    obstacle_grid = make_occupancy_grid(
        px_coords_all,
        rgb255_all,
        img_w,
        img_h,
        dilate_iters=3,
        occ_thresh=1
    )
    print("[Occupancy] grid size:", obstacle_grid.shape, "(h,w)")

    # ===== Figure out GOAL from class-name
    palette_map = load_palette_from_excel(args.palette_xlsx)
    goal_px = pick_goal_for_class(
        args.class_name,
        palette_map,
        rgb255_all,
        px_coords_all,
        goal_offset_px=10.0
    )
    print(f"[Goal] for class '{args.class_name}': pixel {goal_px}")

    # ===== Pick START (interactive or fallback)
    if not args.no_interactive:
        print("Showing map for start-point selection...")
        start_px = interactive_pick_start(map_png_path)
    else:
        # fallback: center-ish
        start_px = (img_w*0.5, img_h*0.5)
    print("[Start] pixel:", start_px)

    # ===== Run RRT
    path_px = run_rrt(
        start_px=start_px,
        goal_px=goal_px,
        obstacle=obstacle_grid,
        step_size=100.0,
        goal_sample_rate=0.2,
        max_iter=50000,
        goal_thresh=30.0,
        rng_seed=0,
    )

    if path_px is None:
        print("[RRT] Failed to find a path.")
    else:
        print(f"[RRT] Path length {len(path_px)} waypoints.")
        # ===== Visualize + save
        result_png = os.path.join(args.out_dir, "rrt_result.png")
        visualize_rrt_result(
            map_png_path,
            start_px,
            goal_px,
            path_px,
            out_path=result_png,
            show=True
        )
        print("  rrt_result.png:", result_png)

        # ===== Convert pixel path -> apartment_0 world coords
        world_path = convert_path_pixel_to_world(path_px, tf_all)
        print("[RRT] Waypoints in pixel coords:")
        for i, p in enumerate(path_px):
            print(f"  {i}: (px_x={p[0]:.2f}, px_y={p[1]:.2f})")

        print("[RRT] Waypoints converted to apartment_0 coords (x,~,z):")
        for i, pw in enumerate(world_path):
            x, y_approx, z = pw
            print(f"  {i}: x={x:.3f}, z={z:.3f}, y≈{y_approx}")
            # ===== Save navigation plan for Habitat (Part 3)
        # We'll only keep x/z since Habitat's Y is "up".
        # We'll also store the target/category name so load.py can name the video.
        nav_plan = {
            "target_name": args.class_name,
            "waypoints_world": [
                {"x": float(x), "z": float(z)}
                for (x, y_approx, z) in world_path
            ]
        }

        nav_plan_path = os.path.join(
            args.out_dir,
            f"{args.class_name}_nav_plan.json"
        )
        with open(nav_plan_path, "w", encoding="utf-8") as f:
            json.dump(nav_plan, f, indent=2)

        print(f"[NavPlan] wrote navigation plan -> {nav_plan_path}")

    print("Done.")


if __name__ == "__main__":
    main()
