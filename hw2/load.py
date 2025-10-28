import argparse
import json
import math
import os
from typing import List, Dict, Tuple

import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
from PIL import Image


###############################################################################
# 0. Scene / sim defaults
###############################################################################

# NOTE: you already set this path before; keep it pointing at apartment_0.
TEST_SCENE = "replica_v1/apartment_0/habitat/mesh_semantic.ply"

SIM_SETTINGS = {
    "scene": TEST_SCENE,
    "default_agent": 0,
    "sensor_height": 1.5,   # meters
    "width": 512,
    "height": 512,
    "sensor_pitch": 0.0,    # radians (x-rotation down/up)
}

# How far we move per "move_forward"
FORWARD_STEP_M = 0.25
# How many degrees per "turn_left"/"turn_right"
TURN_DEG = 10.0
TURN_RAD = math.radians(TURN_DEG)

# How close (meters) we consider "at a waypoint"
WAYPOINT_REACHED_DIST = 0.80

# FPS for the output mp4
VIDEO_FPS = 8


###############################################################################
# 1. Basic frame transforms / visualization helpers
###############################################################################

def transform_rgb_bgr(rgb_img: np.ndarray) -> np.ndarray:
    """
    Habitat returns COLOR obs as uint8 RGB.
    We want BGR for OpenCV.
    """
    return rgb_img[:, :, [2, 1, 0]]


def transform_depth(depth_img_meters: np.ndarray) -> np.ndarray:
    """
    Just for debugging if you still want to view depth.
    Scales depth to 0~255 for visualization.
    """
    depth_vis = (depth_img_meters / 10.0 * 255.0).clip(0, 255).astype(np.uint8)
    return depth_vis


def transform_semantic(semantic_obs: np.ndarray) -> np.ndarray:
    """
    Convert semantic IDs -> pretty color palette (like the starter code).
    Returned frame is BGR for cv2.imshow if needed.
    """
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_bgr = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_bgr


def overlay_target_mask_bgr(
    bgr_frame: np.ndarray,
    semantic_frame_raw: np.ndarray,
    target_ids: List[int],
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Requirement #2:
    "highlight the target with a red transparent mask while navigating"
    We create a red overlay ONLY where semantic == one of target_ids.

    - bgr_frame:     (H,W,3) uint8
    - semantic_frame_raw: (H,W) int semantic id per pixel
    - target_ids: list of ints to highlight (e.g. all instance IDs of 'rack')
    - alpha: transparency strength of the red mask
    """
    mask = np.isin(semantic_frame_raw, target_ids)

    # Build an overlay that's the same as original except target pixels are red (BGR: [0,0,255])
    overlay = bgr_frame.copy()
    overlay[mask] = np.array([0, 0, 255], dtype=np.uint8)  # pure red in BGR

    # Blend overlay (with red) and original to get transparent red
    blended = cv2.addWeighted(overlay, alpha, bgr_frame, 1.0 - alpha, 0)
    return blended


###############################################################################
# 2. Simulator configuration
###############################################################################

def make_simple_cfg(settings: Dict) -> habitat_sim.Configuration:
    """
    Build a Habitat-Sim config:
    - scene
    - 1 agent
    - attach RGB, depth, semantic sensors
    - OVERRIDE the action space so that
      move_forward = 0.25 m,
      turn_left/right = 10 deg each step,
      which matches Part 3 requirement.
    """
    # ---- simulator backend ----
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    # ---- agent ----
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # RGB sensor
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [settings["sensor_pitch"], 0.0, 0.0]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # Depth sensor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [settings["sensor_pitch"], 0.0, 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # Semantic sensor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [settings["sensor_pitch"], 0.0, 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # attach sensors
    agent_cfg.sensor_specifications = [
        rgb_sensor_spec,
        depth_sensor_spec,
        semantic_sensor_spec,
    ]

    # >>> Part 3 requirement: define motion primitives <<<
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward",
            habitat_sim.agent.ActuationSpec(amount=FORWARD_STEP_M),
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left",
            habitat_sim.agent.ActuationSpec(amount=TURN_DEG),
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right",
            habitat_sim.agent.ActuationSpec(amount=TURN_DEG),
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


###############################################################################
# 3. Navigation math helpers
###############################################################################

def quat_to_yaw(agent_rot) -> float:
    """
    從 agent_state.rotation (四元數) 抽取 heading yaw (以 +Y 為轉軸).
    """
    w = agent_rot.w
    x = agent_rot.x
    y = agent_rot.y
    z = agent_rot.z

    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + x * x)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw

def normalize_angle(angle: float) -> float:
    """
    Wrap to [-pi, pi]
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def compute_desired_yaw(from_pos: np.ndarray, to_pos: np.ndarray) -> float:
    """
    We want the agent to face 'to_pos' using Habitat's yaw convention.

    Habitat's default forward direction for yaw=0 is along -Z.
    After yaw=psi, forward points along [-sin(psi), 0, -cos(psi)].

    Given delta = to_pos - from_pos = (dx, dy, dz), we want that forward vector
    aligns with (dx, dz).  Solve:
      forward = (-sin psi, -cos psi)  ~  (dx, dz)
    => sin psi = -dx / L, cos psi = -dz / L
    => psi = atan2(-dx, -dz)

    We're ignoring y because navigation is planar.
    """
    dx = to_pos[0] - from_pos[0]
    dz = to_pos[2] - from_pos[2]
    desired_yaw = math.atan2(-dx, -dz)
    return desired_yaw


def turn_towards(sim, agent, yaw_target: float):
    """
    Spin in-place with discrete 10° left/right until we face yaw_target.

    We'll:
    - read current yaw
    - compute shortest signed delta
    - if delta>0: step 'turn_left'
      else      : step 'turn_right'
    - repeat until |delta| < TURN_RAD/2 (~5 deg)

    Every step we also capture + log frame (for video).
    We'll NOT move forward here.
    """
    while True:
        agent_state = agent.get_state()
        cur_yaw = quat_to_yaw(agent_state.rotation)
        delta = normalize_angle(yaw_target - cur_yaw)
        if abs(delta) <= TURN_RAD / 2.0:
            break  # we are facing the waypoint closely enough

        action = "turn_left" if delta > 0.0 else "turn_right"
        yield action  # tell caller "execute one turn step"


def walk_forward_towards(sim, agent, goal_pos: np.ndarray):
    """
    After we're roughly facing the waypoint,
    repeatedly issue "move_forward" until we are within WAYPOINT_REACHED_DIST.

    We re-check distance every step, and also yield the action names out.
    """
    while True:
        agent_state = agent.get_state()
        cur_pos = agent_state.position  # np.ndarray [x,y,z]
        dist = math.sqrt(
            (goal_pos[0] - cur_pos[0]) ** 2 +
            (goal_pos[2] - cur_pos[2]) ** 2
        )
        if dist <= WAYPOINT_REACHED_DIST:
            break

        # take one small forward step
        yield "move_forward"


###############################################################################
# 4. Core rollout: follow the full RRT path
###############################################################################
def run_navigation_and_record(
    sim: habitat_sim.Simulator,
    agent: habitat_sim.agent.Agent,
    waypoints_world: List[Dict[str, float]],
    target_name: str,
    target_semantic_ids: List[int],
    video_out_dir: str = ".",
):
    """
    新策略（閉迴路控制）：
    我們維護一個 waypoint index = cur_idx。
    每一個 sim step:
      1. 取得目前 agent 位置與朝向 (yaw)。
      2. 檢查目前距離當前 waypoint 有多近，如果很近就切到下一個 waypoint。
      3. 算出理想朝向 (desired_yaw) 指向該 waypoint。
      4. 算 yaw_error = desired_yaw - current_yaw，整理到 [-pi,pi] 範圍。
         - 如果 |yaw_error| > 轉向容忍角，就 turn_left / turn_right (10度) 一步
         - 否則 move_forward (0.25m) 一步
      5. 每執行一步，就擷取 RGB/semantic -> 疊紅色半透明遮罩 -> 寫進 mp4
    持續到最後一個 waypoint 也到達為止。
    """

    assert len(waypoints_world) >= 1, "Need at least one waypoint."

    # 0. 把 agent 放到第一個 waypoint (起點)
    first_wp = waypoints_world[0]
    init_state = habitat_sim.AgentState()
    init_state.position = np.array([first_wp["x"], 0.0, first_wp["z"]], dtype=np.float32)
    agent.set_state(init_state)

    # 視訊 writer
    H = SIM_SETTINGS["height"]
    W = SIM_SETTINGS["width"]
    out_path = os.path.join(video_out_dir, f"{target_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, VIDEO_FPS, (W, H))
    print(f"[Recorder] Writing video to: {out_path}")

    # 參數
    YAW_TOL_RAD = TURN_RAD / 2.0  # 例如 ~5度，允許的朝向誤差
    MAX_STEPS = 2000            # 安全保險，避免無限迴圈
    cur_idx = 0                  # 目前在追的 waypoint index

    def step_and_record(action_name: str):
        """
        執行一步 Habitat action，存一幀影片（含紅色遮罩）。
        """
        observations = sim.step(action_name)

        rgb_bgr = transform_rgb_bgr(observations["color_sensor"])
        semantic_raw = observations["semantic_sensor"]

        # 疊紅色半透明 mask
        rgb_bgr_hl = overlay_target_mask_bgr(
            rgb_bgr,
            semantic_raw,
            target_semantic_ids,
            alpha=0.4,
        )

        writer.write(rgb_bgr_hl)

        # 可選 debug
        agent_state_dbg = agent.get_state()
        sensor_state_dbg = agent_state_dbg.sensor_states["color_sensor"]
        print(
            "pose:",
            float(sensor_state_dbg.position[0]),
            float(sensor_state_dbg.position[1]),
            float(sensor_state_dbg.position[2]),
            float(sensor_state_dbg.rotation.w),
            float(sensor_state_dbg.rotation.x),
            float(sensor_state_dbg.rotation.y),
            float(sensor_state_dbg.rotation.z),
            "| action:", action_name,
            "| chasing wp", cur_idx
        )

    # 主迴圈
    steps = 0
    while steps < MAX_STEPS:
        steps += 1

        # 如果我們已經在最後一個 waypoint 附近，就結束
        if cur_idx >= len(waypoints_world):
            print("[Nav] Reached final waypoint list end.")
            break

        # 1. 拿 agent 現在的位置 / 朝向
        agent_state = agent.get_state()
        cur_pos = agent_state.position  # np.array([x,y,z])
        cur_yaw = quat_to_yaw(agent_state.rotation)

        # 2. 取得我們目前要走向的 waypoint
        goal_wp = waypoints_world[cur_idx]
        goal_pos = np.array([goal_wp["x"], 0.0, goal_wp["z"]], dtype=np.float32)

        #   檢查是不是已經很接近這個 waypoint
        dist_xz = math.sqrt(
            (goal_pos[0] - cur_pos[0]) ** 2 +
            (goal_pos[2] - cur_pos[2]) ** 2
        )
        if dist_xz <= WAYPOINT_REACHED_DIST:
            # 到了這個 waypoint，換下一個
            print(f"[Nav] Waypoint {cur_idx} reached (dist={dist_xz:.3f}m).")
            cur_idx += 1

            # 如果剛好已經是最後一個也到達了，就可以在下一輪跳出
            continue

        # 3. 算理想朝向（希望朝向下一個 waypoint）
        desired_yaw = compute_desired_yaw(cur_pos, goal_pos)

        # 4. 算偏差角
        yaw_error = normalize_angle(desired_yaw - cur_yaw)

        # 5. 決定這一小步要做什麼
        #    如果偏很多，先轉，不急著走
        if abs(yaw_error) > YAW_TOL_RAD:
            action_name = "turn_left" if yaw_error > 0.0 else "turn_right"
            step_and_record(action_name)
            continue

        #    方向OK了，往前走一格
        action_name = "move_forward"
        step_and_record(action_name)

    # loop 結束後再抓一張最後畫面(可選)
    obs_final = sim.get_sensor_observations()
    rgb_bgr = transform_rgb_bgr(obs_final["color_sensor"])
    semantic_raw = obs_final["semantic_sensor"]
    rgb_bgr_hl = overlay_target_mask_bgr(
        rgb_bgr,
        semantic_raw,
        target_semantic_ids,
        alpha=0.4,
    )
    writer.write(rgb_bgr_hl)

    writer.release()
    print("[Recorder] Done, video closed.")

###############################################################################
# 5. Utility: figure out which semantic IDs to highlight
###############################################################################

def infer_target_ids(sim: habitat_sim.Simulator, agent, n_samples: int = 5) -> List[int]:
    """
    We don't actually know from code alone which integer semantic IDs
    correspond to your desired category (e.g. 'rack') in Replica.
    Replica's semantic sensor returns per-object instance IDs, not names.

    Here's a helper you can *temporarily* call at the start of a run:
    we'll take a few random spins, look at semantic_sensor,
    and print the unique IDs. Then you can manually map your category.

    In practice you should build a dictionary like:
        SEMANTIC_ID_MAP = {
            "rack": [17, 42],   # all instance IDs for racks
            "sofa": [5],
            ...
        }
    and just return SEMANTIC_ID_MAP[target_name].

    For now, we leave this function here documented and
    we'll *not* call it automatically to avoid messing up nav.
    """
    ids = set()
    for _ in range(n_samples):
        obs = sim.step("turn_left")
        sem = obs["semantic_sensor"]
        ids |= set(np.unique(sem).tolist())
    print("[Debug] unique semantic IDs seen:", sorted(list(ids)))
    return list(ids)


###############################################################################
# 6. Main entry point
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous navigation along RRT path + video logging"
    )
    parser.add_argument(
        "--plan-json",
        required=True,
        help="Path to the {target_name}_nav_plan.json dumped by main.py"
    )
    parser.add_argument(
        "--video-dir",
        default=".",
        help="Where to save {target_name}.mp4"
    )
    args = parser.parse_args()

    # ---------------------------------
    # Load navigation plan from Part 2
    # ---------------------------------
    with open(args.plan_json, "r", encoding="utf-8") as f:
        plan = json.load(f)

    target_name = plan["target_name"]
    waypoints_world = plan["waypoints_world"]
    print("[Plan] target_name:", target_name)
    print("[Plan] num waypoints:", len(waypoints_world))
    # (optional) print them:
    for i, wp in enumerate(waypoints_world):
        print(f"  {i}: x={wp['x']:.3f}, z={wp['z']:.3f}")

    # ------------------------------------------------
    # Bring up the simulator and initialize the agent
    # ------------------------------------------------
    cfg = make_simple_cfg(SIM_SETTINGS)
    sim = habitat_sim.Simulator(cfg)

    agent = sim.initialize_agent(SIM_SETTINGS["default_agent"])

    # We'll still print the discrete action space for sanity.
    action_names = list(cfg.agents[SIM_SETTINGS["default_agent"]].action_space.keys())
    print("Discrete action space:", action_names)

    # ------------------------------------------------
    # Figure out which semantic IDs to highlight in red
    # ------------------------------------------------
    #
    # IMPORTANT:
    #   You MUST map `target_name` (e.g. "rack") to the semantic IDs
    #   of that category in Replica. This depends on the dataset labels.
    #
    #   For now we create a placeholder dict you should fill
    #   once you inspect the IDs in your scene.
    #
    SEMANTIC_ID_MAP = {
        # "rack": [17, 42],  # <-- example only; you must fix these
        # "sofa": [5],
        # ...
    }
    target_semantic_ids = SEMANTIC_ID_MAP.get(target_name, [])
    if len(target_semantic_ids) == 0:
        print(
            f"[WARN] No semantic IDs configured for '{target_name}'. "
            "The red mask overlay will just do nothing."
        )

    # ------------------------------------------------
    # Run nav + record video
    # ------------------------------------------------
    run_navigation_and_record(
        sim=sim,
        agent=agent,
        waypoints_world=waypoints_world,
        target_name=target_name,
        target_semantic_ids=target_semantic_ids,
        video_out_dir=args.video_dir,
    )

    print("[Done] Navigation complete.")


if __name__ == "__main__":
    main()
