import argparse, time, os, json
import numpy as np
import math as m
from scipy.spatial.transform import Rotation as R
from scipy.linalg import pinv

# for simulator
import pybullet as p
import pybullet_data

# for geometry information
from hw3_utils.bullet_utils import draw_coordinate, get_matrix_from_pose, get_pose_from_matrix, pose_7d_to_6d, pose_6d_to_7d

# you may use your forward kinematic algorithm to compute 
from fk import your_fk, get_ur5_DH_params

SIM_TIMESTEP = 1.0 / 240.0
TASK2_SCORE_MAX = 40
IK_ERROR_THRESH = 0.02

def cross(a : np.ndarray, b : np.ndarray) -> np.ndarray :
    return np.cross(a, b)

# this is the pybullet version
def pybullet_ik(robot_id, new_pose : list or tuple or np.ndarray, 
                max_iters : int=1000, stop_thresh : float=.001, base_pos=None):
    
    new_pos, new_rot = new_pose[:3], new_pose[3:]

    joint_poses = p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=10,
        targetPosition=new_pos,
        targetOrientation=new_rot,
        lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
        upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
        jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
        restPoses=np.float32(np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi).tolist(),
        maxNumIterations=max_iters,
        residualThreshold=stop_thresh)
    joint_poses = np.array(joint_poses)

    
    return joint_poses


def your_ik(robot_id, new_pose : list or tuple or np.ndarray, 
                base_pos=None, max_iters : int=1000, stop_thresh : float=.001):


    joint_limits = np.asarray([
            [-3*np.pi/2, -np.pi/2], # joint1
            [-2.3562, -1],           # joint2
            [-17, 17],               # joint3
            [-17, 17],               # joint4
            [-17, 17],               # joint5
            [-17, 17],               # joint6
        ])

    # get current joint angles for the 6 revolute joints (robust across URDFs)
    n_joints = p.getNumJoints(robot_id)
    joint_infos = [p.getJointInfo(robot_id, i) for i in range(n_joints)]
    revolute_ids = [j[0] for j in joint_infos if j[2] == p.JOINT_REVOLUTE]
    if len(revolute_ids) < 6:
        # fallback: assume first 6 joints form the arm
        revolute_ids = list(range(6))
    q_states = p.getJointStates(robot_id, revolute_ids)
    q = np.asarray([x[0] for x in q_states], dtype=float)  # 6-DoF

    # Target pose parsing
    target_pose = np.asarray(new_pose, dtype=float)
    assert target_pose.shape[0] == 7, "new_pose must be 7D: [x,y,z,qx,qy,qz,qw]"
    tgt_pos = target_pose[:3]
    tgt_quat = target_pose[3:]

    # FK model
    DH = get_ur5_DH_params()

    # Hyperparameters
    alpha = 0.4                 # step size for joint update
    damping = 1e-3              # damping factor for DLS pseudoinverse
    w_pos = 1.0                 # weight for positional error
    w_ori = 1.0                 # weight for orientation error
    max_step = 0.2              # limit per-iteration joint update (rad)

    # Iterative IK using Jacobian pseudo-inverse (damped least squares)
    for _ in range(max_iters):
        # determine base pose if not provided
        if base_pos is None:
            base_ref, _ = p.getBasePositionAndOrientation(robot_id)
        else:
            base_ref = base_pos

        cur_pose, J = your_fk(DH, q, base_ref)

        cur_pos = cur_pose[:3]
        cur_quat = cur_pose[3:]

        # Position error (in world frame)
        e_pos = tgt_pos - cur_pos

        # Orientation error via rotation vector (current -> target)
        # Ensure shortest quaternion path by flipping sign if needed
        cq = np.array(cur_quat, dtype=float)
        tq = np.array(tgt_quat, dtype=float)
        if np.dot(cq, tq) < 0.0:
            tq = -tq
        R_err = R.from_quat(tq) * R.from_quat(cq).inv()
        e_ori = R_err.as_rotvec()

        # Stack 6D task-space error
        e = np.hstack([w_pos * e_pos, w_ori * e_ori])

        # Stopping criterion on combined error
        if np.linalg.norm(e, ord=2) < stop_thresh:
            break

        # Damped least squares pseudoinverse: J^T (J J^T + λ^2 I)^{-1}
        JJt = J @ J.T
        lamb2I = (damping ** 2) * np.eye(JJt.shape[0])
        J_pinv = J.T @ np.linalg.inv(JJt + lamb2I)

        dq = alpha * (J_pinv @ e)

        # Limit step size to avoid instability
        dq = np.clip(dq, -max_step, max_step)

        # Update and clamp to joint limits
        q = q + dq
        q = np.clip(q, joint_limits[:, 0], joint_limits[:, 1])

    return list(q) # 6 DoF


# TODO: [for your information]
# This function is the scoring function, we will use the same code 
# to score your algorithm using all the testcases
def score_ik(robot, testcase_files : str, visualize : bool=False):

    testcase_file_num = len(testcase_files)
    ik_score = [TASK2_SCORE_MAX / testcase_file_num for _ in range(testcase_file_num)]
    ik_error_cnt = [0 for _ in range(testcase_file_num)]


    p.addUserDebugText(text = "Scoring Your Inverse Kinematic Algorithm ...", 
                        textPosition = [0.1, -0.6, 1.5],
                        textColorRGB = [1,1,1],
                        textSize = 1.0,
                        lifeTime = 0)

    print("============================ Task 2 : Inverse Kinematic ============================\n")
    for file_id, testcase_file in enumerate(testcase_files):

        f_in = open(testcase_file, 'r')
        ik_dict = json.load(f_in)
        f_in.close()
        
        test_case_name = os.path.split(testcase_file)[-1]

        poses = ik_dict['next_poses']
        cases_num = len(ik_dict['current_joint_poses'])
        
        penalty = (TASK2_SCORE_MAX / testcase_file_num) / (0.3 * cases_num)
        ik_errors = []

        for i in range(cases_num):

            # TODO: check your default arguments of `max_iters` and `stop_thresh` are your best parameters.
            #       We will only pass default arguments of your `max_iters` and `stop_thresh`.
            your_joint_poses = your_ik(robot.robot_id, poses[i], base_pos=robot._base_position) 
            

            # You can use `pybullet_ik` to see the correct version 
            # your_joint_poses = pybullet_ik(robot.robot_id, poses[i]) 

            gt_pose = poses[i]        

            p.setJointMotorControlArray(bodyUniqueId=robot.robot_id,
                                        jointIndices=robot._joint_name_to_ids.values(),
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=your_joint_poses,
                                        positionGains=[0.2] * len(your_joint_poses),
                                        velocityGains=[1] * len(your_joint_poses),
                                        physicsClientId=robot._physics_client_id)
            
            # warmup for 0.1 sec
            for _ in range(int(1 / SIM_TIMESTEP * 0.1)):
                p.stepSimulation()
                time.sleep(SIM_TIMESTEP)


            your_pose = robot.get_eef_pose()

            if visualize:
                color_yours = [[1,0,0], [1,0,0], [1,0,0]]
                color_gt = [[0,1,0], [0,1,0], [0,1,0]]
                draw_coordinate(your_pose, size=0.01, color=color_yours)
                draw_coordinate(gt_pose, size=0.01, color=color_gt)

            ik_error = np.linalg.norm(your_pose - np.asarray(gt_pose), ord=2)
            ik_errors.append(ik_error)
            if ik_error > IK_ERROR_THRESH:
                ik_score[file_id] -= penalty
                ik_error_cnt[file_id] += 1

        ik_score[file_id] = 0.0 if ik_score[file_id] < 0.0 else ik_score[file_id]
        ik_errors = np.asarray(ik_errors)

        score_msg = "- Testcase file : {}\n".format(test_case_name) + \
                    "- Mean Error : {:0.06f}\n".format(np.mean(ik_errors)) + \
                    "- Error Count : {:3d} / {:3d}\n".format(ik_error_cnt[file_id], cases_num) + \
                    "- Your Score Of Inverse Kinematic : {:00.03f} / {:00.03f}\n".format(
                            ik_score[file_id], TASK2_SCORE_MAX / testcase_file_num)
        
        print(score_msg)
    p.removeAllUserDebugItems()

    total_ik_score = 0.0
    for file_id in range(testcase_file_num):
        total_ik_score += ik_score[file_id]
    
    print("====================================================================================")
    print("- Your Total Score : {:00.03f} / {:00.03f}".format(total_ik_score , TASK2_SCORE_MAX))
    print("====================================================================================")

def main(args):

    # ------------------------ #
    # --- Setup simulation --- #
    # ------------------------ #

    # Create pybullet GUI
    physics_client_id = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=90,
        cameraPitch=0,
        cameraTargetPosition=[0.5, 0.0, 1.0]
    )
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(SIM_TIMESTEP)
    p.setGravity(0, 0, -9.8)
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [0.9, 0.0, 0.0])

    # ------------------- #
    # --- Setup robot --- #
    # ------------------- #

    # goto initial pose
    from pybullet_robot_envs.envs.ur5_envs.ur5_env import ur5Env
    robot = ur5Env(physics_client_id, use_IK=1)

    # -------------------------------------------- #
    # --- Test your Forward Kinematic function --- #
    # -------------------------------------------- #
    
    # warmup for 1 sec
    for _ in range(int(1 / SIM_TIMESTEP * 1)):
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)

    # ------------------------------------------------------------------ #
    # --- Test your Inverse Kinematic function using one target pose --- #
    # ------------------------------------------------------------------ #
    
    # warmup for 2 secs
    p.addUserDebugText(text = "Warmup for 2 secs ...", 
                        textPosition = [0.1, -0.6, 1.5],
                        textColorRGB = [1,1,1],
                        textSize = 1.0,
                        lifeTime = 0)
    for _ in range(int(1 / SIM_TIMESTEP * 2)):
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)
    p.removeAllUserDebugItems()

    # test your ik solver
    testcase_files = [
        'test_case/ik_test_case_easy.json',
        'test_case/ik_test_case_medium.json',
        'test_case/ik_test_case_hard.json',
        # 'test_case/ik_test_case_ta1.json',
        # 'test_case/ik_test_case_ta2.json',
    ]

    # ------------------------------------------------------------- #
    # --- Test your Inverse Kinematic function using test cases --- #
    # ------------------------------------------------------------- #

    # scoring your algorithm
    score_ik(robot, testcase_files, visualize=args.visualize_pose)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize-pose', '-vp', action='store_true', default=False, help='whether show the poses of end effector')
    args = parser.parse_args()
    main(args)
