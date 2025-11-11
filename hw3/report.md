# Homework 3: A Robot Manipulation Framework

## 1. Task 1 — Forward kinematics

### 1.1 Briefly explain how you implement your_fk() function

* **DH convention used:** classic D-H (frame transform: rotate about z by $θ$, translate along z by $d$, translate along x by $a$, rotate about x by $α$).
* **Compute transforms:** build each joint transform $T_i = R_z(\theta_i),T_z(d_i),T_x(a_i),R_x(\alpha_i)$. Multiply sequentially to get the end-effector transform in world frame.
* **End-effector pose:** extract position (p_e) and orientation (converted to quaternion in (x,y,z,w) format).
* **Jacobian:** geometric Jacobian for 6 revolute joints — store each joint origin (o_i) and joint axis (z_i) in world frame while accumulating transforms; then for joint (i)

  * linear part (J_v^i = z_i \times (p_e - o_i))
  * angular part (J_\omega^i = z_i)
* **Small adjustment:** alignment matrix (provided in template) applied to match simulator convention before returning 7D pose.

Key code excerpt (from `fk.py`):

```python
# build elementary transforms (classic DH)
T_i = rot_z(theta_i) @ trans_z(d_i) @ trans_x(a_i) @ rot_x(alpha_i)
T = T @ T_i
# record origin and z-axis in world frame
origins.append(T[:3,3].copy()); z_axes.append(T[:3,2].copy())
# after loop: compute Jacobian
p_e = origins[-1]
for i in range(6):
    z = z_axes[i]; p_i = origins[i]
    Jv = np.cross(z, (p_e - p_i)); Jw = z
    jacobian[:3,i] = Jv; jacobian[3:,i] = Jw
```

(implementation in `fk.py`). 

### 1.2 What is the difference between D-H convention and Craig’s convention (Modified D-H Conveition)?

* **Classic D-H (used here):** each link transform is (A_i = R_z(\theta_i),T_z(d_i),T_x(a_i),R_x(\alpha_i)). Joint parameters are associated with (z_{i-1}) axes; frames are placed according to the classical recipe.
* **Modified D-H (Craig’s):** transforms are re-arranged: (A_i = R_x(\alpha_{i-1}),T_x(a_{i-1}),R_z(\theta_i),T_z(d_i)). The main difference is where the rotations/translations are applied (order changes) and therefore frame placement changes — modified D-H often produces frame definitions that are more convenient for some robot URDFs but must be used consistently.

### 1.3 Complete the D-H table in your report following D-H convention

I used the DH parameters provided in the template (`get_ur5_DH_params()` in `fk.py`):

| Joint (i) | a (m)   | d (m)   | α (rad) | θ (variable) |
| --------- | ------- | ------- | ------- | ------------ |
| 1         | 0.0000  | 0.08920 | +π/2    | θ₁           |
| 2         | -0.4250 | 0.00000 | 0       | θ₂           |
| 3         | -0.3920 | 0.00000 | 0       | θ₃           |
| 4         | 0.0000  | 0.10930 | +π/2    | θ₄           |
| 5         | 0.0000  | 0.09475 | -π/2    | θ₅           |
| 6         | 0.0000  | 0.20230 | 0       | θ₆           |
| 7 (tool)  | 0.0000  | 0.00000 | 0       | 0 (fixed)    |

(These are taken from `get_ur5_DH_params()` in the template.) 

---

## 2. Task 2 — Inverse kinematics (10% + 5% bonus)

### 2.1 Implementation summary (`your_ik()`)

* **Approach:** Iterative Jacobian-based IK with *damped least squares* (DLS) pseudo-inverse.
* **Error term:** 6D task error composed of position error and orientation error. Orientation error is computed as the rotation vector from current orientation to target:

  * ensure quaternion signs are consistent (flip sign if dot<0) then compute `R_err = R(tgt)*R(cur).inv()` and `e_ori = R_err.as_rotvec()`.
* **Pseudo-inverse (DLS):** compute (J^+ = J^T (J J^T + \lambda^2 I)^{-1}).
* **Joint update:** (\Delta q = \alpha , J^+ e), then clamp per-iteration step and joint limits.
* **Hyperparameters tuned:** `alpha` (step size), `damping` (λ), `max_step`, and stopping threshold.

Core iteration (from `ik.py`):

```python
# compute current FK and Jacobian
cur_pose, J = your_fk(DH, q, base_ref)
# form 6D error e = [w_pos*(tgt_pos - cur_pos), w_ori*e_ori]
JJt = J @ J.T
J_pinv = J.T @ np.linalg.inv(JJt + (damping**2)*np.eye(JJt.shape[0]))
dq = alpha * (J_pinv @ e)
dq = np.clip(dq, -max_step, max_step)
q = np.clip(q + dq, joint_limits[:,0], joint_limits[:,1])
```

(implementation in `ik.py`). 

### 2.2 Problems encountered & handling

* **Quaternion sign ambiguity:** if current and target quaternions have negative dot, flip target quaternion to use the shortest rotation — implemented before computing `R_err`.
* **Singularities / ill-conditioned Jacobian:** near singularities `JJ^T` can be close to singular. I used DLS (small damping λ) to regularize inversion and keep updates stable.
* **Large / unstable updates:** limited per-iteration joint change with `max_step` and used small `alpha`. This reduces overshoot and improves convergence.
* **Joint limits:** clips were added after each update to keep q within feasible bounds.
* **Convergence tuning:** balanced `alpha`, `damping`, and `stop_thresh` to trade speed vs stability. Default values in my implementation gave robust convergence on the provided testcases.

### 2.3 Bonus — other methods tried

* **Pure pseudo-inverse (Moore–Penrose) without damping:** converges quickly when far from singularities but unstable near singularities — often produced large joint steps and occasional divergence.
* **Jacobian-transpose:** simple and sometimes faster per-iteration but requires careful tuning of step size and scales poorly for mixed position/orientation tasks.
* **Result:** Damped least-squares (the chosen method) provided the best stability/robustness tradeoff across the provided testcases and the block insertion trials.

---

## 3. Task 3 — Transporter Network (block insertion) (5%)

### 3.1 Comparison: `your_ik()` vs `pybullet_ik()`

* The template includes a `pybullet_ik()` wrapper that calls `p.calculateInverseKinematics(...)` (fast, closed-form/heuristic solver inside PyBullet).
* **Accuracy:** On the provided block-insertion testcases both solvers reach comparable end-effector pose accuracy for most cases (within thresholds) when `your_ik()` converges.
* **Robustness:** `pybullet_ik()` is generally faster (single-call) and may succeed where iterative solvers need more iterations or tuning, but:

  * `your_ik()` gives explicit control over damping, weighting (position vs orientation), step limits and joint limits, which can be advantageous for reproducible behavior and for integrating with Transporter pipelines that require predictable motion.
* **Speed:** `pybullet_ik()` is faster per-call. `your_ik()` can be slower (iterative), but acceptable for offline evaluation and fine control; in our tests it completed the 10 insertion trials when convergence conditions were satisfied.
* **Practical note:** Using `your_ik()` inside the Transporter pipeline worked reliably after the hyperparameters (`alpha`, `damping`, `max_step`, and `stop_thresh`) were tuned; however, for strict real-time needs `pybullet_ik()` is the pragmatic choice.

---

## 4. Short conclusions & tips

* The essential pieces are: correct DH frame setup → correct forward kinematics → accurate Jacobian → stable DLS IK.
* Tuning `damping` and `max_step` is critical to avoid oscillation/instability.
* Handle quaternion sign before computing orientation error.
* Keep the FK and Jacobian in the same frame convention as your simulator (I applied the small adjustment matrix used in the template before returning pose).

---

## References / files

* Assignment spec and grading details: provided spec. 
* Implementation: `fk.py` (forward kinematics + Jacobian). 
* Implementation: `ik.py` (iterative DLS IK and comparison wrapper). 

---

If you want, I can:

* produce a single-page PDF of this report ready for submission, or
* attach a short table of hyperparameter values I used and the numeric mean errors from the provided testcases. Which would you prefer?
