# AIC Project Handoff Document

## What This Project Is

I am a solo competitor in the **AI for Industry Challenge (AIC)** by Intrinsic. The task: program a robot arm (UR5e) to insert fiber optic cable connectors (SFP and SC types) into ports on a randomized task board. The qualification deadline is **May 15, 2026**. I am not a CS major — CS is a side pursuit, so I benefit from concept explanations alongside practical instructions.

## My Approach: Classical Control + Vision

Instead of pure imitation learning, I'm using:
1. **Vision** — a neural network that looks at camera images and predicts where the target port is
2. **Classical control** — scripted motion (approach → align → insert with force feedback) modeled after the provided `CheatCode` example policy

The only thing I need from vision is the port's 3D position and orientation relative to the camera. Everything else (motion planning, insertion, force monitoring) follows the CheatCode pattern.

## Hardware & Environment

- **GPU:** NVIDIA RTX 5090 (32 GB VRAM, Blackwell sm_120)
- **OS:** Ubuntu 22.04 on this machine (eval container runs Ubuntu 24.04 internally)
- **Evaluation target:** NVIDIA L4 (24 GB VRAM) — inference model must fit under ~20 GB
- **ROS 2:** Kilted Kaiju (via Pixi + Docker/Distrobox)

### PyTorch/CUDA Situation
The RTX 5090 (sm_120) is NOT compatible with PyTorch installed via Pixi (which pulls cu126). We use **two separate Python environments**:
- **Pixi environment** — for running ROS 2 policies. PyTorch is available but only works on CPU. This is fine because inference on a single image is fast enough on CPU.
- **`~/aic-workspace/train-env/`** — a plain venv with PyTorch cu128 that works on the 5090 GPU. Used only for training.

To activate the training env: `source ~/aic-workspace/train-env/bin/activate`

## Repository Structure

GitHub repo: `git@github.com:rushav/aic-workspace.git`

```
~/aic-workspace/
├── aic/                          # AIC toolkit (git submodule from github.com/intrinsic-dev/aic)
│   ├── aic_example_policies/     # Example policies including our DataCollector
│   │   └── aic_example_policies/ros/
│   │       ├── WaveArm.py        # Minimal example (just waves arm)
│   │       ├── CheatCode.py      # Ground-truth insertion (scores ~279/300)
│   │       ├── RunACT.py         # ACT imitation learning example
│   │       └── DataCollector.py  # OUR data collection policy (copied here for pixi access)
│   ├── aic_model/                # Policy base class and ROS node
│   ├── aic_interfaces/           # ROS message definitions
│   ├── docker/                   # Dockerfiles for submission
│   ├── docs/                     # Official documentation
│   └── pixi.toml                # Dependencies (has our PyTorch override appended)
├── my_policy/
│   └── my_policy/
│       ├── __init__.py
│       ├── MyPolicy.py           # Starter template (not yet functional)
│       └── DataCollector.py      # Source copy of data collection policy
├── scripts/
│   ├── apply-fixes.sh            # Applies RTX 5090 PyTorch fix to aic/pixi.toml
│   ├── capture_labeled.py        # Captures single labeled image with port projection
│   ├── train_detector.py         # Trains ResNet-18 port detector
│   └── visualize_predictions.py  # Visualizes model predictions vs ground truth
├── datasets/
│   └── port_detection/           # 83 labeled samples (sample_XXXX.png + .json pairs)
├── checkpoints/
│   ├── port_detector_best.pth    # Best trained model weights
│   └── port_detector_final.pth   # Final epoch weights
├── data/                         # Misc captured images and prediction visualizations
├── docs/
│   └── aic_compliance_guidelines.md  # Comprehensive rules/regulations reference
├── train-env/                    # Python venv with GPU-compatible PyTorch (gitignored)
├── CLAUDE.md                     # Claude Code project context
├── .env                          # Environment variables (gitignored)
├── .gitignore
└── .gitmodules
```

## What Has Been Done

### Setup (complete)
- Docker, NVIDIA Container Toolkit, Distrobox, Pixi, Node.js 22, Claude Code, Ruflo all installed
- AIC evaluation container pulled and verified end-to-end (WaveArm scores ~60, CheatCode scores ~279)
- SSH key configured for GitHub

### Week 1: Exploration (complete)
- Ran WaveArm (minimal example) and CheatCode (ground truth insertion) — understood both codebases
- Read and understood the Policy base class API: `set_pose_target()`, `sleep_for()`, `time_now()`, TF buffer
- Explored available ROS topics — identified whitelist vs forbidden topics
- Teleoperated the robot manually — experienced how hard alignment and insertion is
- Captured camera images from all three wrist cameras — center camera chosen as primary
- Examined F/T sensor baseline readings

### Week 2: Perception (in progress)
- Built `DataCollector.py` — a policy that moves the arm to different viewpoints and captures camera images paired with ground truth port positions (projected to pixel coordinates)
- Collected **83 labeled samples** across multiple randomized board configurations
- Each sample: 1152×1024 PNG image + JSON with pixel coords (u,v), 3D position (x,y,z), and orientation (qx,qy,qz,qw) for each visible SFP port
- Trained a **ResNet-18 based port detector** that predicts (u, v, x, y, z) from an image
- Results: **~20px best pixel error, ~33px mean** — translates to ~5mm real-world, needs improvement
- Visualized predictions — model works well on close-up frontal views, struggles with varied angles

## Key Files to Read

### Policy base class
`~/aic-workspace/aic/aic_model/aic_model/policy.py`
- `set_pose_target(move_robot, pose)` — sends Cartesian command with default stiffness/damping
- `sleep_for(seconds)` — sim-time aware sleep
- `time_now()` — sim-time aware clock
- `self._parent_node._tf_buffer` — TF transform tree access

### CheatCode (the blueprint)
`~/aic-workspace/aic/aic_example_policies/aic_example_policies/ros/CheatCode.py`
- Two-phase motion: smooth approach (5s interpolation to position above port) → descent (0.5mm steps with integral XY correction)
- The ONLY thing it gets from ground truth is `port_transform` — the port's position/orientation
- Everything else uses standard TF data available during evaluation

### DataCollector
`~/aic-workspace/aic/aic_example_policies/aic_example_policies/ros/DataCollector.py`
(also source at `~/aic-workspace/my_policy/my_policy/DataCollector.py`)
- Moves arm to ~7 different viewpoints per trial
- Captures center camera image + ground truth port pixel/3D labels
- Saves to `~/aic-workspace/datasets/port_detection/`

### Training script
`~/aic-workspace/scripts/train_detector.py`
- ResNet-18 backbone, regression head outputs 5 values: (u_norm, v_norm, x, y, z)
- Images resized to 256×256, normalized with ImageNet stats
- 80/20 train/val split, MSE loss, Adam optimizer, 100 epochs
- Trained in `train-env` (not pixi)

### Dataset format
Each sample in `~/aic-workspace/datasets/port_detection/`:
```
sample_0000.png  — 1152x1024 RGB image from center camera
sample_0000.json — labels:
{
  "image": "sample_0000.png",
  "width": 1152, "height": 1024,
  "fx": 1236.63, "fy": 1236.63, "cx": 576.0, "cy": 512.0,
  "ports": {
    "sfp_port_0": {
      "u": 534.1, "v": 525.1,        // pixel coordinates
      "x": -0.012, "y": 0.004, "z": 0.359,  // 3D in camera frame
      "qx": -0.124, "qy": 0.0, "qz": 0.0, "qw": -0.992  // orientation
    },
    "sfp_port_1": { ... }
  }
}
```

## Compliance Rules (critical)

Full doc at `~/aic-workspace/docs/aic_compliance_guidelines.md`. Key points:

### Allowed inputs (subscribe)
- `/left_camera/image`, `/center_camera/image`, `/right_camera/image`
- `/left_camera/camera_info`, `/center_camera/camera_info`, `/right_camera/camera_info`
- `/fts_broadcaster/wrench` (force/torque sensor)
- `/joint_states`, `/gripper_state`
- `/tf`, `/tf_static`
- `/aic_controller/controller_state`

### Allowed outputs (publish)
- `/aic_controller/pose_commands` (MotionUpdate)
- `/aic_controller/joint_commands` (JointMotionUpdate)

### Allowed services
- `/aic_controller/change_target_mode`

### Must implement
- `/insert_cable` action server (handled by aic_model base class)

### FORBIDDEN — do not touch
- `/scoring/*`, `/gazebo/*`, `/gz_server/*`, `/clock`, `/model/*`, `/world_stats`, `/pause_physics`
- Ground truth TF frames are NOT available during evaluation

## How to Run Things

### Start evaluation environment (Terminal 1)
```bash
export DBX_CONTAINER_MANAGER=docker
# With ground truth (for data collection/debugging):
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=true start_aic_engine:=true
# Without ground truth (for real testing):
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true
```

### Run a policy (Terminal 2)
```bash
cd ~/aic-workspace/aic
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.DataCollector
```

### Run training (in train-env)
```bash
source ~/aic-workspace/train-env/bin/activate
cd ~/aic-workspace
python3 scripts/train_detector.py
```

### Teleoperation (no engine)
Terminal 1:
```bash
export DBX_CONTAINER_MANAGER=docker
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=true start_aic_engine:=false spawn_task_board:=true spawn_cable:=true attach_cable_to_gripper:=true nic_card_mount_0_present:=true cable_type:=sfp_sc_cable
```
Terminal 2:
```bash
cd ~/aic-workspace/aic
pixi run ros2 run aic_teleoperation cartesian_keyboard_teleop
```

## What Needs to Happen Next

### Immediate: Improve perception accuracy
Current model has ~20-33px error (~5mm real-world). Need to get this under ~10px (~2-3mm) for reliable insertion. Two paths:
1. **More training data** — collect 200-500+ more samples with wider viewpoint variation, different board randomizations. The DataCollector policy only captures ~14 samples per engine run. Need to run it many more times, or modify it to capture more viewpoints per trial.
2. **Better model architecture or training** — could try larger input resolution, data augmentation, or predicting both ports simultaneously.

### After perception: Build the insertion policy
Combine the trained detector with CheatCode's motion strategy:
1. Capture image → run detector → get port 3D position in camera frame
2. Transform to robot base frame using TF
3. Approach phase: smooth interpolation to position above port
4. Insertion phase: descend in small steps with force feedback + spiral search
5. Return when insertion depth reached or time runs out

### SC port generalization (Trial 3)
Need separate detection data for SC ports. The DataCollector already tries to find SC frames but they weren't spawned in our data collection runs. Need to run with SC port configuration.

## Scoring Reference
- Max 100 points per trial, 3 trials = 300 total
- Tier 1 (validity): 0 or 1
- Tier 2 (smoothness 0-6, speed 0-12, efficiency 0-6, force penalty 0 to -12, collision penalty 0 to -24)
- Tier 3 (correct insertion 75, wrong port -12, proximity 0-25, partial insertion 38-50)
- CheatCode scored 279/300 — our target baseline
- WaveArm scored ~60/300 — no insertion attempt

## Git Info
- Repo: `git@github.com:rushav/aic-workspace.git`
- Username: rushav
- Email: rushavsd@gmail.com
