# AIC Workspace — Claude Code Configuration

> AI for Industry Challenge workspace with Ruflo orchestration.
> GPU: RTX 5090 (32 GB VRAM, sm_120 — requires PyTorch >= 2.7.1)
> Evaluation target: NVIDIA L4 (24 GB VRAM) — keep inference models under ~20 GB

## Behavioral Rules

- ALWAYS consult docs/aic_compliance_guidelines.md before implementing any ROS 2 interaction
- NEVER use topics/services outside the AIC whitelist (see compliance doc Section 3)
- NEVER hardcode poses, port locations, or simulation-specific exploits
- NEVER publish robot commands in unconfigured/configured/shutdown lifecycle states
- ALWAYS ensure aic_model goals are cancellable in active state
- ALWAYS keep model inference VRAM under 20 GB (evaluation runs on 24 GB L4)
- Read files before editing them
- Do not commit .env, credentials, or secrets

## File Organization

| Directory | Purpose |
|-----------|---------|
| aic/ | AIC toolkit (git submodule — do NOT edit directly) |
| my_policy/ | Your custom policy package (this gets submitted) |
| scripts/ | Setup, training, evaluation, and utility scripts |
| docs/ | Compliance guidelines, notes, design docs |
| config/ | Training configs, hyperparameters, experiment settings |
| checkpoints/ | Model weights (gitignored) |
| datasets/ | Training data (gitignored) |
| rosbags/ | Recorded ROS bag files (gitignored) |

## Key Commands

Source environment: source .env

Start eval environment (Terminal 1):
  distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true

Run your policy (Terminal 2):
  cd aic && pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=my_policy.MyPolicy

Build submission container:
  docker compose -f aic/docker/docker-compose.yaml build model

Test submission locally:
  docker compose -f aic/docker/docker-compose.yaml up

## AIC Interface Whitelist (Quick Reference)

Allowed Inputs (subscribe):
  /left_camera/image, /center_camera/image, /right_camera/image
  /left_camera/camera_info, /center_camera/camera_info, /right_camera/camera_info
  /fts_broadcaster/wrench, /joint_states, /gripper_state
  /tf, /tf_static
  /aic_controller/controller_state

Allowed Outputs (publish):
  /aic_controller/pose_commands (MotionUpdate)
  /aic_controller/joint_commands (JointMotionUpdate)

Allowed Services (call):
  /aic_controller/change_target_mode

Must Implement:
  /insert_cable action server (accept + cancellable in active state)

EVERYTHING ELSE IS FORBIDDEN:
  /scoring/*, /gazebo/*, /gz_server/*, /clock, /model/*, /world_stats, /pause_physics
