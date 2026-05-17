# AI for Industry Challenge 2026 — A Solo Journey

This is the story of how I — a mechanical engineering undergrad turned master's student — spent 10 weeks trying to teach a robot arm to insert fiber optic connectors into ports it couldn't see very well. I didn't achieve full insertion. But I learned more about robotics, controls, computer vision, and my own stubbornness than any coursework could have taught me.

## About Me

I'm a mechanical engineering undergraduate by training, currently pursuing a **Master of Science in Technology Innovation (MSTI)** at the University of Washington. I recently started learning robotics and computer science as part of the master's program. Before this competition, I had never written a ROS node, never trained a neural network for a real task, and never debugged why a robot arm refuses to move the last 7 centimeters toward a target.

I entered this competition to push myself into unfamiliar territory. ROS 2, Gazebo simulation, impedance control, computer vision, imitation learning — all of it was new to me. I competed solo against established robotics teams and labs.

## The Competition

The **AI for Industry Challenge (AIC)** is organized by **Intrinsic** (an Alphabet company) with support from **Google DeepMind**. The task: program a **UR5e robot arm** to autonomously insert SFP and SC fiber optic connectors into ports on a randomized task board in Gazebo simulation.

**Structure:**
- **3 phases:** Qualification (simulation) → Phase 1 (Flowstate cloud platform) → Phase 2 (real robot)
- **Qualification:** 3 trials per run — 2 SFP insertions + 1 SC insertion, scored up to **300 points** (100 per trial)
- **Key challenge:** Board position, orientation, and component placement are **randomized every run** — no hardcoding allowed
- **Sensor constraints:** Only cameras (3 wrist-mounted), force/torque sensor, and joint states. No ground truth positions during evaluation.

**Scoring per trial (100 pts max):**
- **Tier 1 (1 pt):** Model loads and responds to commands
- **Tier 2 (up to 24 pts):** Trajectory smoothness, task duration, efficiency, force/contact penalties
- **Tier 3 (up to 75 pts):** Task success — full insertion = 75, partial = 38–50, proximity = 0–25

The binary gap between teams that "get close" (~25 pts tier-3) and teams that "actually insert" (75 pts tier-3) is where the competition lives. I never crossed that gap.

## Final Results

| Metric | Value |
|--------|-------|
| Best local score | **134.0/300** (proximity only) |
| Best local with partial insertion | **160.9/300** (2 SFP partial insertions at yaw=4.71) |
| Best portal score | **123/300** |
| Docker images pushed | **46+** |
| Policy versions written | **20+** (V1 through V16, Servo, ACT, Diagnostic, Calibration) |
| Detector versions trained | **8** (v1 through v8) |
| Full insertions achieved | **0** |
| Ranking | **14th** (evaluation period May 18–27, final results May 28) |

## The Journey

### Week 1–2: Learning the Stack

I started from zero. The first week was entirely spent understanding the toolchain: **ROS 2 Kilted** running inside **Distrobox** containers, the **Pixi** package manager for Python environments, **Gazebo** simulation with physics plugins, and the competition's custom **impedance controller** architecture.

The hardest part wasn't any single concept — it was how they all connected. The robot publishes joint states on one topic, accepts pose commands on another, and the impedance controller translates desired poses into joint torques through a Cartesian impedance control law. Understanding that pipeline took days of reading source code and running the example `CheatCode` policy.

CheatCode was the reference implementation — a policy that uses **ground truth transforms** (TF frames) to know exactly where every port is. It achieves near-perfect scores. My job was to replicate that behavior using only camera images. The gap between "I know the answer" and "I have to figure out the answer from pixels" defined the entire competition for me.

**Key files created:** `DataCollector.py` — my first data collection policy, capturing labeled images from diverse viewpoints with ground truth port positions projected into pixel coordinates.

### Week 3–4: Classical Control — MyPolicy V1 (120 pts)

My first real policy (`MyPolicy_v1_120pts.py`) followed a straightforward pipeline:

1. **Move to a survey pose** — position the arm above the task board looking down
2. **Run a port detector** — 4 separate ResNet-18 models (one per port: `sfp_port_0`, `sfp_port_1`, `sc_port_0`, `sc_port_1`), each predicting normalized (u, v) pixel coordinates
3. **Back-project to 3D** — use camera intrinsics + assumed depth to convert pixels to a 3D position in the camera frame, then transform to the robot's base frame
4. **Approach** — interpolate the TCP toward the detected port position over 5 seconds
5. **Descend** — step downward in 0.5mm increments until reaching the target depth

The detector training pipeline was my first ML project. I collected ~1,200 labeled images using `DataCollector.py` with 20 diverse viewpoints per trial (grid sampling across X/Y/Z with random perturbations and camera tilts). Training used `train_detector.py` with ImageNet-pretrained ResNet-18, outputting 2 normalized pixel coordinates per model.

**The score: ~120 points.** Almost entirely from proximity credit and trajectory quality. No insertions.

**The Z=0.14 wall:** During SC trials, the TCP would stall at Z≈0.14m — about 7cm above the port at Z≈0.065m. The UR5e physically could not reach that position with the gripper pointing straight down. CheatCode solves this with a ~123-degree orientation change computed from ground truth frames. Without ground truth, I was stuck.

I spent days analyzing CheatCode's `calc_gripper_pose` function, understanding its incremental orientation update: `q_diff = q_port * q_plug_inverse`. I traced through the controller source code and found the tracking error reset mechanism. I tried **7+ approaches** to fix the SC descent: removing orientation constraints, lowering stiffness, slerping toward various targets, computing orientation from the TCP-to-port vector. None worked. The required rotation was 123 degrees — my 15-degree nudges were laughably insufficient, and I couldn't compute the correct direction without knowing the port's actual orientation.

**What I learned:** Impedance control is not magic. The controller follows `F = K(x_desired - x_actual) + D(v_desired - v_actual)`. If the robot can't reach your desired pose due to joint limits, it just... stops. Lowering stiffness doesn't help — it just makes the robot stop more gently.

### Week 5–6: The Imitation Learning Pivot

Frustrated with the SC wall, I pivoted to **imitation learning** using **ACT (Action Chunking with Transformers)** via the **LeRobot** library.

**The data collection pipeline** was a significant engineering effort:

- **Config generator** (`generate_configs.py`): Randomized board positions, yaw angles, NIC rail placements, and SC port positions — producing YAML files matching the competition's format
- **CheatCodeRecorder**: A wrapper around CheatCode that records observations (3 camera images + TCP state + joint positions) and actions (commanded poses) at 20 Hz
- **Automated batch runner** (`run_data_collection.py`): Iterates through config files, launches docker-compose with each config, records demos, and manages simulator lifecycle with timeout handling
- **Dataset converter** (`convert_to_lerobot.py`): Converts raw recordings to LeRobot's HuggingFace-compatible format with proper normalization statistics

Getting the batch runner to work was its own adventure. Distrobox subprocess management, sudo password prompts breaking automation, and a trial-1-only bug that took hours to find. Eventually: **530 episodes collected (432 SFP + 98 SC) over 11 hours, totaling 169 GB.**

**Training** (`train_act.py`) ran on my RTX 5090 — a Blackwell architecture GPU (sm_120) that required PyTorch >= 2.7.1 with CUDA 12.8. Getting the CUDA toolchain working was a multi-day side quest. Training hyperparameters: 100k steps, batch size 64, chunk size 100 (~5 seconds at 20 Hz), learning rate 1e-5.

**The result:** `RunTrainedACTHybrid.py` — ACT for coarse approach + force-responsive spiral descent for insertion. The model learned to move toward ports reasonably well. Portal score: **~50 pts** (v17–v20 submissions). The model could approach but couldn't insert. Open-loop action chunks fundamentally can't handle sub-millimeter precision — the 5-second action chunks drift too much over their horizon.

**What I learned:** Imitation learning is powerful for learning *what* to do but struggles with *precision*. The gap between "get within 2cm of the port" and "actually insert a plug into a 14mm opening" is vast, and action chunking across that gap loses the fine-grained feedback needed.

### Week 7–8: The Docker Submission Grind

This period was defined by the **v21–v36 submission arc** — pivoting back to classical control but now inside Docker containers, with systematic benchmarking.

**v21–v27: The MyPolicyV2 Baseline**

I moved `MyPolicyV2.py` (computed insertion orientation for SFP, blind descent for SC) into the Docker container. Key tricks:
- Defer all `torch` imports into `__init__()` — the 30-second model discovery timeout kills policies that import heavy libraries at module level
- Copy policy files directly into pixi's site-packages (the build cache otherwise serves stale code)
- Pack 4 ResNet-18 checkpoint files (~180 MB total) into the container

v22 added **XY spiral search on stall** and **force-drop chamfer detection** (when force drops >4N over 8 steps, interpret as chamfer entry). v27 cleaned up all failed SC tilt experiments and became the **recommended fallback baseline at 98.33 pts on portal.**

**v28–v36: Systematic Experimentation**

I built `benchmark_random_configs.py` — a script that runs N random configs through docker-compose and parses scores. This replaced testing on the single sample config (which my detector had basically memorized).

Key versions and what I learned:

| Version | Innovation | Local Score | Portal | Lesson |
|---------|-----------|-------------|--------|--------|
| v27 | Clean V2 baseline | 131 | **98** | Simple and reliable beats clever and fragile |
| v28 | Re-detect from closer | 126 | 94 | Second detection adds noise, doesn't help |
| v29 | Ray-plane Z fix | 132 | — | Port Z is constant in base frame (0.195 SFP, 0.072 SC) |
| v30 | 15mm spiral search | 132 | — | Wider search needed, but bumps into components |
| v32 | v2 detectors + ALIGN + force modulation | 94 (5-config mean) | — | **Infrastructure milestone** |
| v33–v36 | Tilt-search, wider spiral, various force thresholds | 52–93 | — | Diminishing returns without better perception |

**The v32 breakthrough wasn't in score — it was in infrastructure:**
- **v2 detectors** with strong augmentation (rotation ±25°, scale, translate, color jitter, random erasing)
- **ALIGN phase**: Hover 50mm above port for 1.2 seconds to let oscillations settle before descent
- **Force-modulated descent**: Full step below 5N, half step 5–10N, halt above 12N
- **Critical discovery:** `SFP_DESCENT_Z_END` was set to -0.015 (15mm *into* the card surface). Changing to +0.005 eliminated finger-on-card collisions that cost a **-24 contact penalty per trial**

**The honest realization:** Testing on 5 diverse configs revealed my detector's true accuracy. Mean scores dropped from 131 (on familiar configs) to 94 (on random configs). The detector was overfitting to a narrow range of board yaw angles.

### Week 9–10: Better Detectors and the Final Push

**v38–v46: Detector retraining with diverse data**

The real bottleneck was clear: **detector accuracy.** I built a new data collection pipeline targeting full 360° yaw coverage:

- Collected **149 additional training samples** with configs spanning yaw 0–2π
- Retrained through multiple detector versions (v3–v7), each with progressively more data and augmentation
- **v6 detectors** (trained on the expanded dataset) dramatically improved generalization

**The v44 breakthrough:** MyPolicyV3 + v6 detectors achieved **160.9 pts on a yaw=4.71 config** — including **2 partial SFP insertions** (58.7 + 58.5 pts). This was the first time I saw tier-3 scores above 25. The detector was finally accurate enough at that yaw angle for the plug to actually enter the port bounding box.

But this was config-dependent. On other yaw angles, scores remained in the 100–134 range. The portal averages across many configs, so my portal score stayed around **123 pts**.

**What I tried in the final days:**
- **Visual servoing** (`MyPolicyServo.py`): Detect port in image → drive pixel error to zero using feedback control. The servo converged beautifully (150px error → ~0 in 3 seconds), but the **camera-to-plug offset** computation introduced its own error. The plug hangs at ~25° from the gripper TCP, making the geometric relationship between "where the camera sees the port" and "where the plug tip actually is" non-trivial.
- **Calibration policy** (`CalibrationPolicy.py`): Used ground truth to measure the exact plug offset in TCP frame. Found a sign error in my Z offset (+0.042 should have been -0.054). But the offset varies with orientation, so a single constant doesn't generalize.
- **Grid scan insertion** (MyPolicyV5, V7, V10): Systematically probe a grid of XY positions with gentle Z pushes, looking for force drops that indicate hole entry. The grid was too slow (60+ seconds for a 7x7 grid at 1.5mm spacing) and the scoring penalized duration.
- **Tilt-search insertion** (v36): Probe ±3° in pitch and roll after contact, pick the lowest-force direction, commit. A good idea in principle, but the plug never made reliable contact in the first place.

## Technical Deep Dives

### The Impedance Controller

The AIC uses a **Cartesian impedance controller** — probably the single most important thing I had to understand. The control law is:

```
F = K * (x_desired - x_actual) + D * (dx_desired - dx_actual) + F_feedforward
```

Where `K` is the 6×6 stiffness matrix (translational + rotational) and `D` is damping. The controller accepts commands in two modes:

- **Position mode** (`MODE_CARTESIAN`): You specify a target pose. The controller drives the robot toward it. Good for large motions, but the robot can get stuck if the target is near kinematic limits.
- **Velocity mode** (`MODE_VELOCITY`): You specify a desired velocity. The controller integrates it. Better for compliant contact behavior because there's no "target" to get stuck on — you just keep pushing gently.

I discovered the **tracking error reset mechanism** in the controller source: when switching modes or after large jumps, the controller resets its internal tracking error to avoid sudden jerks. This explained why my mode switches sometimes caused the robot to lurch.

**Stiffness tuning** was critical. Default stiffness `[150,150,150,50,50,50]` (N/m translational, Nm/rad rotational) was too stiff for insertion — the robot would push hard against misaligned ports rather than complying. I experimented with values as low as `[60,60,60,30,30,30]` for insertion phases, which allowed more compliance but also more positional drift.

### The Grasp Geometry Problem

The plug doesn't point straight down from the gripper. It's grasped at an angle defined by:

```python
GRASP_ROLL  = 0.4432   # radians
GRASP_PITCH = -0.4838
GRASP_YAW   = 1.3303
GRASP_POS_SFP = [0.0, 0.015385, 0.04245]  # meters, TCP-to-plug offset
```

This means the plug tip is offset ~42mm below and ~15mm to the side of the TCP frame origin, tilted about 25° from vertical. When you command the TCP to a position, the plug tip ends up somewhere else entirely. CheatCode handles this elegantly because it knows both the plug and port orientations from ground truth. I had to approximate it.

The camera adds another layer: it's mounted on the wrist at a ~75° tilt. So the camera sees the port at an angle, the plug hangs at a different angle, and converting "port pixel in image" to "where should the TCP go so the plug tip lands on the port" requires chaining multiple coordinate transforms with imperfect calibration at every step.

### The Detection Accuracy Wall

This was the fundamental bottleneck. The numbers tell the story:

| Metric | Value |
|--------|-------|
| ResNet-18 pixel error (median) | ~22–50 px |
| ResNet-18 pixel error (worst case) | 115 px |
| World XY error at 30cm distance | 6–40 mm |
| SFP port opening | 13.75 mm × 4.225 mm |
| SC port opening | ~2.5 mm diameter |

At best, my detector achieved **6mm XY error** — just barely within the 14mm SFP port width. At worst, **40mm error** — completely hopeless. And this varied wildly depending on board yaw, NIC rail position, and lighting conditions.

The v6 detectors improved generalization with 360° yaw training data, but the fundamental architecture (ResNet-18 regressing 2 pixel coordinates) has a ceiling. The model has no notion of port geometry — it's just fitting a function from image to (u,v). A keypoint detector, template matching approach, or even a simple edge detector might have been more reliable.

**Multi-camera attempts:** I tried running detectors on all 3 cameras (left, center, right) and averaging or triangulating. Averaging helped slightly. Triangulation was a disaster — pixel-level noise compounds across the stereo baseline, producing wildly wrong 3D points.

## What I Would Do Differently

### If Starting Over

1. **Start with docker-compose testing from day 1.** I wasted weeks testing on the single `sample_config.yaml` where my detector performed well, then was surprised when portal scores were 30 points lower. The benchmark script should have been the first thing I built.

2. **Visual servoing from the beginning.** The wrist camera moves with the arm — use it for closed-loop control during approach, not just a single snapshot from survey height.

3. **Diverse training data immediately.** My v1 detector was trained on ~1,200 images from a narrow yaw range. When I finally collected 360° yaw data and retrained, scores jumped 60 points on hard configs. Data diversity matters more than model architecture.

4. **Study the scoring plugins first.** I spent weeks trying to "insert" without fully understanding what triggers the insertion event in Gazebo. Understanding the scoring collision geometry would have focused my insertion strategy.

5. **Velocity mode for all contact phases.** Position mode fights you near kinematic limits. Velocity mode just keeps pushing gently. I should have used it from the start for descent and insertion.

### What Would Actually Achieve 200+

Based on watching top teams and analyzing the scoring:

1. **Better perception:** Keypoint detection or template matching instead of regression CNN — need consistent <5mm accuracy across all board configurations
2. **Closed-loop visual servoing** during the final approach with the port visible in the wrist camera
3. **Force-guided insertion:** Tilt-search patterns from manipulation literature — probe ±3° in pitch/roll after contact, find the chamfer, commit
4. **Or: massive diverse dataset** (5,000+ demos) with a diffusion policy instead of ACT — the action chunk horizon matters less with diffusion's denoising approach

## Repository Structure

```
aic-workspace/
├── my_policy/my_policy/        # Policy development versions
│   ├── MyPolicy.py             # V1 baseline (120 pts)
│   ├── MyPolicy_v1_120pts.py   # V1 snapshot
│   ├── MyPolicyV2.py           # Computed orientation + CheatCode SC mode
│   ├── MyPolicyV8.py           # No-detector, yaw-independent
│   ├── MyPolicyV9.py           # Ray-plane + spiral search + force feedback
│   ├── RunTrainedACT.py        # Pure ACT inference policy
│   └── DataCollector.py        # Training data collection
│
├── aic/aic_model/aic_model/    # Deployed Docker policies (V3–V16 + specialized)
│   ├── MyPolicyV3.py           # Expanding spiral search
│   ├── MyPolicyV5.py           # Fast raster scan
│   ├── MyPolicyV7.py           # Lawn-mower grid + verification
│   ├── MyPolicyV11.py          # 3-camera detection + close-range refinement
│   ├── MyPolicyServo.py        # Visual servoing + grid
│   ├── MyPolicyDocker.py       # Production v36 (tilt-search + ALIGN)
│   ├── RunTrainedACTHybrid.py  # ACT approach + spiral descent
│   ├── DiagnosticPolicy.py     # Ground-truth error measurement
│   ├── CalibrationPolicy.py    # Plug offset extraction
│   └── SurveyCollector.py      # Labeled dataset collection
│
├── checkpoints/                # Trained ResNet-18 weights
│   ├── v2/ through v7/         # Per-port detector versions
│   └── *.pth                   # v1 detector weights
│
├── scripts/                    # Automation and training
│   ├── train_detector_v5.py    # ResNet-18 training with augmentation
│   ├── train_act.py            # ACT model training via LeRobot
│   ├── run_data_collection.py  # Automated batch data collection
│   ├── convert_to_lerobot.py   # Dataset format conversion
│   ├── generate_configs.py     # Random config generation
│   └── benchmark_random_configs.py  # Multi-config scoring benchmark
│
├── configs*/                   # Generated config YAML files (334+ configs)
├── benchmark_results/          # Benchmark score JSONs and trend.csv
├── docs/                       # Submission history, compliance notes
└── aic/                        # Competition toolkit (git submodule)
```

## Tools & Technologies

- **ROS 2 Kilted**, Gazebo Harmonic, Python 3.12
- **PyTorch** — ResNet-18 for port detection, ACT (Action Chunking with Transformers) for imitation learning
- **LeRobot** — HuggingFace's robot learning library for ACT training
- **Docker**, AWS ECR — 46+ container image versions submitted
- **RTX 5090** (Blackwell architecture, CUDA sm_120, 32 GB VRAM, PyTorch cu128)
- **Ubuntu 24.04**, Pixi package manager, Distrobox containers
- **Claude Code** (Anthropic) — AI pair programming for rapid iteration across all phases

## Score History

### Portal Submissions

| Version | Date | Portal Score | Strategy |
|---------|------|-------------|----------|
| v17 | Apr 22 | 41.3 | First ACT submission |
| v18 | Apr 23 | 50.1 | ACT + spiral descent |
| v19 | Apr 23 | 47.7 | ACT + force-aware backoff |
| v20 | Apr 24 | 47.7 | ACT + stricter stall detection |
| v27 | Apr 25 | **98.3** | Clean V2 baseline + XY search |
| v28 | Apr 26 | 94.5 | Iterative re-detection (regressed) |
| v38 | May 1 | 98.0 | V3 original |
| v44 | May 3 | **123.0** | V3 + v6 detectors (360° yaw training) |

### Local Benchmarks (Multi-Config)

| Tag | Date | Configs | Mean | Median | Max | Notes |
|-----|------|---------|------|--------|-----|-------|
| v30_smoke | Apr 26 | 2 | 83.6 | 83.6 | 85.6 | 15mm spiral search |
| v32_wide | Apr 27 | 5 | 93.8 | 92.8 | 114.2 | v2 detectors + ALIGN phase |
| v33b | Apr 27 | 5 | 92.4 | 93.3 | 107.0 | Tilt experiments |
| v34_wideyaw | Apr 27 | 4 | 61.5 | 67.0 | 82.9 | Wide yaw configs exposed detector weakness |
| v36final | Apr 28 | 4 | 56.8 | 54.0 | 92.7 | Tilt-search (over-engineered) |
| v2_final | May 5 | 3 | 101.7 | 107.5 | 112.4 | V2 with v6 detectors |
| v9_test | May 6 | 3 | 92.5 | 99.4 | 103.4 | V9 ray-plane + spiral |

### Best Single-Config Results

| Config | Version | Score | T1 | T2 | T3 | Notes |
|--------|---------|-------|-----|-----|-----|-------|
| yaw=4.71 | v44 | **160.9** | 58.7 | 58.5 | 43.7 | 2 partial insertions! |
| config_166 | v32 | **112.3** | — | — | — | Collision fix helped |
| Default | v44 | 134.0 | ~45 | ~45 | ~44 | Proximity only |

## Acknowledgments

- **Intrinsic** and **Google DeepMind** for organizing the AIC and providing the simulation infrastructure
- The AIC open-source toolkit, example policies (especially CheatCode), and the competitor community
- **University of Washington MSTI program** for the environment and motivation to take on challenges like this
- **Claude** (Anthropic) for AI-assisted development — from debugging quaternion math at 2am to writing data collection pipelines, Claude Code was my teammate throughout this project

---

*This repository represents ~10 weeks of work, ~20 policy versions, 8 detector versions, 530 imitation learning episodes, thousands of simulation runs, and one person's stubborn refusal to accept that a 14mm port opening is really that hard to hit. It is.*
