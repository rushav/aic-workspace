#!/usr/bin/env python3
"""Generate randomized trial config YAML files for automated data collection.

Each config has 3 trials: 2 SFP insertions + 1 SC insertion, matching the
qualification phase structure. Board pose, NIC rail placement, SC rail
placement, and mount rail decorations are all randomized within official limits.

Usage:
    python scripts/generate_configs.py --num-configs 50
    python scripts/generate_configs.py --num-configs 334 --output-dir configs
"""

import argparse
import os
import random

import yaml


# ── Official randomization limits (from qualification_phase.md + sample_config.yaml) ──

BOARD_X_RANGE = (0.05, 0.30)
BOARD_Y_RANGE = (-0.30, 0.10)
BOARD_Z = 1.14
BOARD_YAW_RANGE = (0.0, 6.2832)  # full 360° (2*pi)

NIC_TRANSLATION_RANGE = (-0.0215, 0.0234)
NIC_YAW_RANGE = (-0.175, 0.175)  # ±10 degrees
NUM_NIC_RAILS = 5

SC_TRANSLATION_RANGE = (-0.06, 0.055)
NUM_SC_RAILS = 2

MOUNT_RAIL_TRANSLATION_RANGE = (-0.09425, 0.09425)

# Fixed grasp offsets (from sample_config.yaml / qualification_phase.md)
SFP_GRASP = {
    "x": 0.0, "y": 0.015385, "z": 0.04245,
    "roll": 0.4432, "pitch": -0.4838, "yaw": 1.3303,
}
SC_GRASP = {
    "x": 0.0, "y": 0.015385, "z": 0.04045,
    "roll": 0.4432, "pitch": -0.4838, "yaw": 1.3303,
}

# Robot home joint positions (fixed)
HOME_JOINTS = {
    "shoulder_pan_joint": -0.1597,
    "shoulder_lift_joint": -1.3542,
    "elbow_joint": -1.6648,
    "wrist_1_joint": -1.6933,
    "wrist_2_joint": 1.5710,
    "wrist_3_joint": 1.4110,
}


def rand_range(lo, hi, decimals=4):
    return round(random.uniform(lo, hi), decimals)


def generate_board_pose():
    return {
        "x": rand_range(*BOARD_X_RANGE),
        "y": rand_range(*BOARD_Y_RANGE),
        "z": BOARD_Z,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": rand_range(*BOARD_YAW_RANGE),
    }


def generate_nic_rails(active_rail):
    """Generate nic_rail_0..4 with one active NIC card."""
    rails = {}
    for i in range(NUM_NIC_RAILS):
        if i == active_rail:
            rails[f"nic_rail_{i}"] = {
                "entity_present": True,
                "entity_name": f"nic_card_{i}",
                "entity_pose": {
                    "translation": rand_range(*NIC_TRANSLATION_RANGE),
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": rand_range(*NIC_YAW_RANGE),
                },
            }
        else:
            rails[f"nic_rail_{i}"] = {"entity_present": False}
    return rails


def generate_sc_rails(active_rail):
    """Generate sc_rail_0..1 with one active SC port."""
    rails = {}
    for i in range(NUM_SC_RAILS):
        if i == active_rail:
            rails[f"sc_rail_{i}"] = {
                "entity_present": True,
                "entity_name": f"sc_mount_{i}",
                "entity_pose": {
                    "translation": rand_range(*SC_TRANSLATION_RANGE),
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": round(random.uniform(-0.1, 0.1), 4),
                },
            }
        else:
            rails[f"sc_rail_{i}"] = {"entity_present": False}
    return rails


def generate_mount_rails():
    """Generate random decorative mount rails for visual diversity."""
    mount_rails = {}

    # LC mount rails
    for i in range(2):
        present = random.random() < 0.6
        mount_rails[f"lc_mount_rail_{i}"] = {
            "entity_present": present,
        }
        if present:
            mount_rails[f"lc_mount_rail_{i}"]["entity_name"] = f"lc_mount_{i}"
            mount_rails[f"lc_mount_rail_{i}"]["entity_pose"] = {
                "translation": rand_range(*MOUNT_RAIL_TRANSLATION_RANGE),
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            }

    # SFP mount rails
    for i in range(2):
        present = random.random() < 0.5
        mount_rails[f"sfp_mount_rail_{i}"] = {
            "entity_present": present,
        }
        if present:
            mount_rails[f"sfp_mount_rail_{i}"]["entity_name"] = f"sfp_mount_{i}"
            mount_rails[f"sfp_mount_rail_{i}"]["entity_pose"] = {
                "translation": rand_range(*MOUNT_RAIL_TRANSLATION_RANGE),
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            }

    # SC mount rails
    for i in range(2):
        present = random.random() < 0.5
        mount_rails[f"sc_mount_rail_{i}"] = {
            "entity_present": present,
        }
        if present:
            # Use offset index to avoid name collision with sc_rail entities
            mount_rails[f"sc_mount_rail_{i}"]["entity_name"] = f"sc_mount_{i + 2}"
            mount_rails[f"sc_mount_rail_{i}"]["entity_pose"] = {
                "translation": rand_range(*MOUNT_RAIL_TRANSLATION_RANGE),
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            }

    return mount_rails


def generate_sfp_trial(trial_num, nic_rail=None, sc_rail=None, sfp_port=None):
    """Generate one SFP insertion trial."""
    if nic_rail is None:
        nic_rail = random.randint(0, NUM_NIC_RAILS - 1)
    if sc_rail is None:
        sc_rail = random.randint(0, NUM_SC_RAILS - 1)
    if sfp_port is None:
        sfp_port = random.randint(0, 1)

    board_pose = generate_board_pose()
    nic_rails = generate_nic_rails(nic_rail)
    sc_rails = generate_sc_rails(sc_rail)
    mount_rails = generate_mount_rails()

    scene = {
        "task_board": {
            "pose": board_pose,
            **nic_rails,
            **sc_rails,
            **mount_rails,
        },
        "cables": {
            "cable_0": {
                "pose": {
                    "gripper_offset": {
                        "x": SFP_GRASP["x"],
                        "y": SFP_GRASP["y"],
                        "z": SFP_GRASP["z"],
                    },
                    "roll": SFP_GRASP["roll"],
                    "pitch": SFP_GRASP["pitch"],
                    "yaw": SFP_GRASP["yaw"],
                },
                "attach_cable_to_gripper": True,
                "cable_type": "sfp_sc_cable",
            },
        },
    }

    tasks = {
        "task_1": {
            "cable_type": "sfp_sc",
            "cable_name": "cable_0",
            "plug_type": "sfp",
            "plug_name": "sfp_tip",
            "port_type": "sfp",
            "port_name": f"sfp_port_{sfp_port}",
            "target_module_name": f"nic_card_mount_{nic_rail}",
            "time_limit": 180,
        }
    }

    return {"scene": scene, "tasks": tasks}


def generate_sc_trial(sc_rail=None):
    """Generate one SC insertion trial."""
    if sc_rail is None:
        sc_rail = random.randint(0, NUM_SC_RAILS - 1)

    board_pose = generate_board_pose()
    # No NIC cards needed for SC trial, but may add some for visual diversity
    nic_rails = {}
    if random.random() < 0.3:
        # Occasionally add a NIC card for visual diversity
        extra_nic = random.randint(0, NUM_NIC_RAILS - 1)
        nic_rails = generate_nic_rails(extra_nic)
    else:
        for i in range(NUM_NIC_RAILS):
            nic_rails[f"nic_rail_{i}"] = {"entity_present": False}

    sc_rails = generate_sc_rails(sc_rail)
    mount_rails = generate_mount_rails()

    scene = {
        "task_board": {
            "pose": board_pose,
            **nic_rails,
            **sc_rails,
            **mount_rails,
        },
        "cables": {
            "cable_1": {
                "pose": {
                    "gripper_offset": {
                        "x": SC_GRASP["x"],
                        "y": SC_GRASP["y"],
                        "z": SC_GRASP["z"],
                    },
                    "roll": SC_GRASP["roll"],
                    "pitch": SC_GRASP["pitch"],
                    "yaw": SC_GRASP["yaw"],
                },
                "attach_cable_to_gripper": True,
                "cable_type": "sfp_sc_cable_reversed",
            },
        },
    }

    tasks = {
        "task_1": {
            "cable_type": "sfp_sc",
            "cable_name": "cable_1",
            "plug_type": "sc",
            "plug_name": "sc_tip",
            "port_type": "sc",
            "port_name": "sc_port_base",
            "target_module_name": f"sc_port_{sc_rail}",
            "time_limit": 180,
        }
    }

    return {"scene": scene, "tasks": tasks}


def generate_scoring_section():
    """Generate the fixed scoring topics section."""
    return {
        "topics": [
            {"topic": {"name": "/joint_states", "type": "sensor_msgs/msg/JointState"}},
            {"topic": {"name": "/tf", "type": "tf2_msgs/msg/TFMessage"}},
            {"topic": {"name": "/tf_static", "type": "tf2_msgs/msg/TFMessage", "latched": True}},
            {"topic": {"name": "/scoring/tf", "type": "tf2_msgs/msg/TFMessage"}},
            {"topic": {"name": "/aic/gazebo/contacts/off_limit", "type": "ros_gz_interfaces/msg/Contacts"}},
            {"topic": {"name": "/fts_broadcaster/wrench", "type": "geometry_msgs/msg/WrenchStamped"}},
            {"topic": {"name": "/aic_controller/joint_commands", "type": "aic_control_interfaces/msg/JointMotionUpdate"}},
            {"topic": {"name": "/aic_controller/pose_commands", "type": "aic_control_interfaces/msg/MotionUpdate"}},
            {"topic": {"name": "/scoring/insertion_event", "type": "std_msgs/msg/String"}},
            {"topic": {"name": "/aic_controller/controller_state", "type": "aic_control_interfaces/msg/ControllerState"}},
        ]
    }


def generate_config():
    """Generate a full config with 3 trials (2 SFP + 1 SC)."""
    # Trial 1: SFP insertion
    trial_1 = generate_sfp_trial(1)

    # Trial 2: SFP insertion (different randomization)
    trial_2 = generate_sfp_trial(2)

    # Trial 3: SC insertion
    trial_3 = generate_sc_trial()

    config = {
        "scoring": generate_scoring_section(),
        "task_board_limits": {
            "nic_rail": {
                "min_translation": NIC_TRANSLATION_RANGE[0],
                "max_translation": NIC_TRANSLATION_RANGE[1],
            },
            "sc_rail": {
                "min_translation": SC_TRANSLATION_RANGE[0],
                "max_translation": SC_TRANSLATION_RANGE[1],
            },
            "mount_rail": {
                "min_translation": MOUNT_RAIL_TRANSLATION_RANGE[0],
                "max_translation": MOUNT_RAIL_TRANSLATION_RANGE[1],
            },
        },
        "trials": {
            "trial_1": trial_1,
            "trial_2": trial_2,
            "trial_3": trial_3,
        },
        "robot": {
            "home_joint_positions": HOME_JOINTS,
        },
    }

    return config


def main():
    parser = argparse.ArgumentParser(description="Generate randomized trial configs")
    parser.add_argument("--num-configs", type=int, default=50, help="Number of configs to generate")
    parser.add_argument("--output-dir", type=str, default="configs", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, args.num_configs + 1):
        config = generate_config()
        path = os.path.join(output_dir, f"config_{i:03d}.yaml")
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Generated {args.num_configs} configs in {output_dir}/")
    print(f"  config_001.yaml .. config_{args.num_configs:03d}.yaml")
    print(f"  Each config: 2 SFP trials + 1 SC trial = 3 episodes")
    print(f"  Total episodes: {args.num_configs * 3}")


if __name__ == "__main__":
    main()
