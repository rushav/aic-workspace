"""
DataCollector — captures labeled training data during trials.

Instead of attempting insertion, this policy:
1. Moves the arm to several different viewing positions
2. At each position, captures camera images + ground truth port pose
3. Saves image-label pairs to disk

Run with ground_truth:=true so port TF frames are available.
"""

import numpy as np
import os
import json
import time as pytime

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.time import Time
from rclpy.parameter import Parameter
from tf2_ros import TransformException


class DataCollector(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._save_dir = os.path.expanduser("~/aic-workspace/datasets/port_detection")
        os.makedirs(self._save_dir, exist_ok=True)

        # Count existing samples to avoid overwriting
        existing = [f for f in os.listdir(self._save_dir) if f.endswith('.png')]
        self._sample_id = len(existing)
        self.get_logger().info(f"DataCollector: saving to {self._save_dir}, starting at sample {self._sample_id}")

    def _get_port_pixel(self, camera_frame, port_frame, cam_info):
        """Project a port's 3D position into pixel coordinates."""
        try:
            t = self._parent_node._tf_buffer.lookup_transform(
                camera_frame, port_frame, Time()
            )
            p = t.transform.translation
            r = t.transform.rotation

            fx = cam_info.k[0]
            fy = cam_info.k[4]
            cx = cam_info.k[2]
            cy = cam_info.k[5]

            if p.z <= 0:
                return None

            u = fx * p.x / p.z + cx
            v = fy * p.y / p.z + cy

            return {
                "u": float(u),
                "v": float(v),
                "x": float(p.x),
                "y": float(p.y),
                "z": float(p.z),
                "qx": float(r.x),
                "qy": float(r.y),
                "qz": float(r.z),
                "qw": float(r.w),
            }
        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed for {port_frame}: {e}")
            return None

    def _capture_sample(self, get_observation, port_frames):
        """Capture one labeled sample from all cameras."""
        obs = get_observation()
        if obs is None:
            return

        cam_info = obs.center_camera_info
        camera_frame = "center_camera/optical"

        # Get labels for all visible ports
        labels = {}
        for name, frame in port_frames.items():
            result = self._get_port_pixel(camera_frame, frame, cam_info)
            if result is not None:
                # Check if projection is within image bounds
                if 0 <= result["u"] < cam_info.width and 0 <= result["v"] < cam_info.height:
                    labels[name] = result

        if not labels:
            self.get_logger().info("No ports visible in frame, skipping")
            return

        # Save image
        img_data = np.frombuffer(obs.center_image.data, dtype=np.uint8)
        img = img_data.reshape(obs.center_image.height, obs.center_image.width, 3)

        from PIL import Image as PILImage
        img_path = os.path.join(self._save_dir, f"sample_{self._sample_id:04d}.png")
        PILImage.fromarray(img).save(img_path)

        # Save labels
        label_path = os.path.join(self._save_dir, f"sample_{self._sample_id:04d}.json")
        meta = {
            "image": f"sample_{self._sample_id:04d}.png",
            "width": int(cam_info.width),
            "height": int(cam_info.height),
            "fx": float(cam_info.k[0]),
            "fy": float(cam_info.k[4]),
            "cx": float(cam_info.k[2]),
            "cy": float(cam_info.k[5]),
            "ports": labels,
        }
        with open(label_path, "w") as f:
            json.dump(meta, f, indent=2)

        self.get_logger().info(
            f"Sample {self._sample_id}: saved {len(labels)} port labels"
        )
        self._sample_id += 1

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"DataCollector: task={task}")
        send_feedback("Collecting training data")

        # Build port frame names from the task
        port_frames = {}
        # Try both SFP ports
        for i in range(2):
            frame = f"task_board/{task.target_module_name}/sfp_port_{i}_link_entrance"
            port_frames[f"sfp_port_{i}"] = frame
        # Also try SC ports if present
        for i in range(2):
            frame = f"task_board/sc_port_{i}/sc_port_{i}_link_entrance"
            port_frames[f"sc_port_{i}"] = frame

        # Capture from the starting position first
        self._capture_sample(get_observation, port_frames)

        # Move to several different viewing positions and capture at each
        positions = [
            # (x, y, z) — different viewpoints around the port area
            (-0.40, 0.30, 0.30),  # center high
            (-0.40, 0.30, 0.22),  # center low
            (-0.38, 0.28, 0.25),  # slight offset
            (-0.42, 0.32, 0.25),  # other offset
            (-0.40, 0.30, 0.18),  # close up
            (-0.36, 0.30, 0.28),  # shifted left
            (-0.44, 0.30, 0.28),  # shifted right
        ]

        for px, py, pz in positions:
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=Pose(
                        position=Point(x=px, y=py, z=pz),
                        orientation=Quaternion(x=1.0, y=0.0, z=0.0, w=0.0),
                    ),
                )
                self.sleep_for(1.5)  # wait for arm to settle
                self._capture_sample(get_observation, port_frames)
            except Exception as e:
                self.get_logger().warn(f"Failed at position ({px},{py},{pz}): {e}")

        self.get_logger().info(
            f"DataCollector: trial complete, {self._sample_id} total samples"
        )
        return True
