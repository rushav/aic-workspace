"""
DataCollector v2 — captures diverse labeled training data during trials.

Improvements over v1:
- 20 viewpoints per trial (up from 7) with random perturbations
- Slightly varied camera orientations (small tilts)
- Wider spatial coverage for better generalization
- SC port support via frame name detection

Run with ground_truth:=true so port TF frames are available.
"""

import numpy as np
import os
import json
import random

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.time import Time
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
        """Capture one labeled sample from center camera."""
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

    def _build_viewpoints(self):
        """Generate ~20 diverse viewpoints with random perturbations.

        The viewing positions form a grid over the port area, with each
        position randomly perturbed to increase data diversity across runs.
        """
        viewpoints = []

        # Grid of base positions: x in [-0.46, -0.34], y in [0.26, 0.34], z in [0.17, 0.32]
        # x: lateral (left/right), y: forward/back, z: height
        x_vals = [-0.44, -0.40, -0.36]
        y_vals = [0.27, 0.30, 0.33]
        z_vals = [0.18, 0.23, 0.28, 0.32]

        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    # Random perturbation: ±15mm in x/y, ±10mm in z
                    px = x + random.uniform(-0.015, 0.015)
                    py = y + random.uniform(-0.015, 0.015)
                    pz = z + random.uniform(-0.010, 0.010)
                    viewpoints.append((px, py, pz))

        # Shuffle so if we time out mid-run, we still get spatial diversity
        random.shuffle(viewpoints)

        # Take up to 20 viewpoints (full grid is 3*3*4=36, subsample for time)
        return viewpoints[:20]

    def _orientation_with_tilt(self, tilt_x=0.0, tilt_y=0.0):
        """Build a quaternion for looking down with small tilts.

        Base orientation: (1, 0, 0, 0) = looking straight down.
        Apply small rotations around x and y axes for varied viewing angles.
        """
        # Small angle approximation for quaternion rotation
        # q = cos(theta/2) + sin(theta/2) * axis
        cx = np.cos(tilt_x / 2)
        sx = np.sin(tilt_x / 2)
        cy = np.cos(tilt_y / 2)
        sy = np.sin(tilt_y / 2)

        # Rotation around x-axis then y-axis, composed with base (1,0,0,0)
        # Base q = (x=1, y=0, z=0, w=0) means 180° around x-axis
        # We apply small additional rotations
        # q_tilt = q_y * q_x (in wxyz convention for multiplication)
        qw = cx * cy
        qx = sx * cy
        qy = cx * sy
        qz = -sx * sy

        # Compose with base orientation (x=1, y=0, z=0, w=0)
        # Hamilton product: q_base * q_tilt
        # q_base = (w=0, x=1, y=0, z=0)
        bw, bx, by, bz = 0.0, 1.0, 0.0, 0.0
        rw = bw * qw - bx * qx - by * qy - bz * qz
        rx = bw * qx + bx * qw + by * qz - bz * qy
        ry = bw * qy - bx * qz + by * qw + bz * qx
        rz = bw * qz + bx * qy - by * qx + bz * qw

        # Normalize
        norm = np.sqrt(rw**2 + rx**2 + ry**2 + rz**2)
        return Quaternion(x=rx/norm, y=ry/norm, z=rz/norm, w=rw/norm)

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
        # SFP ports
        for i in range(2):
            frame = f"task_board/{task.target_module_name}/sfp_port_{i}_link_entrance"
            port_frames[f"sfp_port_{i}"] = frame
        # SC ports
        for i in range(2):
            frame = f"task_board/sc_port_{i}/sc_port_{i}_link_entrance"
            port_frames[f"sc_port_{i}"] = frame

        # Capture from the starting position first
        self._capture_sample(get_observation, port_frames)

        # Generate diverse viewpoints
        viewpoints = self._build_viewpoints()
        self.get_logger().info(f"DataCollector: will visit {len(viewpoints)} viewpoints")

        for idx, (px, py, pz) in enumerate(viewpoints):
            try:
                # Add small random tilt (±5 degrees = ±0.087 rad)
                tilt_x = random.uniform(-0.087, 0.087)
                tilt_y = random.uniform(-0.087, 0.087)
                orientation = self._orientation_with_tilt(tilt_x, tilt_y)

                self.set_pose_target(
                    move_robot=move_robot,
                    pose=Pose(
                        position=Point(x=px, y=py, z=pz),
                        orientation=orientation,
                    ),
                )
                self.sleep_for(1.2)  # slightly shorter settle time for more samples
                self._capture_sample(get_observation, port_frames)
            except Exception as e:
                self.get_logger().warn(f"Failed at viewpoint {idx} ({px:.3f},{py:.3f},{pz:.3f}): {e}")

        self.get_logger().info(
            f"DataCollector: trial complete, {self._sample_id} total samples"
        )
        return True
