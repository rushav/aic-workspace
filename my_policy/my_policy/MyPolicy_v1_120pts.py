"""
MyPolicy — perception bridge + vertical descent (CheatCode strategy).

Pipeline:
1. Move to survey pose, capture image, run regression detector
2. Back-project detected port pixel to 3D, transform to base_link
3. Approach: interpolate to port XY, 100mm above port Z (5s)
4. Vertical descent: decrease Z in 0.5mm steps, XY locked to port position
5. Descend from z_offset=+100mm to z_offset=-15mm (past port Z)
6. Stabilize 5s, return True

Four per-port detectors (sfp_port_0, sfp_port_1, sc_port_0, sc_port_1),
each a ResNet-18 with a regression head predicting (u, v) for one port.
Architecture matches train_regression_detector.py.
"""

import numpy as np
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
from torchvision.transforms import Normalize
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
from transforms3d._gohlketransforms import quaternion_slerp


# ── Constants ────────────────────────────────────────────────────────────────

# Must match train_regression_detector.py
IMG_SIZE = 256

# Original camera resolution (for denormalizing predictions)
ORIG_W, ORIG_H = 1152, 1024

# Survey pose: arm positioned above the task board looking straight down.
# Orientation (1, 0, 0, 0) = 180° around X = camera facing down.
# Position chosen to give a centered view of the port area from ~30cm above.
SURVEY_POSE = Pose(
    position=Point(x=-0.40, y=0.30, z=0.30),
    orientation=Quaternion(x=1.0, y=0.0, z=0.0, w=0.0),
)

# Depth estimates (meters) for back-projecting pixels to 3D, per port type.
# Camera-to-port distance from the survey pose.
SFP_DEPTH_Z = 0.33    # SFP ports: mean z in camera frame from training data
SC_DEPTH_Z = 0.45     # SC ports: mean z in camera frame from training data

# Approach interpolation parameters (matches CheatCode's strategy)
APPROACH_STEPS = 100      # number of interpolation steps
APPROACH_STEP_DT = 0.05   # seconds per step → 100 * 0.05 = 5.0s total

# Descent parameters
DESCENT_STEP_M = 0.0005         # 0.5mm per step
DESCENT_STEP_DT = 0.05          # seconds per step (~20 Hz)

# Empirical plug-tip-to-TCP offset in world Y (measured with ground_truth:=true)
# The plug tip is this far ahead of the TCP in the Y direction.
# To place the plug at the port, shift TCP target backward (negative Y).
SFP_PLUG_Y_OFFSET = 0.0206   # 20.6mm — consistent across SFP trials
SC_PLUG_Y_OFFSET = 0.0132    # 13.2mm — from SC trial
SC_PLUG_HANG_OFFSET_Z = 0.0164  # SC plug tip hangs 16.4mm below TCP in Z


# ── Model definition (must match train_regression_detector.py exactly) ───────

class PortDetector(nn.Module):
    """ResNet-18 backbone → regression head → (u_norm, v_norm) for ONE port.

    Simple and effective: no heatmaps, no decoder. Directly regresses
    normalized pixel coordinates for a single port.
    """

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),  # just u_norm, v_norm
        )

    def forward(self, x):
        feat = self.features(x)
        return self.head(feat)


# ── Utility functions ────────────────────────────────────────────────────────

def quat_to_rotation_matrix(qx, qy, qz, qw):
    """Convert a quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    return np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),       1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),       2*(qy*qz + qw*qx),     1 - 2*(qx**2 + qy**2)],
    ])


def lerp(a, b, t):
    """Linear interpolation between scalars or arrays a and b by fraction t."""
    return a + (b - a) * t




# ── Policy ───────────────────────────────────────────────────────────────────

class MyPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)

        # ── Load all 4 per-port detectors ────────────────────────────────
        ckpt_dir = os.path.expanduser("~/aic-workspace/checkpoints")
        model_keys = ["sfp_port_0", "sfp_port_1", "sc_port_0", "sc_port_1"]

        self._models = {}
        for key in model_keys:
            path = os.path.join(ckpt_dir, f"{key}_best.pth")
            self.get_logger().info(f"Loading detector: {path}")
            m = PortDetector()
            m.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True))
            m.eval()
            self._models[key] = m

        # Convenience aliases
        self._sfp0_model = self._models["sfp_port_0"]
        self._sfp1_model = self._models["sfp_port_1"]
        self._sc0_model = self._models["sc_port_0"]
        self._sc1_model = self._models["sc_port_1"]

        # ImageNet normalization (same as training)
        self._normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.get_logger().info("All 4 per-port detectors loaded successfully")

    # ── Helper methods ───────────────────────────────────────────────────

    def _run_detector(self, img_rgb_np, model):
        """Run a single-port regression detector on a raw RGB numpy image.

        Args:
            img_rgb_np: (H, W, 3) uint8 numpy array from the camera
            model: PortDetector regression model (single port, 2 outputs)

        Returns:
            (u_px, v_px) tuple in original image coordinates.
        """
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(img_rgb_np)
        resized = TF.resize(pil_img, [IMG_SIZE, IMG_SIZE])

        tensor = TF.to_tensor(resized)
        tensor = self._normalize(tensor)
        batch = tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(batch)  # (1, 2) → u_norm, v_norm

        u_px = pred[0, 0].item() * ORIG_W
        v_px = pred[0, 1].item() * ORIG_H
        return (u_px, v_px)

    def _get_tcp_pose(self, get_observation):
        """Get the current TCP pose from controller_state in the observation."""
        obs = get_observation()
        if obs is None:
            self.get_logger().error("Failed to get observation for TCP pose")
            return None
        return obs.controller_state.tcp_pose

    def _make_pose(self, position, orientation):
        """Build a geometry_msgs/Pose from a numpy position array and Quaternion."""
        return Pose(
            position=Point(
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
            ),
            orientation=orientation,
        )

    # ── Main entry point ─────────────────────────────────────────────────

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"MyPolicy.insert_cable() task: {task}")
        send_feedback("Perception bridge — moving to survey pose")

        # ── Step 1: Move arm to survey pose ──────────────────────────────
        self.set_pose_target(move_robot=move_robot, pose=SURVEY_POSE)
        self.sleep_for(2.0)

        # Capture the TCP orientation at survey pose — we'll reuse this
        # for approach and insertion (the gripper is already facing the board).
        survey_tcp = self._get_tcp_pose(get_observation)
        if survey_tcp is None:
            self.get_logger().error("Failed to get TCP pose at survey pose")
            return False
        survey_orientation = survey_tcp.orientation

        # ── Step 2: Capture image and camera intrinsics ──────────────────
        obs = get_observation()
        if obs is None:
            self.get_logger().error("Failed to get observation")
            return False

        img_msg = obs.center_image
        img_np = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, 3
        )

        cam_info = obs.center_camera_info
        fx = cam_info.k[0]
        fy = cam_info.k[4]
        cx = cam_info.k[2]
        cy = cam_info.k[5]

        self.get_logger().info(
            f"Captured image: {img_msg.width}x{img_msg.height}, "
            f"intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}"
        )

        # ── Step 3: Run the detector ─────────────────────────────────────
        # Select the per-port model based on task
        if task.port_type == 'sfp':
            if 'port_0' in task.port_name:
                model = self._sfp0_model
                port_label = "sfp_port_0"
            else:
                model = self._sfp1_model
                port_label = "sfp_port_1"
        elif task.port_type == 'sc':
            if 'sc_port_0' in task.target_module_name:
                model = self._sc0_model
                port_label = "sc_port_0"
            else:
                model = self._sc1_model
                port_label = "sc_port_1"

        u_px, v_px = self._run_detector(img_np, model)

        self.get_logger().info(
            f"Detector ({port_label}): pixel=({u_px:.1f}, {v_px:.1f})"
        )

        # ── Step 4: Back-project pixel to 3D point in camera frame ───────
        if task.port_type == 'sfp':
            z_cam = SFP_DEPTH_Z
        else:
            z_cam = SC_DEPTH_Z
        x_cam = (u_px - cx) * z_cam / fx
        y_cam = (v_px - cy) * z_cam / fy

        self.get_logger().info(
            f"Camera-frame 3D point: x={x_cam:.4f} y={y_cam:.4f} z={z_cam:.4f} (meters)"
        )

        # ── Step 5: Transform from camera frame to robot base frame ──────
        camera_frame = "center_camera/optical"
        base_frame = "base_link"

        try:
            tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                base_frame, camera_frame, Time(),
            )
        except TransformException as ex:
            self.get_logger().error(
                f"TF lookup {camera_frame} → {base_frame} failed: {ex}"
            )
            return False

        tf_trans = tf_stamped.transform.translation
        tf_rot = tf_stamped.transform.rotation
        R = quat_to_rotation_matrix(tf_rot.x, tf_rot.y, tf_rot.z, tf_rot.w)

        p_cam = np.array([x_cam, y_cam, z_cam])
        port_pos_base = R @ p_cam + np.array([tf_trans.x, tf_trans.y, tf_trans.z])

        self.get_logger().info(
            f"Base-frame 3D point: x={port_pos_base[0]:.4f} y={port_pos_base[1]:.4f} z={port_pos_base[2]:.4f}"
        )

        # ── Perception summary ───────────────────────────────────────────
        self.get_logger().info(
            f"=== PERCEPTION SUMMARY for {task.port_name} ===\n"
            f"  Pixel prediction:   ({u_px:.1f}, {v_px:.1f})\n"
            f"  Camera-frame 3D:    ({x_cam:.4f}, {y_cam:.4f}, {z_cam:.4f})\n"
            f"  Base-frame 3D:      ({port_pos_base[0]:.4f}, {port_pos_base[1]:.4f}, {port_pos_base[2]:.4f})"
        )

        # ── Step 6: Approach — move to port XY, 100mm above port Z ──────
        # CheatCode strategy: no insertion axis. Go directly above the port
        # (same XY), then descend vertically in world Z.
        approach_orientation = survey_orientation

        # Apply empirical plug-tip Y offset so the plug tip (not TCP) targets the port
        if task.port_type == 'sfp':
            plug_y_offset = SFP_PLUG_Y_OFFSET
        else:
            plug_y_offset = SC_PLUG_Y_OFFSET

        corrected_port_y = port_pos_base[1] - plug_y_offset
        send_feedback(
            f"Plug Y offset: {plug_y_offset*1000:.1f}mm, "
            f"corrected port Y: {corrected_port_y:.4f} (was {port_pos_base[1]:.4f})"
        )

        approach_pos = np.array([
            port_pos_base[0],
            corrected_port_y,
            port_pos_base[2] + 0.1,  # 100mm above port Z
        ])

        send_feedback(
            f"Approach target: XY=({approach_pos[0]:.4f}, {approach_pos[1]:.4f}), "
            f"Z={approach_pos[2]:.4f} (port Z + 100mm)"
        )
        self.get_logger().info(
            f"Approach target: ({approach_pos[0]:.4f}, {approach_pos[1]:.4f}, {approach_pos[2]:.4f}), "
            f"port Z={port_pos_base[2]:.4f}"
        )

        # Smooth interpolation from current TCP to approach position (5s)
        send_feedback("Approaching detected port")

        start_tcp = self._get_tcp_pose(get_observation)
        if start_tcp is None:
            return False

        start_pos = np.array([
            start_tcp.position.x,
            start_tcp.position.y,
            start_tcp.position.z,
        ])

        self.get_logger().info(
            f"Starting approach interpolation: {APPROACH_STEPS} steps over "
            f"{APPROACH_STEPS * APPROACH_STEP_DT:.1f}s"
        )

        for step in range(APPROACH_STEPS):
            frac = step / float(APPROACH_STEPS)
            interp_pos = lerp(start_pos, approach_pos, frac)
            self.set_pose_target(
                move_robot=move_robot,
                pose=self._make_pose(interp_pos, approach_orientation),
            )
            self.sleep_for(APPROACH_STEP_DT)

        # Settle at approach pose
        self.set_pose_target(
            move_robot=move_robot,
            pose=self._make_pose(approach_pos, approach_orientation),
        )
        self.sleep_for(1.0)

        # Log approach result
        final_tcp = self._get_tcp_pose(get_observation)
        if final_tcp is not None:
            final_pos = np.array([
                final_tcp.position.x,
                final_tcp.position.y,
                final_tcp.position.z,
            ])
            pos_error = np.linalg.norm(final_pos - approach_pos)
            self.get_logger().info(
                f"=== APPROACH COMPLETE ===\n"
                f"  Target:  ({approach_pos[0]:.4f}, {approach_pos[1]:.4f}, {approach_pos[2]:.4f})\n"
                f"  Actual:  ({final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f})\n"
                f"  Error:   {pos_error*1000:.1f}mm"
            )

        # Use the actual achieved orientation for descent (avoids controller limit violations)
        descent_orientation = final_tcp.orientation if final_tcp is not None else approach_orientation

        # ── Step 7: Vertical descent in world Z ─────────────────────────
        # Descend from 100mm above port Z past the port.
        # XY stays locked to port XY. Exactly like CheatCode.
        if task.port_type == 'sfp':
            descent_z_end = -0.015   # 15mm past port Z
        else:
            descent_z_end = -0.10    # 100mm past port Z

        send_feedback(f"Descent: z_offset 0.1 → {descent_z_end}, port_z={port_pos_base[2]:.4f}")

        z_offset = 0.1  # start 100mm above port Z
        step_count = 0
        current_orientation = descent_orientation  # start with approach orientation

        self.get_logger().info(
            f"Starting vertical descent: z_offset 0.1 → {descent_z_end}, "
            f"step={DESCENT_STEP_M*1000:.1f}mm, port_z={port_pos_base[2]:.4f}"
        )

        while z_offset > descent_z_end:
            z_offset -= DESCENT_STEP_M
            if task.port_type == 'sc':
                target_z = port_pos_base[2] + z_offset + SC_PLUG_HANG_OFFSET_Z
            else:
                target_z = port_pos_base[2] + z_offset
            target_pos = np.array([port_pos_base[0], corrected_port_y, target_z])

            tcp_now = self._get_tcp_pose(get_observation)
            if tcp_now is not None:
                current_orientation = tcp_now.orientation

            if task.port_type == 'sc' and tcp_now is not None:
                # Compute direction from current TCP to detected port
                tcp_pos = np.array([
                    tcp_now.position.x,
                    tcp_now.position.y,
                    tcp_now.position.z,
                ])
                port_target = np.array([
                    port_pos_base[0], corrected_port_y, port_pos_base[2],
                ])
                direction = port_target - tcp_pos
                d_norm = np.linalg.norm(direction)
                if d_norm > 1e-6:
                    direction = direction / d_norm
                    # Quaternion rotating gripper Z-axis (0,0,-1) to direction
                    down = np.array([0.0, 0.0, -1.0])
                    cross = np.cross(down, direction)
                    dot = np.dot(down, direction)
                    cross_norm = np.linalg.norm(cross)
                    if cross_norm > 1e-6:
                        axis = cross / cross_norm
                        angle = np.arccos(np.clip(dot, -1.0, 1.0))
                        half = angle / 2.0
                        w = np.cos(half)
                        s = np.sin(half)
                        q_look = (w, axis[0]*s, axis[1]*s, axis[2]*s)
                    else:
                        # Vectors are parallel — identity or 180° flip
                        if dot > 0:
                            q_look = (1.0, 0.0, 0.0, 0.0)
                        else:
                            q_look = (0.0, 1.0, 0.0, 0.0)  # 180° around X
                    # Slerp gently from current orientation toward look-at target
                    q_current = (
                        tcp_now.orientation.w,
                        tcp_now.orientation.x,
                        tcp_now.orientation.y,
                        tcp_now.orientation.z,
                    )
                    q_slerped = quaternion_slerp(q_current, q_look, 0.05)
                    current_orientation = Quaternion(
                        w=q_slerped[0], x=q_slerped[1],
                        y=q_slerped[2], z=q_slerped[3],
                    )

            self.set_pose_target(
                move_robot=move_robot,
                pose=self._make_pose(target_pos, current_orientation),
            )
            self.sleep_for(DESCENT_STEP_DT)

            if step_count % 20 == 0:
                self.get_logger().info(
                    f"Descent step {step_count}: z_offset={z_offset:.4f}, target_z={target_z:.4f}"
                )
            step_count += 1

        # ── Step 8: Stabilize ────────────────────────────────────────────
        self.get_logger().info(
            f"Descent complete ({step_count} steps). Stabilizing for 5s..."
        )
        send_feedback("Insertion complete — stabilizing")
        self.sleep_for(5.0)

        # Log final state
        final_tcp = self._get_tcp_pose(get_observation)
        if final_tcp is not None:
            fpos = np.array([final_tcp.position.x, final_tcp.position.y, final_tcp.position.z])
            self.get_logger().info(
                f"=== INSERTION COMPLETE ===\n"
                f"  Final TCP:        ({fpos[0]:.4f}, {fpos[1]:.4f}, {fpos[2]:.4f})\n"
                f"  Port position:    ({port_pos_base[0]:.4f}, {port_pos_base[1]:.4f}, {port_pos_base[2]:.4f})\n"
                f"  XY error:         {np.linalg.norm(fpos[:2] - port_pos_base[:2])*1000:.1f}mm\n"
                f"  Z below port:     {(port_pos_base[2] - fpos[2])*1000:.1f}mm"
            )

        self.get_logger().info("MyPolicy.insert_cable() exiting")
        return True
