"""
MyPolicyV2 — V1 positioning + computed insertion orientation for SFP.

Key insight: The initial TCP orientation is the same as survey (straight down).
We must COMPUTE the correct insertion angle from:
  1. The known grasp geometry (plug offset from TCP)
  2. The estimated insertion direction (horizontal, toward the port)

For SFP: during descent, gradually rotate the gripper so the plug axis aligns
with the horizontal direction toward the port, with TCP position compensation
to keep the plug tip aimed at the port.

For SC: use V1 behavior exactly (proven to score ~30pts).
"""

import math
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

IMG_SIZE = 256
ORIG_W, ORIG_H = 1152, 1024

SURVEY_POSE = Pose(
    position=Point(x=-0.40, y=0.30, z=0.30),
    orientation=Quaternion(x=1.0, y=0.0, z=0.0, w=0.0),
)

SFP_DEPTH_Z = 0.33
SC_DEPTH_Z = 0.45

APPROACH_STEPS = 100
APPROACH_STEP_DT = 0.05

# Empirical plug-tip-to-TCP offsets (V1 — proven good)
SFP_PLUG_Y_OFFSET = 0.0206
SC_PLUG_Y_OFFSET = 0.0132
SC_PLUG_HANG_OFFSET_Z = 0.0164

# Grasp offset RPY from sample_config.yaml (TCP → plug frame)
GRASP_ROLL = 0.4432
GRASP_PITCH = -0.4838
GRASP_YAW = 1.3303
GRASP_POS_SFP = np.array([0.0, 0.015385, 0.04245])
GRASP_POS_SC = np.array([0.0, 0.015385, 0.04045])

# Descent
DESCENT_STEP_M = 0.0005
DESCENT_STEP_DT = 0.05
DESCENT_MAX_TIME = 50.0

# Orientation transition for SFP
ORIENTATION_TRANSITION_Z = 0.04   # complete slerp by 40mm above port
MAX_SLERP_ANGLE = 0.7            # radians (~40°) max rotation from survey

# Stall detection
STALL_TIMEOUT = 2.0
STALL_Z_THRESHOLD = 0.0003
MAX_STALL_RETRIES = 5


# ── Model definition ────────────────────────────────────────────────────────

class PortDetector(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        feat = self.features(x)
        return self.head(feat)


# ── Utility functions ────────────────────────────────────────────────────────

def quat_to_rotation_matrix(qx, qy, qz, qw):
    return np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),       1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),       2*(qy*qz + qw*qx),     1 - 2*(qx**2 + qy**2)],
    ])


def rpy_to_rotation_matrix(roll, pitch, yaw):
    """Extrinsic XYZ (ROS convention): R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ])


def rotation_matrix_to_quat(R):
    """Rotation matrix → (w, x, y, z) quaternion."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (w, x, y, z)


def axis_angle_to_rotation_matrix(axis, angle):
    """Rodrigues formula: rotation matrix from axis (unit) and angle (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c],
    ])


def lerp(a, b, t):
    return a + (b - a) * t


# ── Policy ───────────────────────────────────────────────────────────────────

class MyPolicyV2(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)

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

        self._sfp0_model = self._models["sfp_port_0"]
        self._sfp1_model = self._models["sfp_port_1"]
        self._sc0_model = self._models["sc_port_0"]
        self._sc1_model = self._models["sc_port_1"]

        self._normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # Precompute grasp rotation matrix
        self._R_grasp = rpy_to_rotation_matrix(GRASP_ROLL, GRASP_PITCH, GRASP_YAW)

        self.get_logger().info("MyPolicyV2: all detectors loaded")

    # ── Helpers ──────────────────────────────────────────────────────────

    def _run_detector(self, img_rgb_np, model):
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img_rgb_np)
        resized = TF.resize(pil_img, [IMG_SIZE, IMG_SIZE])
        tensor = TF.to_tensor(resized)
        tensor = self._normalize(tensor)
        batch = tensor.unsqueeze(0)
        with torch.no_grad():
            pred = model(batch)
        return (pred[0, 0].item() * ORIG_W, pred[0, 1].item() * ORIG_H)

    def _get_tcp_pose(self, get_observation):
        obs = get_observation()
        if obs is None:
            return None
        return obs.controller_state.tcp_pose

    def _make_pose(self, position, orientation):
        return Pose(
            position=Point(
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
            ),
            orientation=orientation,
        )

    def _compute_insertion_orientation(self, q_survey, insert_dir_3d):
        """Compute gripper orientation that aligns the plug axis with insert_dir_3d.

        Uses the grasp geometry to determine where the plug axis points at the
        survey orientation, then computes the rotation to align it with the
        desired insertion direction. Limits rotation to MAX_SLERP_ANGLE.

        Returns: target quaternion as (w, x, y, z) tuple.
        """
        # Plug Z axis in base_link at survey orientation
        R_survey = quat_to_rotation_matrix(
            q_survey[1], q_survey[2], q_survey[3], q_survey[0],
        )
        plug_axis = R_survey @ self._R_grasp @ np.array([0.0, 0.0, 1.0])
        plug_axis = plug_axis / np.linalg.norm(plug_axis)

        # Desired plug direction
        desired = insert_dir_3d / np.linalg.norm(insert_dir_3d)

        # Rotation from current plug axis to desired direction
        cross = np.cross(plug_axis, desired)
        cross_norm = np.linalg.norm(cross)
        dot = np.dot(plug_axis, desired)

        if cross_norm < 1e-6:
            # Already aligned (or anti-aligned)
            return q_survey

        axis = cross / cross_norm
        angle = np.arccos(np.clip(dot, -1.0, 1.0))

        # Clamp rotation to MAX_SLERP_ANGLE
        angle = min(angle, MAX_SLERP_ANGLE)

        self.get_logger().info(
            f"Insertion orientation: plug_axis=({plug_axis[0]:.3f}, {plug_axis[1]:.3f}, "
            f"{plug_axis[2]:.3f}), desired=({desired[0]:.3f}, {desired[1]:.3f}, "
            f"{desired[2]:.3f}), angle={np.degrees(angle):.1f}°"
        )

        # Apply rotation to survey orientation
        R_correction = axis_angle_to_rotation_matrix(axis, angle)
        R_target = R_correction @ R_survey
        return rotation_matrix_to_quat(R_target)

    # ── Main entry point ─────────────────────────────────────────────────

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"MyPolicyV2.insert_cable() task: {task}")

        # ── Phase 1: Survey pose (perception) ───────────────────────────
        send_feedback("V2: Survey for perception")
        self.set_pose_target(move_robot=move_robot, pose=SURVEY_POSE)
        self.sleep_for(2.0)

        survey_tcp = self._get_tcp_pose(get_observation)
        if survey_tcp is None:
            return False
        survey_orientation = survey_tcp.orientation
        survey_pos = np.array([
            survey_tcp.position.x, survey_tcp.position.y, survey_tcp.position.z,
        ])

        # ── Phase 2: Detect port ────────────────────────────────────────
        obs = get_observation()
        if obs is None:
            return False

        img_msg = obs.center_image
        img_np = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, 3
        )

        cam_info = obs.center_camera_info
        fx, fy = cam_info.k[0], cam_info.k[4]
        cx, cy = cam_info.k[2], cam_info.k[5]

        if task.port_type == 'sfp':
            if 'port_0' in task.port_name:
                model, port_label = self._sfp0_model, "sfp_port_0"
            else:
                model, port_label = self._sfp1_model, "sfp_port_1"
        elif task.port_type == 'sc':
            if 'sc_port_0' in task.target_module_name:
                model, port_label = self._sc0_model, "sc_port_0"
            else:
                model, port_label = self._sc1_model, "sc_port_1"

        u_px, v_px = self._run_detector(img_np, model)
        self.get_logger().info(f"Detector ({port_label}): pixel=({u_px:.1f}, {v_px:.1f})")

        z_cam = SFP_DEPTH_Z if task.port_type == 'sfp' else SC_DEPTH_Z
        x_cam = (u_px - cx) * z_cam / fx
        y_cam = (v_px - cy) * z_cam / fy

        try:
            tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link", "center_camera/optical", Time(),
            )
        except TransformException as ex:
            self.get_logger().error(f"TF lookup failed: {ex}")
            return False

        tf_t = tf_stamped.transform.translation
        tf_r = tf_stamped.transform.rotation
        R_cam = quat_to_rotation_matrix(tf_r.x, tf_r.y, tf_r.z, tf_r.w)
        port_pos = R_cam @ np.array([x_cam, y_cam, z_cam]) + np.array([tf_t.x, tf_t.y, tf_t.z])

        self.get_logger().info(
            f"Port in base_link: ({port_pos[0]:.4f}, {port_pos[1]:.4f}, {port_pos[2]:.4f})"
        )

        # ── Phase 3: V1 approach positioning ─────────────────────────────
        plug_y_offset = SFP_PLUG_Y_OFFSET if task.port_type == 'sfp' else SC_PLUG_Y_OFFSET
        corrected_port_y = port_pos[1] - plug_y_offset

        approach_pos = np.array([
            port_pos[0],
            corrected_port_y,
            port_pos[2] + 0.1,
        ])

        start_tcp = self._get_tcp_pose(get_observation)
        if start_tcp is None:
            return False
        start_pos = np.array([
            start_tcp.position.x, start_tcp.position.y, start_tcp.position.z,
        ])

        for step in range(APPROACH_STEPS):
            frac = step / float(APPROACH_STEPS)
            interp_pos = lerp(start_pos, approach_pos, frac)
            self.set_pose_target(
                move_robot=move_robot,
                pose=self._make_pose(interp_pos, survey_orientation),
            )
            self.sleep_for(APPROACH_STEP_DT)

        self.set_pose_target(
            move_robot=move_robot,
            pose=self._make_pose(approach_pos, survey_orientation),
        )
        self.sleep_for(1.0)

        final_tcp = self._get_tcp_pose(get_observation)
        if final_tcp is not None:
            fpos = np.array([
                final_tcp.position.x, final_tcp.position.y, final_tcp.position.z,
            ])
            self.get_logger().info(
                f"Approach done: error={np.linalg.norm(fpos - approach_pos)*1000:.1f}mm"
            )

        # ── Phase 4: Descent ────────────────────────────────────────────
        if task.port_type == 'sfp':
            # SFP: computed insertion orientation (proven ~45pts)
            self._descent_with_orientation(
                move_robot, get_observation, send_feedback,
                port_pos, corrected_port_y, survey_orientation, survey_pos,
                GRASP_POS_SFP, -0.015, 'sfp',
            )
        else:
            # SC: V1 descent (survey orientation, read current each step)
            self._descent_sc_v1(
                move_robot, get_observation, send_feedback,
                port_pos, corrected_port_y, survey_orientation,
            )

        # ── Phase 5: Stabilize ───────────────────────────────────────────
        self.get_logger().info("Descent complete. Stabilizing...")
        send_feedback("Insertion complete — stabilizing")
        self.sleep_for(5.0)

        final_tcp = self._get_tcp_pose(get_observation)
        if final_tcp is not None:
            fpos = np.array([
                final_tcp.position.x, final_tcp.position.y, final_tcp.position.z,
            ])
            self.get_logger().info(
                f"=== INSERTION COMPLETE ===\n"
                f"  Final TCP:  ({fpos[0]:.4f}, {fpos[1]:.4f}, {fpos[2]:.4f})\n"
                f"  Port pos:   ({port_pos[0]:.4f}, {port_pos[1]:.4f}, {port_pos[2]:.4f})\n"
                f"  Z below port: {(port_pos[2] - fpos[2])*1000:.1f}mm"
            )

        self.get_logger().info("MyPolicyV2.insert_cable() exiting")
        return True

    # ── Descent with computed orientation (SFP and SC) ──────────────────

    def _descent_with_orientation(self, move_robot, get_observation, send_feedback,
                                   port_pos, corrected_port_y, survey_orientation,
                                   survey_pos, grasp_pos, descent_z_end, port_type):
        """Descent with slerp to computed insertion orientation + position compensation."""

        z_offset_start = 0.1

        q_survey = (
            survey_orientation.w, survey_orientation.x,
            survey_orientation.y, survey_orientation.z,
        )

        # Estimate insertion direction: horizontal, from survey toward port
        insert_dir_xy = port_pos[:2] - survey_pos[:2]
        insert_dir_3d = np.array([insert_dir_xy[0], insert_dir_xy[1], 0.0])
        if np.linalg.norm(insert_dir_3d) < 1e-6:
            insert_dir_3d = np.array([0.0, 1.0, 0.0])  # fallback

        # Compute target orientation
        q_target = self._compute_insertion_orientation(q_survey, insert_dir_3d)

        # Plug offset at survey orientation (baseline for delta)
        R_survey = quat_to_rotation_matrix(
            survey_orientation.x, survey_orientation.y,
            survey_orientation.z, survey_orientation.w,
        )
        offset_at_survey = R_survey @ grasp_pos

        self.get_logger().info(
            f"{port_type.upper()} descent: target_q=({q_target[0]:.3f}, {q_target[1]:.3f}, "
            f"{q_target[2]:.3f}, {q_target[3]:.3f}), z_end={descent_z_end}"
        )

        z_offset = z_offset_start
        step_count = 0
        descent_start = self.time_now()
        last_z = None
        last_z_change_time = self.time_now()
        stall_count = 0

        while z_offset > descent_z_end:
            elapsed = (self.time_now() - descent_start).nanoseconds / 1e9
            if elapsed > DESCENT_MAX_TIME or stall_count > MAX_STALL_RETRIES:
                break

            z_offset -= DESCENT_STEP_M

            # Slerp fraction
            if z_offset >= ORIENTATION_TRANSITION_Z:
                slerp_frac = (z_offset_start - z_offset) / (z_offset_start - ORIENTATION_TRANSITION_Z)
                slerp_frac = np.clip(slerp_frac, 0.0, 1.0)
            else:
                slerp_frac = 1.0

            q_current = quaternion_slerp(q_survey, q_target, slerp_frac)
            orientation = Quaternion(
                w=q_current[0], x=q_current[1],
                y=q_current[2], z=q_current[3],
            )

            # Position compensation for orientation change
            R_current = quat_to_rotation_matrix(
                q_current[1], q_current[2], q_current[3], q_current[0],
            )
            offset_current = R_current @ grasp_pos
            delta = offset_current - offset_at_survey

            if port_type == 'sc':
                base_z = port_pos[2] + z_offset + SC_PLUG_HANG_OFFSET_Z
            else:
                base_z = port_pos[2] + z_offset
            base_pos = np.array([port_pos[0], corrected_port_y, base_z])
            target_pos = base_pos - delta

            self.set_pose_target(
                move_robot=move_robot,
                pose=self._make_pose(target_pos, orientation),
            )
            self.sleep_for(DESCENT_STEP_DT)

            # Stall detection
            tcp_now = self._get_tcp_pose(get_observation)
            if tcp_now is not None:
                tcp_z = tcp_now.position.z
                if last_z is not None:
                    if abs(tcp_z - last_z) > STALL_Z_THRESHOLD:
                        last_z_change_time = self.time_now()
                    else:
                        time_since = (self.time_now() - last_z_change_time).nanoseconds / 1e9
                        if time_since > STALL_TIMEOUT:
                            stall_count += 1
                            self.get_logger().info(
                                f"Stall #{stall_count} at z={tcp_z:.4f}"
                            )
                            last_z_change_time = self.time_now()
                last_z = tcp_z

            if step_count % 40 == 0:
                tcp_z_str = f"{last_z:.4f}" if last_z else "?"
                self.get_logger().info(
                    f"Step {step_count}: z_offset={z_offset:.4f}, "
                    f"tcp_z={tcp_z_str}, slerp={slerp_frac:.2f}, "
                    f"delta=({delta[0]*1000:.1f}, {delta[1]*1000:.1f}, {delta[2]*1000:.1f})mm"
                )
            step_count += 1

    # ── SC descent — V1 behavior ─────────────────────────────────────────

    def _descent_sc_v1(self, move_robot, get_observation, send_feedback,
                        port_pos, corrected_port_y, survey_orientation):
        """SC descent: V1 behavior. Survey orientation, read current each step."""

        z_offset = 0.1
        descent_z_end = -0.10
        step_count = 0
        descent_start = self.time_now()
        last_z = None
        last_z_change_time = self.time_now()
        stall_count = 0
        current_orientation = survey_orientation

        while z_offset > descent_z_end:
            elapsed = (self.time_now() - descent_start).nanoseconds / 1e9
            if elapsed > DESCENT_MAX_TIME or stall_count > MAX_STALL_RETRIES:
                break

            z_offset -= DESCENT_STEP_M
            target_z = port_pos[2] + z_offset + SC_PLUG_HANG_OFFSET_Z
            target_pos = np.array([port_pos[0], corrected_port_y, target_z])

            tcp_now = self._get_tcp_pose(get_observation)
            if tcp_now is not None:
                current_orientation = tcp_now.orientation

            self.set_pose_target(
                move_robot=move_robot,
                pose=self._make_pose(target_pos, current_orientation),
            )
            self.sleep_for(DESCENT_STEP_DT)

            if tcp_now is not None:
                tcp_z = tcp_now.position.z
                if last_z is not None:
                    if abs(tcp_z - last_z) > STALL_Z_THRESHOLD:
                        last_z_change_time = self.time_now()
                    else:
                        time_since = (self.time_now() - last_z_change_time).nanoseconds / 1e9
                        if time_since > STALL_TIMEOUT:
                            stall_count += 1
                            self.get_logger().info(
                                f"SC stall #{stall_count} at z={tcp_z:.4f}"
                            )
                            last_z_change_time = self.time_now()
                last_z = tcp_z

            if step_count % 40 == 0:
                tcp_z_str = f"{last_z:.4f}" if last_z else "?"
                self.get_logger().info(
                    f"SC step {step_count}: z_offset={z_offset:.4f}, tcp_z={tcp_z_str}"
                )
            step_count += 1

