"""
MyPolicy — perception bridge + approach + insertion with force feedback.

Pipeline:
1. Move to survey pose, capture image, run heatmap detector
2. Back-project detected port pixel to 3D, transform to base_link
3. Compute approach pose (80mm above port along insertion axis)
4. Smooth interpolation to approach pose (5s, ~20 Hz)
5. Descent in 0.5mm steps with force monitoring and XY integral correction
6. Spiral search if contact detected before insertion depth reached
7. Return True when insertion depth reached or spiral exhausted

The detector is a ResNet-34 encoder-decoder (HeatmapDetector) that outputs
a 2-channel 96x96 heatmap. Soft-argmax extracts sub-pixel (u, v) coordinates
for each SFP port. Architecture must match train_detector.py exactly.
"""

import numpy as np
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
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


# ── Constants ────────────────────────────────────────────────────────────────

# Must match train_detector.py exactly
IMG_SIZE = 384
HEATMAP_SIZE = 96

# Original camera resolution (for denormalizing predictions)
ORIG_W, ORIG_H = 1152, 1024

# Survey pose: arm positioned above the task board looking straight down.
# Orientation (1, 0, 0, 0) = 180° around X = camera facing down.
# Position chosen to give a centered view of the port area from ~30cm above.
SURVEY_POSE = Pose(
    position=Point(x=-0.40, y=0.30, z=0.30),
    orientation=Quaternion(x=1.0, y=0.0, z=0.0, w=0.0),
)

# Default depth estimate (meters) used when back-projecting pixels to 3D.
# This is the typical camera-to-port distance from the survey pose.
# Learned from dataset statistics: mean z = 0.326m.
DEFAULT_DEPTH_Z = 0.33

# Approach offset: how far above the port (along insertion axis) to stop.
APPROACH_OFFSET_M = 0.08  # 80mm

# Approach interpolation parameters (matches CheatCode's strategy)
APPROACH_STEPS = 100      # number of interpolation steps
APPROACH_STEP_DT = 0.05   # seconds per step → 100 * 0.05 = 5.0s total

# Descent parameters
DESCENT_STEP_M = 0.0005         # 0.5mm per step
DESCENT_STEP_DT = 0.05          # seconds per step (~20 Hz)
INSERTION_DEPTH_M = 0.030       # 30mm past approach pose = insertion complete
CONTACT_FORCE_N = 15.0          # force threshold to trigger spiral search
CLEAR_FORCE_N = 8.0             # force below this means port opening found
XY_INTEGRAL_GAIN = 0.001        # gain for lateral force integral correction
XY_INTEGRAL_WINDUP = 0.005      # max accumulated correction (5mm)

# Spiral search parameters
SPIRAL_PUSH_M = 0.002           # 2mm push during each spiral probe
SPIRAL_RADII_M = [0.002, 0.004] # two rings: 2mm, 4mm
SPIRAL_POINTS_PER_RING = 8      # 8 evenly-spaced points per ring


# ── Model definition (must match train_detector.py exactly) ──────────────────

def soft_argmax_2d(heatmap, temperature=10.0):
    """Extract (x, y) coordinates from heatmap via differentiable soft-argmax.

    Args:
        heatmap: (B, C, H, W) tensor — raw heatmap logits from the detector
        temperature: scaling factor; higher = sharper peak selection

    Returns:
        (B, C, 2) tensor of (x, y) coordinates in heatmap pixel space [0, H-1]
    """
    B, C, H, W = heatmap.shape
    flat = heatmap.view(B, C, -1)
    weights = F.softmax(flat * temperature, dim=-1).view(B, C, H, W)

    device = heatmap.device
    y_coords = torch.arange(H, dtype=torch.float32, device=device).view(1, 1, H, 1)
    x_coords = torch.arange(W, dtype=torch.float32, device=device).view(1, 1, 1, W)

    x = (weights * x_coords).sum(dim=(2, 3))
    y = (weights * y_coords).sum(dim=(2, 3))
    return torch.stack([x, y], dim=-1)  # (B, C, 2)


class HeatmapDetector(nn.Module):
    """ResNet-34 encoder + upsampling decoder → 2-channel heatmap (96x96).

    This is the exact architecture from train_detector.py. The encoder
    uses pretrained ResNet-34 layers (conv1 through layer4). The decoder
    upsamples with transposed convolutions and skip connections from
    the encoder layers.

    Channel 0 = SFP port 0, Channel 1 = SFP port 1.
    """

    def __init__(self, num_keypoints=2):
        super().__init__()
        # Use weights=None since we'll load our own trained weights
        backbone = models.resnet34(weights=None)

        self.conv1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Dropout2d(0.15))
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Dropout2d(0.1))
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.head = nn.Conv2d(64, num_keypoints, 1)

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d3 = self.up4(x4)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))
        d2 = self.up3(d3)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))
        d1 = self.up2(d2)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))
        return self.head(d1)


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


def build_insertion_frame(insertion_axis):
    """Build an orthonormal frame aligned with the insertion axis.

    Returns (x_perp, y_perp, z_ins) where z_ins = insertion_axis (points
    away from port toward TCP), and x_perp/y_perp span the plane
    perpendicular to the insertion axis (used for spiral search and
    lateral force correction).
    """
    z_ins = insertion_axis / np.linalg.norm(insertion_axis)

    # Pick a reference vector not parallel to z_ins to cross-product against
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z_ins, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])

    x_perp = np.cross(z_ins, ref)
    x_perp = x_perp / np.linalg.norm(x_perp)
    y_perp = np.cross(z_ins, x_perp)
    y_perp = y_perp / np.linalg.norm(y_perp)

    return x_perp, y_perp, z_ins


# ── Policy ───────────────────────────────────────────────────────────────────

class MyPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)

        # ── Load the trained detector ────────────────────────────────────
        ckpt_path = os.path.expanduser(
            "~/aic-workspace/checkpoints/port_detector_best.pth"
        )
        self.get_logger().info(f"Loading detector from {ckpt_path}")

        self._model = HeatmapDetector(num_keypoints=2)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state_dict)
        self._model.eval()

        # ImageNet normalization (same as training)
        self._normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.get_logger().info("Detector loaded successfully")

    # ── Helper methods ───────────────────────────────────────────────────

    def _run_detector(self, img_rgb_np):
        """Run the heatmap detector on a raw RGB numpy image.

        Args:
            img_rgb_np: (H, W, 3) uint8 numpy array from the camera

        Returns:
            List of (u_px, v_px) tuples in original image coordinates,
            one per detected port (port 0, port 1).
        """
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(img_rgb_np)
        resized = TF.resize(pil_img, [IMG_SIZE, IMG_SIZE])

        tensor = TF.to_tensor(resized)
        tensor = self._normalize(tensor)
        batch = tensor.unsqueeze(0)

        with torch.no_grad():
            pred_hm = self._model(batch)

        pred_coords = soft_argmax_2d(pred_hm)

        results = []
        for port_idx in range(2):
            hm_x = pred_coords[0, port_idx, 0].item()
            hm_y = pred_coords[0, port_idx, 1].item()
            u_px = hm_x / HEATMAP_SIZE * ORIG_W
            v_px = hm_y / HEATMAP_SIZE * ORIG_H
            results.append((u_px, v_px))

        return results

    def _get_tcp_pose(self, get_observation):
        """Get the current TCP pose from controller_state in the observation."""
        obs = get_observation()
        if obs is None:
            self.get_logger().error("Failed to get observation for TCP pose")
            return None
        return obs.controller_state.tcp_pose

    def _get_wrench(self, get_observation):
        """Get the current wrist wrench from the observation.

        Returns:
            (fx, fy, fz) force vector in the wrench sensor frame, or None.
        """
        obs = get_observation()
        if obs is None:
            return None
        w = obs.wrist_wrench.wrench.force
        return np.array([w.x, w.y, w.z])

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

    # ── Spiral search ────────────────────────────────────────────────────

    def _spiral_search(
        self,
        center_pos,
        orientation,
        insertion_axis,
        x_perp,
        y_perp,
        move_robot,
        get_observation,
    ):
        """Probe points on concentric rings perpendicular to the insertion axis.

        At each probe point, push SPIRAL_PUSH_M along the insertion axis.
        If the axial force drops below CLEAR_FORCE_N during the push, the
        port opening has been found — return the new position to resume
        descent from.

        Args:
            center_pos:     numpy (3,) — the XY center in base frame (where
                            contact was first detected)
            orientation:    Quaternion — gripper orientation (held fixed)
            insertion_axis: numpy (3,) unit vector (port → TCP direction)
            x_perp, y_perp: numpy (3,) orthonormal basis of the perpendicular plane
            move_robot:     MoveRobotCallback
            get_observation: GetObservationCallback

        Returns:
            numpy (3,) — the new descent position if a clear path was found,
            or None if the full spiral completed without success.
        """
        # Descent direction is opposite to insertion_axis
        descent_dir = -insertion_axis

        for ring_idx, radius in enumerate(SPIRAL_RADII_M):
            self.get_logger().info(
                f"Spiral: ring {ring_idx + 1}, radius={radius * 1000:.1f}mm, "
                f"{SPIRAL_POINTS_PER_RING} points"
            )

            for pt_idx in range(SPIRAL_POINTS_PER_RING):
                # Evenly-spaced angle around the ring
                angle = 2.0 * math.pi * pt_idx / SPIRAL_POINTS_PER_RING

                # Compute the probe position: offset from center in the
                # perpendicular plane
                offset = radius * (math.cos(angle) * x_perp + math.sin(angle) * y_perp)
                probe_pos = center_pos + offset

                # Move to the probe position (laterally, same depth as contact)
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self._make_pose(probe_pos, orientation),
                )
                self.sleep_for(DESCENT_STEP_DT)

                # Attempt a small push along the insertion direction
                push_steps = int(SPIRAL_PUSH_M / DESCENT_STEP_M)
                for push_step in range(push_steps):
                    push_pos = probe_pos + (push_step + 1) * DESCENT_STEP_M * descent_dir
                    self.set_pose_target(
                        move_robot=move_robot,
                        pose=self._make_pose(push_pos, orientation),
                    )
                    self.sleep_for(DESCENT_STEP_DT)

                    # Check if force has dropped — meaning we found the opening
                    force_vec = self._get_wrench(get_observation)
                    if force_vec is not None:
                        axial_force = abs(np.dot(force_vec, insertion_axis))
                        if axial_force < CLEAR_FORCE_N:
                            self.get_logger().info(
                                f"Spiral: opening found at ring {ring_idx + 1} "
                                f"point {pt_idx}, force={axial_force:.1f}N"
                            )
                            # Return the position where force dropped —
                            # descent will resume from here
                            return push_pos

                # Retract back to contact depth before trying next point
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self._make_pose(probe_pos, orientation),
                )
                self.sleep_for(DESCENT_STEP_DT)

            self.get_logger().info(
                f"Spiral: ring {ring_idx + 1} complete, no opening found"
            )

        # Full spiral exhausted without finding the opening
        return None

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
        port_pixels = self._run_detector(img_np)

        target_port_idx = 0
        if "port_1" in task.port_name:
            target_port_idx = 1

        u_px, v_px = port_pixels[target_port_idx]

        self.get_logger().info(
            f"Detector prediction for {task.port_name} (port {target_port_idx}): "
            f"pixel=({u_px:.1f}, {v_px:.1f})"
        )

        other_idx = 1 - target_port_idx
        ou, ov = port_pixels[other_idx]
        self.get_logger().info(
            f"Other port (port {other_idx}): pixel=({ou:.1f}, {ov:.1f})"
        )

        # ── Step 4: Back-project pixel to 3D point in camera frame ───────
        z_cam = DEFAULT_DEPTH_Z
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

        # ── Step 6: Compute approach pose ────────────────────────────────
        send_feedback("Computing approach pose")

        tcp_pose = self._get_tcp_pose(get_observation)
        if tcp_pose is None:
            return False

        tcp_pos = np.array([
            tcp_pose.position.x,
            tcp_pose.position.y,
            tcp_pose.position.z,
        ])

        self.get_logger().info(
            f"Current TCP position: x={tcp_pos[0]:.4f} y={tcp_pos[1]:.4f} z={tcp_pos[2]:.4f}"
        )

        # Insertion axis: unit vector from port toward TCP
        axis_vec = tcp_pos - port_pos_base
        axis_len = np.linalg.norm(axis_vec)
        if axis_len < 1e-6:
            self.get_logger().error("TCP is at port position — cannot compute insertion axis")
            return False
        insertion_axis = axis_vec / axis_len

        self.get_logger().info(
            f"Insertion axis (port→TCP): ({insertion_axis[0]:.4f}, {insertion_axis[1]:.4f}, {insertion_axis[2]:.4f}), "
            f"distance: {axis_len*1000:.1f}mm"
        )

        # Build perpendicular frame for spiral search and lateral force
        x_perp, y_perp, z_ins = build_insertion_frame(insertion_axis)

        # Approach position: port position + 80mm along insertion axis
        approach_pos = port_pos_base + APPROACH_OFFSET_M * insertion_axis
        approach_orientation = tcp_pose.orientation

        approach_pose = self._make_pose(approach_pos, approach_orientation)

        self.get_logger().info(
            f"Approach pose: pos=({approach_pos[0]:.4f}, {approach_pos[1]:.4f}, {approach_pos[2]:.4f}), "
            f"offset={APPROACH_OFFSET_M*1000:.0f}mm from port"
        )

        # ── Step 7: Smooth interpolation to approach pose ────────────────
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

        # ── Step 8: Settle at approach pose ──────────────────────────────
        self.set_pose_target(move_robot=move_robot, pose=approach_pose)
        self.sleep_for(1.0)

        # ── Step 9: Log approach result ──────────────────────────────────
        final_tcp = self._get_tcp_pose(get_observation)
        if final_tcp is not None:
            final_pos = np.array([
                final_tcp.position.x,
                final_tcp.position.y,
                final_tcp.position.z,
            ])
            pos_error = np.linalg.norm(final_pos - approach_pos)
            dist_to_port = np.linalg.norm(final_pos - port_pos_base)

            self.get_logger().info(
                f"=== APPROACH COMPLETE ===\n"
                f"  Target approach:  ({approach_pos[0]:.4f}, {approach_pos[1]:.4f}, {approach_pos[2]:.4f})\n"
                f"  Actual TCP:       ({final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f})\n"
                f"  Position error:   {pos_error*1000:.1f}mm\n"
                f"  Distance to port: {dist_to_port*1000:.1f}mm"
            )

        # ── Step 10: Descent with force feedback ─────────────────────────
        # Move along the insertion axis in 0.5mm steps toward the port.
        # Descent direction is the negative insertion axis (toward the port).
        # At each step:
        #   - Read the FTS wrench
        #   - Project force onto insertion axis → axial force
        #   - Project force onto perpendicular plane → lateral force
        #   - Accumulate lateral force as XY integral correction
        #   - If axial force exceeds threshold → spiral search
        #   - Stop when insertion depth reached
        send_feedback("Inserting — descending with force feedback")

        descent_dir = -insertion_axis  # toward the port
        total_steps = int(INSERTION_DEPTH_M / DESCENT_STEP_M)  # 30mm / 0.5mm = 60 steps

        # Current target position starts at the approach pose.
        # The XY integral correction accumulates lateral offsets.
        current_target = approach_pos.copy()
        xy_integrator = np.zeros(3)  # accumulated lateral correction in base frame

        self.get_logger().info(
            f"Starting descent: {total_steps} steps, "
            f"{INSERTION_DEPTH_M*1000:.0f}mm target depth, "
            f"contact threshold {CONTACT_FORCE_N:.0f}N"
        )

        for step_i in range(total_steps):
            # Advance one step along the descent direction
            current_target = current_target + DESCENT_STEP_M * descent_dir

            # Apply XY integral correction to the target position.
            # This nudges the gripper laterally based on accumulated side forces,
            # helping it find the port opening — same concept as CheatCode's
            # integral XY correction.
            corrected_target = current_target + xy_integrator

            # Command the pose
            self.set_pose_target(
                move_robot=move_robot,
                pose=self._make_pose(corrected_target, approach_orientation),
            )
            self.sleep_for(DESCENT_STEP_DT)

            # Read force/torque sensor
            force_vec = self._get_wrench(get_observation)
            if force_vec is None:
                continue

            # Project force onto insertion axis to get axial force.
            # Positive = pushing against the port surface.
            axial_force = np.dot(force_vec, insertion_axis)

            # Project force onto the perpendicular plane for lateral correction.
            # These are the side forces that indicate misalignment.
            lateral_x = np.dot(force_vec, x_perp)
            lateral_y = np.dot(force_vec, y_perp)

            # Accumulate lateral integral correction (clamped to prevent windup).
            # The correction is in the perpendicular plane, expressed in base frame.
            xy_integrator += XY_INTEGRAL_GAIN * (lateral_x * x_perp + lateral_y * y_perp)
            xy_integrator = np.clip(xy_integrator, -XY_INTEGRAL_WINDUP, XY_INTEGRAL_WINDUP)

            # Log every 10 steps
            depth_mm = (step_i + 1) * DESCENT_STEP_M * 1000
            if step_i % 10 == 0:
                self.get_logger().info(
                    f"Descent step {step_i}/{total_steps}: depth={depth_mm:.1f}mm, "
                    f"axial_F={axial_force:.1f}N, "
                    f"lateral_F=({lateral_x:.1f}, {lateral_y:.1f})N, "
                    f"xy_corr=({np.linalg.norm(xy_integrator)*1000:.2f}mm)"
                )

            # Check if axial force exceeds contact threshold
            if abs(axial_force) > CONTACT_FORCE_N:
                self.get_logger().info(
                    f"Contact detected at step {step_i}, depth={depth_mm:.1f}mm, "
                    f"axial_F={axial_force:.1f}N — starting spiral search"
                )

                # ── Step 11: Spiral search ───────────────────────────────
                # The gripper has hit the surface but hasn't found the port.
                # Search a spiral pattern in the perpendicular plane to find
                # the opening, then resume descent.
                send_feedback("Spiral search — looking for port opening")

                found_pos = self._spiral_search(
                    center_pos=corrected_target,
                    orientation=approach_orientation,
                    insertion_axis=insertion_axis,
                    x_perp=x_perp,
                    y_perp=y_perp,
                    move_robot=move_robot,
                    get_observation=get_observation,
                )

                if found_pos is not None:
                    # Resume descent from the position where force dropped.
                    # Reset XY integrator since we've physically relocated.
                    self.get_logger().info(
                        f"Spiral search succeeded — resuming descent from "
                        f"({found_pos[0]:.4f}, {found_pos[1]:.4f}, {found_pos[2]:.4f})"
                    )
                    current_target = found_pos.copy()
                    xy_integrator = np.zeros(3)
                    send_feedback("Inserting — resuming descent")
                    continue
                else:
                    # Full spiral exhausted. Log warning and exit gracefully.
                    self.get_logger().warn(
                        "Spiral search exhausted — could not find port opening. "
                        "Giving up on this trial."
                    )
                    send_feedback("Spiral search failed — giving up")
                    return True

        # ── Step 12: Insertion depth reached ─────────────────────────────
        self.get_logger().info(
            f"Insertion depth reached ({INSERTION_DEPTH_M*1000:.0f}mm). "
            "Waiting for connector to stabilize..."
        )
        send_feedback("Insertion complete — stabilizing")

        # Hold position and wait for the connector to seat
        self.sleep_for(3.0)

        # Log final state
        final_tcp = self._get_tcp_pose(get_observation)
        if final_tcp is not None:
            fpos = np.array([final_tcp.position.x, final_tcp.position.y, final_tcp.position.z])
            self.get_logger().info(
                f"=== INSERTION COMPLETE ===\n"
                f"  Final TCP:         ({fpos[0]:.4f}, {fpos[1]:.4f}, {fpos[2]:.4f})\n"
                f"  Distance to port:  {np.linalg.norm(fpos - port_pos_base)*1000:.1f}mm\n"
                f"  XY correction:     {np.linalg.norm(xy_integrator)*1000:.2f}mm"
            )

        self.get_logger().info("MyPolicy.insert_cable() exiting")
        return True
