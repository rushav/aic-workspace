"""
MyPolicy — perception bridge + vertical descent (CheatCode strategy).

Pipeline:
1. Move to survey pose, capture image, run heatmap detector
2. Back-project detected port pixel to 3D, transform to base_link
3. Approach: interpolate to port XY, 100mm above port Z (5s)
4. Vertical descent: decrease Z in 0.5mm steps, XY locked to port position
5. Descend from z_offset=+100mm to z_offset=-15mm (past port Z)
6. Stabilize 5s, return True

The detector is a ResNet-34 encoder-decoder (HeatmapDetector) that outputs
a 2-channel 96x96 heatmap. Soft-argmax extracts sub-pixel (u, v) coordinates
for each SFP port. Architecture must match train_detector.py exactly.
"""

import numpy as np
import os

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

# Approach interpolation parameters (matches CheatCode's strategy)
APPROACH_STEPS = 100      # number of interpolation steps
APPROACH_STEP_DT = 0.05   # seconds per step → 100 * 0.05 = 5.0s total

# Descent parameters
DESCENT_STEP_M = 0.0005         # 0.5mm per step
DESCENT_STEP_DT = 0.05          # seconds per step (~20 Hz)


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

        # ── Step 6: Approach — move to port XY, 100mm above port Z ──────
        # CheatCode strategy: no insertion axis. Go directly above the port
        # (same XY), then descend vertically in world Z.
        approach_orientation = survey_orientation
        approach_pos = np.array([
            port_pos_base[0],
            port_pos_base[1],
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

        # ── Step 7: Vertical descent in world Z ─────────────────────────
        # Descend from 100mm above port Z to 15mm below port Z.
        # XY stays locked to port XY. Exactly like CheatCode.
        send_feedback("Descending vertically toward port")

        z_offset = 0.1  # start 100mm above port Z
        step_count = 0

        self.get_logger().info(
            f"Starting vertical descent: z_offset 0.1 → -0.015, "
            f"step={DESCENT_STEP_M*1000:.1f}mm, port_z={port_pos_base[2]:.4f}"
        )

        while z_offset > -0.015:
            z_offset -= DESCENT_STEP_M
            target_z = port_pos_base[2] + z_offset
            target_pos = np.array([port_pos_base[0], port_pos_base[1], target_z])

            self.set_pose_target(
                move_robot=move_robot,
                pose=self._make_pose(target_pos, approach_orientation),
            )
            self.sleep_for(DESCENT_STEP_DT)

            # Log every 20 steps
            if step_count % 20 == 0:
                send_feedback(f"Descent z_offset={z_offset:.4f}, target_z={target_z:.4f}")
                self.get_logger().info(
                    f"Descent step {step_count}: z_offset={z_offset:.4f}, "
                    f"target_z={target_z:.4f}"
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
