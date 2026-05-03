#!/usr/bin/env python3
"""Extract detector training data from CheatCodeRecorder demos.

For each demo episode, extracts the FIRST frame (survey-pose-like) where
the port is visible, and creates a training sample with ground truth pixel
coordinates for all visible ports.

Input: ~/aic-workspace/datasets/demos/config_XXX/trial_Y/
  - episode.npz (with gt_port_pixels_center, etc.)
  - step_0000_center.png, etc.

Output: ~/aic-workspace/datasets/port_detection_v2/
  - sample_XXXXX.json + sample_XXXXX.png

Usage:
    python3 scripts/extract_training_data.py
    python3 scripts/extract_training_data.py --demos-dir datasets/demos --output-dir datasets/port_detection_v2
"""

import argparse
import glob
import json
import os
import shutil

import numpy as np


def extract_from_episode(episode_dir, output_dir, sample_idx):
    """Extract training sample from one episode directory."""
    npz_path = os.path.join(episode_dir, "episode.npz")
    meta_path = os.path.join(episode_dir, "metadata.json")

    if not os.path.exists(npz_path):
        return sample_idx, False

    try:
        data = np.load(npz_path, allow_pickle=True)
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception as e:
        print(f"  Error loading {episode_dir}: {e}")
        return sample_idx, False

    # Get port pixel coordinates from ground truth
    # Use step 0 (first frame, arm at survey/home position)
    # Also try a few early steps in case step 0 has bad data
    for step_idx in [0, 1, 2, 5, 10]:
        try:
            gt_center = data["gt_port_pixels_center"]
            if step_idx >= len(gt_center):
                continue

            # The gt_port_pixels arrays have shape (n_steps, 2) = (u, v)
            u_center = float(gt_center[step_idx, 0])
            v_center = float(gt_center[step_idx, 1])

            # Check if valid (within image bounds)
            if u_center < 0 or u_center > 1200 or v_center < 0 or v_center > 1100:
                continue

            # Find the image file for this step
            img_name = f"step_{step_idx:04d}_center.png"
            img_path = os.path.join(episode_dir, img_name)
            if not os.path.exists(img_path):
                continue

            # Extract port info from metadata
            task = meta.get("task", {})
            port_type = task.get("port_type", "sfp")
            port_name = task.get("port_name", "sfp_port_0")
            plug_type = task.get("plug_type", "sfp")

            # Determine port key
            if port_type == "sfp":
                if "port_0" in port_name:
                    port_key = "sfp_port_0"
                else:
                    port_key = "sfp_port_1"
            else:
                module = task.get("target_module_name", "sc_port_0")
                if "sc_port_0" in module:
                    port_key = "sc_port_0"
                else:
                    port_key = "sc_port_1"

            # Get camera intrinsics from metadata if available
            fx = meta.get("camera_info", {}).get("fx", 1236.63)
            fy = meta.get("camera_info", {}).get("fy", 1236.63)
            cx = meta.get("camera_info", {}).get("cx", 576.0)
            cy = meta.get("camera_info", {}).get("cy", 512.0)

            # Also get port 3D position if available
            gt_port_pos = data.get("gt_port_positions", None)
            port_3d = {}
            if gt_port_pos is not None and step_idx < len(gt_port_pos):
                p = gt_port_pos[step_idx]
                port_3d = {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}

            # Create training sample
            sample_name = f"sample_{sample_idx:05d}"
            out_img = os.path.join(output_dir, f"{sample_name}.png")
            out_json = os.path.join(output_dir, f"{sample_name}.json")

            # Copy image
            shutil.copy2(img_path, out_img)

            # Create label
            label = {
                "image": f"{sample_name}.png",
                "width": 1152,
                "height": 1024,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "ports": {
                    port_key: {
                        "u": u_center,
                        "v": v_center,
                        **port_3d,
                    }
                },
                "_source_episode": episode_dir,
                "_source_step": step_idx,
            }

            with open(out_json, "w") as f:
                json.dump(label, f, indent=2)

            return sample_idx + 1, True

        except (IndexError, KeyError, ValueError) as e:
            continue

    return sample_idx, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos-dir", default=os.path.expanduser("~/aic-workspace/datasets/demos"))
    parser.add_argument("--output-dir", default=os.path.expanduser("~/aic-workspace/datasets/port_detection_v2"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all episode directories
    episodes = sorted(glob.glob(os.path.join(args.demos_dir, "config_*/trial_*")))
    print(f"Found {len(episodes)} episodes in {args.demos_dir}")

    # Count existing samples in output dir
    existing = len(glob.glob(os.path.join(args.output_dir, "sample_*.json")))
    sample_idx = existing
    success = 0
    fail = 0

    for ep_dir in episodes:
        new_idx, ok = extract_from_episode(ep_dir, args.output_dir, sample_idx)
        if ok:
            sample_idx = new_idx
            success += 1
        else:
            fail += 1

    print(f"\nExtracted {success} samples, {fail} failed")
    print(f"Total samples in {args.output_dir}: {sample_idx}")


if __name__ == "__main__":
    main()
