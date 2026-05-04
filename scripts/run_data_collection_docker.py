#!/usr/bin/env python3
"""Automated batch data collection using docker compose (no sudo needed).

For each config, launches eval+model via docker compose, waits for all 3
trials to complete, then moves to the next config.

Usage:
    cd ~/aic-workspace/aic
    python3 ../scripts/run_data_collection_docker.py --config-dir ../configs_v3 --start-idx 1 --end-idx 200
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time


WORKSPACE = os.path.expanduser("~/aic-workspace")
AIC_DIR = os.path.join(WORKSPACE, "aic")
ENGINE_CONFIG = os.path.join(AIC_DIR, "aic_engine/config/sample_config.yaml")
ENGINE_CONFIG_BAK = ENGINE_CONFIG + ".datacollect_bak"
DEMOS_DIR = os.path.join(WORKSPACE, "datasets/demos_v2")
LOG_PATH = os.path.join(DEMOS_DIR, "collection_log.json")

COMPOSE_BASE = os.path.join(AIC_DIR, "docker/docker-compose.yaml")
COMPOSE_COLLECT = os.path.join(AIC_DIR, "docker/docker-compose.collect.yaml")

TRIAL_TIMEOUT = 480  # 8 min per config (3 trials, SC can be slow)
CLEANUP_WAIT = 5


def load_log():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "total_episodes": 0}


def save_log(log):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH + ".tmp", "w") as f:
        json.dump(log, f, indent=2)
    os.replace(LOG_PATH + ".tmp", LOG_PATH)


def run_one_config(config_idx, config_path, log):
    config_str = f"{config_idx:03d}"
    print(f"\n{'='*60}")
    print(f"Config {config_str}: {config_path}")
    print(f"{'='*60}")

    # Skip if already completed
    if config_str in log["completed"]:
        print(f"  Already completed, skipping")
        return True

    # Copy config
    shutil.copy2(config_path, ENGINE_CONFIG)

    # Create demos output dir
    config_demos = os.path.join(DEMOS_DIR, f"config_{config_str}")
    os.makedirs(config_demos, exist_ok=True)

    env = {
        **os.environ,
        "AIC_CONFIG_IDX": config_str,
        "HOST_DEMOS_DIR": DEMOS_DIR,
    }

    try:
        # Build model (in case Dockerfile changed)
        print("  Building model...")
        build_result = subprocess.run(
            ["docker", "compose", "-f", COMPOSE_BASE, "-f", COMPOSE_COLLECT,
             "build", "model"],
            cwd=AIC_DIR, env=env, capture_output=True, text=True, timeout=600,
        )
        if build_result.returncode != 0:
            print(f"  Build failed: {build_result.stderr[-500:]}")
            return False

        # Launch eval + model
        print("  Launching eval + model...")
        proc = subprocess.Popen(
            ["docker", "compose", "-f", COMPOSE_BASE, "-f", COMPOSE_COLLECT,
             "up", "--abort-on-container-exit"],
            cwd=AIC_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        # Monitor output for completion
        start = time.time()
        all_done = False
        lines_buffer = []

        while proc.poll() is None:
            elapsed = time.time() - start
            if elapsed > TRIAL_TIMEOUT:
                print(f"  TIMEOUT after {TRIAL_TIMEOUT}s")
                break

            try:
                line = proc.stdout.readline()
                if line:
                    text = line.decode("utf-8", errors="replace").rstrip()
                    lines_buffer.append(text)
                    # Print key lines
                    if any(k in text for k in ["Score:", "Trial", "completed", "ERROR",
                                                "CheatCode", "insert_cable", "All Trials"]):
                        print(f"  {text[-120:]}")
                    if "All Trials Processed" in text:
                        all_done = True
                        print("  All trials processed!")
                        # Give 10s grace for cleanup
                        time.sleep(10)
                        break
            except Exception:
                time.sleep(0.5)

        # Stop containers
        print("  Stopping containers...")
        subprocess.run(
            ["docker", "compose", "-f", COMPOSE_BASE, "-f", COMPOSE_COLLECT, "down"],
            cwd=AIC_DIR, env=env, capture_output=True, timeout=30,
        )

        # Kill process if still running
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass

        # Check if demos were saved
        trial_dirs = [d for d in os.listdir(config_demos)
                      if os.path.isdir(os.path.join(config_demos, d))]
        print(f"  Saved {len(trial_dirs)} trial dirs: {trial_dirs}")

        if all_done or len(trial_dirs) >= 1:
            log["completed"].append(config_str)
            log["total_episodes"] += len(trial_dirs)
            save_log(log)
            return True
        else:
            log["failed"].append(config_str)
            save_log(log)
            return False

    except Exception as e:
        print(f"  ERROR: {e}")
        log["failed"].append(config_str)
        save_log(log)
        return False

    finally:
        # Cleanup
        subprocess.run(
            ["docker", "compose", "-f", COMPOSE_BASE, "-f", COMPOSE_COLLECT, "down"],
            cwd=AIC_DIR, env=env, capture_output=True, timeout=30,
        )
        time.sleep(CLEANUP_WAIT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, default=os.path.join(WORKSPACE, "configs_v3"))
    parser.add_argument("--start-idx", type=int, default=1)
    parser.add_argument("--end-idx", type=int, required=True)
    args = parser.parse_args()

    # Backup original config
    if not os.path.exists(ENGINE_CONFIG_BAK):
        shutil.copy2(ENGINE_CONFIG, ENGINE_CONFIG_BAK)

    os.makedirs(DEMOS_DIR, exist_ok=True)
    log = load_log()

    print(f"Data collection: configs {args.start_idx}-{args.end_idx}")
    print(f"Output: {DEMOS_DIR}")
    print(f"Already completed: {len(log['completed'])}")

    success = 0
    fail = 0

    for idx in range(args.start_idx, args.end_idx + 1):
        config_path = os.path.join(args.config_dir, f"config_{idx:03d}.yaml")
        if not os.path.exists(config_path):
            print(f"Config {idx:03d} not found, skipping")
            continue

        ok = run_one_config(idx, config_path, log)
        if ok:
            success += 1
        else:
            fail += 1

        print(f"Progress: {success} success, {fail} fail, "
              f"{len(log['completed'])} total completed")

    # Restore original config
    if os.path.exists(ENGINE_CONFIG_BAK):
        shutil.copy2(ENGINE_CONFIG_BAK, ENGINE_CONFIG)

    print(f"\n{'='*60}")
    print(f"Collection complete: {success} success, {fail} fail")
    print(f"Total episodes: {log['total_episodes']}")
    print(f"Output: {DEMOS_DIR}")


if __name__ == "__main__":
    main()
