#!/usr/bin/env python3
"""Fully automated batch data collection runner.

For each config, launches the simulator and recording policy, waits for
completion, then moves to the next config. Handles crashes gracefully and
supports resumption from where it left off.

Usage:
    python scripts/run_data_collection.py --config-dir configs --start-idx 1 --end-idx 50
    python scripts/run_data_collection.py --config-dir configs --start-idx 1 --end-idx 334
"""

import argparse
import atexit
import getpass
import json
import os
import select
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time


WORKSPACE = os.path.expanduser("~/aic-workspace")
AIC_DIR = os.path.join(WORKSPACE, "aic")
ENGINE_CONFIG = os.path.join(AIC_DIR, "aic_engine/config/sample_config.yaml")
DEMOS_DIR = os.path.join(WORKSPACE, "datasets/demos")
LOG_PATH = os.path.join(DEMOS_DIR, "collection_log.json")

SIM_READY_TIMEOUT = 180  # seconds (first launch can be slow)
SIM_KILL_TIMEOUT = 10    # seconds before SIGKILL
CLEANUP_WAIT = 10        # seconds between configs
SIM_LOG_INTERVAL = 10    # seconds between verbose sim log dumps


def load_log():
    """Load collection log, or create a new one."""
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            return json.load(f)
    return {
        "completed": [],
        "failed": [],
        "total_episodes": 0,
        "total_time_sec": 0.0,
    }


def save_log(log):
    """Save collection log atomically."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    tmp_path = LOG_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(log, f, indent=2)
    os.replace(tmp_path, LOG_PATH)


def kill_process_tree(proc, timeout=SIM_KILL_TIMEOUT):
    """Kill a process and all its children."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def kill_sim_container():
    """Stop the aic_eval Docker container that distrobox manages.

    kill_process_tree only kills the bash wrapper — the actual simulator
    runs inside a Docker container that survives. This stops it directly.
    """
    try:
        # Find the aic_eval container
        result = subprocess.run(
            ["sudo", "-A", "docker", "ps", "-q", "--filter", "name=aic_eval"],
            capture_output=True, text=True, timeout=10,
            env=os.environ,
        )
        container_ids = result.stdout.strip().split()
        if not container_ids:
            return
        for cid in container_ids:
            # Use 'kill' not 'stop' — immediate SIGKILL, no grace period.
            # We don't need graceful shutdown between data collection runs.
            print(f"  Killing Docker container {cid[:12]}...")
            subprocess.run(
                ["sudo", "-A", "docker", "kill", cid],
                capture_output=True, timeout=10,
                env=os.environ,
            )
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        print(f"  WARNING: Failed to kill Docker container: {e}")


def drain_sim_output(sim_proc, label="sim"):
    """Read and return any available stdout from the simulator (non-blocking)."""
    lines = []
    if sim_proc is None or sim_proc.stdout is None:
        return lines
    while True:
        ready, _, _ = select.select([sim_proc.stdout], [], [], 0)
        if not ready:
            break
        line = sim_proc.stdout.readline()
        if not line:
            break
        lines.append(line.decode("utf-8", errors="replace").rstrip())
    return lines


def wait_for_simulator(sim_proc, timeout=SIM_READY_TIMEOUT):
    """Poll until aic_engine node is visible in ROS.

    Periodically dumps simulator stdout so we can diagnose startup issues.
    The ros2 node list command runs from AIC_DIR (the pixi workspace).
    """
    start = time.time()
    last_log_time = start
    poll_cmd = ["pixi", "run", "ros2", "node", "list"]
    attempt = 0

    print(f"  Poll command: {' '.join(poll_cmd)}")
    print(f"  Poll cwd: {AIC_DIR}")

    while time.time() - start < timeout:
        elapsed = time.time() - start
        attempt += 1

        # Dump simulator output periodically for debugging
        if time.time() - last_log_time >= SIM_LOG_INTERVAL:
            last_log_time = time.time()
            sim_lines = drain_sim_output(sim_proc)
            if sim_lines:
                print(f"  --- sim stdout ({elapsed:.0f}s elapsed) ---")
                for line in sim_lines[-20:]:  # last 20 lines
                    print(f"  [sim] {line}")
                print(f"  --- end sim stdout ({len(sim_lines)} lines) ---")
            else:
                print(f"  [wait] {elapsed:.0f}s elapsed, no new sim output (attempt {attempt})")

            # Also check if sim process died
            if sim_proc.poll() is not None:
                print(f"  ERROR: Simulator process exited with code {sim_proc.returncode}")
                # Drain remaining output
                remaining = drain_sim_output(sim_proc)
                for line in remaining:
                    print(f"  [sim] {line}")
                return False

        try:
            result = subprocess.run(
                poll_cmd,
                cwd=AIC_DIR,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if "aic_engine" in result.stdout:
                print(f"  aic_engine detected after {elapsed:.0f}s (attempt {attempt})")
                print(f"  Node list: {result.stdout.strip()}")
                return True
            if attempt <= 3 or attempt % 10 == 0:
                # Log early attempts and periodic checks
                nodes = result.stdout.strip() or "(none)"
                print(f"  [poll #{attempt}] {elapsed:.0f}s — nodes: {nodes}")
                if result.stderr.strip():
                    print(f"  [poll stderr] {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"  [poll #{attempt}] {elapsed:.0f}s — ros2 node list timed out")
        except subprocess.SubprocessError as e:
            print(f"  [poll #{attempt}] {elapsed:.0f}s — error: {e}")

        time.sleep(3)

    # Timeout — dump final sim output
    print(f"  TIMEOUT after {timeout}s. Final sim output:")
    sim_lines = drain_sim_output(sim_proc)
    for line in sim_lines[-30:]:
        print(f"  [sim] {line}")
    return False


ASKPASS_PATH = None  # global so atexit can clean it up


def cleanup_askpass():
    """Remove the temporary askpass script."""
    if ASKPASS_PATH and os.path.exists(ASKPASS_PATH):
        os.unlink(ASKPASS_PATH)


def create_askpass_script(password):
    """Create a temp script that echoes the sudo password.

    When a subprocess has no TTY (stdout piped), sudo can't prompt
    interactively. But if SUDO_ASKPASS is set, sudo calls that program
    to get the password instead. This works for ALL sudo calls in the
    entire subprocess tree — including distrobox's internal sudo calls.
    """
    global ASKPASS_PATH
    fd, path = tempfile.mkstemp(prefix=".askpass_", suffix=".sh",
                                dir=os.path.expanduser("~"))
    with os.fdopen(fd, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"printf '%s\\n' {shlex.quote(password)}\n")
    os.chmod(path, stat.S_IRWXU)  # 700 — owner only
    ASKPASS_PATH = path
    atexit.register(cleanup_askpass)
    return path


def setup_sudo(skip=False):
    """Prompt for sudo password, create askpass helper, return its path.

    distrobox -r calls sudo internally (for docker). Those internal sudo
    calls have no TTY, so credential caching doesn't help. Instead we
    create a SUDO_ASKPASS script that feeds the password automatically.

    Returns the askpass script path, or None if --no-sudo.
    Exits on bad password.
    """
    if skip:
        print("--no-sudo: skipping sudo setup.")
        return None

    print("distrobox -r requires sudo. Enter your password once; it will")
    print("be provided automatically to all sudo calls via SUDO_ASKPASS.")
    password = getpass.getpass("Sudo password: ")

    # Validate the password
    askpass_path = create_askpass_script(password)
    try:
        result = subprocess.run(
            ["sudo", "-A", "true"],
            env={**os.environ, "SUDO_ASKPASS": askpass_path},
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            print(f"ERROR: wrong password. {result.stderr.strip()}")
            cleanup_askpass()
            sys.exit(1)
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        print(f"ERROR: sudo validation failed: {e}")
        cleanup_askpass()
        sys.exit(1)

    print("sudo password validated. ASKPASS helper created.")
    return askpass_path


def run_one_config(config_idx, config_path, log, askpass_path=None):
    """Run simulator + recording policy for one config. Returns True on success."""
    config_str = f"{config_idx:03d}"
    print(f"\n{'='*60}")
    print(f"Starting config {config_str}: {config_path}")
    print(f"{'='*60}")

    # Copy config to sample_config.yaml
    shutil.copy2(config_path, ENGINE_CONFIG)

    # Build env with SUDO_ASKPASS so distrobox's internal sudo calls work
    sim_env = dict(os.environ)
    sim_env["DBX_CONTAINER_MANAGER"] = "docker"
    if askpass_path is not None:
        sim_env["SUDO_ASKPASS"] = askpass_path

    sim_proc = None
    policy_proc = None

    try:
        # Wait for any leftover aic_engine from previous run to disappear
        print("  Waiting for old ROS nodes to clean up...")
        stale_start = time.time()
        while time.time() - stale_start < 30:
            try:
                result = subprocess.run(
                    ["pixi", "run", "ros2", "node", "list"],
                    cwd=AIC_DIR, capture_output=True, text=True, timeout=10,
                )
                if "aic_engine" not in result.stdout:
                    break
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                break
            print(f"    stale aic_engine still present ({time.time() - stale_start:.0f}s)...")
            time.sleep(3)
        else:
            print("  WARNING: stale aic_engine still present after 30s, proceeding anyway")

        # Launch simulator headless with ground_truth:=true
        sim_cmd = (
            "distrobox enter -r aic_eval -- "
            "/entrypoint.sh ground_truth:=true start_aic_engine:=true "
            "gazebo_gui:=false launch_rviz:=false"
        )
        print(f"  Sim launch command: {sim_cmd}")
        sim_proc = subprocess.Popen(
            ["bash", "-c", sim_cmd],
            env=sim_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # new process group for clean killing
        )
        print(f"  Simulator launched (PID {sim_proc.pid})")

        # Wait for simulator ready
        print(f"  Waiting for aic_engine node (timeout {SIM_READY_TIMEOUT}s)...")
        if not wait_for_simulator(sim_proc):
            print(f"  ERROR: Simulator did not become ready in {SIM_READY_TIMEOUT}s")
            return False

        print("  Simulator ready, launching recording policy...")

        # Launch recording policy with config index in env
        policy_env = {
            **os.environ,
            "AIC_CONFIG_IDX": config_str,
        }
        policy_proc = subprocess.Popen(
            [
                "pixi", "run", "ros2", "run", "aic_model", "aic_model",
                "--ros-args",
                "-p", "use_sim_time:=true",
                "-p", "policy:=aic_example_policies.ros.CheatCodeRecorder",
            ],
            cwd=AIC_DIR,
            env=policy_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        print(f"  Policy launched (PID {policy_proc.pid})")

        # Stream policy output while waiting for it to finish.
        # The ROS2 lifecycle node may not exit on its own after finalization.
        # We detect shutdown via: on_shutdown message, on_deactivate after all
        # trials, or silence (no output for IDLE_TIMEOUT seconds).
        done_seen = False
        done_time = None
        last_output_time = time.time()
        DONE_GRACE = 15       # seconds after shutdown/deactivate to wait
        IDLE_TIMEOUT = 120    # seconds of no output = assumed done (was 60, too short for 3 trials)

        while policy_proc.poll() is None:
            now = time.time()

            # Kill if grace period after done signal expired
            if done_seen and now - done_time > DONE_GRACE:
                print(f"  Policy did not exit {DONE_GRACE}s after done signal, killing...")
                kill_process_tree(policy_proc)
                break

            # Kill if no output for too long (policy is hung/idle)
            if not done_seen and now - last_output_time > IDLE_TIMEOUT:
                print(f"  Policy silent for {IDLE_TIMEOUT}s, assuming done, killing...")
                kill_process_tree(policy_proc)
                break

            try:
                ready, _, _ = select.select([policy_proc.stdout], [], [], 1.0)
                if ready:
                    line = policy_proc.stdout.readline()
                    if line:
                        text = line.decode("utf-8", errors="replace").rstrip()
                        if text:
                            last_output_time = now
                            print(f"  [policy] {text}")
                            # Detect end-of-work signals
                            # BUG FIX: on_deactivate fires BETWEEN trials, not just at the end.
                            # Only treat on_shutdown or "All Trials Processed" as done.
                            if "on_shutdown" in text or "All Trials Processed" in text:
                                done_seen = True
                                done_time = now
                                print(f"  Policy done signal detected, waiting up to {DONE_GRACE}s for exit...")
            except Exception:
                pass

        policy_exit = policy_proc.returncode if policy_proc.returncode is not None else -1
        print(f"  Policy exited with code {policy_exit}")

        if policy_exit != 0:
            print(f"  WARNING: Policy exited with non-zero code {policy_exit}")

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    finally:
        # Clean up both processes
        if policy_proc is not None and policy_proc.poll() is None:
            print("  Killing policy process...")
            kill_process_tree(policy_proc)

        if sim_proc is not None and sim_proc.poll() is None:
            print("  Killing simulator wrapper...")
            kill_process_tree(sim_proc)

        # The actual simulator runs inside a Docker container that survives
        # kill_process_tree. Stop it directly.
        kill_sim_container()

        print(f"  Waiting {CLEANUP_WAIT}s for cleanup...")
        time.sleep(CLEANUP_WAIT)


def main():
    parser = argparse.ArgumentParser(description="Automated batch data collection")
    parser.add_argument("--config-dir", type=str, default="configs",
                        help="Directory containing config YAML files")
    parser.add_argument("--start-idx", type=int, default=1)
    parser.add_argument("--end-idx", type=int, required=True)
    parser.add_argument("--demos-per-config", type=int, default=1,
                        help="Number of times to run each config (default 1)")
    parser.add_argument("--no-sudo", action="store_true",
                        help="Skip sudo password prompt (if your setup doesn't need it)")
    args = parser.parse_args()

    config_dir = os.path.join(WORKSPACE, args.config_dir)
    os.makedirs(DEMOS_DIR, exist_ok=True)

    # Always prompt for sudo password up front (unless --no-sudo).
    # Returns path to SUDO_ASKPASS script, or None.
    askpass_path = setup_sudo(skip=args.no_sudo)

    # Set SUDO_ASKPASS globally so kill_sim_container() can use sudo -A
    if askpass_path is not None:
        os.environ["SUDO_ASKPASS"] = askpass_path

    log = load_log()
    completed_set = set(log["completed"])

    total_configs = args.end_idx - args.start_idx + 1
    overall_start = time.time()
    configs_done = 0
    configs_failed = 0

    # Handle Ctrl+C cleanly
    interrupted = False

    def sigint_handler(sig, frame):
        nonlocal interrupted
        if interrupted:
            print("\nForce quit!")
            sys.exit(1)
        interrupted = True
        print("\n\nInterrupted! Finishing cleanup and saving progress...")

    signal.signal(signal.SIGINT, sigint_handler)

    print(f"Data collection: configs {args.start_idx}..{args.end_idx} "
          f"({total_configs} configs, {total_configs * 3} episodes)")
    print(f"Config dir: {config_dir}")
    print(f"Output dir: {DEMOS_DIR}")
    print(f"Already completed: {len(completed_set)} configs")

    for config_idx in range(args.start_idx, args.end_idx + 1):
        if interrupted:
            break

        config_str = f"{config_idx:03d}"

        # Check if already completed (resume support)
        if config_str in completed_set:
            print(f"Skipping config {config_str} (already completed)")
            continue

        config_path = os.path.join(config_dir, f"config_{config_str}.yaml")
        if not os.path.exists(config_path):
            print(f"WARNING: {config_path} not found, skipping")
            continue

        for demo_run in range(args.demos_per_config):
            if interrupted:
                break

            config_start = time.time()
            success = run_one_config(config_idx, config_path, log,
                                     askpass_path=askpass_path)
            config_time = time.time() - config_start

            if success:
                configs_done += 1
                log["completed"].append(config_str)
                completed_set.add(config_str)
                log["total_episodes"] += 3  # 3 trials per config
                log["total_time_sec"] += config_time
            else:
                configs_failed += 1
                log["failed"].append({
                    "config": config_str,
                    "demo_run": demo_run,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                })

            # Save progress after EVERY config
            save_log(log)

            elapsed = time.time() - overall_start
            done_so_far = configs_done + configs_failed
            print(
                f"\nCompleted config {config_str} "
                f"({'OK' if success else 'FAILED'}) "
                f"in {config_time:.0f}s | "
                f"Progress: {done_so_far}/{total_configs} | "
                f"Episodes: {log['total_episodes']} | "
                f"Failures: {configs_failed} | "
                f"Elapsed: {elapsed:.0f}s"
            )

    # Final summary
    total_time = time.time() - overall_start
    log["total_time_sec"] = total_time
    save_log(log)

    print(f"\n{'='*60}")
    print(f"DATA COLLECTION {'INTERRUPTED' if interrupted else 'COMPLETE'}")
    print(f"{'='*60}")
    print(f"  Total configs attempted: {configs_done + configs_failed}")
    print(f"  Successful: {configs_done}")
    print(f"  Failed: {configs_failed}")
    print(f"  Total episodes: {log['total_episodes']}")
    print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    if configs_done > 0:
        print(f"  Avg time per config: {total_time / (configs_done + configs_failed):.0f}s")
    print(f"  Log saved to: {LOG_PATH}")
    print(f"  Data saved to: {DEMOS_DIR}")

    if interrupted:
        print(f"\nResume with the same command to continue from where you left off.")


if __name__ == "__main__":
    main()
