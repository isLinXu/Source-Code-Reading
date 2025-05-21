#!/usr/bin/env python

#
# This tool deletes checkpoints found at given path that are no longer needed
#
# we have 2 parts to each checkpoints to cleanup
#
# 1. the original deepspeed checkpoint
# 2. the converted hf checkpoint
#
# we will start with a combined requirement for eval to be completed and s3 synced to nuke the checkpoint
#
# Example:
#
# ./cleanup-checkpoints.py checkpoints-path
#
# Use `-h` for more options

import argparse
import shutil  # noqa
import subprocess
import sys
import time
from pathlib import Path


repo_path = Path(__file__).parents[2]

# we have to deal with potentially overlapping slurm jobs running on different nodes, so we can't
# rely on PIDs of a running process. Will use a control file instead as the filesystem is shared.
#
# If that file is there it means:
#
# 1. either the cleanup is still running
# 2. the cleanup got aborted (e.g. cpu-oom)
#
# to detect aborted cleanups we will check if the control file is older than a reasonable time to perform such a cleanup
control_file_name = "started-cleanup-checkpoint"
finished_uploading_file_name = "finished-upload-checkpoint"
# should fine tune - but surely 1h per checkpoint is plenty
reasonable_cleanup_time_in_secs = 1 * 60 * 60


def run_cmd(cmd, check=True):
    try:
        response = subprocess.run(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=check,
            encoding="utf-8",
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)

    return response


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints_path", type=str, help="base dir with checkpoints")
    parser.add_argument("--skip-evals-check", action="store_true", help="skip evals done checks")
    return parser.parse_args()


def exit(msg):
    print(msg)
    sys.exit()


def should_process(path, control_file_path, args):
    """Heuristics to decide whether to cleanup this opt_step-XXX checkpoint or not"""

    s3_completed_path = path / finished_uploading_file_name
    eval_completed_paths = [
        path / "run_evals_0_shots_done",
        path / "run_evals_4_shots_done",
        path / "run_evals_perplexity_validation_done",
        path / "run_evals_0_shots_a_la_flamingo_done",
    ]

    # check s3 sync is completed
    if not s3_completed_path.exists():
        print(f"[N] {path} hasn't been synced to s3 yet. Skipping")
        return False

    # check evals are completed
    if not args.skip_evals_check:
        for eval_path in eval_completed_paths:
            if not eval_path.exists():
                print(f"[N] {path} hasn't been evaled yet. Skipping")
                return False

    # complicated checks - has another job already started processing? or did it crash?
    if control_file_path.exists():
        if control_file_path.stat().st_mtime < time.time() - reasonable_cleanup_time_in_secs:
            print(f"[Y] {path} looks stale - probably aborted cleanup job. Deleting")
            return True
        else:
            print(
                f"[N] {path} either another job is doing the cleanup or less than"
                f" {reasonable_cleanup_time_in_secs} secs has passed since it was launched. Skipping"
            )
            return False
    else:
        print(f"[Y] {path} completed s3 sync + eval. Deleting")
        return True


def main():
    args = get_args()

    checkpoints_path = Path(args.checkpoints_path)
    if not (checkpoints_path.exists() and checkpoints_path.is_dir()):
        raise FileNotFoundError(f"can't find a directory '{checkpoints_path}'")

    checkpoint_dirs = list(checkpoints_path.glob("opt_step-*"))
    if len(checkpoint_dirs) == 0:
        exit("No checkpoints found, exiting")

    # Check each checkpoint folder in real time to allow for overlapping jobs starting at different times
    # Additionally do not delete the last 2 checkpoints
    #
    # sort numerically to sort correctly different number of digits: opt_step-10, opt_step-100
    checkpoint_dirs_sorted = sorted(checkpoint_dirs, key=lambda x: int(str(x).split("-")[-1]))
    for i, checkpoint_dir in enumerate(checkpoint_dirs_sorted):
        print(f"\n*** Checking {checkpoint_dir}")

        if i + 1 == len(checkpoint_dirs_sorted):
            print(f"[N] {checkpoint_dir} is a last checkpoint. Skipping")
            continue

        if i + 2 == len(checkpoint_dirs_sorted):
            print(f"[N] {checkpoint_dir} is a second to last checkpoint. Skipping")
            continue

        control_file_path = checkpoint_dir / "unwrapped_model" / control_file_name

        if not should_process(checkpoint_dir, control_file_path, args):
            continue

        print(f"Launching cleanup for {checkpoint_dir}")
        # we could use flock here, to avoid a race condition, but it'd be pointless since each
        # cronjob is likely to run on a different node and flock only works within a single node
        control_file_path.touch()

        # cleanup
        # XXX: enable the actual delete once tested a lot
        # The delete should be relatively safe since it'll only run if it finds 2 files:
        # save_dir/opt_step-XXX/s3_sync_is_completed save_dir/opt_step-XXX/eval_is_completed
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        print(f"Checkpoint {checkpoint_dir} deleted")


if __name__ == "__main__":
    main()
