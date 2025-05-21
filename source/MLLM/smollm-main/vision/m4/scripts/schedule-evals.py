#!/usr/bin/env python

#
# This tool checks if evaluation is needed
#

import argparse
import os
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
# 1. either the eval is still running
# 2. the eval got aborted (e.g. gpu-oom)
#

# should fine tune - but surely 9h per checkpoint is plenty
reasonable_eval_time_in_secs = 9 * 60 * 60


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
    return parser.parse_args()


def exit(msg):
    print(msg)
    sys.exit()


def check_eval_crash(path):
    """Heuristics to decide whether to restart this opt_step-XXX checkpoint evaluation or not"""
    eval_start_paths = map(
        lambda x: path / x,
        [
            "start_run_evals_0_shots",
            "start_run_evals_4_shots",
            "start_run_evals_perplexity_validation",
            "start_run_evals_0_shots_a_la_flamingo",
        ],
    )
    # complicated checks - has another job already started processing? or did it crash?
    for eval_start_path in eval_start_paths:
        if eval_start_path.exists():
            if eval_start_path.stat().st_mtime < time.time() - reasonable_eval_time_in_secs:
                print(f"[Y] {path} looks stale - Probably crashed - Restart evals")
                os.remove(eval_start_path)


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
        print(f"\n*** Checking {checkpoint_dir} for evals")
        check_eval_crash(checkpoint_dir)


if __name__ == "__main__":
    main()
