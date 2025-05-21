#!/usr/bin/env python

#
# This tool uploads any new deepspeed checkpoints found at given path to s3 (and also various non-checkpoint files, like logs)
#
# Example:
#
# ./s3-upload-checkpoints.py checkpoints-path
#
# Use `-h` for more options
#


import argparse
import subprocess
import sys
import time
from pathlib import Path


repo_path = Path(__file__).resolve().parents[2]
zero_checkpoint_to_hf_path = repo_path / "m4/models/zero_checkpoint_to_hf.py"

RETRIES = 5

# what dir/file glob patterns to include in the upload besides checkpoints
include_patterns = ["tb_run_*", "logs", "config.yaml"]


# we have to deal with potentially overlapping slurm jobs running on different nodes, so we can't
# rely on PIDs of a running process. Will use a control file instead as the filesystem is shared.
#
# If that file is there it means:
#
# 1. either the upload is still running
# 2. the upload got aborted (e.g. cpu-oom)
#
# to detect aborted uploads we will check if the control file is older than a reasonable time to perform such a upload
control_file_name = "started-upload-checkpoint"
finished_uploading_file_name = "finished-upload-checkpoint"
# should fine tune - but surely 2h per checkpoint is plenty
reasonable_upload_time_in_secs = 2 * 60 * 60


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
    # parser.add_argument("experiment_name", type=str, help="experiment name as a s3 sub-dir")
    parser.add_argument("-f", "--force", action="store_true", help="force uploading of all checkpoints")
    parser.add_argument(
        "--skip-conversion-check", action="store_true", help="skip checkpoint conversion is done check"
    )
    return parser.parse_args()


def exit(msg):
    print(msg)
    sys.exit()


def should_process(path, force, control_file_path, finished_uploading_file_path, args):
    """Heuristics to decide whether to upload this opt_step-XXX checkpoint or not"""

    # check if checkpoint is fully saved
    finished_saving_path = path / "finished-saving"  # defined in from trainer.py
    if not finished_saving_path.exists():
        print(f"[N] {path} isn't finished saving. Skipping")
        return False

    if force:
        print("[Y] Forced to re-process {checkpoint_dir}")
        return True

    # check if already uploaded
    if finished_uploading_file_path.exists():
        print(f"[N] {path} has already been uploaded. Skipping")
        return False

    # check conversion is completed
    if not args.skip_conversion_check:
        target_dir = path / "unwrapped_model"
        checklist = [
            target_dir / "pytorch_model.bin.index.json",
            target_dir / "pytorch_model.bin",
            target_dir / "model.safetensors.index.json",
            target_dir / "model.safetensors",
        ]
        checklist = [model_path.exists() for model_path in checklist]
        if not any(checklist):
            print(f"[N] {path} doesn't have a converted model. Skipping")
            return False

    # complicated checks - has another job already started uploading? or did it crash?
    if control_file_path.exists():
        if control_file_path.stat().st_mtime < time.time() - reasonable_upload_time_in_secs:
            print(f"[Y] {path} looks stale - probably aborted job. Re-uploading")
            return True
        else:
            print(
                f"[N] {path} either another job is uploading it or less than"
                f" {reasonable_upload_time_in_secs} secs has passed since it was launched. Skipping"
            )
            return False
    else:
        print(f"[Y] {path} is a new checkpoint. Uploading")
        return True


def main():
    args = get_args()

    checkpoints_path = Path(args.checkpoints_path)
    if not (checkpoints_path.exists() and checkpoints_path.is_dir()):
        raise FileNotFoundError(f"can't find a directory '{checkpoints_path}'")

    checkpoint_dirs = list(checkpoints_path.glob("opt_step-*"))
    if len(checkpoint_dirs) == 0:
        exit("No checkpoints found, exiting")

    exp_name = checkpoints_path.name

    # Check each folder in real time to allow for overlapping jobs starting at different times
    for checkpoint_dir in checkpoint_dirs:
        print(f"\n*** Checking {checkpoint_dir}")

        control_file_path = checkpoint_dir / control_file_name
        finished_uploading_file_path = checkpoint_dir / finished_uploading_file_name

        if not should_process(checkpoint_dir, args.force, control_file_path, finished_uploading_file_path, args):
            continue

        opt_step = checkpoint_dir.name
        bucket_name = "m4-exps-us-east-1"
        bucket_path = f"{exp_name}/{opt_step}"

        print(f"Launching upload for {checkpoint_dir} - it could take a long time")
        cmd = f"s5cmd sync {checkpoint_dir}/ s3://{bucket_name}/{bucket_path}/".split()
        # we could use flock here, to avoid a race condition, but it'd be pointless since each
        # cronjob is likely to run on a different node and flock only works within a single node
        control_file_path.touch()
        # print(f"mock running {cmd}")

        # s5cmd will fail with an error like this when MD5 checksum doesn't match on upload (it won't retry)
        # ERROR "cp data4.tar s3://m4-datasets/cm4-test/data4.tar": InvalidDigest: The Content-MD5
        # you specified was invalid. status code: 400, request id: SZEHBJ4QQ33JSMH7, host id:
        # XTeMYKd2KECiVKbFnwVbXo3LgnuA2OHWk5S+tHKAOKO95Os/pje2ZEbCfO5pojQtCTFOovvnVME=

        tries = 0
        while tries < RETRIES:
            tries += 1
            try:
                response = run_cmd(cmd)
                print(response)
                break
            except EnvironmentError as e:
                if "InvalidDigest" in str(e):
                    print(f"MD5 checksum failed, upload retry {tries}")
                    continue
            except Exception:
                # some other possible failure?
                raise

        # for now disable this as large files don't have sha256 checksums
        # result = integrity_check_recursive(checkpoint_dir, bucket_name, bucket_path)
        # print(f"Integrity check was {result}")

        control_file_path.unlink()
        finished_uploading_file_path.touch()

    # now upload non-checkpoint files
    print("\n*** Uploading non-checkpoint files")
    upload_dirs = []
    for pat in include_patterns:
        upload_dirs += list(checkpoints_path.glob(pat))

    for dir in upload_dirs:
        print(f"Launching upload for {dir}")
        cmd = f"s5cmd sync {dir} s3://m4-exps-us-east-1/{exp_name}/".split()
        print(f"running {cmd}")
        response = run_cmd(cmd)
        print(response)


if __name__ == "__main__":
    main()
