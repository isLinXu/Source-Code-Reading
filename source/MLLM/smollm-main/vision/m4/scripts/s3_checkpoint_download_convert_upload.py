#!/usr/bin/env python

#
# This tool converts any deepspeed checkpoints found at given path to hf format
#
# Example:
#
# ./convert-checkpoints.py checkpoints-path
#

import argparse
import subprocess
import sys
from pathlib import Path

import boto3


def check_s3_directory(directory_path):
    s3 = boto3.client("s3")

    # Add a trailing slash to the directory path
    if not directory_path.endswith("/"):
        directory_path += "/"

    # Check if any objects exist with the given directory prefix
    response = s3.list_objects_v2(Bucket="m4-exps", Prefix=directory_path)

    # If any objects are found, the directory exists
    if "Contents" in response:
        return True

    return False


def check_s3_file(file_key):
    s3 = boto3.client("s3")

    try:
        s3.head_object(Bucket="m4-exps", Key=file_key)
        return True
    except Exception:
        return False


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
    parser.add_argument("run_name", type=str, help="run name")
    parser.add_argument("opt_step_num_list", nargs="+", help="list of opt-steps to download")
    parser.add_argument("repo_path", type=str, help="repo path")

    parser.add_argument("-f", "--force", action="store_true", help="force rebuilding of all checkpoints")
    return parser.parse_args()


def exit(msg):
    print(msg)
    sys.exit()


def cmd_retry_loop(cmd, max_retries=5):
    # s5cmd will fail with an error like this when MD5 checksum doesn't match on upload (it won't retry)
    # ERROR "cp data4.tar s3://m4-datasets/cm4-test/data4.tar": InvalidDigest: The Content-MD5
    # you specified was invalid. status code: 400, request id: SZEHBJ4QQ33JSMH7, host id:
    # XTeMYKd2KECiVKbFnwVbXo3LgnuA2OHWk5S+tHKAOKO95Os/pje2ZEbCfO5pojQtCTFOovvnVME=

    tries = 0
    while tries < max_retries:
        tries += 1
        try:
            response = run_cmd(cmd)
            print(response)
            break
        except EnvironmentError as e:
            if "InvalidDigest" in str(e):
                print(f"MD5 checksum failed, download retry {tries}")
                continue
        except Exception:
            # some other possible failure?
            raise
    return response


def main():
    args = get_args()

    run_name = args.run_name
    opt_step_num_list = args.opt_step_num_list
    repo_path = Path(args.repo_path)
    zero_checkpoint_to_hf_path = repo_path / "m4/models/zero_checkpoint_to_hf.py"
    bucket_name = "m4-exps"
    opt_step_s3_file_keys = [f"{run_name}/opt_step-{opt_step_num}" for opt_step_num in opt_step_num_list]

    check_s3_directory(run_name)

    # Check each folder in real time to allow for overlapping jobs starting at different times
    for opt_step_s3_file_key in opt_step_s3_file_keys:
        print(f"\n*** Checking {opt_step_s3_file_key}")
        if not check_s3_directory(opt_step_s3_file_key):
            print(f"The checkpoint {opt_step_s3_file_key} does not exist - skipping")
            continue
        unwrapped_model_s3_file_key = f"{opt_step_s3_file_key}/unwrapped_model"
        bin_s3_file_key = f"{unwrapped_model_s3_file_key}/model.safetensors"
        index_s3_file_key = f"{unwrapped_model_s3_file_key}/model.safetensors.index.json"
        is_not_converted = not check_s3_file(bin_s3_file_key) and not check_s3_file(index_s3_file_key)
        if is_not_converted:
            print(
                f"The checkpoint hasn't been converted, launching download for {opt_step_s3_file_key} - it could take"
                " a long time"
            )

            opt_step_dirname = opt_step_s3_file_key.split("/")[-1]
            cluster_opt_step_dir = f"/fsx/m4/experiments/local_experiment_dir/s3_async_temporary_checkpoint_folder/{run_name}/{opt_step_dirname}"
            cmd = f"s5cmd sync s3://{bucket_name}/{opt_step_s3_file_key}/* {cluster_opt_step_dir}".split()
            download_response_opt_step_dir = cmd_retry_loop(cmd, max_retries=5)
            print(f"download_response_opt_step_dir: {download_response_opt_step_dir}")
        else:
            print(
                "The checkpoint has been converted already, downloading only the unwrapped checkpoint and"
                " tokenizer dir"
            )
            opt_step_dirname = opt_step_s3_file_key.split("/")[-1]
            cluster_opt_step_dir = f"/fsx/m4/experiments/local_experiment_dir/s3_async_temporary_checkpoint_folder/{run_name}/{opt_step_dirname}"
            unwrapped_model_dir = f"{cluster_opt_step_dir}/unwrapped_model"
            tokenizer_dir = f"{cluster_opt_step_dir}/tokenizer"
            cmd_model = (
                f"s5cmd sync s3://{bucket_name}/{opt_step_s3_file_key}/unwrapped_model/* {unwrapped_model_dir}".split()
            )
            cmd_tokenizer = f"s5cmd sync s3://{bucket_name}/{opt_step_s3_file_key}/tokenizer/* {tokenizer_dir}".split()
            download_response_model = cmd_retry_loop(cmd_model, max_retries=5)
            print(f"download_response_model: {download_response_model}")
            download_response_tokenizer = cmd_retry_loop(cmd_tokenizer, max_retries=5)
            print(f"download_response_tokenizer: {download_response_tokenizer}")

            # If there is an adapter, load it as well
            if check_s3_file(f"{opt_step_s3_file_key}/unwrapped_adapter"):
                unwrapped_adapter_dir = f"{cluster_opt_step_dir}/unwrapped_adapter"
                cmd_adapter = (
                    f"s5cmd sync s3://{bucket_name}/{opt_step_s3_file_key}/unwrapped_adapter/* {unwrapped_adapter_dir}"
                    .split()
                )
                download_response_adapter = cmd_retry_loop(cmd_adapter, max_retries=5)
                print(f"download_response_adapter: {download_response_adapter}")

        print(f"opt_step_dirname: {opt_step_dirname} downloaded to cluster_opt_step_dir: {cluster_opt_step_dir}")

        if is_not_converted:
            print(f"Converting {cluster_opt_step_dir}")
            convert_cmd = [zero_checkpoint_to_hf_path, cluster_opt_step_dir]
            conversion_response = run_cmd(convert_cmd)
            print(f"conversion_response: {conversion_response}")
            print(f"upload converted checkpoint: {cluster_opt_step_dir}")
            upload_cmd = (
                f"s5cmd sync {cluster_opt_step_dir}/unwrapped_model/"
                f" s3://{bucket_name}/{opt_step_s3_file_key}/unwrapped_model/ ".split()
            )
            upload_response = cmd_retry_loop(upload_cmd, max_retries=5)
            print(f"upload_response: {upload_response}")

            if Path(f"{cluster_opt_step_dir}/unwrapped_adapter").exists():
                upload_cmd_lora = (
                    f"s5cmd sync {cluster_opt_step_dir}/unwrapped_adapter/"
                    f" s3://{bucket_name}/{opt_step_s3_file_key}/unwrapped_adapter/ ".split()
                )
                upload_response_lora = cmd_retry_loop(upload_cmd_lora, max_retries=5)
                print(f"upload_response_lora: {upload_response_lora}")
                print(
                    f"Uploaded {cluster_opt_step_dir}/unwrapped_adapter to"
                    f" s3://{bucket_name}/{opt_step_s3_file_key}/unwrapped_adapter"
                )


if __name__ == "__main__":
    main()
