import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import boto3
import requests
from huggingface_hub import HfFileSystem


logger = logging.getLogger(__name__)


def load_rendered_text_into_s3(start_idx, end_idx):
    # Get list of tarfile URLs
    fs = HfFileSystem(token=os.getenv("HF_TOKEN", True))
    file_list_1 = fs.ls("datasets/wendlerc/RenderedText", detail=False)
    file_list_2 = fs.ls("datasets/wendlerc/RenderedText/remaining", detail=False)
    file_list_1 = [full_filename.split("/")[-1] for full_filename in file_list_1 if full_filename.endswith(".tar")]
    file_list_2 = [full_filename.split("/")[-1] for full_filename in file_list_2 if full_filename.endswith(".tar")]

    prefix_1 = "https://huggingface.co/datasets/wendlerc/RenderedText/resolve/main/"
    prefix_2 = "https://huggingface.co/datasets/wendlerc/RenderedText/resolve/main/remaining/"
    url_list_1 = [prefix_1 + filename for filename in file_list_1]
    url_list_2 = [prefix_2 + filename for filename in file_list_2]
    tar_file_urls = url_list_1 + url_list_2
    tar_file_urls = tar_file_urls[start_idx:end_idx]

    s3_bucket_name = "m4-datasets-us-east-1"
    s3_folder_name = "rendered_text"  # Folder name inside the S3 bucket

    # Function to download and upload tarfile
    def download_upload_tarfile(tar_file_url):
        response = requests.get(tar_file_url)
        tar_file_name = "shard_" + os.path.basename(tar_file_url)
        tar_file_path = "/scratch/m4data/" + os.path.basename(tar_file_url)
        with open(tar_file_path, "wb") as tar_file:
            tar_file.write(response.content)

        # Upload the tarfile to S3
        s3 = boto3.client("s3")
        s3_key = f"{s3_folder_name}/{tar_file_name}"
        s3.upload_file(tar_file_path, s3_bucket_name, s3_key)

        # Remove the downloaded tarfile
        os.remove(tar_file_path)
        logger.info(f"Uploaded {tar_file_name} to S3 and removed it from local storage.")

    # Create a ThreadPoolExecutor with a maximum of 5 worker threads
    with ThreadPoolExecutor(max_workers=95) as executor:
        # Submit download_upload_tarfile function for each tarfile URL
        futures = [executor.submit(download_upload_tarfile, tar_file_url) for tar_file_url in tar_file_urls]

        # Wait for all tasks to complete
        for future in futures:
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_idx",
        type=int,
        help="idx at which to start downloading and uploading tarfiles",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        help="idx at which to end downloading and uploading tarfiles",
    )

    args = parser.parse_args()

    load_rendered_text_into_s3(args.start_idx, args.end_idx)
