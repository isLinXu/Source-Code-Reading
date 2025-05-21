import logging
import os
import sys
from multiprocessing import cpu_count

import numpy as np
from datasets import load_dataset
from PIL import Image, ImageFile
from tqdm import tqdm


# Useful to avoid DecompressionBombError and truncated image error
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


IDX_JOB = int(sys.argv[1])
PATH_SAVE_DISK_TMP_FILES = f"/scratch/storage_hugo_{IDX_JOB}/"

NUM_JOB_TOT = 200
# tar files indexed from 50_000 to 231_349 are the ones missing from the previous 1/4 of LAION
LIST_IDX_LAION_TAR_FILES = [el.tolist() for el in np.array_split(range(50_000, 231_349), NUM_JOB_TOT)][IDX_JOB]

PATH_LAION_TAR_FILES_S3 = "s3://hf-datasets-laion-5b/glacier/laion-data/laion2B-data/"
PATH_LAION_TAR_FILES_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "LAION_tar_files")

PATH_TEMPLATE_LOADING_SCRIPT_LAION = (  # script at m4/datasets_processing_scripts/build_laion_dataset/python_scripts/01_02_template_loading_script_laion.py
    "/fsx/hugo/download_prepare_laion/template_loading_script_laion.py"
)
PATH_LOADING_SCRIPT_LAION = os.path.join(PATH_SAVE_DISK_TMP_FILES, "loading_script_laion.py")

NUM_PROC = cpu_count()

PATH_SAVE_DISK_LAION_DATASET = os.path.join(PATH_SAVE_DISK_TMP_FILES, "laion_dataset")
PATH_SAVE_S3_LAION_DATASET = os.path.join("s3://m4-datasets/LAION_data/laion_raw_hf_dataset", str(IDX_JOB))


if __name__ == "__main__":
    if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
        os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_LAION_TAR_FILES_LOCAL}")

    logger.info("Starting downloading the LAION tar files")
    for idx in tqdm(LIST_IDX_LAION_TAR_FILES):
        path_laion_tar_file = os.path.join(PATH_LAION_TAR_FILES_S3, f"{idx:06d}.tar")
        command_sync_s3 = f"aws s3 cp {path_laion_tar_file} {PATH_LAION_TAR_FILES_LOCAL} --profile hugo-sandbox"
        os.system(command_sync_s3)
    logger.info("Finished downloading the LAION tar files")

    logger.info("Starting converting the tar files to HF datasets")
    f = open(PATH_TEMPLATE_LOADING_SCRIPT_LAION, "r")
    template_loading_script_laion = f.read()
    f.close()

    str_loading_script = f'DL_DATASET_PATH = "{PATH_LAION_TAR_FILES_LOCAL}"\n' + template_loading_script_laion
    f = open(PATH_LOADING_SCRIPT_LAION, "w")
    f.write(str_loading_script)
    f.close()

    laion_dataset = load_dataset(PATH_LOADING_SCRIPT_LAION)
    laion_dataset = laion_dataset["train"]
    logger.info("Finished converting the tar files to HF datasets")

    logger.info("Starting saving the LAION dataset")
    laion_dataset.save_to_disk(PATH_SAVE_DISK_LAION_DATASET, num_proc=NUM_PROC)

    command_sync_s3 = f"aws s3 sync {PATH_SAVE_DISK_LAION_DATASET} {PATH_SAVE_S3_LAION_DATASET}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    logger.info("Finished saving the LAION dataset")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")
