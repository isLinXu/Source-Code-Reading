import glob
import os
from pathlib import Path

from m4.sourcing.data_collection.processors.web_document_extractor import create_dataset_images_from_tar


path_save_dir_downloaded_images = Path("/home/lucile/local_datasets/enwiki/enwiki-v2-downloaded-images")
path_save_dir_tmp_datasets_images = Path("/home/lucile/local_datasets/enwiki/enwiki-v2-ds-images-tmp")
num_proc = 16
path_save_file_map_url_idx = Path("/home/lucile/local_datasets/enwiki/enwiki-v2-map-url-idx.json")
path_save_dir_dataset_images = Path("/home/lucile/local_datasets/enwiki/enwiki-v2-ds-images")

tar_paths = []
for path_save_dir_downloaded_images_shard in path_save_dir_downloaded_images.glob("*"):
    if path_save_dir_downloaded_images_shard.is_dir():
        tar_paths.extend(glob.glob(os.path.join(path_save_dir_downloaded_images_shard, "*.tar")))

create_dataset_images_from_tar(
    tar_paths,
    path_save_dir_tmp_datasets_images,
    num_proc,
    path_save_file_map_url_idx,
    path_save_dir_dataset_images,
)
