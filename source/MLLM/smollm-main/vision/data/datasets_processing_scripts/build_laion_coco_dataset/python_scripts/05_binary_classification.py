import logging
import os
import sys

import torch
import torch.nn as nn
from datasets import load_from_disk
from PIL import Image, ImageFile
from transformers import AutoModel, AutoProcessor

from m4.training.utils import _convert_to_rgb


class MyCustomBinaryClassification(nn.Module):
    def __init__(self, freeze_siglip=True):
        super().__init__()
        self.siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
        if freeze_siglip:
            self.siglip.requires_grad_(False)
        self.freeze_siglip = freeze_siglip
        input_size = self.siglip.config.text_config.hidden_size * 2
        self.fc1 = nn.Linear(input_size, int(input_size / 2))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(int(input_size / 2), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, pixel_values):
        if self.freeze_siglip:
            with torch.no_grad():
                self.siglip.eval()
                outputs = self.siglip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        else:
            outputs = self.siglip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        text_features = outputs.text_embeds
        image_features = outputs.image_embeds
        feature = torch.cat([image_features, text_features], dim=-1)
        return self.sigmoid(self.fc2(self.dropout(self.activation(self.fc1(feature)))).squeeze(-1))


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


IDX_JOB = sys.argv[1]
PATH_SAVE_DISK_TMP_FILES = f"/scratch/storage_hugo_{IDX_JOB}/"

MAX_NUM_RETRIES_SYNC = 3

PATH_DS_LAION_COCO_S3 = (
    f"s3://m4-datasets-us-east-1/LAION_data/laion_coco_dataset_optoutrmv_nsfwfiltered_smallimgrmv/{IDX_JOB}/"
)
# PATH_DS_LAION_COCO_S3 = "s3://m4-datasets-us-east-1/trash/ds_laion_coco_100000/"  # For testing
PATH_DS_LAION_COCO_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "ds_laion_coco")

PATH_BINARY_CLASSIFIER = (
    "/fsx/m4/victor/bin_classif_models/fold-3_epoch-14_precision-0.55_recall-0.38_f1-0.44.model.pt"
)
DEVICE_BINARY_CLASSIFIER = "cuda"
PATH_PROCESSOR_BINARY_CLASSIFIER = "google/siglip-so400m-patch14-384"
BATCH_SIZE = 512
THRESHOLD_BINARY_CLASSIFICATION = 0.5
NUM_PROC = 1
NUM_PROC_SAVING = 12

PATH_SAVE_DISK_DS_LAION_COCO_BINARY_CLASSIFICATION_FILTERED = os.path.join(
    PATH_SAVE_DISK_TMP_FILES, "dl_laion_coco_binclassifrmv"
)
PATH_SAVE_S3_DS_LAION_COCO_BINARY_CLASSIFICATION_FILTERED = f"s3://m4-datasets-us-east-1/LAION_data/laion_coco_dataset_optoutrmv_nsfwfiltered_smallimgrmv_binclassifrmv/{IDX_JOB}/"


class BinaryClassifierFiltering:
    __slots__ = (
        "path_binary_classifier",
        "binary_classifier",
        "path_processor_binary_classifier",
        "processor_binary_classifier",
    )

    def __init__(
        self,
        path_binary_classifier,
        path_processor_binary_classifier,
    ):
        self.path_binary_classifier = path_binary_classifier
        self.binary_classifier = MyCustomBinaryClassification()
        self.binary_classifier.load_state_dict(torch.load(path_binary_classifier))
        self.binary_classifier.to(DEVICE_BINARY_CLASSIFIER)
        self.binary_classifier.eval()
        self.binary_classifier = self.binary_classifier.to(torch.bfloat16)
        self.path_processor_binary_classifier = path_processor_binary_classifier
        self.processor_binary_classifier = AutoProcessor.from_pretrained(path_processor_binary_classifier)

    def __call__(self, batch):
        texts = batch["text"]
        images = batch["image"]
        inputs = self.process_inputs_binary_classification(texts=texts, images=images)
        inputs = inputs.to(DEVICE_BINARY_CLASSIFIER)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        with torch.no_grad():
            logits = self.binary_classifier(**inputs)
            bool_keep_examples = (logits < 0.5).tolist()
        return bool_keep_examples

    def process_inputs_binary_classification(self, texts, images):
        inputs = self.processor_binary_classifier.tokenizer(
            texts,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
            truncation=True,
        )
        inputs.update(self.processor_binary_classifier(images=[_convert_to_rgb(image) for image in images]))
        return inputs

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.path_binary_classifier,
                self.path_processor_binary_classifier,
            ),
        )


if __name__ == "__main__":
    if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
        os.system(f"rm -rf {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_SAVE_DISK_TMP_FILES}")

    logger.info("Starting loading the LAION COCO dataset")
    command_sync_s3 = f"aws s3 sync {PATH_DS_LAION_COCO_S3} {PATH_DS_LAION_COCO_LOCAL}"
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)

    ds_laion_coco = load_from_disk(PATH_DS_LAION_COCO_LOCAL)
    num_pairs_before_filtering = ds_laion_coco.num_rows
    logger.info("Finished loading the LAION COCO dataset")

    logger.info("Starting removing the bad images with the binary classifier")
    binary_classifier_filtering = BinaryClassifierFiltering(
        path_binary_classifier=PATH_BINARY_CLASSIFIER,
        path_processor_binary_classifier=PATH_PROCESSOR_BINARY_CLASSIFIER,
    )
    ds_laion_coco = ds_laion_coco.filter(
        binary_classifier_filtering, num_proc=NUM_PROC, batched=True, batch_size=BATCH_SIZE
    )
    logger.info("Finished removing the bad images with the binary classifier")

    logger.info("Starting saving the LAION COCO dataset with the bad images removed")
    ds_laion_coco.save_to_disk(PATH_SAVE_DISK_DS_LAION_COCO_BINARY_CLASSIFICATION_FILTERED, num_proc=NUM_PROC_SAVING)

    command_sync_s3 = (
        "aws s3 sync"
        f" {PATH_SAVE_DISK_DS_LAION_COCO_BINARY_CLASSIFICATION_FILTERED} {PATH_SAVE_S3_DS_LAION_COCO_BINARY_CLASSIFICATION_FILTERED}"
    )
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)
    logger.info("Finished saving the LAION COCO dataset with the bad images removed")

    logger.info(
        "Number of pairs in the LAION COCO dataset before the filtering of the bad images:"
        f" {num_pairs_before_filtering}"
    )
    logger.info(
        f"Number of pairs in the LAION COCO dataset after the filtering of the bad images: {ds_laion_coco.num_rows}"
    )

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -rf {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")
