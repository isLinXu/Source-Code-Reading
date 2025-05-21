from enum import Enum


class DatasetNames(Enum):
    PMD = "pmd"
    LAION = "laion"
    LAION_COCO = "laion_coco"
    TIKZ = "tikz"
    CM4 = "cm4"
    WIKI = "wiki"
    IMAGE_WEBSITE_CODE = "image_website_code"
    VQAV2_TASK_FINETUNING = "vqav2_task_finetuning"
    OCR = "ocr"
    DOCVQA = "docvqa"
    SFT = "sft"


class DatasetTypes(Enum):
    WEB_DOCUMENTS = "wd"
    IMAGE_CAPTION_PAIRS = "icp"
    VQAV2_TASK_FINETUNING = "vqav2_task_finetuning"
    OCR = "ocr"
    DOCVQA = "docvqa"
    SFT = "sft"
