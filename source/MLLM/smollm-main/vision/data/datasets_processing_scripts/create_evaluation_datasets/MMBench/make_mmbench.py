import base64
from copy import deepcopy
from io import BytesIO

import datasets
import pandas as pd
from datasets import Dataset
from PIL import Image


PATH_MMBENCH_DATA = (  # DL from https://opencompass.org.cn/mmbench
    "/Users/hugolaurencon/Desktop/mmbench_dev_20230712.tsv"
)
NUM_PROC = 10
REPO_ID = "HuggingFaceM4/MMBench_dev"


data_frame = pd.read_csv(PATH_MMBENCH_DATA, sep="\t", header=0)


ds = Dataset.from_pandas(data_frame)
ds = ds.remove_columns(["index", "category", "source", "l2-category", "comment", "split"])
ds = ds.rename_column("answer", "label")


def map_func_transform_image_column(example):
    example["image"] = Image.open(BytesIO(base64.b64decode(example["image"])))
    return example


new_features = deepcopy(ds.features)
new_features["image"] = datasets.Image()
new_features["label"] = datasets.features.ClassLabel(names=["A", "B", "C", "D"])

ds = ds.map(map_func_transform_image_column, features=new_features, num_proc=NUM_PROC)

ds.push_to_hub(REPO_ID)


def map_func_modif_context(example):
    question = example["question"]
    hint = example["hint"]
    context = []
    if hint:
        context.append(f"Context: {hint}")
    context.append(f"Question: {question}")
    context.append("Possible answers:")
    for key in ["A", "B", "C", "D"]:
        ans = example[key]
        if ans:
            context.append(f"{key}: {ans}")
    context.append("Correct answer: ")
    example["context"] = "\n".join(context)
    return example


ds = ds.map(map_func_modif_context, num_proc=NUM_PROC)
ds = ds.remove_columns(["question", "hint", "A", "B", "C", "D"])
ds.push_to_hub(REPO_ID + "_modif", private=True)
