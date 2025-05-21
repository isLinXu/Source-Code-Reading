import argparse
import re

import numpy as np
from datasets import load_dataset
from scipy import stats

from m4.sourcing.data_collection.utils import compute_clip_score, fetch_single_image


def get_scores(dataset_name, save_filename, nb_pairs):
    if dataset_name == "red_caps":
        dataset = load_dataset(
            "red_caps",
            "all",
            split="train",
            streaming=True,
        )
    elif dataset_name == "sbu_captions":
        dataset = load_dataset(
            "sbu_captions",
            split="train",
            streaming=True,
        )
    elif dataset_name == "laion/laion400m":
        dataset = load_dataset(
            "laion/laion400m",
            split="train",
            streaming=True,
        )
    shuffled_dataset = iter(dataset.shuffle(seed=42, buffer_size=10_000))

    counter = 0
    counter_successful = 0
    scores = []

    print("Start collection.")
    while counter_successful < nb_pairs:
        example = next(shuffled_dataset)
        counter += 1

        if dataset_name in ["red_caps", "sbu_captions"]:
            url = example["image_url"]
            caption = example["caption"]
        else:
            url = example["URL"]
            caption = example["TEXT"]

        if dataset_name == "red_caps":
            if len(re.findall(r"http\S+", url)) > 1:
                continue

        image = fetch_single_image(url, timeout=1)
        if image is not None:
            try:
                score = compute_clip_score(texts=[caption], images=image, num_max_words=50).item()
                scores.append(score)
                counter_successful += 1
            except ValueError:
                print("Skipping image. Bug.")
            except RuntimeError:
                print("Skipping image. Model error.")

        if counter_successful % 100 == 0:
            print(f"Done: {counter_successful}/{nb_pairs}")

    print(f"Nb temptatives: {counter}")
    print(f"Number successful temptatives: {counter_successful}")
    print(f"Describe stats: {stats.describe(scores)}")
    np.save(save_filename, scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracting clip score on `nb_pairs` pairs (image, text).")
    parser.add_argument(
        "--dataset_name", type=str, default="red_caps", choices=["red_caps", "sbu_captions", "laion/laion400m"]
    )
    parser.add_argument(
        "--nb_pairs",
        type=int,
        default=10_000,
    )
    args = parser.parse_args()

    get_scores(
        args.dataset_name,
        f"./m4/sourcing/data_collection/outputs/clip_scores_{args.dataset_name.split('/')[-1]}_{args.nb_pairs}.npy",
        args.nb_pairs,
    )
