import argparse
import logging
from pathlib import Path

import datasets
from datasets import Features, load_dataset
from datasets.utils.logging import set_verbosity_info


logger = logging.getLogger(__name__)


set_verbosity_info()

features = {
    "c4_shard": {"dtype": "int64", "id": None, "_type": "Value"},
    "c4_timestamp": {"dtype": "string", "id": None, "_type": "Value"},
    "html": {"dtype": "string", "id": None, "_type": "Value"},
    "url": {"dtype": "string", "id": None, "_type": "Value"},
    "metadata_html": [
        {
            "char_end_idx": {"dtype": "int64", "id": None, "_type": "Value"},
            "char_start_idx": {"dtype": "int64", "id": None, "_type": "Value"},
            "html_attrs": {
                "attrs": [{"dtype": "string", "id": None, "_type": "Value"}],
                "values": [{"dtype": "string", "id": None, "_type": "Value"}],
            },
            "key": {"dtype": "string", "id": None, "_type": "Value"},
            "relative_end_pos": {"dtype": "int64", "id": None, "_type": "Value"},
            "relative_start_pos": {"dtype": "int64", "id": None, "_type": "Value"},
            "type": {"dtype": "string", "id": None, "_type": "Value"},
            "value": {"dtype": "string", "id": None, "_type": "Value"},
        }
    ],
    "text": {"dtype": "string", "id": None, "_type": "Value"},
    "html_footer": [{"dtype": "string", "id": None, "_type": "Value"}],
    "html_head": [{"dtype": "string", "id": None, "_type": "Value"}],
    "html_title": [{"dtype": "string", "id": None, "_type": "Value"}],
    "HtmlPreprocessor_error": {"dtype": "int64", "id": None, "_type": "Value"},
    "HtmlPreprocessor_error_comment": {"dtype": "string", "id": None, "_type": "Value"},
    "metadata_url": [
        {
            "key": {"dtype": "string", "id": None, "_type": "Value"},
            "type": {"dtype": "string", "id": None, "_type": "Value"},
            "value": {"dtype": "string", "id": None, "_type": "Value"},
        }
    ],
    "metadata_timestamp": [
        {
            "key": {"dtype": "string", "id": None, "_type": "Value"},
            "type": {"dtype": "string", "id": None, "_type": "Value"},
            "value": {"dtype": "string", "id": None, "_type": "Value"},
        }
    ],
    "metadata_generation_length_text": [
        {
            "key": {"dtype": "string", "id": None, "_type": "Value"},
            "type": {"dtype": "string", "id": None, "_type": "Value"},
            "value": {"dtype": "string", "id": None, "_type": "Value"},
        }
    ],
    "metadata_generation_length_sentence": [
        {
            "char_end_idx": {"dtype": "int64", "id": None, "_type": "Value"},
            "char_start_idx": {"dtype": "int64", "id": None, "_type": "Value"},
            "key": {"dtype": "string", "id": None, "_type": "Value"},
            "type": {"dtype": "string", "id": None, "_type": "Value"},
            "value": {"dtype": "string", "id": None, "_type": "Value"},
        }
    ],
    "metadata_generation_datasource": [
        {
            "key": {"dtype": "string", "id": None, "_type": "Value"},
            "type": {"dtype": "string", "id": None, "_type": "Value"},
            "value": {"dtype": "string", "id": None, "_type": "Value"},
        }
    ],
    "metadata_website_desc": [
        {
            "key": {"dtype": "string", "id": None, "_type": "Value"},
            "type": {"dtype": "string", "id": None, "_type": "Value"},
            "value": {"dtype": "string", "id": None, "_type": "Value"},
        }
    ],
}


def convert_types(features):
    if isinstance(features, dict) and "_type" in features:
        return getattr(datasets, features["_type"])(features["dtype"])
    elif isinstance(features, dict):
        return {key: convert_types(value) for key, value in features.items()}
    elif isinstance(features, list):
        return [convert_types(value) for value in features]


def load_modelling_metadata_dataset(dataset_path: str, shard_name: str, cache_path: str = None):
    final_features = convert_types(features)

    if cache_path is None:
        cache_path = datasets.config.HF_DATASETS_CACHE
    ds = load_dataset(
        dataset_path,
        data_files=[f"{shard_name}.jsonl.gz"],
        cache_dir=cache_path,
        features=Features(final_features),
        # this loading script only work for this version of the dataset
        # revision="827d53b6718b9bcdd88335e0c16630fe97d3b956",
    )

    return ds


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--shard-name", type=str)
    args = parser.parse_args()

    logger.info(" ===== Loading dataset =====")
    ds = load_modelling_metadata_dataset(dataset_path=str(args.dataset_path), shard_name=args.shard_name)
    logger.info(f"ds_final info: {ds}")
    logger.info(" ===== Saving Final dataset =====")
    save_path = args.save_dir / args.shard_name
    logger.info(f"Saving to final dataset at {save_path}.")
    tmp_save_path = Path(save_path.parent, f"tmp-{save_path.name}")
    ds.save_to_disk(tmp_save_path)
    tmp_save_path.rename(save_path)
    logger.info(" ===== Final dataset saved successfully =====")
