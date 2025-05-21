from pathlib import Path
from urllib.parse import urlparse

from datasets import Features, Image, Sequence, Value, load_dataset, load_from_disk


NUM_SHARDS = 68
DATA_DIR = Path("/home/lucile/local_datasets/enwiki/enwiki-NS0-20230220-ENTERPRISE-HTML-EXTRACTION")
DATASET_NAME_COMPLETE_EXAMPLES = "wikipedia_html_enterprise-with-images-full-v1"
DATASET_NAME_INCOMPLETE_EXAMPLES = "wikipedia_html_enterprise-with-images-incomplete-v1"
NUM_PROC = 32

EXCLUDE_SHARD_IDS = [34]  # This shard is corrupted since the beginning

# Logging generates a lot of output from elasticsearch which slow down the script
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
# )
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


def add_image_name(example):
    example["image_name"] = urlparse(example["image_url"]).path.split("/")[-1]
    return example


print("Loading wit dataset")
wit_ds = load_dataset("wikimedia/wit_base")["train"]
print("Adding image name")
wit_ds = wit_ds.map(add_image_name, num_proc=NUM_PROC)
print("Adding elasticsearch index")
# The order matters here, we need to load the index before we define the function that uses it
wit_ds.add_elasticsearch_index(
    "image_name",
    host="localhost",
    port="9200",
)
# wit_ds.load_elasticsearch_index("image_name", host="localhost", port="9200", es_index_name="huggingface_datasets_tmpzdmi8uft")

print("Index created")


def get_image_from_wit(example):
    max_retries = 10
    num_found = 0
    num_not_found = 0
    mismatches = []
    images = []
    for image in example["images"]:
        if image == "":
            images.append(None)
        else:
            image_name = urlparse(image).path.split("/")[-2]

            trials = 0
            while True:
                try:
                    scores, retrieved_examples = wit_ds.get_nearest_examples(
                        "image_name", image_name, k=1, request_timeout=1800, allow_partial_search_results=False
                    )
                    break
                except Exception as e:
                    trials += 1
                    if trials > max_retries:
                        print("Max retries reached")
                        break
                    print("Retrying...")
                    print(f"Error: {e}")
                    continue

            if trials > max_retries or len(retrieved_examples["image_name"]) == 0:
                images.append(None)
                num_not_found += 1
                continue

            found_image_name = retrieved_examples["image_name"][0]
            if found_image_name == image_name:
                images.append(retrieved_examples["image"][0])
                num_found += 1
            else:
                images.append(None)
                mismatches.append((image_name, found_image_name))
                num_not_found += 1
    example["images_urls"] = example["images"]
    example["images"] = images
    example["num_found"] = num_found
    example["num_not_found"] = num_not_found
    example["mismatches"] = mismatches
    return example


def process_shard(shard_id, data_dir):
    shard_dir = data_dir / f"shard_{shard_id}"
    shard_ds = load_from_disk(shard_dir / "wikipedia_html_enterprise")

    shard_ds = shard_ds.map(
        get_image_from_wit,
        features=Features(
            {
                "texts": Sequence(Value("string")),
                "images": Sequence(Image()),
                "metadata": Value("string"),
                "images_urls": Sequence(Value("string")),
                "num_found": Value("int32"),
                "num_not_found": Value("int32"),
                "mismatches": Sequence([Value("string")]),
            }
        ),
        num_proc=NUM_PROC,
    )

    total_num_found = sum(shard_ds["num_found"])
    total_num_not_found = sum(shard_ds["num_not_found"])
    total = total_num_found + total_num_not_found
    print(f"Shard {shard_id}: {total_num_found} images found, {total_num_not_found} images not found, {total} total")

    complete_examples = shard_ds.filter(lambda x: x["num_not_found"] == 0 and x["num_found"] > 0, num_proc=NUM_PROC)
    print(f"Shard {shard_id}: {len(complete_examples)} examples with all images found")
    complete_examples.save_to_disk(shard_dir / DATASET_NAME_COMPLETE_EXAMPLES)

    incomplete_examples = shard_ds.filter(
        lambda x: ~(x["num_not_found"] == 0 and x["num_found"] > 0), num_proc=NUM_PROC
    )
    print(f"Shard {shard_id}: {len(incomplete_examples)} examples with some images not found")
    incomplete_examples.save_to_disk(shard_dir / DATASET_NAME_INCOMPLETE_EXAMPLES)


print("Processing shards")
for shard_id in range(1, 2):
    if shard_id in EXCLUDE_SHARD_IDS:
        continue
    print(f"Processing shard {shard_id}")
    process_shard(shard_id, DATA_DIR)
