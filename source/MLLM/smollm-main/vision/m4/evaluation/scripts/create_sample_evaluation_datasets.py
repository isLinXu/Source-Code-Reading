import random

from datasets import concatenate_datasets, load_dataset

from m4.evaluation.tasks import VGPT2_TASKS, Predictor


MIN_DATASET_SIZE = 100
DEFAULT_NUM_EX_PER_CLASS = 3
MAX_DATASET_SIZE_TO_SAMPLE = 50000

ALREADY_PUSHED_DATASETS = {
    "test_datasets": {
        "HuggingFaceM4/Stanford-Cars-Sample",
        "HuggingFaceM4/cifar10-Sample",
        "HuggingFaceM4/cifar100-Sample",
        "HuggingFaceM4/food101-Sample",
        "HuggingFaceM4/RenderedSST2-Sample",
        "HuggingFaceM4/imagenet-1k-Sample",
        "HuggingFaceM4/DTD_Describable-Textures-Dataset-partition_1-Sample",
        "HuggingFaceM4/sun397-standard-part1-120k-Sample",
        "HuggingFaceM4/Oxford-IIIT-Pet-Sample",
        "HuggingFaceM4/Caltech-101-with_background_category-Sample",
    },
    "train_datasets": (
        "HuggingFaceM4/Caltech-101-with_background_category-Sample",
        "HuggingFaceM4/cifar10-Sample",
        "HuggingFaceM4/cifar100-Sample",
        "HuggingFaceM4/food101-Sample",
        "HuggingFaceM4/DTD_Describable-Textures-Dataset-partition_1-Sample",
    ),
}


def main():
    model_name = "gpt2"  # Not used but necessary to load the task
    tokenizer_name = "t5-base"  # Not used but necessary to load the task

    for task_objet in VGPT2_TASKS[Predictor.in_contexter]:
        task = task_objet(model_name=model_name, tokenizer_name=tokenizer_name)

        if task.dataset_name.startswith("HuggingFaceM4/"):
            prefix = task.dataset_name
        elif "/" in task.dataset_name:
            raise NotImplementedError()
        else:
            prefix = f"HuggingFaceM4/{task.dataset_name}"

        if task.dataset_config is not None:
            prefix = f"{prefix}-{task.dataset_config}"

        repo_id = f"{prefix}-Sample"

        # Test split

        if repo_id not in ALREADY_PUSHED_DATASETS["test_datasets"]:
            split = task.test_split_name

            sample_ds = create_sample_single_label_dataset(task, split)
            sample_ds.push_to_hub(repo_id, split, private=True)

        # Train split

        if repo_id not in ALREADY_PUSHED_DATASETS["train_datasets"]:
            split = task.train_split_name

            sample_ds = create_sample_single_label_dataset(task, split)
            sample_ds.push_to_hub(repo_id, split, private=True)


def create_sample_single_label_dataset(task, split):
    dataset_split = load_dataset(task.dataset_name, name=task.dataset_config, split=split, use_auth_token=True)
    print("********************************************************")
    print(task.__class__.__name__)
    print(len(dataset_split))
    print(f"Dataset name is {task.dataset_name} and split is {split} and config is {task.dataset_config}")
    print(dataset_split.features[task.label_column_name].num_classes)
    print("********************************************************")

    if len(dataset_split) > MAX_DATASET_SIZE_TO_SAMPLE:
        indices = random.choices(range(len(dataset_split)), k=MAX_DATASET_SIZE_TO_SAMPLE)
        dataset_split = dataset_split.select(indices)

    num_classes = dataset_split.features[task.label_column_name].num_classes
    if num_classes * DEFAULT_NUM_EX_PER_CLASS < MIN_DATASET_SIZE:
        num_ex_per_class = int(MIN_DATASET_SIZE / num_classes)
    else:
        num_ex_per_class = DEFAULT_NUM_EX_PER_CLASS

    print(f"Number of examples per class: {num_ex_per_class}")

    all_sub_ds = []
    for class_id in range(num_classes):
        sub_ds = dataset_split.filter(lambda x: [item == class_id for item in x[task.label_column_name]], batched=True)
        num_examples_to_take = min(num_ex_per_class, len(sub_ds))
        if num_examples_to_take != num_ex_per_class:
            print(
                f"Warning: not enough examples for class {class_id} to take {num_ex_per_class} examples. Taking"
                f" {num_examples_to_take} instead."
            )
        sub_ds = sub_ds.select(range(num_examples_to_take))
        all_sub_ds.append(sub_ds)

    sample_ds = concatenate_datasets(all_sub_ds)
    print(f"Number of examples in sample dataset: {len(sample_ds)}")
    return sample_ds


if __name__ == "__main__":
    main()
