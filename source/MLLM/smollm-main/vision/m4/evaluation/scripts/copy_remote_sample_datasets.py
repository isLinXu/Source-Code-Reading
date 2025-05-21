from pathlib import Path

from datasets import load_dataset

from m4.evaluation.tasks import VGPT2_SAMPLE_TASKS, Predictor
from m4.evaluation.utils import EvaluationVersion


MIN_DATASET_SIZE = 100
DEFAULT_NUM_EX_PER_CLASS = 3

ALREADY_COPIED_DATASETS = set()

model_name = "gpt2"  # Not used but necessary to load the task
tokenizer_name = "t5-base"  # Not used but necessary to load the task
image_size = 224  # Not used but necessary to load the task
evaluation_version = EvaluationVersion.v2  # Not used but necessary to load the task

save_dir = Path("/gpfsscratch/rech/cnw/commun/local_datasets")


def load_and_save_dataset(task, split, save_dir):
    dataset_split = load_dataset(task.dataset_name, name=task.dataset_config, split=split, use_auth_token=True)
    print("********************************************************")
    print(task.__class__.__name__)
    print(len(dataset_split))
    print(f"Dataset name is {task.dataset_name} and split is {split} and config is {task.dataset_config}")
    print("********************************************************")

    dataset_split.save_to_disk(save_dir / task.dataset_name / split)


if __name__ == "__main__":
    for task_objet in VGPT2_SAMPLE_TASKS[Predictor.in_contexter]:
        task = task_objet(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            image_size=image_size,
            evaluation_version=evaluation_version,
        )

        load_and_save_dataset(task, task.test_split_name, save_dir)
        load_and_save_dataset(task, task.train_split_name, save_dir)
