import logging
from functools import partial

import evaluate
import numpy as np
import transformers
from datasets import load_dataset

from m4.evaluation import custom_metrics
from m4.evaluation.config import DatasetSplit


logger = logging.getLogger(__name__)


# NOTE: Accelerator is not used because linear probing is using scikit learn's logistic regression
def linear_prober(task, accelerator, args):
    model_name = args.tasks.model_name

    model_class = getattr(transformers, task.model_class)
    model = model_class.from_pretrained(model_name)
    model.eval()
    model = model.to(args.hparams.device)

    metric_class = getattr(custom_metrics, task.metric_name, None)
    if metric_class is not None:
        metric = metric_class()
    else:
        metric = evaluate.load(task.metric_name)

    if args.tasks.dataset_split == DatasetSplit.default:
        support_split_name = task.default_support_split_name
        query_split_name = task.default_query_split_name
    else:
        raise ValueError(f"Dataset split {args.tasks.dataset_split} is not supported for linear prober.")

    assert task.default_support_split_name is not None
    train_dataset = load_dataset(
        task.dataset_name, name=task.dataset_config, split=support_split_name, use_auth_token=True
    )
    if args.hparams.select_n_examples is not None:
        train_dataset = train_dataset.select(
            np.random.choice(
                range(len(train_dataset)), min(args.hparams.select_n_examples, len(train_dataset)), replace=False
            )
        )
    train_dataset = train_dataset.map(
        partial(task.project, model=model), batched=True, batch_size=args.hparams.mini_batch_size
    )

    if not args.hparams.only_load_datasets:
        logger.info(f"Info train dataset {train_dataset}")
        train_dataset = train_dataset.with_format("np")
        task.fit(features=train_dataset["features"], labels=train_dataset[task.label_column_name])

    assert query_split_name is not None
    test_dataset = load_dataset(
        task.dataset_name, name=task.dataset_config, split=query_split_name, use_auth_token=True
    )
    logger.info(f"Info test dataset {test_dataset}")

    if args.hparams.only_load_datasets:
        return

    if args.hparams.select_n_examples is not None:
        test_dataset = test_dataset.select(range(min(args.hparams.select_n_examples, len(test_dataset))))
    test_dataset = test_dataset.map(
        partial(task.project, model=model), batched=True, batch_size=args.hparams.mini_batch_size
    )
    test_dataset = test_dataset.with_format("np")
    predictions = task.predict(features=test_dataset["features"])
    metric.add_batch(
        predictions=predictions,
        references=test_dataset[task.label_column_name],
    )
    score = metric.compute()
    return score
