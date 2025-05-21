import os
import random
import sys
from functools import partial

import torch
from datasets import load_from_disk
from multiprocess import set_start_method
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_tokenizer(model_id, device):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def mapper_tokenize_dataset(sample, tokenizer, max_tokenized_len):
    sample_concat_texts = []
    for texts in sample["texts"]:
        concat_texts = "\n\n".join([text for text in texts if text is not None])
        sample_concat_texts.append(concat_texts)

    encodings = tokenizer(
        sample_concat_texts,
        add_special_tokens=False,
        padding="max_length",
        truncation=False,
        max_length=max_tokenized_len,
        return_attention_mask=True,
    )
    for i, (input_ids, attention_mask) in enumerate(zip(encodings["input_ids"], encodings["attention_mask"])):
        if len(input_ids) > max_tokenized_len:
            start_index = random.randint(0, len(input_ids) - max_tokenized_len)
            # Calculate the end index based on the start index and subset size
            end_index = start_index + max_tokenized_len
            encodings["input_ids"][i] = input_ids[start_index:end_index]
            encodings["attention_mask"][i] = attention_mask[start_index:end_index]
        else:
            encodings["input_ids"][i] = input_ids
            encodings["attention_mask"][i] = attention_mask
    sample["concat_texts"] = sample_concat_texts
    sample["input_ids"] = encodings["input_ids"]
    sample["attention_mask"] = encodings["attention_mask"]

    return sample


def mapper_ppl_dataset(input_ids, attention_mask, model, device, loss_fct):
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = input_ids

    with torch.no_grad():
        out_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = out_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_attention_mask_batch = attention_mask[..., 1:].contiguous()

    perplexity_batch = torch.exp(
        (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
        / shift_attention_mask_batch.sum(1)
    )
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ppl": perplexity_batch.to("cpu").to(torch.float),
    }
    return result


if __name__ == "__main__":
    # # Useful otherwise the `map` hangs in multiprocessing
    set_start_method("spawn")
    IDX_JOB = sys.argv[1]

    PATH_WEBDOCS_S3 = f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_linededup_finalcleaning_setimgurlsdedup_optoutrmv_finalprocessing/{IDX_JOB}/*"
    PATH_NEW_WEBDOCS_S3 = f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_linededup_finalcleaning_setimgurlsdedup_optoutrmv_finalprocessing_pplfilter/{IDX_JOB}/"
    PATH_WEBDOCS_LOCAL = f"/scratch/storage_leo/loaded_dataset/{IDX_JOB}/"
    PATH_SAVE_DISK_TMP_FILES = f"/scratch/storage_leo/saved_dataset/{IDX_JOB}/"
    command_is_in_s3 = (
        "aws s3 ls"
        f" s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_linededup_finalcleaning_setimgurlsdedup_optoutrmv_finalprocessing_pplfilter/{IDX_JOB}/"
    )
    # Check that folder is not in s3 already
    if os.system(command_is_in_s3) == 256:
        if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
            os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
        os.system(f"mkdir -p {PATH_SAVE_DISK_TMP_FILES}")
        os.system(f"mkdir -p {PATH_WEBDOCS_LOCAL}")

        command_sync_s3 = f"s5cmd sync {PATH_WEBDOCS_S3} {PATH_WEBDOCS_LOCAL}"
        os.system(command_sync_s3)

        dataset = load_from_disk(PATH_WEBDOCS_LOCAL)

        # Only use for debugging + determine threshold
        dataset = dataset.select(range(100))

        input_texts = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer = get_model_tokenizer(model_id="tiiuae/falcon-rw-1b", device=device)
        mapper_tokenize_dataset_kwargs = {
            "tokenizer": tokenizer,
            "max_tokenized_len": 512,
        }

        mapper_tokenize = partial(mapper_tokenize_dataset, **mapper_tokenize_dataset_kwargs)
        dataset = dataset.map(
            mapper_tokenize,
            batched=True,
            batch_size=20,
            num_proc=10,
        )

        loss_fct = CrossEntropyLoss(reduction="none")
        mapper_ppl_dataset_kwargs = {
            "model": model,
            "device": device,
            "loss_fct": loss_fct,
        }
        mapper_ppl = partial(mapper_ppl_dataset, **mapper_ppl_dataset_kwargs)
        hf_dataset_ppl = dataset.with_format(
            "torch", columns=["input_ids", "attention_mask"], output_all_columns=True
        ).map(
            mapper_ppl,
            batched=True,
            batch_size=20,
            num_proc=1,
            input_columns=["input_ids", "attention_mask"],
        )
        # use only to check examples manually and determine threshold
        # hf_dataset_ppl = hf_dataset_ppl.sort("ppl")

        hf_dataset_ppl = hf_dataset_ppl.filter(
            lambda ppl: torch.tensor(ppl) < 42, batched=True, batch_size=20, num_proc=10, input_columns=["ppl"]
        )
        hf_dataset_ppl = hf_dataset_ppl.remove_columns(["concat_texts", "input_ids", "attention_mask"])
        print(f"Proportion of dataset filtered: {((len(dataset) - len(hf_dataset_ppl))/len(dataset))*100}%")
        hf_dataset_ppl.save_to_disk(PATH_SAVE_DISK_TMP_FILES)
        command_sync_s3 = f"s5cmd sync {PATH_SAVE_DISK_TMP_FILES} {PATH_NEW_WEBDOCS_S3}"
        command_remove_webdocs_local_files = f"rm -r {PATH_WEBDOCS_LOCAL}"
        command_remove_save_disk_temp_files = f"rm -r {PATH_SAVE_DISK_TMP_FILES}"
        os.system(command_sync_s3)
        os.system(command_remove_webdocs_local_files)
        os.system(command_remove_save_disk_temp_files)
