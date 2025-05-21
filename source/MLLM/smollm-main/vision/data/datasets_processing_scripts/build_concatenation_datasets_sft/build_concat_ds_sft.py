"""
srun --pty --cpus-per-task=96 --mem-per-cpu=20G --partition=hopper-prod bash -i
srun --pty --cpus-per-task=32 --partition=hopper-cpu --qos high bash -i
conda activate shared-m4
"""


from multiprocessing import Pool
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk
from tqdm import tqdm


COMMON_PATH_DATASETS = Path("/fsx/hugo/fine_tuning_datasets_merge_image_individual")

NUM_MAX_MINI_EPOCHS = 50

NUM_PROC = 32

MAPPING_DATASET_TO_PROPORTION = {
    "screen2words": 1,
    "vistext": 0.3,
    "textcaps": 1,
    "vqav2": 1,
    "visual7w": 1,
    "okvqa": 2,
    "cocoqa": 1,
    "vsr": 4,
    "iam": 4,
    "textvqa": 4,
    "st_vqa": 4,
    "diagram_image_to_text": 4,
    "docvqa": 1,
    "infographic_vqa": 1,
    "chartqa": 1,
    "visualmrc": 1,
    "ai2d": 1,
    "figureqa": 0.01,
    "dvqa": 0.1,
    "plotqa": 0.01,
    "ocrvqa": 0.03,
    "scienceqa": 1,
    "aokvqa": 1,
    "nlvr2": 1,
    "iconqa": 1,
    "clevr": 0.1,
    "tallyqa": 0.1,
    "mimic_cgd": 0.3,
    "spot_the_diff": 1,
    "hateful_memes": 1,
    "intergps": 2,
    "geomverse": 0.3,
    "vqarad": 1,
    "websight": 0.005,
    "datikz": 0.005,
    "raven": 1,
    "lima": 1,
    "openhermes": 0.05,
}

MAPPING_DATASET_TO_PROPORTION_MIX_1 = {
    "screen2words": 1,
    "vistext": 0.3,
    "textcaps": 1,
    "vqav2": 1,
    "visual7w": 1,
    "okvqa": 2,
    "cocoqa": 1,
    "vsr": 4,
    "iam": 4,
    "textvqa": 4,
    "st_vqa": 4,
    "diagram_image_to_text": 4,
    "docvqa": 2,
    "infographic_vqa": 2.5,
    "chartqa": 3,
    "visualmrc": 1,
    "ai2d": 5,
    "figureqa": 0.2,
    "dvqa": 0.15,
    "plotqa": 0.015,
    "ocrvqa": 0.03,
    "scienceqa": 3,
    "aokvqa": 1,
    "nlvr2": 1,
    "iconqa": 1,
    "clevr": 0.1,
    "tallyqa": 0.1,
    "mimic_cgd": 0.3,
    "spot_the_diff": 1,
    "hateful_memes": 1,
    "intergps": 3,
    "geomverse": 0.6,
    "vqarad": 1,
    "websight": 0.005,
    "datikz": 0.005,
    "raven": 1,
    "lima": 1,
    "openhermes": 0.05,
}

MAPPING_DATASET_TO_PROPORTION_MIX_2 = {
    "screen2words": 1,
    "vistext": 0.3,
    "textcaps": 1,
    "vqav2": 0.7,
    "visual7w": 1,
    "okvqa": 2,
    "cocoqa": 1,
    "vsr": 4,
    "iam": 4,
    "textvqa": 4,
    "st_vqa": 4,
    "diagram_image_to_text": 4,
    "docvqa": 1,
    "infographic_vqa": 1,
    "chartqa": 1,
    "visualmrc": 1,
    "ai2d": 1,
    "figureqa": 0.01,
    "dvqa": 0.1,
    "plotqa": 0.0025,
    "ocrvqa": 0.03,
    "scienceqa": 1,
    "aokvqa": 1,
    "nlvr2": 1,
    "iconqa": 1,
    "clevr": 0.1,
    "tallyqa": 0.1,
    "mimic_cgd": 0.15,
    "spot_the_diff": 1,
    "hateful_memes": 1,
    "intergps": 2,
    "geomverse": 0.3,
    "vqarad": 1,
    "websight": 0.0025,
    "datikz": 0.005,
    "raven": 1,
    "lima": 1,
    "openhermes": 0.005,
}

MAPPING_DATASET_TO_PROPORTION_MIX_3 = {
    "screen2words": 1,
    "vistext": 0.3,
    "textcaps": 1,
    "vqav2": 1,
    "visual7w": 1,
    "okvqa": 2,
    "cocoqa": 1,
    "vsr": 4,
    "iam": 4,
    "textvqa": 8,
    "st_vqa": 8,
    "diagram_image_to_text": 4,
    "docvqa": 6,
    "infographic_vqa": 8,
    "chartqa": 8,
    "visualmrc": 1,
    "ai2d": 13,
    "figureqa": 0.7,
    "dvqa": 0.5,
    "plotqa": 0.02,
    "ocrvqa": 0.03,
    "scienceqa": 10,
    "aokvqa": 1,
    "nlvr2": 1,
    "iconqa": 6,
    "clevr": 0.1,
    "tallyqa": 0.1,
    "mimic_cgd": 0.3,
    "spot_the_diff": 1,
    "hateful_memes": 1,
    "intergps": 10,
    "geomverse": 1,
    "vqarad": 1,
    "websight": 0.005,
    "datikz": 0.005,
    "raven": 1,
    "lima": 1,
    "openhermes": 0.05,
}


MAPPING_DATASET_TO_PROPORTION_MIX_5 = {
    "screen2words": 1,
    "vistext": 0.3,
    "textcaps": 1,
    "vqav2": 1,
    "visual7w": 1,
    "okvqa": 2,
    "cocoqa": 1,
    "vsr": 4,
    "iam": 4,
    "textvqa": 4,
    "st_vqa": 4,
    "diagram_image_to_text": 4,
    "docvqa": 2,
    "infographic_vqa": 2.5,
    "chartqa": 3,
    "visualmrc": 1,
    "ai2d": 5,
    "figureqa": 0.2,
    "dvqa": 0.15,
    "plotqa": 0.015,
    "ocrvqa": 0.03,
    "scienceqa": 3,
    "aokvqa": 1,
    "nlvr2": 1,
    "iconqa": 1,
    "clevr": 0.1,
    "tallyqa": 0.1,
    "mimic_cgd": 0.3,
    "spot_the_diff": 1,
    "hateful_memes": 1,
    "intergps": 3,
    "geomverse": 0.6,
    "vqarad": 1,
    "websight": 0.005,
    "datikz": 0.005,
    "raven": 1,
    "lima": 1,
    "openhermes": 0.012,
    "orca_math": 0.02,
    "metamathqa": 0.038,
    "math_instruct": 0.06,
    "camel_ai_math": 0.01,
    "atlas_math_sets": 0.004,
    "goat": 0.009,
}

MAPPING_DATASET_TO_PROPORTION_MIX_6 = {
    "screen2words": 1,
    "vistext": 0.3,
    "textcaps": 1,
    "vqav2": 1,
    "visual7w": 1,
    "okvqa": 2,
    "cocoqa": 1,
    "vsr": 4,
    "iam": 4,
    "textvqa": 4,
    "st_vqa": 4,
    "diagram_image_to_text": 4,
    "docvqa": 2,
    "infographic_vqa": 2.5,
    "chartqa": 3,
    "visualmrc": 1,
    "ai2d": 5,
    "figureqa": 0.2,
    "dvqa": 0.15,
    "plotqa": 0.015,
    "ocrvqa": 0.03,
    "scienceqa": 3,
    "aokvqa": 1,
    "nlvr2": 1,
    "iconqa": 1,
    "clevr": 0.1,
    "tallyqa": 0.1,
    "mimic_cgd": 0.3,
    "spot_the_diff": 1,
    "hateful_memes": 1,
    "intergps": 3,
    "geomverse": 0.6,
    "vqarad": 1,
    "websight": 0.005,
    "datikz": 0.005,
    "raven": 1,
    "tat_qa": 3,
    "robut_wikisql": 0.5,
    "robut_wtq": 0.5,
    "lima": 1,
    "openhermes": 0.05,
}


MAPPING_DATASET_TO_PROPORTION_MIX_7 = {
    "screen2words": 1,
    "vistext": 0.3,
    "textcaps": 1,
    "vqav2": 1,
    "visual7w": 1,
    "okvqa": 2,
    "cocoqa": 1,
    "vsr": 4,
    "iam": 4,
    "textvqa": 4,
    "st_vqa": 4,
    "diagram_image_to_text": 4,
    "docvqa": 2,
    "infographic_vqa": 2.5,
    "chartqa": 3,
    "visualmrc": 1,
    "ai2d": 5,
    "figureqa": 0.2,
    "dvqa": 0.15,
    "plotqa": 0.015,
    "ocrvqa": 0.03,
    "scienceqa": 3,
    "aokvqa": 1,
    "nlvr2": 1,
    "iconqa": 1,
    "clevr": 0.1,
    "tallyqa": 0.1,
    "mimic_cgd": 0.3,
    "spot_the_diff": 1,
    "hateful_memes": 1,
    "intergps": 3,
    "geomverse": 0.6,
    "vqarad": 1,
    "websight": 0.005,
    "datikz": 0.005,
    "raven": 1,
    "lima": 1,
    "openhermes": 0.05,
    "clevr_math": 0.15,
    "mapqa": 0.05,
    "tabmwp": 0.5,
    "tqa": 5,
}


MAPPING_DATASET_TO_PROPORTION_MIX_8 = {
    "screen2words": 0.5,
    "vistext": 0.3,
    "textcaps": 1,
    "localized_narratives": 0.04,
    "ny_cc_explanation": 0.05,
    "ny_cc_matching": 0.01,
    "ny_cc_ranking": 0.01,
    "vqav2": 0.7,
    "visual7w": 1,
    "okvqa": 2,
    "cocoqa": 1,
    "vsr": 4,
    "rendered_text": 0.04,
    "iam": 1,
    "textvqa": 2,
    "st_vqa": 2,
    "diagram_image_to_text": 1,
    "docvqa": 2,
    "infographic_vqa": 2,
    "chartqa": 2,
    "visualmrc": 1,
    "ai2d": 4,
    "figureqa": 0.03,
    "dvqa": 0.1,
    "plotqa": 0.015,
    "ocrvqa": 0.03,
    "scienceqa": 2,
    "aokvqa": 1,
    "nlvr2": 1,
    "iconqa": 2,
    "clevr": 0.1,
    "clevr_math": 0.2,
    "tallyqa": 0.15,
    "mapqa": 0.01,
    "mimic_cgd": 0.02,
    "spot_the_diff": 0.5,
    "hateful_memes": 1,
    "intergps": 4,
    "geomverse": 0.3,
    "vqarad": 2,
    "websight": 0.0002,
    "datikz": 0.0001,
    "raven": 1.25,
    "lima": 0.25,
    "openhermes": 0.01,
    "orca_math": 0.005,
    "metamathqa": 0.01,
    "math_instruct": 0.01,
    "camel_ai_math": 0.0005,
    "atlas_math_sets": 0.0015,
    "goat": 0.001,
    "dolly": 0.1,
    "tabmwp": 0.25,
    "tqa": 4,
    "hitab": 1,
    "multihiertt": 1,
    "robut_sqa": 0.1,
    "tat_qa": 1.5,
    "chart2text": 0.3,
    "finqa": 0.8,
    "robut_wikisql": 0.02,
    "robut_wtq": 0.015,
}


MAPPING_DATASET_TO_PROPORTION_MIX_9 = {
    "screen2words": 0.5,
    "vistext": 0.3,
    "textcaps": 1,
    "sharegpt4o": 0.1,
    "iiw400": 2,
    "localized_narratives": 0.02,
    "ny_cc_explanation": 0.05,
    "ny_cc_matching": 0.01,
    "ny_cc_ranking": 0.01,
    "vqav2": 0.4,
    "lnqa": 0.05,
    "visual7w": 1,
    "okvqa": 2,
    "cocoqa": 1,
    "vsr": 4,
    "rendered_text": 0.08,
    "iam": 1,
    "cord": 2,
    "textvqa_new_prompt": 2,
    "st_vqa": 2,
    "diagram_image_to_text": 1,
    "large_docvqa": 0.008,
    "docvqa_new_prompt": 2,
    "infographic_vqa": 2,
    "chartqa_new_prompt": 2,
    "visualmrc": 1,
    "ai2d": 4,
    "figureqa": 0.06,
    "dvqa": 0.2,
    "plotqa": 0.03,
    "ocrvqa": 0.03,
    "scienceqa": 2,
    "aokvqa": 1,
    "nlvr2": 0.5,
    "iconqa": 2,
    "clevr": 0.1,
    "clevr_math": 0.2,
    "tallyqa": 0.15,
    "mapqa": 0.01,
    "mimic_cgd": 0.02,
    "spot_the_diff": 0.5,
    "hateful_memes": 1,
    "raven": 1.25,
    "intergps": 4,
    "geomverse": 0.3,
    "geo170k": 0.05,
    "vqarad": 2,
    "websight": 0.001,
    "datikz": 0.0001,
    "lima": 0.25,
    "openhermes_filtered_code_filtered_bad_punctuation": 0.01,
    "orca_math": 0.005,
    "metamathqa": 0.01,
    "math_instruct": 0.01,
    "camel_ai_math": 0.0005,
    "atlas_math_sets": 0.0015,
    "goat": 0.001,
    "dolly": 0.1,
    "tabmwp": 0.25,
    "tqa": 4,
    "hitab": 1,
    "multihiertt": 1,
    "robut_sqa": 0.1,
    "tat_qa": 1.5,
    "chart2text": 0.3,
    "finqa": 0.8,
    "robut_wikisql": 0.02,
    "robut_wtq": 0.015,
}


MAPPING_DATASET_TO_PROPORTION_LLAVA_CONV_AND_TEXT = {
    "lima": 1,
    "llava_conv": 1,
}

# To modify depending on mixture #
CHOSEN_MIXTURE = MAPPING_DATASET_TO_PROPORTION_MIX_9
PATH_SAVE_DATASET = Path("/fsx/hugo/concat_ds_sft/ds_mixture_9_mini_epochs_50")
PATH_SAVE_TMP_DATASETS = Path("/fsx/hugo/to_delete/tmp_ds_mixture_9_mini_epochs_50")
# To modify depending on mixture #

ALL_DS_NAMES = list(CHOSEN_MIXTURE.keys())
max_proportion_value = max(list(CHOSEN_MIXTURE.values()))
MAPPING_DATASET_TO_PROPORTION_NORMALIZED = {key: value / max_proportion_value for key, value in CHOSEN_MIXTURE.items()}


all_datasets = {ds_name: load_from_disk(COMMON_PATH_DATASETS / ds_name) for ds_name in tqdm(ALL_DS_NAMES)}


def map_func_replace_wrong_punctuation(example):
    texts = example["texts"]
    for idx in range(len(texts)):
        texts[idx]["user"] = texts[idx]["user"].replace("?.", "?")
        texts[idx]["user"] = texts[idx]["user"].replace("!.", "?")
        texts[idx]["assistant"] = texts[idx]["assistant"].replace("?.", "?")
        texts[idx]["assistant"] = texts[idx]["assistant"].replace("!.", "?")
    example["texts"] = texts
    return example


for ds_name in tqdm(all_datasets):
    if (ds_name != "rendered_text") and (ds_name != "large_docvqa"):
        all_datasets[ds_name] = all_datasets[ds_name].map(map_func_replace_wrong_punctuation, num_proc=NUM_PROC)
# all_datasets = {ds_name: ds.map(map_func_replace_wrong_punctuation, num_proc=NUM_PROC) for ds_name, ds in tqdm(all_datasets.items())}

for ds_name in tqdm(all_datasets):
    normalized_proportion_dataset = MAPPING_DATASET_TO_PROPORTION_NORMALIZED[ds_name]
    num_shards = int(1 / normalized_proportion_dataset)
    ds = all_datasets[ds_name]
    all_datasets[ds_name] = [ds.shard(num_shards=num_shards, index=idx_shard) for idx_shard in range(num_shards)]
    all_datasets[ds_name] = (all_datasets[ds_name] * (NUM_MAX_MINI_EPOCHS // len(all_datasets[ds_name]) + 1))[
        :NUM_MAX_MINI_EPOCHS
    ]

all_concat_datasets = []
for idx_mini_epoch in tqdm(range(NUM_MAX_MINI_EPOCHS)):
    all_concat_datasets_mini_epoch = [ds[idx_mini_epoch] for ds in list(all_datasets.values())]
    all_concat_datasets_mini_epoch = concatenate_datasets(all_concat_datasets_mini_epoch)
    all_concat_datasets_mini_epoch = all_concat_datasets_mini_epoch.shuffle(seed=idx_mini_epoch)
    all_concat_datasets.append(all_concat_datasets_mini_epoch)

# Save to disk and then load from disk because otherwise the last save_to_disk is too long
# Doing it with a single process is too long too

# for idx_mini_epoch, ds_mini_epoch in enumerate(tqdm(all_concat_datasets)):
#     ds_mini_epoch.save_to_disk(PATH_SAVE_TMP_DATASETS / str(idx_mini_epoch), num_proc=NUM_PROC)


def save_dataset(idx_mini_epoch):
    ds_mini_epoch = all_concat_datasets[idx_mini_epoch]
    ds_mini_epoch.save_to_disk(PATH_SAVE_TMP_DATASETS / str(idx_mini_epoch), num_proc=NUM_PROC // NUM_MAX_MINI_EPOCHS)


with Pool(processes=min(NUM_MAX_MINI_EPOCHS, NUM_PROC)) as pool:
    result = pool.map_async(save_dataset, range(NUM_MAX_MINI_EPOCHS))
    result.wait()

all_concat_datasets = [
    load_from_disk(PATH_SAVE_TMP_DATASETS / str(idx_mini_epoch))
    for idx_mini_epoch in tqdm(range(len(all_concat_datasets)))
]

all_concat_datasets = concatenate_datasets(all_concat_datasets)

all_concat_datasets.save_to_disk(PATH_SAVE_DATASET, num_proc=NUM_PROC)

# Check
# [all_concat_datasets[0][idx]["texts"][0]["source"] for idx in range(100)]
# [all_concat_datasets[1][idx]["texts"][0]["source"] for idx in range(100)]
