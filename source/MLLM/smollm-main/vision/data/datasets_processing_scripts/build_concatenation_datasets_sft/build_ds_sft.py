"""
srun --pty --cpus-per-task=96 --mem-per-cpu=20G --gpus=1 --partition=hopper-prod --qos high bash -i
srun --pty --cpus-per-task=96 --partition=hopper-cpu --qos high bash -i
srun --pty --cpus-per-task=32 --partition=hopper-cpu --qos high bash -i
conda activate shared-m4
"""

import glob
import json
import math
import os
import pickle
import random
import re
from collections import Counter
from copy import deepcopy
from functools import partial
from io import BytesIO

import datasets
import requests
from datasets import concatenate_datasets, load_dataset, load_from_disk
from PIL import Image, ImageDraw, ImageFile, ImageFont
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


NUM_PROC = 48

FEATURES = datasets.Features(
    {
        "images": datasets.Sequence(datasets.Image(decode=True)),
        "texts": [
            {
                "user": datasets.Value("string"),
                "assistant": datasets.Value("string"),
                "source": datasets.Value("string"),
            }
        ],
    }
)

PROMPTS_ANSWER_SHORTLY = [
    "\nGive a very brief answer.",
    "\nAnswer briefly.",
    "\nWrite a very short answer.",
    "\nQuick response, please.",
    "\nShort answer required.",
    "\nKeep it brief.",
    "\nConcise answer only.",
    "\nBe succinct.",
]

QUESTION_BRIEVETY_HINT = [
    "\nMake the answer very short.",
    "\nGive a very brief answer.",
    "\nYour answer should be very brief.",
    "\nOffer a terse response.",
    "\nProvide a succinct answer.",
    "\nKeep it short and to the point.",
    "\nEnsure brevity in your answer. ",
    "\nOffer a very short reply.",
    "\nYour response must be concise.",
    "\nYour answer should be compact.",
    "\nProvide a short and direct response.",
]

QUESTION_BRIEVETY_HINT_MULTI_TURN = [
    "For each of the following questions, make the answer very short.\n",
    "Provide succinct responses to the questions below.\n",
    "Answer each question with brevity.\n",
    "Keep your answers concise for the following inquiries.\n",
    "Keep responses brief for each question.\n",
    "Offer short answers to the listed questions.\n",
    "Respond to each question with a brief answer.\n",
    "Keep your answers to the questions concise.\n",
    "Provide succinct answers for each question.\n",
    "Offer short and to-the-point responses to the questions provided.\n",
    "Keep your replies short for the following questions.\n",
    "Respond concisely to the questions presented.\n",
    "Provide terse responses to the following questions.\n",
    "Keep it snappy and straightforward when answering each question.\n",
    "Make your answers short.\n",
    "Keep your answers brief.\n",
    "Answer each question with brevity.\n",
]


def convert_img_to_bytes(img_path, format):
    img = Image.open(img_path)
    buffer = BytesIO()
    img.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    img.close()
    return img_bytes


def _convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates
    # a wrong background for transparent images. The call to `alpha_composite`
    # handles this case
    if image.mode == "RGB":
        return image
    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


def correct_casing(text, is_question=False):
    if text and text[0].islower():
        text = text.capitalize()
    if not text.endswith(".") and not is_question:
        text += "."
    return text


def lowercase_first_letter(text):
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    return text


def format_numbers_in_string(input_str):
    def format_number(match):
        number = float(match.group(0))
        # Check if the number is greater than 1000
        if abs(number) > 10000:
            # Format number to scientific notation with maximum 2 decimals if necessary
            formatted_number = "{:.2e}".format(number)
            # Remove trailing zeros after the decimal point
            formatted_number = (
                formatted_number.rstrip("0").rstrip(".") if "." in formatted_number else formatted_number
            )
            return formatted_number
        else:
            # Format number with a maximum of 2 decimals if necessary
            return "{:.2f}".format(number).rstrip("0").rstrip(".") if "." in input_str else "{:.0f}".format(number)

    return re.sub(r"\b\d+(\.\d+)?\b", format_number, input_str)


# --------------------------- Hugo --------------------------------------

# -------------------------------------------------------------------------------
# --------------------------- Screen2Words --------------------------------------
# -------------------------------------------------------------------------------

ds_screen2words = load_dataset("pinkmooncake/rico-screen2words", split="train")

prompts_screen2words = [
    "Provide a description of this screenshot.",
    "Describe the content in this image.",
    "Tell me what you see in this picture.",
    "Give me a summary of this screen capture.",
    "Provide a textual representation of this image.",
    "Describe the visual elements of this screenshot.",
    "What can you discern from this picture?",
    "Summarize the information in this screenshot.",
    "Explain what's happening in this screen capture.",
    "Give me a narrative description of this picture.",
    "Provide a detailed account of this screenshot.",
    "What details can you identify in this image?",
    "Describe the key features of this screenshot.",
    "Summarize the main components in this picture.",
    "Please provide a description for this image.",
    "Tell me about the visual elements in this screen capture.",
    "What is the overall content of this screenshot?",
    "Describe this image in words.",
    "Explain the elements present in this screenshot.",
]


def map_transform_screen2words(example):
    example["images"] = [example["image"]]
    example["texts"] = [
        {
            "user": random.choice(prompts_screen2words),
            "assistant": correct_casing(example["text"]),
            "source": "Screen2Words",
        }
    ]
    return example


ds_screen2words = ds_screen2words.map(
    map_transform_screen2words, remove_columns=ds_screen2words.column_names, features=FEATURES, num_proc=NUM_PROC
)

ds_screen2words.save_to_disk("/fsx/hugo/fine_tuning_datasets/screen2words", num_proc=NUM_PROC)  # 15_743 examples


# -------------------------------------------------------------------------------
# --------------------------- TextCaps --------------------------------------
# -------------------------------------------------------------------------------

prompts_textcaps = [
    "Summarize this image.",
    "Provide a caption for this picture.",
    "Give a brief description of this image.",
    "Title this photo.",
    "Interpret this scene.",
    "What does this picture show?",
    "Detail this image in one sentence.",
    "Outline the contents of this picture.",
    "Decode this image.",
    "Illustrate what's depicted here.",
    "Frame this scene in words.",
    "Translate this image to text.",
    "Caption this image.",
]

ds_textcaps = load_dataset("HuggingFaceM4/TextCaps", split="train")


def map_transform_textcaps(example):
    example["images"] = [example["image"]]
    example["texts"] = [
        {
            "user": random.choice(prompts_textcaps),
            "assistant": correct_casing(random.choice(example["reference_strs"])),
            "source": "TextCaps",
        }
    ]
    return example


ds_textcaps = ds_textcaps.map(
    map_transform_textcaps, remove_columns=ds_textcaps.column_names, features=FEATURES, num_proc=NUM_PROC
)

ds_textcaps.save_to_disk("/fsx/hugo/fine_tuning_datasets/textcaps", num_proc=NUM_PROC)  # 21_953 examples


# -------------------------------------------------------------------------------
# --------------------------- VisText ---------------------------------------
# -------------------------------------------------------------------------------

"""
Download:
https://vis.csail.mit.edu/vistext/tabular.zip -> /fsx/hugo/vistext/tabular
https://vis.csail.mit.edu/vistext/images.zip -> /fsx/hugo/vistext/images
"""

with open("/fsx/hugo/vistext/tabular/data_train.json", "r") as f:
    data_vistext = json.load(f)

prompts_vistext = [
    "Summarize the key information in this chart.",
    "Explain the trends shown in this chart.",
    "Identify the main components of this chart.",
    "What does this chart reveal about the data?",
    "Describe the relationship between variables in this chart.",
    "Highlight the significant data points in this chart.",
    "What insights can be drawn from this chart?",
    "Analyze the distribution shown in this chart.",
    "Estimate the changes over time shown in this chart.",
    "Explain the correlation depicted in this chart.",
    "Describe the pattern or trend evident in this chart.",
    "What is the chart's main message or takeaway?",
    "Describe this chart.",
]


dict_vistext = {
    "images": [
        [
            {
                "bytes": convert_img_to_bytes(
                    img_path="/fsx/hugo/vistext/images/" + el["img_id"] + ".png", format="png"
                ),
                "path": None,
            }
        ]
        for el in tqdm(data_vistext)
    ],
    "texts": [
        [
            {
                "user": random.choice(prompts_vistext),
                "assistant": el["caption_L1"] + " " + el["caption_L2L3"],
                "source": "VisText",
            }
        ]
        for el in data_vistext
    ],
}

ds_vistext = datasets.Dataset.from_dict(dict_vistext, features=FEATURES)

ds_vistext.save_to_disk("/fsx/hugo/fine_tuning_datasets/vistext", num_proc=NUM_PROC)  # 9_969 examples


# -------------------------------------------------------------------------------
# --------------------------- VQAv2 ---------------------------------------
# -------------------------------------------------------------------------------

ds_vqav2 = load_dataset("HuggingFaceM4/VQAv2_modif", "train", split="train")


def map_transform_vqav2(example):
    example["images"] = [example["image"]]
    question = example["question"] + random.choice(PROMPTS_ANSWER_SHORTLY)
    answers = Counter(example["answers"])
    most_common_answer = max(answers, key=answers.get)
    example["texts"] = [{"user": question, "assistant": correct_casing(most_common_answer), "source": "VQAv2"}]
    return example


ds_vqav2 = ds_vqav2.map(
    map_transform_vqav2, remove_columns=ds_vqav2.column_names, features=FEATURES, num_proc=NUM_PROC
)

ds_vqav2.save_to_disk("/fsx/hugo/fine_tuning_datasets/vqav2", num_proc=NUM_PROC)  # 443_757 examples


# -------------------------------------------------------------------------------
# --------------------------- OK-VQA ---------------------------------------
# -------------------------------------------------------------------------------

ds_okvqa = load_dataset("HuggingFaceM4/OK-VQA_modif", split="train")


def map_transform_okvqa(example):
    example["images"] = [example["image"]]
    question = example["question"] + random.choice(PROMPTS_ANSWER_SHORTLY)
    answers = Counter(example["answers"])
    most_common_answer = max(answers, key=answers.get)
    example["texts"] = [{"user": question, "assistant": correct_casing(most_common_answer), "source": "OK-VQA"}]
    return example


ds_okvqa = ds_okvqa.map(
    map_transform_okvqa, remove_columns=ds_okvqa.column_names, features=FEATURES, num_proc=NUM_PROC
)

ds_okvqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/okvqa", num_proc=NUM_PROC)  # 9_009 examples


# -------------------------------------------------------------------------------
# --------------------------- GQA ---------------------------------------
# -------------------------------------------------------------------------------

ds_gqa = load_dataset("Graphcore/gqa", split="train")


def map_transform_gqa(example):
    path_image = os.path.join("/fsx/hugo/gqa/images", os.path.basename(example["image_id"]))
    image_bytes = convert_img_to_bytes(img_path=path_image, format="JPEG")
    question = example["question"] + random.choice(PROMPTS_ANSWER_SHORTLY)
    example["images"] = [{"path": None, "bytes": image_bytes}]
    example["texts"] = [{"user": question, "assistant": correct_casing(example["label"]), "source": "GQA"}]
    return example


ds_gqa = ds_gqa.map(map_transform_gqa, remove_columns=ds_gqa.column_names, features=FEATURES, num_proc=NUM_PROC)

ds_gqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/gqa", num_proc=NUM_PROC)  # 943_000 examples


# -------------------------------------------------------------------------------
# --------------------------- COCO-QA ---------------------------------------
# -------------------------------------------------------------------------------

"""
Download:
http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/cocoqa-2015-05-17.zip -> /fsx/hugo/cocoqa
"""

with open("/fsx/hugo/cocoqa/train/answers.txt", "r") as f:
    data_cocoqa_answers = f.read().strip().split("\n")

with open("/fsx/hugo/cocoqa/train/questions.txt", "r") as f:
    data_cocoqa_questions = f.read().strip().split("\n")

with open("/fsx/hugo/cocoqa/train/img_ids.txt", "r") as f:
    data_cocoqa_img_ids = f.read().strip().split("\n")

assert len(data_cocoqa_answers) == len(data_cocoqa_questions) == len(data_cocoqa_img_ids)

dict_cocoqa = {
    "images": [
        [
            {
                "bytes": convert_img_to_bytes(
                    img_path="/fsx/hugo/coco/train2017/" + img_ids.zfill(12) + ".jpg", format="JPEG"
                ),
                "path": None,
            }
        ]
        for img_ids in tqdm(data_cocoqa_img_ids)
    ],
    "texts": [
        [
            {
                "user": question.capitalize() + random.choice(PROMPTS_ANSWER_SHORTLY),
                "assistant": correct_casing(answer),
                "source": "COCO-QA",
            }
        ]
        for question, answer in zip(data_cocoqa_questions, data_cocoqa_answers)
    ],
}

ds_cocoqa = datasets.Dataset.from_dict(dict_cocoqa, features=FEATURES)

ds_cocoqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/cocoqa", num_proc=NUM_PROC)  # 78_736 examples


# -------------------------------------------------------------------------------
# --------------------------- VSR ---------------------------------------
# -------------------------------------------------------------------------------

ds_vsr = load_dataset("cambridgeltl/vsr_zeroshot", split="train")

prompts_vsr = [
    'Is this affirmation: "{caption}" correct?',
    'Does the description: "{caption}" accurately reflect the image?',
    'Is the caption "{caption}" a true representation of the image?',
    'Verify the accuracy of this image caption: "{caption}".',
    'Is "{caption}" an appropriate description for the image?',
    'Evaluate: Does the caption "{caption}" match the image?',
    'Is the given caption "{caption}" fitting for the image?',
    'Does the caption "{caption}" correctly depict the image?',
    'Is the statement "{caption}" accurate regarding the image?',
    'Does the image validate the caption "{caption}"?',
]


def map_transform_vsr(example):
    try:
        path_image = os.path.join("/fsx/hugo/coco/train2017/", os.path.basename(example["image"]))
        image_bytes = convert_img_to_bytes(img_path=path_image, format="JPEG")
        example["images"] = [{"path": None, "bytes": image_bytes}]
        question = random.choice(prompts_vsr).format(caption=example["caption"])
        question += "\nAnswer yes or no."
        answer = "Yes." if (example["label"] == 1) else "No."
        example["texts"] = [{"user": question, "assistant": answer, "source": "VSR"}]
    except Exception:
        example["images"] = None
        example["texts"] = None
    return example


ds_vsr = ds_vsr.map(map_transform_vsr, remove_columns=ds_vsr.column_names, features=FEATURES, num_proc=NUM_PROC)
ds_vsr = ds_vsr.filter(lambda example: example["images"] is not None, num_proc=NUM_PROC)

ds_vsr.save_to_disk("/fsx/hugo/fine_tuning_datasets/vsr", num_proc=NUM_PROC)  # 3_354 examples


# -------------------------------------------------------------------------------
# --------------------------- Visual7W ---------------------------------------
# -------------------------------------------------------------------------------

"""
Download:
http://vision.stanford.edu/yukezhu/visual7w_images.zip -> /fsx/hugo/visual7w/images
https://ai.stanford.edu/~yukez/papers/resources/dataset_v7w_telling.zip -> /fsx/hugo/visual7w/dataset_v7w_telling.json
"""

with open("/fsx/hugo/visual7w/dataset_v7w_telling.json", "r") as f:
    data_visual7w = json.load(f)["images"]


def make_question_answer_visual7w(question, multiple_choices, answer):
    letters_cap = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    prompt = f"Question: {question}\nChoices:\n"
    all_choices = deepcopy(multiple_choices)
    index_answer = random.randint(0, len(all_choices))
    all_choices.insert(index_answer, answer)
    for idx, choice in enumerate(all_choices):
        letter = letters_cap[idx]
        prompt += f"{letter}. {choice}\n"
    prompt += "Answer with the letter."
    letter_answer = letters_cap[index_answer]
    output = f"Answer: {letter_answer}"
    return prompt, output


dict_visual7w = {"images": [], "texts": []}
for data_point_visual7w in tqdm(data_visual7w):
    if data_point_visual7w["split"] == "train":
        img_path = os.path.join("/fsx/hugo/visual7w/images", data_point_visual7w["filename"])
        images = [
            {
                "bytes": convert_img_to_bytes(img_path=img_path, format="png"),
                "path": None,
            }
        ]
        qa_pairs = data_point_visual7w["qa_pairs"]
        texts = []
        for qa_pair in qa_pairs:
            question, answer = make_question_answer_visual7w(
                question=qa_pair["question"], multiple_choices=qa_pair["multiple_choices"], answer=qa_pair["answer"]
            )
            text = {
                "user": question,
                "assistant": answer,
                "source": "Visual7W",
            }
            texts.append(text)
        dict_visual7w["images"].append(images)
        dict_visual7w["texts"].append(texts)


ds_visual7w = datasets.Dataset.from_dict(dict_visual7w, features=FEATURES)

ds_visual7w.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets/visual7w", num_proc=NUM_PROC
)  # 14_366 examples (multiple questions per example)


# -------------------------------------------------------------------------------
# --------------------------- IAM ---------------------------------------
# -------------------------------------------------------------------------------


ds_iam = load_dataset("alpayariyak/IAM_Sentences", split="train")

prompts_iam = [
    "Identify the text in this image.",
    "Decode the message shown.",
    "Extract text from the given image.",
    "What does the handwriting in this picture say?",
    "Reveal the contents of this note.",
    "Translate this image's handwriting into text.",
    "What message is written in the photograph?",
    "Output the text in this image.",
    "Convert the handwriting in this image to text.",
    "What words are inscribed in this image?",
    "Read the script in this image.",
    "What is the handwriting in this image about?",
    "Uncover the written words in this picture.",
    "Transcribe the handwriting seen in this image.",
    "Describe the text written in this photo.",
    "Detail the handwritten content in this image.",
    "What text does this image contain?",
    "Elucidate the handwriting in this image.",
    "What is scribbled in this image?",
    "What's written in this image?",
]


def map_transform_iam(example):
    example["images"] = [example["image"]]
    example["texts"] = [{"user": random.choice(prompts_iam), "assistant": example["text"], "source": "IAM"}]
    return example


ds_iam = ds_iam.map(map_transform_iam, remove_columns=ds_iam.column_names, features=FEATURES, num_proc=NUM_PROC)

ds_iam.save_to_disk("/fsx/hugo/fine_tuning_datasets/iam", num_proc=NUM_PROC)  # 5_663 examples


# -------------------------------------------------------------------------------
# --------------------------- Diagram-Image-to-Text ---------------------------------------
# -------------------------------------------------------------------------------

ds_diagram_image_to_text = load_dataset("Kamizuru00/diagram_image_to_text", split="train")

prompts_diagram_image_to_text = [
    "Identify and explain the connections between elements in this diagram.",
    "Summarize the interactions among the components shown in the diagram.",
    "Describe the flow of information or energy in this diagram.",
    "Explain how the parts of this diagram work together to achieve a purpose.",
    "Detail the hierarchical structure of the components in this diagram.",
    "Map out and interpret the links among diagram components.",
    "Analyze the diagram and describe the dependency between its elements.",
    "Illustrate the network of connections presented in this diagram.",
    "Break down the diagram into its components and explain their interrelations.",
    "Elucidate the sequence of operations depicted in the diagram.",
    "Delineate the roles of the components within this diagram.",
    "Examine the diagram and outline how each part contributes to the whole.",
    "Clarify the mechanism of action represented by the diagram.",
    "Dissect the diagram, highlighting the interaction between elements.",
    "Narrate the process illustrated by the diagram, focusing on component links.",
    "Decode the diagram's representation of relationships between its parts.",
    "Chart the connections and roles of the diagram’s components.",
    "Interpret the system depicted in the diagram, detailing component functions.",
    "Detail the cause-and-effect relationships within this diagram.",
    "Review the diagram and comment on the linkage and flow among entities.",
]


def map_transform_diagram_image_to_text(example):
    example["images"] = [example["image"]]
    example["texts"] = [
        {
            "user": random.choice(prompts_diagram_image_to_text),
            "assistant": example["text"],
            "source": "Diagram-Image-to-Text",
        }
    ]
    return example


ds_diagram_image_to_text = ds_diagram_image_to_text.map(
    map_transform_diagram_image_to_text,
    remove_columns=ds_diagram_image_to_text.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_diagram_image_to_text.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets/diagram_image_to_text", num_proc=NUM_PROC
)  # 300 examples


# -------------------------------------------------------------------------------
# --------------------------- ST-VQA ---------------------------------------
# -------------------------------------------------------------------------------

"""
Download:
https://rrc.cvc.uab.es/?ch=11&com=downloads
Weakly Contextualised Task-2a and Training images -> /fsx/hugo/st_vqa
"""

with open("/fsx/hugo/st_vqa/train_task_2.json", "r") as f:
    data_st_vqa = json.load(f)["data"]


dict_st_vqa = {"images": [], "texts": []}
for example in tqdm(data_st_vqa):
    if not (example["set_name"] == "train"):
        continue
    question = example["question"] + random.choice(PROMPTS_ANSWER_SHORTLY)
    answers = example["answers"]
    if not (len(answers) == 1):
        continue
    answer = correct_casing(answers[0])

    text = [{"user": question, "assistant": answer, "source": "ST-VQA"}]
    dict_st_vqa["texts"].append(text)

    image = [
        {
            "bytes": convert_img_to_bytes(
                img_path=os.path.join("/fsx/hugo/st_vqa", example["file_path"]), format="JPEG"
            ),
            "path": None,
        }
    ]
    dict_st_vqa["images"].append(image)


ds_st_vqa = datasets.Dataset.from_dict(dict_st_vqa, features=FEATURES)

ds_st_vqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/st_vqa", num_proc=NUM_PROC)  # 23_121 examples


# -------------------------------------------------------------------------------
# --------------------------- Infographic-VQA ---------------------------------------
# -------------------------------------------------------------------------------

"""
Download:
https://rrc.cvc.uab.es/?ch=17&com=downloads -> /fsx/hugo/infographic_vqa
"""

with open("/fsx/hugo/infographic_vqa/infographicsVQA_train_v1.0.json", "r") as f:
    data_infographic_vqa = json.load(f)["data"]


dict_infographic_vqa = {"images": [], "texts": []}
for example in tqdm(data_infographic_vqa):
    question = example["question"] + random.choice(PROMPTS_ANSWER_SHORTLY)
    answers = example["answers"]
    if not (len(answers) == 1):
        continue
    answer = correct_casing(answers[0])

    text = [{"user": question, "assistant": answer, "source": "Infographic-VQA"}]
    dict_infographic_vqa["texts"].append(text)

    image = [
        {
            "bytes": convert_img_to_bytes(
                img_path=os.path.join("/fsx/hugo/infographic_vqa/images", example["image_local_name"]), format="JPEG"
            ),
            "path": None,
        }
    ]
    dict_infographic_vqa["images"].append(image)


ds_infographic_vqa = datasets.Dataset.from_dict(dict_infographic_vqa, features=FEATURES)

ds_infographic_vqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/infographic_vqa", num_proc=NUM_PROC)  # 22_498 examples

ds_infographic_vqa = load_from_disk("/fsx/hugo/fine_tuning_datasets/infographic_vqa")


def filter_big_images_infovqa(example):
    images = example["images"]
    for image in images:
        width, height = image.size
        if width / height > 4 or height / width > 4:
            return False
        if width > 2 * 980 or height > 2 * 980:
            return False
    return True


ds_infographic_vqa = ds_infographic_vqa.filter(filter_big_images_infovqa, num_proc=NUM_PROC)

ds_infographic_vqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/infographic_vqa", num_proc=NUM_PROC)  # 18_052 examples


# -------------------------------------------------------------------------------
# --------------------------- ChartQA ---------------------------------------
# -------------------------------------------------------------------------------

"""
Download:
git clone https://github.com/vis-nlp/ChartQA.git -> /fsx/hugo/ChartQA/
"""

with open("/fsx/hugo/ChartQA/ChartQA_Dataset/train/train_human.json", "r") as f:
    data_chartqa = json.load(f)

with open("/fsx/hugo/ChartQA/ChartQA_Dataset/train/train_augmented.json", "r") as f:
    data_chartqa.extend(json.load(f))


dict_chartqa = {"images": [], "texts": []}
for example in tqdm(data_chartqa):
    question = example["query"] + random.choice(PROMPTS_ANSWER_SHORTLY)
    answer = correct_casing(example["label"])
    text = [{"user": question, "assistant": answer, "source": "ChartQA"}]
    dict_chartqa["texts"].append(text)
    image = [
        {
            "bytes": convert_img_to_bytes(
                img_path=os.path.join("/fsx/hugo/ChartQA/ChartQA_Dataset/train/png", example["imgname"]), format="png"
            ),
            "path": None,
        }
    ]
    dict_chartqa["images"].append(image)


ds_chartqa = datasets.Dataset.from_dict(dict_chartqa, features=FEATURES)

ds_chartqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/chartqa", num_proc=NUM_PROC)  # 28_299 examples


# -------------------------------------------------------------------------------
# --------------------------- ChartQA new prompt ---------------------------------------
# -------------------------------------------------------------------------------

"""
Download:
git clone https://github.com/vis-nlp/ChartQA.git -> /fsx/hugo/ChartQA/
"""

with open("/fsx/hugo/ChartQA/ChartQA_Dataset/train/train_human.json", "r") as f:
    data_chartqa_new_prompt = json.load(f)

with open("/fsx/hugo/ChartQA/ChartQA_Dataset/train/train_augmented.json", "r") as f:
    data_chartqa_new_prompt.extend(json.load(f))


prompt_template_chartqa_new_prompt = """For the question below, follow the following instructions:
-The answer should contain as few words as possible.
-Don’t paraphrase or reformat the text you see in the image.
-Answer a binary question with Yes or No.
-When asked to give a numerical value, provide a number like 2 instead of Two.
-If the final answer has two or more items, provide it in the list format like [1, 2].
-When asked to give a ratio, give out the decimal value like 0.25 instead of 1:4.
-When asked to give a percentage, give out the whole value like 17 instead of decimal like 0.17%.
-Don’t include any units in the answer.
-Do not include any full stops at the end of the answer.
-Try to include the full label from the graph when asked about an entity.
Question: {question}"""

dict_chartqa_new_prompt = {"images": [], "texts": []}
for example in tqdm(data_chartqa_new_prompt):
    question = prompt_template_chartqa_new_prompt.format(question=example["query"])
    answer = example["label"]
    text = [{"user": question, "assistant": answer, "source": "ChartQA"}]
    dict_chartqa_new_prompt["texts"].append(text)
    image = [
        {
            "bytes": convert_img_to_bytes(
                img_path=os.path.join("/fsx/hugo/ChartQA/ChartQA_Dataset/train/png", example["imgname"]), format="png"
            ),
            "path": None,
        }
    ]
    dict_chartqa_new_prompt["images"].append(image)


ds_chartqa_new_prompt = datasets.Dataset.from_dict(dict_chartqa_new_prompt, features=FEATURES)

ds_chartqa_new_prompt.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets/chartqa_new_prompt", num_proc=NUM_PROC
)  # 28_299 examples


# -------------------------------------------------------------------------------
# --------------------------- AI2D ---------------------------------------
# -------------------------------------------------------------------------------

"""
Download:
https://ai2-public-datasets.s3.amazonaws.com/diagrams/ai2d-all.zip -> /fsx/hugo/AI2D/ai2d/
https://s3-us-east-2.amazonaws.com/prior-datasets/ai2d_test_ids.csv -> /fsx/hugo/AI2D/
"""

dict_ai2d = {"images": [], "texts": []}

paths_common_data_ai2d = glob.glob("/fsx/hugo/AI2D/ai2d/questions/*.json")
paths_test_ids_ai2d = "/fsx/hugo/AI2D/ai2d_test_ids.csv"

with open(paths_test_ids_ai2d) as f:
    test_ids_ai2d = set(f.read().split("\n"))

for path_data_ai2d in tqdm(paths_common_data_ai2d):
    with open(path_data_ai2d, "r") as f:
        example = json.load(f)

    id_example = example["imageName"].replace(".png", "")
    if id_example in test_ids_ai2d:
        continue

    texts = []
    for question, dict_question in example["questions"].items():
        if dict_question["abcLabel"]:
            continue

        letters_cap = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
        prompt = f"Question: {question}\nChoices:\n"
        all_choices = dict_question["answerTexts"]
        index_answer = dict_question["correctAnswer"]
        for idx, choice in enumerate(all_choices):
            letter = letters_cap[idx]
            prompt += f"{letter}. {choice}\n"
        prompt += "Answer with the letter."

        letter_answer = letters_cap[index_answer]
        output = f"Answer: {letter_answer}"

        text = {"user": prompt, "assistant": output, "source": "AI2D"}
        texts.append(text)

    if not texts:
        continue
    dict_ai2d["texts"].append(texts)

    image = [
        {
            "bytes": convert_img_to_bytes(
                img_path=os.path.join("/fsx/hugo/AI2D/ai2d/images/", example["imageName"]), format="png"
            ),
            "path": None,
        }
    ]
    dict_ai2d["images"].append(image)


ds_ai2d = datasets.Dataset.from_dict(dict_ai2d, features=FEATURES)

ds_ai2d.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets/ai2d", num_proc=NUM_PROC
)  # 2_482 examples (multiple questions per example)


# -------------------------------------------------------------------------------
# --------------------------- ScienceQA ---------------------------------------
# -------------------------------------------------------------------------------

ds_scienceqa = load_dataset("derek-thomas/ScienceQA", split="train")

ds_scienceqa = ds_scienceqa.filter(lambda example: example["image"] is not None, num_proc=NUM_PROC)


def map_transform_scienceqa(example):
    question = example["question"]

    all_choices = example["choices"]
    index_answer = example["answer"]

    lecture = example["lecture"]
    hint = example["hint"]

    prompt = ""
    if lecture != "":
        prompt += f"Lecture: {lecture}\n"
    prompt += f"Question: {question}\n"
    if hint != "":
        prompt += f"Hint: {hint}\n"
    prompt += "Choices:\n"

    letters_cap = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    for idx, choice in enumerate(all_choices):
        letter = letters_cap[idx]
        prompt += f"{letter}. {choice}\n"
    prompt += "Answer with the letter."

    letter_answer = letters_cap[index_answer]
    output = f"Answer: {letter_answer}"

    example["texts"] = [
        {
            "user": prompt,
            "assistant": output,
            "source": "ScienceQA",
        }
    ]
    example["images"] = [example["image"]]
    return example


ds_scienceqa = ds_scienceqa.map(
    map_transform_scienceqa, remove_columns=ds_scienceqa.column_names, features=FEATURES, num_proc=NUM_PROC
)

ds_scienceqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/scienceqa", num_proc=NUM_PROC)  # 6_218 examples


# -------------------------------------------------------------------------------
# --------------------------- InterGPS ---------------------------------------
# -------------------------------------------------------------------------------

"""
Download:
https://github.com/lupantech/InterGPS.git -> /fsx/hugo/InterGPS/
"""

dict_intergps = {"images": [], "texts": []}

path_subfolders_intergps = glob.glob("/fsx/hugo/InterGPS/data/geometry3k/train/*")

for path_subfolder_intergps in tqdm(path_subfolders_intergps):
    with open(os.path.join(path_subfolder_intergps, "data.json"), "r") as f:
        data_intergps = json.load(f)

    question = data_intergps["problem_text"]
    all_choices = data_intergps["choices"]
    letter_answer = data_intergps["answer"]

    prompt = f"Question: {question}\nChoices:\n"

    letters_cap = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    for idx, choice in enumerate(all_choices):
        letter = letters_cap[idx]
        prompt += f"{letter}. {choice}\n"
    prompt += "Answer with the letter."

    output = f"Answer: {letter_answer}"

    dict_intergps["texts"].append(
        [
            {
                "user": prompt,
                "assistant": output,
                "source": "InterGPS",
            }
        ]
    )

    image = [
        {
            "bytes": convert_img_to_bytes(
                img_path=os.path.join(path_subfolder_intergps, "img_diagram.png"), format="png"
            ),
            "path": None,
        }
    ]
    dict_intergps["images"].append(image)

ds_intergps = datasets.Dataset.from_dict(dict_intergps, features=FEATURES)

ds_intergps.save_to_disk("/fsx/hugo/fine_tuning_datasets/intergps", num_proc=NUM_PROC)  # 2_101 examples


# -------------------------------------------------------------------------------
# --------------------------- NLVR2 ---------------------------------------
# -------------------------------------------------------------------------------

ds_nlvr2 = load_dataset("HuggingFaceM4/NLVR2", split="train")
int2str_nlvr2 = ds_nlvr2.features["label"].int2str

prompts_nlvr2 = [
    'Evaluate the accuracy of this statement regarding the images: "{caption}". Is it true?',
    'Given the left and right images, does the statement "{caption}" hold true?',
    'Assess this claim about the two images: "{caption}". Correct or not?',
    'For the images shown, is this caption "{caption}" true?',
    'Considering the images on both sides, is "{caption}" valid?',
    'Analyze the images presented: Is the assertion "{caption}" valid?',
    'For the images displayed, is the sentence "{caption}" factually correct?',
    'Examine the images to the left and right. Is the description "{caption}" accurate?',
]
prefix = "The first image is the image on the left, the second image is the image on the right."
suffix = "Answer yes or no."
prompts_nlvr2 = [f"{prefix} {prompt} {suffix}" for prompt in prompts_nlvr2]


def map_transform_nlvr2(example):
    example["images"] = [example["left_image"], example["right_image"]]
    answer = "Yes." if (int2str_nlvr2(values=example["label"]) == "True") else "No."
    example["texts"] = [
        {
            "user": random.choice(prompts_nlvr2).format(caption=example["sentence"]),
            "assistant": answer,
            "source": "NLVR2",
        }
    ]
    return example


ds_nlvr2 = ds_nlvr2.map(
    map_transform_nlvr2,
    remove_columns=ds_nlvr2.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_nlvr2.save_to_disk("/fsx/hugo/fine_tuning_datasets/nlvr2", num_proc=NUM_PROC)  # 86_373 examples


# -------------------------------------------------------------------------------
# --------------------------- GeomVerse ---------------------------------------
# -------------------------------------------------------------------------------

"""
Download:
https://storage.googleapis.com/gresearch/GeomVerseV0/GeomVerse.zip -> /fsx/hugo/GeomVerse/
"""

data_geomverse = []
with open("/fsx/hugo/GeomVerse/GeomVerse/TRAIN/TRAIN_MIX/data.jsonl", "r") as f:
    for line in f:
        data_geomverse.append(json.loads(line))

dict_geomverse = {"images": [], "texts": []}

for example in tqdm(data_geomverse):
    dict_geomverse["texts"].append(
        [
            {
                "user": example["question"],
                "assistant": example["cot"],
                "source": "GeomVerse",
            }
        ]
    )
    image = [
        {
            "bytes": convert_img_to_bytes(
                img_path=os.path.join("/fsx/hugo/GeomVerse/", example["image_path"]), format="JPEG"
            ),
            "path": None,
        }
    ]
    dict_geomverse["images"].append(image)

ds_geomverse = datasets.Dataset.from_dict(dict_geomverse, features=FEATURES)

ds_geomverse.save_to_disk("/fsx/hugo/fine_tuning_datasets/geomverse", num_proc=NUM_PROC)  # 9_339 examples


# -------------------------------------------------------------------------------
# --------------------------- HatefulMemes ---------------------------------------
# -------------------------------------------------------------------------------

ds_hm = load_dataset("HuggingFaceM4/Hateful-Memes", "seen", split="train")

prompts_hm = [
    "Is the sentiment of this meme offensive?",
    "Does this meme promote hate speech?",
    "Can this meme be considered disrespectful?",
    "Does this meme carry a negative message?",
    "Is this meme spreading toxicity?",
    "Can this meme be harmful to a community?",
    "Is the humor in this meme in bad taste?",
    "Is the message of this meme aggressive?",
    "Does this meme support discrimination?",
    "Is the language used in this meme hateful?",
    "Can this meme be interpreted as derogatory?",
]


def map_transform_hm(example):
    example["images"] = [example["image"]]
    question = random.choice(prompts_hm)
    question += "\nAnswer yes or no."
    answer = "Yes." if (example["label"] == 1) else "No."
    example["texts"] = [
        {
            "user": question,
            "assistant": answer,
            "source": "Hateful Memes",
        }
    ]
    return example


ds_hm = ds_hm.map(
    map_transform_hm,
    remove_columns=ds_hm.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_hm.save_to_disk("/fsx/hugo/fine_tuning_datasets/hateful_memes", num_proc=NUM_PROC)  # 8_500 examples


# -------------------------------------------------------------------------------
# --------------------------- TallyQA ---------------------------------------
# -------------------------------------------------------------------------------

"""
Download:
https://github.com/manoja328/TallyQA_dataset -> /fsx/hugo/tallyqa/
"""

with open("/fsx/hugo/tallyqa/train_tallyqa.json", "r") as f:
    data_tallyqa = json.load(f)

dict_tallyqa = {"images": [], "texts": []}

for example in tqdm(data_tallyqa):
    if "VG" in example["image"]:
        common_path = "/fsx/hugo/VisualGenome"
    elif "COCO_train2014" in example["image"]:
        common_path = "/fsx/hugo/coco"
    elif "val" in example["image"]:
        continue
    else:
        raise ValueError("No dataset matched")
    image = [
        {
            "bytes": convert_img_to_bytes(img_path=os.path.join(common_path, example["image"]), format="JPEG"),
            "path": None,
        }
    ]
    question = example["question"] + random.choice(PROMPTS_ANSWER_SHORTLY)
    dict_tallyqa["images"].append(image)
    dict_tallyqa["texts"].append(
        [
            {
                "user": question,
                "assistant": correct_casing(str(example["answer"])),
                "source": "TallyQA",
            }
        ]
    )

ds_tallyqa = datasets.Dataset.from_dict(dict_tallyqa, features=FEATURES)

ds_tallyqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/tallyqa", num_proc=NUM_PROC)  # 183_986 examples


# -------------------------------------------------------------------------------
# --------------------------- IconQA ---------------------------------------
# -------------------------------------------------------------------------------

"""
Download:
https://iconqa2021.s3.us-west-1.amazonaws.com/iconqa_data.zip -> /fsx/hugo/iconqa/
"""

dict_iconqa = {"images": [], "texts": []}

paths_data_iconqa_choose_txt = glob.glob("/fsx/hugo/iconqa/iconqa_data/iconqa/train/choose_txt/*")
for path_data_iconqa_choose_txt in tqdm(paths_data_iconqa_choose_txt):
    image = [
        {
            "bytes": convert_img_to_bytes(
                img_path=os.path.join(path_data_iconqa_choose_txt, "image.png"), format="png"
            ),
            "path": None,
        }
    ]
    dict_iconqa["images"].append(image)

    with open(os.path.join(path_data_iconqa_choose_txt, "data.json"), "r") as f:
        data_iconqa = json.load(f)

    question = data_iconqa["question"]
    all_choices = data_iconqa["choices"]
    index_answer = data_iconqa["answer"]

    prompt = f"Question: {question}\nChoices:\n"
    letters_cap = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    for idx, choice in enumerate(all_choices):
        letter = letters_cap[idx]
        prompt += f"{letter}. {choice}\n"
    prompt += "Answer with the letter."

    letter_answer = letters_cap[index_answer]
    output = f"Answer: {letter_answer}"

    dict_iconqa["texts"].append(
        [
            {
                "user": prompt,
                "assistant": output,
                "source": "IconQA",
            }
        ]
    )

paths_data_iconqa_fill_in_blank = glob.glob("/fsx/hugo/iconqa/iconqa_data/iconqa/train/fill_in_blank/*")
for path_data_iconqa_fill_in_blank in tqdm(paths_data_iconqa_fill_in_blank):
    image = [
        {
            "bytes": convert_img_to_bytes(
                img_path=os.path.join(path_data_iconqa_fill_in_blank, "image.png"), format="png"
            ),
            "path": None,
        }
    ]
    dict_iconqa["images"].append(image)

    with open(os.path.join(path_data_iconqa_fill_in_blank, "data.json"), "r") as f:
        data_iconqa = json.load(f)

    question = data_iconqa["question"]
    if "(_)" in question:
        prompt = "Fill in the blank. " + question
    else:
        prompt = question

    dict_iconqa["texts"].append(
        [
            {
                "user": prompt,
                "assistant": data_iconqa["answer"],
                "source": "IconQA",
            }
        ]
    )

ds_iconqa = datasets.Dataset.from_dict(dict_iconqa, features=FEATURES)

ds_iconqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/iconqa", num_proc=NUM_PROC)  # 29_859 examples


# -------------------------------------------------------------------------------
# --------------------------- DaTikz ---------------------------------------
# -------------------------------------------------------------------------------

ds_datikz = load_dataset("HuggingFaceM4/datikz", split="train")

prompts_datikz = [
    "Generate TikZ code for this figure.",
    "Convert this image into TikZ code.",
    "Produce TikZ code that replicates this diagram.",
    "Transform this figure into its TikZ equivalent.",
    "Create TikZ code to match this image.",
    "Develop TikZ code that mirrors this figure.",
    "Translate this image into TikZ code.",
    "Formulate TikZ code to reconstruct this figure.",
    "Craft TikZ code that reflects this figure.",
    "Construct TikZ code for the given image.",
    "Form TikZ code corresponding to this image.",
    "Map this image into TikZ code.",
    "Recreate this figure using TikZ code.",
    "Encode this image into TikZ format.",
    "Synthesize TikZ code for this figure.",
    "Replicate this image with TikZ code.",
]


def map_transform_datikz(example):
    example["images"] = [example["image"]]
    example["texts"] = [
        {
            "user": random.choice(prompts_datikz),
            "assistant": example["code"],
            "source": "DaTikz",
        }
    ]
    return example


ds_datikz = ds_datikz.map(
    map_transform_datikz,
    remove_columns=ds_datikz.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_datikz.save_to_disk("/fsx/hugo/fine_tuning_datasets/datikz", num_proc=NUM_PROC)  # 48_296 examples


# -------------------------------------------------------------------------------
# --------------------------- WebSight ---------------------------------------
# -------------------------------------------------------------------------------

ds_websight = load_from_disk("/fsx/hugo/websight_v02")
ds_websight = ds_websight.select(range(500_000))

prompts_websight = [
    "Generate the HTML code corresponding to this website screenshot.",
    "Translate this website image into its HTML code.",
    "Produce the HTML markup to recreate the visual appearance of this website.",
    "Convert this screenshot into its equivalent HTML structure.",
    "Write the HTML that mirrors this website's layout.",
    "Craft the HTML code that would generate this website's look.",
    "Reconstruct the HTML code from this website image.",
    "Formulate the HTML to replicate this web page's design.",
    "Compose the HTML code to achieve the same design as this screenshot.",
    "Transform this website screenshot into HTML code.",
    "Develop the HTML structure to match this website's aesthetics.",
    "Derive the HTML code to reflect this website's interface.",
    "Encode this website's visual representation into HTML.",
    "Assemble the HTML code to mimic this webpage's style.",
    "Outline the HTML required to reproduce this website's appearance.",
    "Render the HTML code that corresponds to this web design.",
    "Illustrate the HTML coding for this website's visual format.",
    "Synthesize the HTML to emulate this website's layout.",
]


def map_transform_websight(example):
    example["images"] = [example["image"]]
    example["texts"] = [
        {
            "user": random.choice(prompts_websight),
            "assistant": example["text"],
            "source": "WebSight_v02",
        }
    ]
    return example


ds_websight = ds_websight.map(
    map_transform_websight,
    remove_columns=ds_websight.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_websight.save_to_disk("/fsx/hugo/fine_tuning_datasets/websight", num_proc=NUM_PROC)  # 500_000 examples


# -------------------------------------------------------------------------------
# --------------------------- FigureQA ---------------------------------------
# -------------------------------------------------------------------------------

"""
Download:
https://www.microsoft.com/en-hk/download/details.aspx?id=100635 -> /fsx/hugo/figureqa/
"""

with open("/fsx/hugo/figureqa/train1/qa_pairs.json", "r") as f:
    data_figureqa = json.load(f)["qa_pairs"]

dict_tmp_figureqa = {}
for example in tqdm(data_figureqa):
    question = example["question_string"]
    question += "\nAnswer yes or no."
    answer = "Yes." if (example["answer"] == 1) else "No."
    text = [
        {
            "user": question,
            "assistant": answer,
            "source": "FigureQA",
        }
    ]
    image_index = example["image_index"]
    dict_tmp_figureqa[image_index] = dict_tmp_figureqa.get(image_index, []) + text

dict_figureqa = {"images": [], "texts": []}
for image_index, texts in tqdm(dict_tmp_figureqa.items()):
    dict_figureqa["texts"].append(texts)
    image = [
        {
            "bytes": convert_img_to_bytes(
                img_path=os.path.join("/fsx/hugo/figureqa/train1/png", f"{image_index}.png"), format="png"
            ),
            "path": None,
        }
    ]
    dict_figureqa["images"].append(image)

ds_figureqa = datasets.Dataset.from_dict(dict_figureqa, features=FEATURES)

ds_figureqa.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets/figureqa", num_proc=NUM_PROC
)  # 100_000 examples (multiple questions per example)


# -------------------------------------------------------------------------------
# --------------------------- PathVQA ---------------------------------------
# -------------------------------------------------------------------------------

ds_pathvqa = load_dataset("flaviagiammarino/path-vqa", split="train")


def map_transform_pathvqa(example):
    example["images"] = [example["image"]]
    example["texts"] = [
        {
            "user": example["question"].capitalize(),
            "assistant": correct_casing(example["answer"]),
            "source": "PathVQA",
        }
    ]
    return example


ds_pathvqa = ds_pathvqa.map(
    map_transform_pathvqa,
    remove_columns=ds_pathvqa.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_pathvqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/pathvqa", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- VQA-RAD ---------------------------------------
# -------------------------------------------------------------------------------

ds_vqarad = load_dataset("flaviagiammarino/vqa-rad", split="train")


def map_transform_vqarad(example):
    example["images"] = [example["image"]]
    example["texts"] = [
        {
            "user": example["question"].capitalize() + random.choice(PROMPTS_ANSWER_SHORTLY),
            "assistant": correct_casing(example["answer"]),
            "source": "VQA-RAD",
        }
    ]
    return example


ds_vqarad = ds_vqarad.map(
    map_transform_vqarad,
    remove_columns=ds_vqarad.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_vqarad.save_to_disk("/fsx/hugo/fine_tuning_datasets/vqarad", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- Spot-the-diff ---------------------------------------
# -------------------------------------------------------------------------------

"""
Download:
https://github.com/harsh19/spot-the-diff/tree/master -> /fsx/hugo/spot_the_diff
"""

with open("/fsx/hugo/spot_the_diff/train_spot_difference.json", "r") as f:
    data_spot_the_diff = json.load(f)

prompts_spot_the_diff = [
    "Identify the discrepancies between these two pictures.",
    "Point out what differs between these two visuals.",
    "List the variances found in these pictures.",
    "Detect the changes between these images.",
    "Find the divergences between these two pictures.",
    "Describe the differences spotted in these photos.",
    "Reveal the deviations in these images.",
    "Enumerate the differences between these visuals.",
    "Discern the dissimilarities in these two pictures.",
    "Pinpoint the contrasts found in these images.",
    "Explain the variances between these photos.",
    "Outline the disparities in these two images.",
    "Identify the non-matching elements in these pictures.",
    "Locate the discrepancies between these visuals.",
    "Discover the changes evident in these two photos.",
    "Assess the differences in these images.",
]


dict_spot_the_diff = {"images": [], "texts": []}
for example in tqdm(data_spot_the_diff):
    question = random.choice(prompts_spot_the_diff)
    sentences = [correct_casing(sentence) for sentence in example["sentences"]]
    answer = " ".join(sentences)
    text = [{"user": question, "assistant": answer, "source": "Spot-the-diff"}]
    dict_spot_the_diff["texts"].append(text)

    img_id = example["img_id"]
    image = [
        {
            "bytes": convert_img_to_bytes(
                img_path=os.path.join("/fsx/hugo/spot_the_diff/resized_images", f"{img_id}.png"), format="png"
            ),
            "path": None,
        },
        {
            "bytes": convert_img_to_bytes(
                img_path=os.path.join("/fsx/hugo/spot_the_diff/resized_images", f"{img_id}_2.png"), format="png"
            ),
            "path": None,
        },
    ]
    dict_spot_the_diff["images"].append(image)


ds_spot_the_diff = datasets.Dataset.from_dict(dict_spot_the_diff, features=FEATURES)

ds_spot_the_diff.save_to_disk("/fsx/hugo/fine_tuning_datasets/spot_the_diff", num_proc=NUM_PROC)  # 9_524 examples


# -------------------------------------------------------------------------------
# --------------------------- Fix PlotQA ---------------------------------------
# -------------------------------------------------------------------------------


ds_plotqa = load_from_disk("/fsx/hugo/fine_tuning_datasets/plotqa")


def map_transform_plotqa(example):
    for idx in range(len(example["texts"])):
        example["texts"][idx]["user"] = example["texts"][idx]["user"].replace(" ?", "?")
    return example


ds_plotqa = ds_plotqa.map(
    map_transform_plotqa,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_plotqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/plotqa", num_proc=NUM_PROC)  # 157_070 examples


# -------------------------------------------------------------------------------
# --------------------------- Fix VisualMRC ---------------------------------------
# -------------------------------------------------------------------------------


ds_visualmrc = load_from_disk("/fsx/hugo/fine_tuning_datasets/visual_mrc")


def map_transform_visualmrc(example):
    texts = example["texts"]
    for idx in range(len(texts)):
        texts[idx]["user"] = texts[idx]["user"].replace(" ?", "?")
        texts[idx]["assistant"] = texts[idx]["assistant"].rstrip("?. ").strip() + "."
        texts[idx]["assistant"] = texts[idx]["assistant"].replace("?.", ".")
    example["texts"] = texts
    return example


ds_visualmrc = ds_visualmrc.map(
    map_transform_visualmrc,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_visualmrc.save_to_disk("/fsx/hugo/fine_tuning_datasets/new_visualmrc", num_proc=NUM_PROC)  # 3_996 examples


# -------------------------------------------------------------------------------
# --------------------------- Fix OpenHermes ---------------------------------------
# -------------------------------------------------------------------------------


ds_openhermes = load_from_disk("/fsx/hugo/fine_tuning_datasets/openhermes")


def map_remove_none_images(example):
    example["images"] = []
    return example


def filter_openhermes(example):
    texts = example["texts"]
    for text in texts:
        assistant_text = text["assistant"].lower()
        if "language model" in assistant_text:
            return False
        if "sorry" in assistant_text:
            return False
        if "openai" in assistant_text:
            return False
    return True


ds_openhermes = ds_openhermes.filter(filter_openhermes, num_proc=NUM_PROC)
ds_openhermes = ds_openhermes.map(map_remove_none_images, num_proc=NUM_PROC)

ds_openhermes.save_to_disk("/fsx/hugo/fine_tuning_datasets/new_openhermes", num_proc=NUM_PROC)  # 999_593 examples


def filter_special_tokens(example):
    special_tokens = ["<unk>", "<s>", "</s>", "<fake_token_around_image>", "<image>", "<end_of_utterance>"]
    texts = example["texts"]
    for text in texts:
        concat_text = text["user"] + "\n" + text["assistant"]
        if any([(special_token in concat_text) for special_token in special_tokens]):
            return False
    return True


ds_openhermes = load_from_disk("/fsx/hugo/fine_tuning_datasets_merge_image_individual/openhermes")
ds_openhermes = ds_openhermes.filter(filter_special_tokens, num_proc=NUM_PROC)
ds_openhermes.save_to_disk("/fsx/hugo/fine_tuning_datasets/openhermes", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- Fix LIMA ---------------------------------------
# -------------------------------------------------------------------------------


ds_lima = load_from_disk("/fsx/hugo/fine_tuning_datasets/lima")

ds_lima = ds_lima.filter(filter_openhermes, num_proc=NUM_PROC)
ds_lima = ds_lima.map(map_remove_none_images, num_proc=NUM_PROC)

ds_lima.save_to_disk("/fsx/hugo/fine_tuning_datasets/new_lima", num_proc=NUM_PROC)  # 1_002 examples

ds_lima = load_from_disk("/fsx/hugo/fine_tuning_datasets_merge_image_individual/lima")
ds_lima = ds_lima.filter(filter_special_tokens, num_proc=NUM_PROC)
ds_lima.save_to_disk("/fsx/hugo/fine_tuning_datasets/lima", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- OrcaMath ---------------------------------------
# -------------------------------------------------------------------------------


ds_orcamath = load_dataset("microsoft/orca-math-word-problems-200k", split="train")


def map_transform_orcamath(example):
    example["images"] = []
    example["texts"] = [
        {
            "user": example["question"],
            "assistant": example["answer"],
            "source": "OrcaMath",
        }
    ]
    return example


ds_orcamath = ds_orcamath.map(
    map_transform_orcamath,
    remove_columns=ds_orcamath.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_orcamath = ds_orcamath.filter(filter_openhermes, num_proc=NUM_PROC)

ds_orcamath.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets_merge_image_individual/orca_math", num_proc=NUM_PROC
)  # 200_031 examples

ds_orcamath = load_from_disk("/fsx/hugo/fine_tuning_datasets_merge_image_individual/orca_math")
ds_orcamath = ds_orcamath.filter(filter_special_tokens, num_proc=NUM_PROC)
ds_orcamath.save_to_disk("/fsx/hugo/fine_tuning_datasets/orcamath", num_proc=NUM_PROC)

# -------------------------------------------------------------------------------
# --------------------------- MetaMathQA ---------------------------------------
# -------------------------------------------------------------------------------


ds_metamathqa = load_dataset("meta-math/MetaMathQA", split="train")


def map_transform_metamathqa(example):
    example["images"] = []
    example["texts"] = [
        {
            "user": example["query"],
            "assistant": example["response"],
            "source": "MetaMathQA",
        }
    ]
    return example


ds_metamathqa = ds_metamathqa.map(
    map_transform_metamathqa,
    remove_columns=ds_metamathqa.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_metamathqa = ds_metamathqa.filter(filter_openhermes, num_proc=NUM_PROC)

ds_metamathqa.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets_merge_image_individual/metamathqa", num_proc=NUM_PROC
)  # 395_000 examples

ds_metamathqa = load_from_disk("/fsx/hugo/fine_tuning_datasets_merge_image_individual/metamathqa")
ds_metamathqa = ds_metamathqa.filter(filter_special_tokens, num_proc=NUM_PROC)
ds_metamathqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/metamathqa", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- MathInstruct ---------------------------------------
# -------------------------------------------------------------------------------


ds_math_instruct = load_dataset("TIGER-Lab/MathInstruct", split="train")


def map_transform_math_instruct(example):
    example["images"] = []
    example["texts"] = [
        {
            "user": example["instruction"],
            "assistant": example["output"],
            "source": "MathInstruct",
        }
    ]
    return example


ds_math_instruct = ds_math_instruct.map(
    map_transform_math_instruct,
    remove_columns=ds_math_instruct.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_math_instruct = ds_math_instruct.filter(filter_openhermes, num_proc=NUM_PROC)

ds_math_instruct.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets_merge_image_individual/math_instruct", num_proc=NUM_PROC
)  # 261_781 examples

ds_math_instruct = load_from_disk("/fsx/hugo/fine_tuning_datasets_merge_image_individual/math_instruct")
ds_math_instruct = ds_math_instruct.filter(filter_special_tokens, num_proc=NUM_PROC)
ds_math_instruct.save_to_disk("/fsx/hugo/fine_tuning_datasets/math_instruct", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- CamelAIMath ---------------------------------------
# -------------------------------------------------------------------------------


ds_camel_ai_math = load_dataset("camel-ai/math", split="train")


def map_transform_camel_ai_math(example):
    example["images"] = []
    example["texts"] = [
        {
            "user": example["message_1"],
            "assistant": example["message_2"],
            "source": "CamelAIMath",
        }
    ]
    return example


ds_camel_ai_math = ds_camel_ai_math.map(
    map_transform_camel_ai_math,
    remove_columns=ds_camel_ai_math.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_camel_ai_math = ds_camel_ai_math.filter(filter_openhermes, num_proc=NUM_PROC)

ds_camel_ai_math.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets_merge_image_individual/camel_ai_math", num_proc=NUM_PROC
)  # 49_744 examples

ds_camel_ai_math = load_from_disk("/fsx/hugo/fine_tuning_datasets_merge_image_individual/camel_ai_math")
ds_camel_ai_math = ds_camel_ai_math.filter(filter_special_tokens, num_proc=NUM_PROC)
ds_camel_ai_math.save_to_disk("/fsx/hugo/fine_tuning_datasets/camel_ai_math", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- AtlasMathSets ---------------------------------------
# -------------------------------------------------------------------------------


ds_atlas_math_sets = load_dataset("AtlasUnified/atlas-math-sets", split="train")


def map_transform_atlas_math_sets(example):
    example["images"] = []
    example["texts"] = [
        {
            "user": example["instruction"],
            "assistant": example["output"],
            "source": "AtlasMathSets",
        }
    ]
    return example


ds_atlas_math_sets = ds_atlas_math_sets.map(
    map_transform_atlas_math_sets,
    remove_columns=ds_atlas_math_sets.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_atlas_math_sets = ds_atlas_math_sets.filter(filter_openhermes, num_proc=NUM_PROC)

ds_atlas_math_sets.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets_merge_image_individual/atlas_math_sets", num_proc=NUM_PROC
)  # 17_807_579 examples

ds_atlas_math_sets = load_from_disk("/fsx/hugo/fine_tuning_datasets_merge_image_individual/atlas_math_sets")
ds_atlas_math_sets = ds_atlas_math_sets.filter(filter_special_tokens, num_proc=NUM_PROC)
ds_atlas_math_sets.save_to_disk("/fsx/hugo/fine_tuning_datasets/atlas_math_sets", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- Goat ---------------------------------------
# -------------------------------------------------------------------------------


ds_goat = load_dataset("tiedong/goat", split="train")


def map_transform_goat(example):
    example["images"] = []
    example["texts"] = [
        {
            "user": example["instruction"],
            "assistant": example["output"],
            "source": "Goat",
        }
    ]
    return example


ds_goat = ds_goat.map(
    map_transform_goat,
    remove_columns=ds_goat.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_goat = ds_goat.filter(filter_openhermes, num_proc=NUM_PROC)

ds_goat.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets_merge_image_individual/goat", num_proc=NUM_PROC
)  # 1_746_300 examples

ds_goat = load_from_disk("/fsx/hugo/fine_tuning_datasets_merge_image_individual/goat")
ds_goat = ds_goat.filter(filter_special_tokens, num_proc=NUM_PROC)
ds_goat.save_to_disk("/fsx/hugo/fine_tuning_datasets/goat", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- CamelAIPhysics ---------------------------------------
# -------------------------------------------------------------------------------


ds_camel_ai_physics = load_dataset("camel-ai/physics", split="train")


def map_transform_camel_ai_physics(example):
    example["images"] = []
    example["texts"] = [
        {
            "user": example["message_1"],
            "assistant": example["message_2"],
            "source": "CamelAIPhysics",
        }
    ]
    return example


ds_camel_ai_physics = ds_camel_ai_physics.map(
    map_transform_camel_ai_physics,
    remove_columns=ds_camel_ai_physics.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_camel_ai_physics = ds_camel_ai_physics.filter(filter_openhermes, num_proc=NUM_PROC)

ds_camel_ai_physics.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets_merge_image_individual/camel_ai_physics", num_proc=NUM_PROC
)  # 19_991 examples

ds_camel_ai_physics = load_from_disk("/fsx/hugo/fine_tuning_datasets_merge_image_individual/camel_ai_physics")
ds_camel_ai_physics = ds_camel_ai_physics.filter(filter_special_tokens, num_proc=NUM_PROC)
ds_camel_ai_physics.save_to_disk("/fsx/hugo/fine_tuning_datasets/camel_ai_physics", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- CamelAIBiology ---------------------------------------
# -------------------------------------------------------------------------------


ds_camel_ai_biology = load_dataset("camel-ai/biology", split="train")


def map_transform_camel_ai_biology(example):
    example["images"] = []
    example["texts"] = [
        {
            "user": example["message_1"],
            "assistant": example["message_2"],
            "source": "CamelAIBiology",
        }
    ]
    return example


ds_camel_ai_biology = ds_camel_ai_biology.map(
    map_transform_camel_ai_biology,
    remove_columns=ds_camel_ai_biology.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_camel_ai_biology = ds_camel_ai_biology.filter(filter_openhermes, num_proc=NUM_PROC)

ds_camel_ai_biology.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets_merge_image_individual/camel_ai_biology", num_proc=NUM_PROC
)  # 19_971 examples

ds_camel_ai_biology = load_from_disk("/fsx/hugo/fine_tuning_datasets_merge_image_individual/camel_ai_biology")
ds_camel_ai_biology = ds_camel_ai_biology.filter(filter_special_tokens, num_proc=NUM_PROC)
ds_camel_ai_biology.save_to_disk("/fsx/hugo/fine_tuning_datasets/camel_ai_biology", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- CamelAIChemistry ---------------------------------------
# -------------------------------------------------------------------------------


ds_camel_ai_chemistry = load_dataset("camel-ai/chemistry", split="train")


def map_transform_camel_ai_chemistry(example):
    example["images"] = []
    example["texts"] = [
        {
            "user": example["message_1"],
            "assistant": example["message_2"],
            "source": "CamelAIChemistry",
        }
    ]
    return example


ds_camel_ai_chemistry = ds_camel_ai_chemistry.map(
    map_transform_camel_ai_chemistry,
    remove_columns=ds_camel_ai_chemistry.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_camel_ai_chemistry = ds_camel_ai_chemistry.filter(filter_openhermes, num_proc=NUM_PROC)

ds_camel_ai_chemistry.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets_merge_image_individual/camel_ai_chemistry", num_proc=NUM_PROC
)  # 19_919 examples

ds_camel_ai_chemistry = load_from_disk("/fsx/hugo/fine_tuning_datasets_merge_image_individual/camel_ai_chemistry")
ds_camel_ai_chemistry = ds_camel_ai_chemistry.filter(filter_special_tokens, num_proc=NUM_PROC)
ds_camel_ai_chemistry.save_to_disk("/fsx/hugo/fine_tuning_datasets/camel_ai_chemistry", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- Dolly ---------------------------------------
# -------------------------------------------------------------------------------


ds_dolly = load_dataset("databricks/databricks-dolly-15k", split="train")


def map_transform_dolly(example):
    example["images"] = []
    instruction = example["instruction"]
    context = example["context"]
    if context:
        instruction = context + "\n" + instruction
    example["texts"] = [
        {
            "user": instruction,
            "assistant": example["response"],
            "source": "Dolly",
        }
    ]
    return example


ds_dolly = ds_dolly.map(
    map_transform_dolly,
    remove_columns=ds_dolly.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_dolly = ds_dolly.filter(filter_openhermes, num_proc=NUM_PROC)

ds_dolly.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets_merge_image_individual/dolly", num_proc=NUM_PROC
)  # 14_972 examples

ds_dolly = load_from_disk("/fsx/hugo/fine_tuning_datasets_merge_image_individual/dolly")
ds_dolly = ds_dolly.filter(filter_special_tokens, num_proc=NUM_PROC)
ds_dolly.save_to_disk("/fsx/hugo/fine_tuning_datasets/dolly", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- CLEVR-Math ---------------------------------------
# -------------------------------------------------------------------------------

ds_clevr_math_1 = load_dataset("dali-does/clevr-math", "general", split="train")
ds_clevr_math_2 = load_dataset("dali-does/clevr-math", "multihop", split="train")
ds_clevr_math = concatenate_datasets([ds_clevr_math_1, ds_clevr_math_2])


image_ids = ds_clevr_math["id"]
map_image_id_to_ds_id = {}
for ds_id, image_id in enumerate(tqdm(image_ids)):
    map_image_id_to_ds_id[image_id] = map_image_id_to_ds_id.get(image_id, []) + [ds_id]


dict_clevr_math = {"images": [], "texts": []}
for image_id, indices_ds in tqdm(map_image_id_to_ds_id.items()):
    images = [ds_clevr_math[indices_ds[0]]["image"]]
    texts = []
    for ds_id in indices_ds:
        example = ds_clevr_math[ds_id]
        text = {
            "user": example["question"] + random.choice(PROMPTS_ANSWER_SHORTLY),
            "assistant": correct_casing(str(example["label"])),
            "source": "CLEVR-Math",
        }
        texts.append(text)
    dict_clevr_math["images"].append(images)
    dict_clevr_math["texts"].append(texts)


def data_generator():
    for images, texts in zip(dict_clevr_math["images"], dict_clevr_math["texts"]):
        yield {"images": images, "texts": texts}


ds_clevr_math = datasets.Dataset.from_generator(data_generator, features=FEATURES, writer_batch_size=100)

ds_clevr_math.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets/clevr_math", num_proc=NUM_PROC
)  # 70_000 examples (multiple questions per example)


# -------------------------------------------------------------------------------
# --------------------------- TQA ---------------------------------------
# -------------------------------------------------------------------------------

"""
Download:
https://ai2-public-datasets.s3.amazonaws.com/tqa/tqa_train_val_test.zip -> /fsx/hugo/TQA/
"""

with open("/fsx/hugo/TQA/tqa_train_val_test/train/tqa_v1_train.json", "r") as f:
    data_tqa = json.load(f)

dict_tqa = {"images": [], "texts": []}
for lesson in tqdm(data_tqa):
    questions = lesson["questions"]["diagramQuestions"]
    for _, question_ex in questions.items():
        #
        img_path = os.path.join("/fsx/hugo/TQA/tqa_train_val_test/train", question_ex["imagePath"])
        images = [
            {
                "bytes": convert_img_to_bytes(img_path=img_path, format="png"),
                "path": None,
            }
        ]
        #
        question_type = question_ex["questionType"]
        if question_type == "Diagram Multiple Choice":
            question = "Question: " + question_ex["beingAsked"]["processedText"] + "\n"
            question += "Choices:\n"
            for _, choice in question_ex["answerChoices"].items():
                question += correct_casing(choice["idStructural"] + " " + choice["processedText"]) + "\n"
            question += "Answer with the letter."
            answer = "Answer: " + question_ex["correctAnswer"]["processedText"].upper()
        else:
            raise ValueError("Unknown question type")
        texts = [
            {
                "user": question,
                "assistant": answer,
                "source": "TQA",
            }
        ]
        dict_tqa["images"].append(images)
        dict_tqa["texts"].append(texts)

ds_tqa = datasets.Dataset.from_dict(dict_tqa, features=FEATURES)

ds_tqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/tqa", num_proc=NUM_PROC)  # 6_501 examples


# -------------------------------------------------------------------------------
# --------------------------- TabMWP (PromptPG) ---------------------------------------
# -------------------------------------------------------------------------------

"""
Download:
https://drive.google.com/drive/folders/1IYgGrY9agwF_qQlh4WNRG_4WF_ggTkvG -> /fsx/hugo/PromptPG
"""

with open("/fsx/hugo/PromptPG/tabmwp/problems_train.json", "r") as f:
    data_tabmwp = json.load(f)

dict_tabmwp = {"images": [], "texts": []}
for table_id, data_ex in tqdm(data_tabmwp.items()):
    question = data_ex["question"]
    answer = data_ex["solution"].replace("\n\n", "\n")
    assert len(answer) > 0
    #
    img_path = os.path.join("/fsx/hugo/PromptPG/tabmwp/tables", f"{table_id}.png")
    images = [
        {
            "bytes": convert_img_to_bytes(img_path=img_path, format="png"),
            "path": None,
        }
    ]
    texts = [
        {
            "user": question,
            "assistant": answer,
            "source": "TabMWP",
        }
    ]
    dict_tabmwp["images"].append(images)
    dict_tabmwp["texts"].append(texts)

ds_tabmwp = datasets.Dataset.from_dict(dict_tabmwp, features=FEATURES)

ds_tabmwp.save_to_disk("/fsx/hugo/fine_tuning_datasets/tabmwp", num_proc=NUM_PROC)  # 23_059 examples


# -------------------------------------------------------------------------------
# --------------------------- UniGeo ---------------------------------------
# -------------------------------------------------------------------------------

"""
Download:
https://drive.google.com/drive/folders/1NifdHLJe5U08u2Zb1sWL6C-8krpV2z2O?usp=share_link -> /fsx/hugo/UniGeo/
"""

with open(os.path.join("/fsx/hugo/UniGeo/UniGeo", "calculation_train.pk"), "rb") as f:
    data_unigeo = pickle.load(f)


dict_unigeo = {"images": [], "texts": []}
for data_ex in tqdm(data_unigeo):
    question = "Question: " + data_ex["English_problem"] + "\n"
    question += "Choices:\n"
    letters_cap = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    choices = data_ex["choices"]
    for idx_choice, choice in enumerate(choices):
        letter = letters_cap[idx_choice]
        question += f"{letter}. {choice}\n"
    question += "Answer with the letter."
    answer = "Answer: " + letters_cap[data_ex["label"]]
    #
    img = Image.fromarray(data_ex["image"])
    bytes_io = BytesIO()
    img.save(bytes_io, format="PNG")
    image_bytes = bytes_io.getvalue()
    images = [
        {
            "bytes": image_bytes,
            "path": None,
        }
    ]
    texts = [
        {
            "user": question,
            "assistant": answer,
            "source": "UniGeo",
        }
    ]
    dict_unigeo["images"].append(images)
    dict_unigeo["texts"].append(texts)

ds_unigeo = datasets.Dataset.from_dict(dict_unigeo, features=FEATURES)


def func_filter_unigeo(example, black_threshold=20, mostly_black_threshold=50):
    # Removes the illisible images containing a lot of black pixels
    img = example["images"][0]
    img = img.convert("RGB")
    width, height = img.size
    total_pixels = width * height
    black_pixels = 0
    for x in range(width):
        for y in range(height):
            r, g, b = img.getpixel((x, y))
            if r < black_threshold and g < black_threshold and b < black_threshold:
                black_pixels += 1
    black_percentage = (black_pixels / total_pixels) * 100
    return black_percentage < mostly_black_threshold


ds_unigeo = ds_unigeo.filter(func_filter_unigeo, num_proc=NUM_PROC)

ds_unigeo.save_to_disk("/fsx/hugo/fine_tuning_datasets/unigeo", num_proc=NUM_PROC)  # 3_482 examples


# -------------------------------------------------------------------------------
# --------------------------- MapQA ---------------------------------------
# -------------------------------------------------------------------------------

"""
Download:
Link in https://github.com/OSU-slatelab/MapQA -> /fsx/hugo/MapQA/
"""

dict_mapqa = {"images": [], "texts": []}
subsets_mapqa = ["MapQA_S", "MapQA_U", "MapQA_V"]
for subset_mapqa in subsets_mapqa:
    common_path_subset_mapqa = os.path.join("/fsx/hugo/MapQA/", subset_mapqa)
    path_questions = os.path.join(common_path_subset_mapqa, "questions/train-QA.json")
    with open(path_questions, "r") as f:
        data_subset_mapqa = json.load(f)
    for data_ex in tqdm(data_subset_mapqa):
        path_image = os.path.join(common_path_subset_mapqa, "images", data_ex["map_id"])
        dict_mapqa["images"].append([path_image])
        question = data_ex["question"].replace(" ?", "?") + random.choice(PROMPTS_ANSWER_SHORTLY)
        answer = data_ex["answer"]
        if isinstance(answer, int):
            answer = str(answer)
        elif isinstance(answer, list):
            answer = ", ".join(answer)
        answer = correct_casing(answer)
        texts = [
            {
                "user": question,
                "assistant": answer,
                "source": "MapQA",
            }
        ]
        dict_mapqa["texts"].append(texts)

mapping_path_image_to_dict_id = {}
for dict_id, (images, texts) in enumerate(zip(dict_mapqa["images"], dict_mapqa["texts"])):
    path_image = images[0]
    mapping_path_image_to_dict_id[path_image] = mapping_path_image_to_dict_id.get(path_image, []) + [dict_id]

merged_dict_mapqa = {"images": [], "texts": []}
for path_image, dict_ids in tqdm(mapping_path_image_to_dict_id.items()):
    images = [
        {
            "bytes": convert_img_to_bytes(img_path=path_image, format="png"),
            "path": None,
        }
    ]
    texts = [dict_mapqa["texts"][dict_id][0] for dict_id in dict_ids]
    merged_dict_mapqa["images"].append(images)
    merged_dict_mapqa["texts"].append(texts)


def data_generator():
    for images, texts in zip(merged_dict_mapqa["images"], merged_dict_mapqa["texts"]):
        yield {"images": images, "texts": texts}


ds_mapqa = datasets.Dataset.from_generator(data_generator, features=FEATURES, writer_batch_size=100)

ds_mapqa.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets/mapqa", num_proc=NUM_PROC
)  # 37_417 examples (multiple questions per example)


# -------------------------------------------------------------------------------
# --------------------------- Llava Conv ---------------------------------------
# -------------------------------------------------------------------------------


ds_llava_conv = load_dataset("jxu124/llava_conversation_58k", split="train")


def map_transform_llava_conv(example):
    example["images"] = [
        {
            "bytes": convert_img_to_bytes(img_path=os.path.join("/fsx/hugo", example["image_path"]), format="JPEG"),
            "path": None,
        }
    ]
    texts = []
    for turn in example["dialog"]:
        assert len(turn) == 2
        texts.append(
            {
                "user": turn[0],
                "assistant": turn[1],
                "source": "Llava Conv",
            }
        )
    example["texts"] = texts
    return example


ds_llava_conv = ds_llava_conv.map(
    map_transform_llava_conv,
    remove_columns=ds_llava_conv.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_llava_conv = ds_llava_conv.filter(filter_openhermes, num_proc=NUM_PROC)
ds_llava_conv = ds_llava_conv.filter(filter_special_tokens, num_proc=NUM_PROC)

ds_llava_conv.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets_merge_image_individual/llava_conv", num_proc=NUM_PROC
)  # 56_676 examples


# -------------------------------------------------------------------------------
# --------------------------- LNQA ---------------------------------------
# -------------------------------------------------------------------------------


ds_lnqa = load_dataset("vikhyatk/lnqa", split="train")


def map_transform_lnqa(example):
    example["images"] = [example["image"]]
    example["texts"] = [
        {
            "user": qa_pair["question"],
            "assistant": qa_pair["answer"],
            "source": "LNQA",
        }
        for qa_pair in example["qa"]
    ]
    return example


ds_lnqa = ds_lnqa.map(
    map_transform_lnqa,
    remove_columns=ds_lnqa.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_lnqa.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets_merge_image_individual/lnqa", num_proc=NUM_PROC
)  # 302_780 examples


# -------------------------------------------------------------------------------
# --------------------------- ShareGPT-4o ---------------------------------------
# -------------------------------------------------------------------------------


ds_sharegpt4o = load_dataset("OpenGVLab/ShareGPT-4o", "image_caption", split="images")


def map_transform_sharegpt4o(example):
    example["images"] = [
        {
            "bytes": convert_img_to_bytes(
                img_path=os.path.join("/fsx/hugo/ShareGPT-4o/images/", example["image"]), format="JPEG"
            ),
            "path": None,
        }
    ]
    conversations = example["conversations"]
    assert len(conversations) % 2 == 0
    texts = []
    for i in range(0, len(conversations), 2):
        assert conversations[i]["from"] == "human"
        assert conversations[i + 1]["from"] == "gpt"
        texts.append(
            {
                "user": conversations[i]["value"].replace("<image>", "").strip(),
                "assistant": conversations[i + 1]["value"].replace("<image>", "").strip(),
                "source": "ShareGPT-4o",
            }
        )
    example["texts"] = texts
    return example


ds_sharegpt4o = ds_sharegpt4o.map(
    map_transform_sharegpt4o,
    remove_columns=ds_sharegpt4o.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_sharegpt4o = ds_sharegpt4o.filter(filter_openhermes, num_proc=NUM_PROC)
ds_sharegpt4o = ds_sharegpt4o.filter(filter_special_tokens, num_proc=NUM_PROC)

ds_sharegpt4o.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets_merge_image_individual/sharegpt4o", num_proc=NUM_PROC
)  # 57_259 examples


# -------------------------------------------------------------------------------
# --------------------------- Geo170K ---------------------------------------
# -------------------------------------------------------------------------------


ds_geo170k = concatenate_datasets(
    [
        load_dataset("Luckyjhg/Geo170K", split="alignment"),
        load_dataset("Luckyjhg/Geo170K", split="qa_tuning"),
    ]
)

mapping_path_image_to_conv = {}
for idx_row in tqdm(range(ds_geo170k.num_rows)):
    path_image = ds_geo170k[idx_row]["image"]
    conv = ds_geo170k[idx_row]["conversations"]
    mapping_path_image_to_conv[path_image] = mapping_path_image_to_conv.get(path_image, []) + conv

dict_geo170k = {
    "image": list(mapping_path_image_to_conv.keys()),
    "conversations": list(mapping_path_image_to_conv.values()),
}
ds_geo170k = datasets.Dataset.from_dict(dict_geo170k)


def map_transform_geo170k(example):
    example["images"] = [
        {
            "bytes": convert_img_to_bytes(
                img_path=os.path.join("/fsx/hugo/Geo170K/images/", example["image"]), format="png"
            ),
            "path": None,
        }
    ]
    conversations = example["conversations"]
    assert len(conversations) % 2 == 0
    texts = []
    for i in range(0, len(conversations), 2):
        assert conversations[i]["from"] == "human"
        assert conversations[i + 1]["from"] == "gpt"
        texts.append(
            {
                "user": conversations[i]["value"].replace("<image>", "").strip(),
                "assistant": conversations[i + 1]["value"].replace("<image>", "").strip(),
                "source": "Geo170K",
            }
        )
    example["texts"] = texts
    return example


ds_geo170k = ds_geo170k.map(
    map_transform_geo170k,
    remove_columns=ds_geo170k.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_geo170k = ds_geo170k.filter(filter_openhermes, num_proc=NUM_PROC)
ds_geo170k = ds_geo170k.filter(filter_special_tokens, num_proc=NUM_PROC)

ds_geo170k.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets_merge_image_individual/geo170k", num_proc=NUM_PROC
)  # 9_067 examples (multiple questions per example)


# -------------------------------------------------------------------------------
# --------------------------- IIW-400 ---------------------------------------
# -------------------------------------------------------------------------------

"""
Download https://github.com/google/imageinwords/tree/main/datasets/IIW-400 -> /fsx/hugo/IIW-400
https://storage.googleapis.com/docci/data/docci_images_aar.tar.gz for the images
"""

prompts_iiw = [
    "Describe this picture in detail.",
    "Give a thorough description of this image.",
    "Provide an in-depth description of what you see in this picture.",
    "Explain everything happening in this image in detail.",
    "Offer a comprehensive description of this picture.",
    "Describe this image with as much detail as possible.",
    "Can you describe every aspect of this picture?",
    "Give a detailed account of this image.",
    "Describe all the details present in this picture.",
    "Provide a complete description of this image.",
    "Describe this picture thoroughly.",
    "Give an extensive description of this image.",
    "Explain this picture in great detail.",
    "Describe all the elements in this image comprehensively.",
    "Provide a detailed narrative of this picture.",
    "Give a long caption of this image.",
    "Give a meticulous description of this image.",
    "Describe everything you see in this picture in detail.",
    "Offer a detailed explanation of this picture.",
    "Can you provide a detailed description of this image?",
]

data_iiw400 = []
with open("/fsx/hugo/IIW-400/data_IIW-400.jsonl", "r") as f:
    for line in f:
        data_iiw400.append(json.loads(line))

dict_iiw400 = {"images": [], "texts": []}

for example in tqdm(data_iiw400):
    dict_iiw400["texts"].append(
        [
            {
                "user": random.choice(prompts_iiw),
                "assistant": example["IIW"],
                "source": "IIW-400",
            }
        ]
    )
    image = [
        {
            "bytes": convert_img_to_bytes(
                img_path=os.path.join("/fsx/hugo/IIW-400/images_aar/", example["image/key"] + ".jpg"), format="JPEG"
            ),
            "path": None,
        }
    ]
    dict_iiw400["images"].append(image)

ds_iiw400 = datasets.Dataset.from_dict(dict_iiw400, features=FEATURES)

ds_iiw400.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets_merge_image_individual/iiw400", num_proc=NUM_PROC
)  # 400 examples


# -------------------------------------------------------------------------------
# --------------------------- Cord-v2 ---------------------------------------
# -------------------------------------------------------------------------------


ds_cord = load_dataset("naver-clova-ix/cord-v2", split="train")

prompts_cord = [
    "Extract information from the image in JSON.",
    "Identify text in the image and format it in JSON.",
    "List visible text in the image in JSON.",
    "Extract written data from the image as JSON.",
    "Format text in the image using JSON.",
    "Summarize image information in JSON.",
    "Extract and format text blocks in JSON.",
    "Provide image details as JSON.",
    "List messages in the image in JSON.",
    "Extract and format text elements in JSON.",
    "Identify and list image text in JSON.",
    "Format any visible text in JSON.",
    "List key information from the image in JSON.",
    "Extract visible numbers and format in JSON.",
    "Summarize visible text in the image using JSON.",
    "Provide text and details from the image in JSON.",
    "Extract the information from this image and output it in a JSON format.",
    "Extract the information in this picture. Output JSON format.",
    "Transcribe everything in this image. Output JSON format.",
]


def map_transform_cord(example):
    example["images"] = [example["image"]]
    output_json_ex = json.loads(example["ground_truth"])["gt_parse"]
    output_json_ex = json.dumps(output_json_ex, indent=4)
    example["texts"] = [
        {
            "user": random.choice(prompts_cord),
            "assistant": output_json_ex,
            "source": "Cord-v2",
        }
    ]
    return example


ds_cord = ds_cord.map(
    map_transform_cord,
    remove_columns=ds_cord.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_cord.save_to_disk("/fsx/hugo/fine_tuning_datasets_merge_image_individual/cord", num_proc=NUM_PROC)  # 800 examples


# -------------------------------------------------------------------------------
# --------------------------- ChartGemma ---------------------------------------
# -------------------------------------------------------------------------------


ds_chartgemma = load_dataset("ahmed-masry/ChartGemma", split="train")


prompts_chart_to_markdown = [
    "Translate this chart to Markdown.",
    "Convert this image to a Markdown table.",
    "Generate a Markdown table from this chart.",
    "Turn this image into Markdown format.",
    "Create a Markdown table based on this chart.",
    "Convert chart data from image to Markdown.",
    "Produce a Markdown table from this chart.",
    "Render this chart in Markdown.",
    "Change this chart image to a Markdown table.",
    "Transform this chart into Markdown format.",
    "Make a Markdown table from this image.",
    "Extract data from this chart and create a Markdown table.",
    "Represent this chart as a Markdown table.",
    "Generate Markdown from this chart image.",
    "Turn chart image data into a Markdown table.",
    "Write a Markdown table using data from this chart.",
    "Convert chart information to Markdown format.",
    "Create Markdown from this image of a chart.",
    "Translate the chart data to a Markdown table.",
    "Produce Markdown format for this chart.",
]

prompts_chart_captioning = [
    "Describe this chart.",
    "Describe this image.",
    "Summarize the chart.",
    "Label this chart.",
    "Explain the chart.",
    "Highlight key points in this image.",
    "What does this image show?",
    "Provide a summary for this picture.",
    "Give an overview of the chart.",
    "Describe the data in this chart.",
    "Write a summary for the chart.",
    "What is this chart illustrating?",
    "Clarify the chart's data.",
    "Explain the chart's meaning.",
    "Summarize the information in the chart.",
    "Interpret this chart.",
    "Highlight the main idea of the chart.",
]


def map_transform_chartgemma(example):
    image = Image.open(BytesIO(example["image"]))
    example["images"] = [image]
    user = example["input"]
    assistant = example["output"]
    if "program of thought" in user:
        user = user.replace("program of thought", "").lstrip(": ")
        user = user + " Answer by giving a code."
    elif "Chart to Markdown" in user:
        user = random.choice(prompts_chart_to_markdown)
    elif "Fact Checking" in user:
        user = user.replace("Fact Checking", "").lstrip(": ")
        user = "Do you support or refute the following statement: " + user
        user = user + "\nAnswer by saying 'Supports' or 'Refutes'."
    elif "Generate a caption for the chart" in user:
        user = random.choice(prompts_chart_captioning)
    elif "Let's think step by step" in user:
        user = user + "\nAfter your reasoning, give your final answer as 'Final Answer: <your_final_answer>'."
    example["texts"] = [
        {
            "user": user,
            "assistant": assistant,
            "source": "ChartGemma",
        }
    ]
    return example


ds_chartgemma = ds_chartgemma.map(
    map_transform_chartgemma,
    remove_columns=ds_chartgemma.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_chartgemma.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets_merge_image_individual/chartgemma", num_proc=NUM_PROC
)  # 163_240 examples


# --------------------------- Victor --------------------------------------

# -------------------------------------------------------------------------------
# --------------------------- A-OKVQA -------------------------------------------
# -------------------------------------------------------------------------------

ds_aokvqa = load_dataset("HuggingFaceM4/A-OKVQA", split="train")

prompts_aokvqa_1 = [
    "Answer the question by selecting the correct answer among the 4 following choices.",
    "Choose the right answer from the provided options to respond to the question.",
    "Select the accurate response from the four choices given to answer the question.",
    "Pick the correct solution from the four options below to address the question.",
    "From the following four choices, select the correct answer to address the question.",
    "Make your selection from the four choices given to correctly answer the question.",
    "From the following set of four choices, select the accurate answer to respond to the question.",
    "Indicate the correct response by choosing from the four available options to answer the question.",
]

prompts_aokvqa_2 = [
    (
        "Answer the question by selecting the correct answer among the 4 following choices and explain your choice"
        " with a short sentence. The answer should be formatted with the following format: `Answer: choice\nRationale:"
        " rationale.`"
    ),
    "Choose the correct response and explain in the format: 'Answer: answer\nRationale: rationale.'",
    "Select the accurate answer and provide justification: `Answer: choice\nRationale: srationale.`",
    "Pick the right solution, then justify: 'Answer: answer\nRationale: rationale.'",
    "Indicate the correct choice and explain in the format: 'Answer: answer\nRationale: rationale.'",
    "Choose the right answer and clarify with the format: 'Answer: answer\nRationale: rationale.'",
    (
        "Select the correct answer and articulate reasoning with the following format: 'Answer: answer\nRationale:"
        " rationale.'"
    ),
    "Make your selection and explain in format: 'Answer: answer\nRationale: rationale.'",
    "Choose the correct response, then elucidate: 'Answer: answer\nRationale: rationale.'",
    "Select the accurate answer and provide explanation: 'Answer: answer\nRationale: rationale.'",
    "Indicate the correct response and explain using: 'Answer: answer\nRationale: rationale.'",
]


def map_transform_aokvqa(example):
    task = random.randint(a=0, b=1)
    if task == 0:
        example["images"] = [example["image"]]
        question = example["question"]
        choices = example["choices"]
        correct_choice_idx = example["correct_choice_idx"]
        example["texts"] = [
            {
                "user": f"{question}\n{random.choice(prompts_aokvqa_1)}\nOptions: {', '.join(choices).capitalize()}.",
                "assistant": correct_casing(choices[correct_choice_idx]),
                "source": "A-OKVQA",
            }
        ]
        return example
    elif task == 1:
        example["images"] = [example["image"]]
        question = example["question"]
        choices = example["choices"]
        correct_choice_idx = example["correct_choice_idx"]
        rationale = random.choice(example["rationales"])
        answer = correct_casing(f"Answer: {choices[correct_choice_idx]}")
        example["texts"] = [
            {
                "user": f"{question}\n{random.choice(prompts_aokvqa_2)}\nOptions: {', '.join(choices).capitalize()}.",
                "assistant": f"{answer}\nRationale: {rationale}",
                "source": "A-OKVQA",
            }
        ]
        return example


ds_aokvqa = ds_aokvqa.map(
    map_transform_aokvqa, remove_columns=ds_aokvqa.column_names, features=FEATURES, num_proc=NUM_PROC
)

ds_aokvqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/aokvqa", num_proc=NUM_PROC)  # 17_056 examples

# -------------------------------------------------------------------------------
# --------------------------- Spot the Diff -------------------------------------
# -------------------------------------------------------------------------------
# The one from https://arxiv.org/pdf/1808.10584.pdf formatted as part of https://arxiv.org/abs/2306.05425

ds_mimic_sd = load_dataset("pufanyi/MIMICIT", name="SD", split="train")
ds_mimic_sd = ds_mimic_sd.rename_column("images", "images_og")


def map_transform_mimic_sd(example):
    instruction = example["instruction"]
    answer = example["answer"]
    example["texts"] = [{"user": instruction, "assistant": answer, "source": "MIMIC-IT - SN"}]
    example["images"] = example["images_og"]
    return example


ds_mimic_sd = ds_mimic_sd.map(
    map_transform_mimic_sd, remove_columns=ds_mimic_sd.column_names, features=FEATURES, num_proc=NUM_PROC
)

ds_mimic_sd.save_to_disk("/fsx/hugo/fine_tuning_datasets/mimic_sd", num_proc=NUM_PROC)  # 15_869 examples


# -------------------------------------------------------------------------------
# --------------------------- Spot the Diff -------------------------------------
# -------------------------------------------------------------------------------
# The one from https://arxiv.org/abs/2306.05425 and GSD (General Scene Difference) which seems to be renamed CGD on the hub dataset

cgd_images = load_dataset("pufanyi/MIMICIT", name="CGD_Images", split="train")
id_to_idx = {id: idx for idx, id in enumerate(cgd_images["id"])}
cgd_instructions = load_dataset("pufanyi/MIMICIT", name="CGD_Instructions", split="train")
cgd_instructions = cgd_instructions.rename_column("images", "images_og")


def map_transform_mimic_cgd(example):
    instruction = example["instruction"]
    answer = example["answer"]
    example["texts"] = [{"user": instruction, "assistant": answer, "source": "MIMIC-IT - SN"}]
    image_1 = cgd_images[id_to_idx[example["images_og"][0]]]["image"]
    image_2 = cgd_images[id_to_idx[example["images_og"][1]]]["image"]
    example["images"] = [image_1, image_2]
    return example


ds_mimic_cgd = cgd_instructions.map(
    map_transform_mimic_cgd,
    remove_columns=cgd_instructions.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_mimic_cgd.save_to_disk("/fsx/hugo/fine_tuning_datasets/mimic_cgd", num_proc=NUM_PROC)  # 141_869 examples


# -------------------------------------------------------------------------------
# --------------------------- PGM -----------------------------------------------
# -------------------------------------------------------------------------------

ds_pgm = load_dataset("HuggingFaceM4/PGM", split="train")
url = "https://banner2.cleanpng.com/20180329/xde/kisspng-computer-icons-question-mark-question-mark-5abc8df9d397b7.7019434615223065538667.jpg"
response = requests.get(url)
question_mark_image = Image.open(BytesIO(response.content))
question_mark_image = question_mark_image.resize((160, 160))
question_mark_image = _convert_to_rgb(question_mark_image)

prompts_pgm_1 = [
    "The image that logically completes the sequence is this one. Is that correct?",
    "Is this the correct image that logically concludes the sequence?",
    "Does this image appropriately finalize the logical sequence?",
    "Is the image provided the accurate completion of the logical sequence?",
    "Can it be affirmed that this image logically concludes the given sequence?",
    "Is the correctness of the image, which logically completes the sequence, confirmed?",
]


def map_transform_pgm(example):
    panels = example["panels"]
    choices = example["choices"]
    target = example["target"]

    border_size = 2
    width, height = panels[0].size
    grid_width, grid_height = 3 * (width + border_size), 3 * (height + border_size)
    grid_image = Image.new("RGB", (grid_width, grid_height))
    draw = ImageDraw.Draw(grid_image)
    for i in range(3):
        for j in range(3):
            x, y = i * (width + border_size), j * (height + border_size)
            if (i, j) == (2, 2):
                grid_image.paste(question_mark_image, (x, y))
            else:
                grid_image.paste(panels[3 * i + j], (x, y))

            draw.rectangle([x, y, x + width + border_size, y + border_size], fill="black")
            draw.rectangle([x, y, x + border_size, y + height + border_size], fill="black")
            draw.rectangle([x + width, y, x + width + border_size, y + height + border_size], fill="black")
            draw.rectangle([x, y + height, x + width + border_size, y + height + border_size], fill="black")

    # Easier problem: the balanced binary classification problem.
    instruction = random.choice(prompts_pgm_1)
    if random.random() < 0.5:
        correct_choice = choices[target]
        example["images"] = [grid_image, correct_choice]
        example["texts"] = [{"user": instruction, "assistant": "Yes", "source": "PGM"}]
    else:
        random_wrong_choice = choices[random.choice([i for i in range(8) if i != target])]
        example["images"] = [grid_image, random_wrong_choice]
        example["texts"] = [{"user": instruction, "assistant": "No", "source": "PGM"}]

    return example


ds_pgm = ds_pgm.map(map_transform_pgm, remove_columns=ds_pgm.column_names, features=FEATURES, num_proc=NUM_PROC)

ds_pgm.save_to_disk("/fsx/hugo/fine_tuning_datasets/pgm", num_proc=NUM_PROC)  # 1_200_000 examples


# -------------------------------------------------------------------------------
# --------------------------- RAVEN ---------------------------------------------
# -------------------------------------------------------------------------------

ds_raven = concatenate_datasets(
    [
        load_dataset("HuggingFaceM4/RAVEN", name=config_name, split="train")
        for config_name in [
            "center_single",
            "distribute_four",
            "distribute_nine",
            "in_center_single_out_center_single",
            "in_distribute_four_out_center_single",
            "left_center_single_right_center_single",
            "up_center_single_down_center_single",
        ]
    ]
)
ds_raven = ds_raven.shuffle()

url = "https://banner2.cleanpng.com/20180329/xde/kisspng-computer-icons-question-mark-question-mark-5abc8df9d397b7.7019434615223065538667.jpg"
response = requests.get(url)
question_mark_image = Image.open(BytesIO(response.content))
question_mark_image = question_mark_image.resize((160, 160))
question_mark_image = _convert_to_rgb(question_mark_image)

prompts_raven_1 = [
    "The image that logically completes the sequence is this one. Is that correct? Answer by yes or no.",
    "Is this the correct image that logically concludes the sequence? Yes or no.",
    "Does this image appropriately finalize the logical sequence? Yes or No?",
    "Answer by yes or no. Is the image provided the accurate completion of the logical sequence?",
    "Can it be affirmed that this image logically concludes the given sequence? Yes or no.",
    "Is the correctness of the image, which logically completes the sequence, confirmed? Yes, no?",
]

prompts_raven_2 = [
    "Choose the figure that would logically complete the sequence.",
    "Which figure would finalize the logical sequence and replace the question mark?",
    "Solve that puzzle by choosing the appropriate letter.",
    "Which figure should complete the logical sequence?",
]


def map_transform_raven(example):
    panels = example["panels"]
    choices = example["choices"]
    target = example["target"]

    border_size = 1
    width, height = panels[0].size

    task = random.randint(1, 2)
    if task == 1:
        # Easier problem: the balanced binary classification problem.
        grid_width, grid_height = border_size + 3 * (width + border_size), border_size + 3 * (height + border_size)
        grid_image = Image.new("RGB", (grid_width, grid_height))
        draw = ImageDraw.Draw(grid_image)
        for i in range(3):
            for j in range(3):
                x, y = border_size + i * (width + border_size), border_size + j * (height + border_size)
                if (i, j) == (2, 2):
                    grid_image.paste(question_mark_image, (x, y))
                else:
                    grid_image.paste(panels[3 * i + j], (x, y))

        total_width = total_height = 4 * border_size + 3 * width
        for i in range(4):  # Vertical borders
            y = i * (width + border_size)
            draw.rectangle([0, y, total_width, y + border_size], fill="black")

        for i in range(4):  # Horizontal borders
            x = i * (height + border_size)
            draw.rectangle([x, 0, x + border_size, total_height], fill="black")

        instruction = random.choice(prompts_raven_1)
        if random.random() < 0.5:
            correct_choice = choices[target]
            choice_draw = ImageDraw.Draw(correct_choice)
            choice_draw.rectangle([0, 0, width, border_size], fill="black")
            choice_draw.rectangle([0, 0, border_size, height], fill="black")
            choice_draw.rectangle([0, height - border_size, width, height], fill="black")
            choice_draw.rectangle([width - border_size, 0, width, height], fill="black")

            example["images"] = [grid_image, correct_choice]
            example["texts"] = [{"user": instruction, "assistant": "Yes.", "source": "RAVEN"}]
        else:
            random_wrong_choice = choices[random.choice([i for i in range(8) if i != target])]
            choice_draw = ImageDraw.Draw(random_wrong_choice)
            choice_draw.rectangle([0, 0, width, border_size], fill="black")
            choice_draw.rectangle([0, 0, border_size, height], fill="black")
            choice_draw.rectangle([0, height - border_size, width, height], fill="black")
            choice_draw.rectangle([width - border_size, 0, width, height], fill="black")

            example["images"] = [grid_image, random_wrong_choice]
            example["texts"] = [{"user": instruction, "assistant": "No.", "source": "RAVEN"}]

    elif task == 2:
        grid_width, grid_height = int(5.25 * (width + border_size)), 7 * (height + border_size)
        grid_image = Image.new("RGB", (grid_width, grid_height), color="white")
        draw = ImageDraw.Draw(grid_image)

        offset_x = width
        offset_y = int(height / 2)
        for i in range(3):
            for j in range(3):
                x, y = offset_x + border_size + i * (width + border_size), offset_y + border_size + j * (
                    height + border_size
                )
                if (i, j) == (2, 2):
                    grid_image.paste(question_mark_image, (x, y))
                else:
                    grid_image.paste(panels[3 * i + j], (x, y))

        total_width = total_height = 4 * border_size + 3 * width
        for i in range(4):  # Vertical borders
            y = offset_y + i * (width + border_size)
            draw.rectangle([offset_x, y, offset_x + total_width, y + border_size], fill="black")

        for i in range(4):  # Horizontal borders
            x = offset_x + i * (height + border_size)
            draw.rectangle([x, offset_y, x + border_size, offset_y + total_height], fill="black")

        font = ImageFont.truetype("/admin/home/victor/.local/share/fonts/Hack-Bold.ttf", size=45)

        # Loop for choices
        for k, choice in enumerate(choices):
            x_choice = int(0.25 * width + (k % 4) * 1.25 * width)
            y_choice = int(4 * (height + border_size) + math.floor(k / 4) * height * 1.5)
            grid_image.paste(choice, (x_choice, y_choice))
            draw.rectangle([x_choice, y_choice, x_choice + width, y_choice + border_size], fill="black")
            draw.rectangle([x_choice, y_choice, x_choice + border_size, y_choice + height], fill="black")
            draw.rectangle(
                [x_choice, y_choice + height - border_size, x_choice + width, y_choice + height], fill="black"
            )
            draw.rectangle(
                [x_choice + width - border_size, y_choice, x_choice + width, y_choice + height], fill="black"
            )

            index_label = chr(ord("A") + k)
            draw.text((x_choice + int(width / 2.1), y_choice - int(height / 3)), index_label, fill="black", font=font)

        instruction = random.choice(prompts_raven_2)
        example["images"] = [grid_image]
        example["texts"] = [{"user": instruction, "assistant": chr(ord("A") + example["target"]), "source": "RAVEN"}]

    return example


ds_raven = ds_raven.map(
    map_transform_raven, remove_columns=ds_raven.column_names, features=FEATURES, num_proc=NUM_PROC
)

ds_raven.save_to_disk("/fsx/hugo/fine_tuning_datasets/raven", num_proc=NUM_PROC)  # 42_000 examples


# --------------------------- Leo --------------------------------------

# -------------------------------------------------------------------------------
# --------------------------- TextVQA ---------------------------------------
# -------------------------------------------------------------------------------


ds_textvqa = load_dataset("textvqa", split="train")


def map_transform_textvqa(example):
    example["images"] = [example["image"]]

    # Get the answer(s) with the highest count (agreement among annotators), or a random one if there is a tie.
    answer_counts = {answer: example["answers"].count(answer) for answer in set(example["answers"])}
    max_count = max(answer_counts.values())
    answers_with_max_count = [string for string, count in answer_counts.items() if count == max_count]
    selected_answer = random.choice(answers_with_max_count)

    question = correct_casing(example["question"], is_question=True) + random.choice(QUESTION_BRIEVETY_HINT)
    selected_answer = correct_casing(selected_answer)
    example["texts"] = [{"user": question, "assistant": selected_answer, "source": "textvqa"}]
    return example


ds_textvqa = ds_textvqa.map(
    map_transform_textvqa, remove_columns=ds_textvqa.column_names, features=FEATURES, num_proc=NUM_PROC
)

ds_textvqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/textvqa", num_proc=NUM_PROC)  # 34602 examples


# -------------------------------------------------------------------------------
# --------------------------- TextVQA new prompt ---------------------------------------
# -------------------------------------------------------------------------------


ds_textvqa_new_prompt = load_dataset("textvqa", split="train")


prompt_template_textvqa_new_prompt = """Answer the following question about the image using as few words as possible. Follow these additional instructions:
-Always answer a binary question with Yes or No.
-When asked what time it is, reply with the time seen in the image.
-Do not put any full stops at the end of the answer.
-Do not put quotation marks around the answer.
-An answer with one or two words is favorable.
-Do not apply common sense knowledge. The answer can be found in the image.
Question: {question}"""


def map_transform_textvqa_new_prompt(example):
    example["images"] = [example["image"]]

    # Get the answer(s) with the highest count (agreement among annotators), or a random one if there is a tie.
    answer_counts = {answer: example["answers"].count(answer) for answer in set(example["answers"])}
    max_count = max(answer_counts.values())
    answers_with_max_count = [string for string, count in answer_counts.items() if count == max_count]
    selected_answer = random.choice(answers_with_max_count)

    question = correct_casing(example["question"], is_question=True)
    prompt_user = prompt_template_textvqa_new_prompt.format(question=question)
    example["texts"] = [{"user": prompt_user, "assistant": selected_answer, "source": "textvqa"}]
    return example


ds_textvqa_new_prompt = ds_textvqa_new_prompt.map(
    map_transform_textvqa_new_prompt,
    remove_columns=ds_textvqa_new_prompt.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)

ds_textvqa_new_prompt.save_to_disk(
    "/fsx/hugo/fine_tuning_datasets/textvqa_new_prompt", num_proc=NUM_PROC
)  # 34602 examples


# -------------------------------------------------------------------------------
# --------------------------- OCRVQA ---------------------------------------
# -------------------------------------------------------------------------------

ds_ocrvqa = load_dataset("howard-hou/OCR-VQA", split="train")


def map_transform_ocrvqa(example):
    example["images"] = [example["image"]]

    # Get an answer at random
    example["texts"] = [
        {
            "user": correct_casing(example["questions"][i], is_question=True) + random.choice(QUESTION_BRIEVETY_HINT),
            "assistant": correct_casing(example["answers"][i]),
            "source": "ocrvqa",
        }
        for i in range(len(example["questions"]))
    ]
    return example


ds_ocrvqa = ds_ocrvqa.map(
    map_transform_ocrvqa, remove_columns=ds_ocrvqa.column_names, features=FEATURES, num_proc=NUM_PROC
)
ds_ocrvqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/ocrvqa", num_proc=NUM_PROC)  # 166022 examples


# -------------------------------------------------------------------------------
# --------------------------- DocumentVQA ---------------------------------------
# -------------------------------------------------------------------------------


ds_docvqa = load_dataset("HuggingFaceM4/DocumentVQA", split="train")


def map_transform_docvqa(example):
    example["images"] = [example["image"]]

    # Get an answer at random
    selected_answer = random.choice(example["answers"])
    question = correct_casing(example["question"], is_question=True) + random.choice(QUESTION_BRIEVETY_HINT)
    selected_answer = correct_casing(selected_answer)
    example["texts"] = [{"user": question, "assistant": selected_answer, "source": "docvqa"}]
    return example


ds_docvqa = ds_docvqa.map(
    map_transform_docvqa, remove_columns=ds_docvqa.column_names, features=FEATURES, num_proc=NUM_PROC
)
print(ds_docvqa)
ds_docvqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/docvqa", num_proc=NUM_PROC)  # 39463 examples


# -------------------------------------------------------------------------------
# --------------------------- DocumentVQA new prompt ---------------------------------------
# -------------------------------------------------------------------------------

ds_docvqa_new_prompt = load_dataset("HuggingFaceM4/DocumentVQA", split="train")

prompt_template_docvqa_new_prompt = """Give a short and terse answer to the following question. Do not paraphrase or reformat the text you see in the image. Do not include any full stops. Just give the answer without additional explanation.
Question: {question}"""


def filter_large_images_docvqa_new_prompt(example, max_side=3 * 980):
    width, height = example["image"].size
    if (width > max_side) or (height > max_side):
        return False
    return True


def map_transform_docvqa_new_prompt(example):
    example["images"] = [example["image"]]
    #
    # Get an answer at random
    selected_answer = random.choice(example["answers"])
    prompt_user = prompt_template_docvqa_new_prompt.format(question=example["question"])
    example["texts"] = [{"user": prompt_user, "assistant": selected_answer, "source": "docvqa"}]
    return example


ds_docvqa_new_prompt = ds_docvqa_new_prompt.filter(filter_large_images_docvqa_new_prompt, num_proc=NUM_PROC)

ds_docvqa_new_prompt = ds_docvqa_new_prompt.map(
    map_transform_docvqa_new_prompt,
    remove_columns=ds_docvqa_new_prompt.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)
print(ds_docvqa_new_prompt[0]["texts"][0]["user"])

# OOM with more processes. It also takes tons of time with 16 but after 1 hour of waiting, it works.
ds_docvqa_new_prompt.save_to_disk("/fsx/hugo/fine_tuning_datasets/docvqa_new_prompt", num_proc=16)  # 38_317 examples


# -------------------------------------------------------------------------------
# --------------------------- CLEVR ---------------------------------------
# -------------------------------------------------------------------------------


ds_clevr = load_dataset("HuggingFaceM4/clevr", "default", split="train")


def map_transform_ds_clevr(example):
    example["images"] = [example["image"]]
    question = correct_casing(example["question"], is_question=True) + random.choice(QUESTION_BRIEVETY_HINT)
    answer = correct_casing(example["answer"])
    example["texts"] = [{"user": question, "assistant": answer, "source": "clevr"}]
    return example


ds_clevr = ds_clevr.map(
    map_transform_ds_clevr, remove_columns=ds_clevr.column_names, features=FEATURES, num_proc=NUM_PROC
)
print(ds_clevr)
ds_clevr.save_to_disk("/fsx/hugo/fine_tuning_datasets/clevr", num_proc=NUM_PROC)  # 699989 examples / 263 shards


# -------------------------------------------------------------------------------
# --------------------------- PMC-VQA ---------------------------------------
# -------------------------------------------------------------------------------

# Build full dataset

# FEATURES_DATASET = datasets.Features(
#     {
#         "image": datasets.Image(decode=True),
#         "Figure_path": datasets.Value("string"),
#         "Caption": datasets.Value("string"),
#         "Question": datasets.Value("string"),
#         "Answer": datasets.Value("string"),
#         "Choice_A": datasets.Value("string"),
#         "Choice_B": datasets.Value("string"),
#         "Choice_C": datasets.Value("string"),
#         "Choice_D": datasets.Value("string"),
#         "Answer_label": datasets.Value("string"),
#     }
# )
# _IMAGES_URLS = {"split_1": "https://huggingface.co/datasets/xmcmic/PMC-VQA/resolve/main/images.zip?download=true", "split_2": "https://huggingface.co/datasets/xmcmic/PMC-VQA/resolve/main/images_2.zip?download=true"}
# _ANNOTATIONS_URLS = {"split_1": "https://huggingface.co/datasets/xmcmic/PMC-VQA/resolve/main/train.csv?download=true", "split_2": "https://huggingface.co/datasets/xmcmic/PMC-VQA/resolve/main/train_2.csv?download=true"}
# ds_pmcvqa = load_dataset("xmcmic/PMC-VQA", split="train", streaming=True)
# dl_manager = datasets.DownloadManager()
# image_folders = {k: Path(v)  for k, v in dl_manager.download_and_extract(_IMAGES_URLS).items()}
# annotations = {k: pd.read_csv(v) for k, v  in _ANNOTATIONS_URLS.items()}
# image_folders["split_1"] = image_folders["split_1"] / "images"
# image_folders["split_2"] = image_folders["split_2"] / "figures"

# dict_pmc_dataset = {"image": [], "Figure_path": [], "Caption": [], "Question": [], "Answer": [], "Choice_A": [], "Choice_B": [], "Choice_C": [], "Choice_D": [], "Answer_label": []}
# for split, image_dir in image_folders.items():
#     for i in tqdm(range(len(annotations[split]))):
#         image_path = image_dir / annotations[split]["Figure_path"][i]
#         image = {
#                 "bytes": convert_img_to_bytes(img_path=image_path, format="JPEG"),
#                 "path": None,
#             }
#         if split == "split_1":
#             dict_pmc_dataset["Caption"].append("")
#             dict_pmc_dataset["Answer_label"].append(annotations[split]["Answer_label"][i])
#         else:
#             dict_pmc_dataset["Caption"].append(annotations[split]["Caption"][i])
#             dict_pmc_dataset["Answer_label"].append(annotations[split]["Answer"][i])
#         dict_pmc_dataset["image"].append(image)
#         dict_pmc_dataset["Figure_path"].append(annotations[split]["Figure_path"][i])
#         dict_pmc_dataset["Question"].append(annotations[split]["Question"][i])
#         dict_pmc_dataset["Answer"].append(annotations[split]["Answer"][i])
#         dict_pmc_dataset["Choice_A"].append(annotations[split]["Choice A"][i][1:])
#         dict_pmc_dataset["Choice_B"].append(annotations[split]["Choice B"][i][1:])
#         dict_pmc_dataset["Choice_C"].append(annotations[split]["Choice C"][i][1:])
#         dict_pmc_dataset["Choice_D"].append(annotations[split]["Choice D"][i][1:])

# full_dataset = datasets.Dataset.from_dict(dict_pmc_dataset, features=FEATURES_DATASET)
# full_dataset = full_dataset.filter(lambda example: example['Answer_label'] in ["A", "B", "C", "D"], num_proc=20) # takes off a few outliers that don't have a label
# full_dataset.push_to_hub("HuggingFaceM4/PMC-VQA", private=True)

ds_pmcvqa = load_dataset("HuggingFaceM4/PMC-VQA", split="train")


def map_transform_ds_pmcvqa(example):
    example["images"] = [example["image"]]

    # When there is a caption, add it as context to the question
    if len(example["Caption"]) == 0:
        question = f'{example["Question"].strip()}\n{example["Choice_A"].strip()}\n{example["Choice_B"].strip()}\n{example["Choice_C"].strip()}\n{example["Choice_D"].strip()}'
    else:
        question = f'{example["Caption"].strip()}\n{example["Question"].strip()}\n{example["Choice_A"].strip()}\n{example["Choice_B"].strip()}\n{example["Choice_C"].strip()}\n{example["Choice_D"].strip()}'
    # Alternate between just using the letter and using the letter with the full answer
    if random.random() < 0.5:
        answer = example[f'Choice_{example["Answer_label"]}'].strip()
    else:
        answer = example["Answer_label"]

    example["texts"] = [{"user": question, "assistant": answer, "source": "pmc-vqa"}]
    return example


ds_pmcvqa = ds_pmcvqa.map(
    map_transform_ds_pmcvqa, remove_columns=ds_pmcvqa.column_names, features=FEATURES, num_proc=NUM_PROC
)
print(ds_pmcvqa)
ds_pmcvqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/ds_pmcvqa", num_proc=NUM_PROC)  # 329537 examples


# -------------------------------------------------------------------------------
# --------------------------- ShareGPT4V ---------------------------------------
# -------------------------------------------------------------------------------
# Following instruction from https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/docs/Data.md Not using webdata because of license

# _IMAGES_URLS = { "coco":"http://images.cocodataset.org/zips/train2017.zip", "LAION-CC-SBU-558K": "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip?download=true", "SAM": "/fsx/leo/datasets/SAM/sam_images_share-sft.zip",
#                  "textvqa_trainval_images": "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip", "visual_genome_1": "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip", "visual_genome_2": "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip" }
# ds_sharegpt4v = load_dataset("Lin-Chen/ShareGPT4V", "ShareGPT4V", split="train")
# dl_manager = datasets.DownloadManager()
# image_folders = {k: Path(v) for k, v in dl_manager.download_and_extract(_IMAGES_URLS).items()}
# image_folders["GQA"] = Path("/fsx/hugo/gqa")
# image_folders["textvqa_trainval_images"] = image_folders["textvqa_trainval_images"] / "train_images"
# ds_ocrvqa = load_dataset("howard-hou/OCR-VQA", split="train")
# ocrvqa_image_id_index = {key: idx for idx, key in enumerate(ds_ocrvqa["image_id"])}
# image_folders["ocr_vqa"] = ds_ocrvqa

# dict_sharegpt4v = {"conversations": [], "image": [], "image_path": [], "image_id": []}
# for example in tqdm(ds_sharegpt4v):
#     image = None
#     image_id = example["id"]
#     if "web" or "wikiart" in example["image"]:
#         pass
#     elif "ocr_vqa" in example["image"]:
#         image = image_folders["ocr_vqa"][ocrvqa_image_id_index[image_id]]["image"]
#         image_path = image_folders["ocr_vqa"][ocrvqa_image_id_index[image_id]]["image_url"]
#     else:
#         if "coco" in example["image"]:
#             image_path = image_folders["coco"] / "/".join(example["image"].split("/")[1:])
#         elif "textvqa" in example["image"]:
#             image_path = image_folders["textvqa_trainval_images"] / "/".join(example["image"].split("/")[1:])
#         elif "VG_100K" in example["image"]:
#             image_path = image_folders["visual_genome_1"] / "/".join(example["image"].split("/")[1:])
#         elif "VG_100K_2" in example["image"]:
#             image_path = image_folders["visual_genome_2"] / "/".join(example["image"].split("/")[1:])
#         elif "sa_" in example["image"]:
#             image_path = image_folders["SAM"] / example["image"].split("/")[-1]
#         elif "gqa" in example["image"]:
#             image_path = image_folders["GQA"] / "/".join(example["image"].split("/")[1:])
#         else:
#             image_path = image_folders["LAION-CC-SBU-558K"] / "/".join(example["image"].split("/")[1:])
#         try:
#             image = {"path": None, "bytes": convert_img_to_bytes(img_path=image_path, format="JPEG")}
#         except Exception:
#             pass
#     if image is not None:
#         human = [ex["value"] for ex in example["conversations"] if ex["from"]=="human"]
#         gpt = [ex["value"] for ex in example["conversations"] if ex["from"]=="gpt"]
#         assert len(human) == len(gpt)
#         [
#             {"human": human[i], "gpt": gpt[i], "source": "sharegpt4v"}
#             for i in range(len(human))
#         ]
#         dict_sharegpt4v["conversations"].append(
#                 [
#                 {"human": human[i], "gpt": gpt[i]}
#                 for i in range(len(human))
#             ]
#         )
#         dict_sharegpt4v["image"].append(image)
#         dict_sharegpt4v["image_path"].append(str(image_path))
#         dict_sharegpt4v["image_id"].append(image_id)
#     else:
#         print(f"Image {example['image']} with id: {example['id']} has no file corresponding or is part of wikiart/webimages")

# full_dataset = datasets.Dataset.from_dict(dict_sharegpt4v, features=FEATURES_DATASET)
# full_dataset.push_to_hub("HuggingFaceM4/sharegpt4v-nowebimages", private=True)


ds_sharegpt4v = load_dataset("HuggingFaceM4/sharegpt4v-nowebimages", split="train")


def map_transform_sharegpt4v(example):
    example["images"] = [example["image"]]

    user_questions = [
        ex["human"].replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
        for ex in example["conversations"]
    ]
    assistant_answers = [
        ex["gpt"].replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
        for ex in example["conversations"]
    ]
    example["texts"] = [
        {"user": user_questions[i], "assistant": assistant_answers[i], "source": "sharegpt4v"}
        for i in range(len(user_questions))
    ]
    return example


ds_sharegpt4v = ds_sharegpt4v.map(
    map_transform_sharegpt4v, remove_columns=ds_sharegpt4v.column_names, features=FEATURES, num_proc=NUM_PROC
)
print(ds_sharegpt4v)
ds_sharegpt4v.save_to_disk("/fsx/hugo/fine_tuning_datasets/sharegpt4v", num_proc=NUM_PROC)  # 59027 examples


# -------------------------------------------------------------------------------
# --------------------------- DVQA ---------------------------------------
# -------------------------------------------------------------------------------

# Dataset creation

# FEATURES_DATASET = datasets.Features(
#     {
#         "image": datasets.Image(decode=True),
#         "image_path": datasets.Value("string"),
#         "question_id": datasets.Value("string"),
#         "question": datasets.Value("string"),
#         "answer_bbox": datasets.Sequence(datasets.Value("float32")),
#         "template_id": datasets.Value("string"),
#         "answer": datasets.Value("string"),
#     }
# )

# _IMAGES_PATH = Path("/fsx/leo/datasets/dvqa/images")
# _ANNOTATIONS_PATH = "/fsx/leo/datasets/dvqa/train_qa.json"


# def process_annotation(annotation):
#     image = {"path": None, "bytes": convert_img_to_bytes(img_path=_IMAGES_PATH / annotation["image"] , format="png")}
#     return {
#         "image": image,
#         "image_path": annotation["image"],
#         "question_id": annotation["question_id"],
#         "question": annotation["question"],
#         "answer_bbox": annotation["answer_bbox"],
#         "template_id": annotation["template_id"],
#         "answer": annotation["answer"],
#     }

# def load_annotations(_ANNOTATIONS_PATH):
#     with open(_ANNOTATIONS_PATH, "r", encoding="utf-8") as file:
#         annotations = json.load(file)
#     return annotations
# annotations = load_annotations(_ANNOTATIONS_PATH)
# with Pool() as pool:
#     ds_dvqa_list = list(tqdm(pool.imap(process_annotation, annotations), total=len(annotations)))
# ds_dvqa_dict = {key: [item[key] for item in ds_dvqa_list] for key in ds_dvqa_list[0]}

# full_dataset = datasets.Dataset.from_dict(ds_dvqa_dict, features=FEATURES_DATASET)
# full_dataset.push_to_hub("HuggingFaceM4/DVQA", private=True)


ds_dvqa = load_dataset("HuggingFaceM4/DVQA", split="train")


def map_transform_ds_dvqa(example):
    example["images"] = [example["image"]]
    question = correct_casing(example["question"], is_question=True) + random.choice(QUESTION_BRIEVETY_HINT)
    answer = correct_casing(example["answer"])
    example["texts"] = [{"user": question, "assistant": answer, "source": "dvqa"}]
    return example


ds_dvqa = ds_dvqa.map(map_transform_ds_dvqa, remove_columns=ds_dvqa.column_names, features=FEATURES, num_proc=NUM_PROC)
print(ds_dvqa)
ds_dvqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/dvqa", num_proc=NUM_PROC)  # 2325316 examples


# -------------------------------------------------------------------------------
# --------------------------- PlotQA ---------------------------------------
# -------------------------------------------------------------------------------

# FEATURES_DATASET = datasets.Features(
#     {
#         "image": datasets.Image(decode=True),
#         "image_index": datasets.Value("string"),
#         "qid": datasets.Value("string"),
#         "question_string": datasets.Value("string"),
#         "answer_bbox": [
#             {
#                 "y": datasets.Value("string"),
#                 "x": datasets.Value("string"),
#                 "w": datasets.Value("string"),
#                 "h": datasets.Value("string"),
#             }
#         ],
#         "template": datasets.Value("string"),
#         "answer": datasets.Value("string"),
#         "answer_id": datasets.Value("string"),
#         "type": datasets.Value("string"),
#         "question_id": datasets.Value("string"),
#     }
# )

# _ANNOTATIONS_PATH = "/fsx/leo/datasets/plotqa/qa_pairs_V2.json"
# image_folder = Path("/fsx/leo/datasets/plotqa/png")

# def process_annotation(annotation):
#     try:
#         image = {"path": None, "bytes": convert_img_to_bytes(img_path=image_folder / f'{annotation["image_index"]}.png' , format="png")}
#         if 'answer_bbox' in annotation:
#             if len(annotation["answer_bbox"]) > 0:
#                 if type(annotation["answer_bbox"]) == dict:
#                     answer_bbox = [annotation["answer_bbox"]]
#                 else:
#                     answer_bbox = annotation["answer_bbox"]
#             else:
#                 answer_bbox = []
#         else:
#             answer_bbox = []
#         if 'answer_id' in annotation:
#             answer_id = str(annotation['answer_id'])
#         else:
#             answer_id = "None"
#         return {
#             "image": image,
#             "image_index": str(annotation["image_index"]),
#             "qid": str(annotation["qid"]),
#             "question_string": str(annotation["question_string"]),
#             "answer_bbox": answer_bbox,
#             "template": str(annotation["template"]),
#             "answer": str(annotation["answer"]),
#             "answer_id": str(answer_id),
#             "type": str(annotation["type"]),
#             "question_id": str(annotation["question_id"])
#         }
#     except Exception as e:
#         print(f"Exception  {e} for image: {annotation['image_index']}")
#         pass

# def load_annotations(_ANNOTATIONS_PATH):
#     with open(_ANNOTATIONS_PATH, "r", encoding="utf-8") as file:
#         annotations = json.load(file)["qa_pairs"]
#     return annotations
# annotations = load_annotations(_ANNOTATIONS_PATH)
# with Pool() as pool:
#     ds_plotqa_list = list(tqdm(pool.imap(process_annotation, annotations), total=len(annotations)))
# ds_plotqa_dict = {key: [item[key] for item in ds_plotqa_list] for key in ds_plotqa_list[0]}


# full_dataset = datasets.Dataset.from_dict(ds_plotqa_dict, features=FEATURES_DATASET)
# full_dataset.push_to_hub("HuggingFaceM4/PlotQA", private=True)


# Make multi-turn dataset on with multiple q/a for same image

# FEATURES_MULTI_TURN_DATASET = datasets.Features(
#     {
#         "image": datasets.Image(decode=True),
#         "image_index": datasets.Value("string"),
#         "qids": datasets.Sequence(datasets.Value("string")),
#         "question_strings": datasets.Sequence(datasets.Value("string")),
#         "answer_bboxes": datasets.Sequence([
#             {
#                 "y": datasets.Value("string"),
#                 "x": datasets.Value("string"),
#                 "w": datasets.Value("string"),
#                 "h": datasets.Value("string"),
#             }
#         ]),
#         "templates": datasets.Sequence(datasets.Value("string")),
#         "answers": datasets.Sequence(datasets.Value("string")),
#         "answer_ids": datasets.Sequence(datasets.Value("string")),
#         "types":datasets.Sequence(datasets.Value("string")),
#         "question_ids": datasets.Sequence(datasets.Value("string")),
#     }
# )


# ds_plotqa = load_dataset("HuggingFaceM4/PlotQA", split="train")
# # Make a ds without image but with image_index for multi-turn dialog on same image.
# ds_plot_qa_no_image = ds_plotqa.remove_columns(["image"])

# # Make a dict mapping keys to an index containing the image
# image_index_to_ds_index = {key: idx for idx, key in enumerate(ds_plotqa["image_index"])}

# new_dict_plot_qa = {}
# for i, example in enumerate(tqdm(ds_plot_qa_no_image)):
#     current_image_index = example["image_index"]

#     image_index = example["image_index"]
#     curr_example = [
#         {
#             "qids": example["qid"],
#             "question_strings": example["question_string"],
#             "answer_bboxes": example["answer_bbox"],
#             "templates": example["template"],
#             "answers": example["answer"],
#             "answer_ids": example["answer_id"],
#             "types": example["type"],
#             "question_ids": example["question_id"]
#         }
#     ]
#     new_dict_plot_qa[image_index] = new_dict_plot_qa.get(image_index, []) + curr_example

# print("Finished creating new_dict without images")

# def process_annotation(image_index):
#     try:
#         annotation = {key: [item[key] for item in new_dict_plot_qa[image_index]] for key in new_dict_plot_qa[image_index][0]}
#         annotation["image"] = ds_plotqa[image_index_to_ds_index[image_index]]["image"]
#         annotation["image_index"] = image_index
#         return annotation
#     except Exception as e:
#         print(f"Exception  {e} for image: {image_index}")
#         pass

# with Pool() as pool:
#     ds_plotqa_list = list(tqdm(pool.imap(process_annotation, new_dict_plot_qa), total=len(new_dict_plot_qa)))
# ds_plotqa_dict = {key: [item[key] for item in ds_plotqa_list] for key in ds_plotqa_list[0]}
# full_dataset = datasets.Dataset.from_dict(ds_plotqa_dict, features=FEATURES_MULTI_TURN_DATASET)
# print(full_dataset)
# full_dataset.push_to_hub("HuggingFaceM4/PlotQA-multi-turn", private=True)

ds_plotqa = load_dataset("HuggingFaceM4/PlotQA-multi-turn", split="train")


def map_transform_ds_plotqa(example):
    example["images"] = [example["image"]]
    if random.random() < 0.5:
        first_example = [
            {
                "user": random.choice(QUESTION_BRIEVETY_HINT_MULTI_TURN) + correct_casing(
                    example["question_strings"][0], is_question=True
                ),
                "assistant": correct_casing(format_numbers_in_string(example["answers"][0])),
                "source": "plotqa",
            }
        ]
        example["texts"] = first_example + [
            {
                "user": correct_casing(example["question_strings"][i], is_question=True),
                "assistant": correct_casing(format_numbers_in_string(example["answers"][i])),
                "source": "plotqa",
            }
            for i in range(1, len(example["question_strings"]))
        ]
    else:
        example["texts"] = [
            {
                "user": correct_casing(example["question_strings"][i], is_question=True) + random.choice(
                    QUESTION_BRIEVETY_HINT
                ),
                "assistant": correct_casing(format_numbers_in_string(example["answers"][i])),
                "source": "plotqa",
            }
            for i in range(len(example["question_strings"]))
        ]
    return example


ds_plotqa = ds_plotqa.map(
    map_transform_ds_plotqa, remove_columns=ds_plotqa.column_names, features=FEATURES, num_proc=NUM_PROC
)
print(ds_plotqa)
ds_plotqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/plotqa_occasional_multi_turn_hint", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- VisualMRC-no-bbox ---------------------------------------
# -------------------------------------------------------------------------------

# FEATURES_DATASET = datasets.Features(
#     {
#         "image": datasets.Image(decode=True),
#         "image_path": datasets.Value("string"),
#         "question_id": datasets.Value("string"),
#         "question": datasets.Value("string"),
#         "answer_bbox": datasets.Sequence(datasets.Value("float32")),
#         "template_id": datasets.Value("string"),
#         "answer": datasets.Value("string"),
#     }
# )

# _IMAGES_PATH = Path("/fsx/leo/datasets/visualmrc/VisualMRC_official")
# _ANNOTATIONS_PATH = "/fsx/leo/datasets/visualmrc/VisualMRC_official/annotation.jsonl"


# def load_annotations(file_path):
#     annotations = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             annotation = json.loads(line)
#             annotations.append(annotation)
#     return annotations

# def process_annotation(annotation):
#     image = {"path": None, "bytes": convert_img_to_bytes(img_path=_IMAGES_PATH / annotation["screenshot_filename"] , format="png")}
#     qa_data = [{"crowdId": qa_datapoint["crowdId"], "question": qa_datapoint["question"]["text"], "answer": qa_datapoint["answer"]["text"]} for qa_datapoint in annotation["qa_data"]]
#     return {
#         "image": image,
#         "image_path": annotation["image_filename"],
#         "screenshot_path": annotation["screenshot_filename"],
#         "id": annotation["id"],
#         "url": annotation["url"],
#         "qa_data": qa_data,
#     }

# annotations = load_annotations(_ANNOTATIONS_PATH)
# with Pool() as pool:
#     ds_vismrc_list = list(tqdm(pool.imap(process_annotation, annotations), total=len(annotations)))
# ds_vismrc_dict = {key: [item[key] for item in ds_vismrc_list] for key in ds_vismrc_list[0]}

# full_dataset = datasets.Dataset.from_dict(ds_vismrc_dict, features=FEATURES_DATASET)
# full_dataset.push_to_hub("HuggingFaceM4/VisualMRC-nobbox", private=True)


# def load_annotations(_ANNOTATIONS_PATH):
#     with open(_ANNOTATIONS_PATH, "r", encoding="utf-8") as file:
#         annotations = json.load(file)
#     return annotations
# annotations = load_annotations(_ANNOTATIONS_PATH)
# with Pool() as pool:
#     ds_dvqa_list = list(tqdm(pool.imap(process_annotation, annotations), total=len(annotations)))
# ds_dvqa_dict = {key: [item[key] for item in ds_dvqa_list] for key in ds_dvqa_list[0]}

# full_dataset = datasets.Dataset.from_dict(ds_dvqa_dict, features=FEATURES_DATASET)
# full_dataset.push_to_hub("HuggingFaceM4/DVQA", private=True)


ds_vismrc = load_dataset("HuggingFaceM4/VisualMRC-nobbox", split="train")


def map_transform_ds_vismrc(example):
    example["images"] = [example["image"]]

    example["texts"] = [
        {
            "user": correct_casing(qa_datapoint["question"], is_question=True),
            "assistant": correct_casing(qa_datapoint["answer"]),
            "source": "vismrc",
        }
        for qa_datapoint in example["qa_data"]
    ]
    return example


def filter_fn_ds_vismrc(example):
    width, height = example["image"].size
    if width / height > 4 or height / width > 4:
        return False
    return True


ds_vismrc = ds_vismrc.filter(filter_fn_ds_vismrc, num_proc=NUM_PROC)
ds_vismrc = ds_vismrc.map(
    map_transform_ds_vismrc, remove_columns=ds_vismrc.column_names, features=FEATURES, num_proc=NUM_PROC
)
print(ds_vismrc)
ds_vismrc.save_to_disk("/fsx/hugo/fine_tuning_datasets/visual_mrc", num_proc=NUM_PROC)

# -------------------------------------------------------------------------------
# --------------------------- OpenHermes-2.5 ---------------------------------------
# -------------------------------------------------------------------------------

ds_openhermes = load_dataset("teknium/OpenHermes-2.5", split="train")


def map_transform_openhermes(example):
    example["images"] = [None]
    questions = [ex["value"] for ex in example["conversations"] if ex["from"] == "human"]
    answers = [ex["value"] for ex in example["conversations"] if ex["from"] == "gpt"]
    assert len(questions) == len(answers)

    example["texts"] = [
        {
            "user": correct_casing(questions[i], is_question=True),
            "assistant": correct_casing(answers[i]),
            "source": "openhermes",
        }
        for i in range(len(questions))
    ]
    return example


ds_openhermes = ds_openhermes.map(
    map_transform_openhermes, remove_columns=ds_openhermes.column_names, features=FEATURES, num_proc=NUM_PROC
)
print(ds_openhermes)
ds_openhermes.save_to_disk("/fsx/hugo/fine_tuning_datasets/openhermes", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- LIMA ---------------------------------------
# -------------------------------------------------------------------------------

ds_lima = load_dataset("GAIR/lima", split="train")


def map_transform_lima(example):
    example["images"] = [None]
    questions = [example["conversations"][i] for i in range(len(example["conversations"])) if i % 2 == 0]
    answers = [example["conversations"][i] for i in range(len(example["conversations"])) if i % 2 != 0]

    # Lima is made of conversations. Sometimes, the last text of the conversations comes from the user (the one asking the question in the first place).
    # We remove it because our dialogues end with the assistant's answer.
    if len(questions) != len(answers):
        questions.pop(-1)
    assert len(questions) == len(answers)

    example["texts"] = [
        {
            "user": correct_casing(questions[i], is_question=True),
            "assistant": correct_casing(answers[i]),
            "source": "openhermes",
        }
        for i in range(len(questions))
    ]
    return example


ds_lima = ds_lima.map(map_transform_lima, remove_columns=ds_lima.column_names, features=FEATURES, num_proc=NUM_PROC)
print(ds_lima)
ds_lima.save_to_disk("/fsx/hugo/fine_tuning_datasets/lima", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- TaTQA ---------------------------------------
# -------------------------------------------------------------------------------

ds_tatqa = datasets.load_dataset("HuggingFaceM4/TATQA-rendered-tables", split="train")


def map_transform_ds_tatqa(example):
    example["images"] = [example["image"]]
    user_texts = [
        (
            example["questions"][i] + f'\nAnswer scale should be: {example["scales"][i]}.'
            if len(example["scales"][i]) > 0
            else example["questions"][i]
        )
        for i in range(1, len(example["questions"]))
    ]

    first_text = (
        "\n".join(example["paragraphs"])
        + "\n"
        + example["questions"][0]
        + f'\nAnswer scale should be: {example["scales"][0]}.'
        if len(example["scales"][0]) > 0
        else "\n".join(example["paragraphs"]) + "\n" + example["questions"][0]
    )
    user_texts = [first_text] + user_texts
    assistant_texts = [
        (
            example["derivations"][i] + "\n" + "Answer: " + example["answers"][i]
            if len(example["derivations"][i]) > 0
            else example["answers"][i]
        )
        for i in range(len(example["answers"]))
    ]
    example["texts"] = [
        {
            "user": correct_casing(user_text, is_question=True),
            "assistant": correct_casing(assistant_text, is_question=False),
            "source": "tatqa",
        }
        for user_text, assistant_text in zip(user_texts, assistant_texts)
    ]
    return example


ds_tatqa = ds_tatqa.map(map_transform_ds_tatqa, remove_columns=ds_tatqa.column_names, features=FEATURES, num_proc=10)
print(ds_tatqa)
ds_tatqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/tat_qa", num_proc=10)


# -------------------------------------------------------------------------------
# --------------------------- Robut wikisql ---------------------------------------
# -------------------------------------------------------------------------------


parse_table_instruct = [
    "Can you parse all the data within this table?",
    "Parse the table in full.",
    "Parse the full table.",
    "Could you parse the entire table?",
    "Would you be able to parse every entry in this table?",
    "Help me parse the entirety of this table.",
    "Would you mind parsing the complete table?",
    "Write the full table.",
    "Can you give me this table as a dict?",
    "Give me the full table as a dictionary.",
    "Could you parse the entire table as a dict?",
    "I'm looking to parse the entire table for insights. Could you assist me with that?",
    "Could you help me parse every detail presented in this table?",
]

ds_robut_wikisql = datasets.load_dataset("HuggingFaceM4/ROBUT-wikisql-rendered-tables", split="train")


def map_transform_ds_robut_wikisql(example):
    example["images"] = [example["image"]]
    assistant_texts = ", ".join(example["answers"])
    example["texts"] = [
        {
            "user": correct_casing(example["question"], is_question=True),
            "assistant": correct_casing(assistant_texts, is_question=False),
            "source": "robut_wikisql",
        }
    ]
    if random.random() < 0.15:
        parse_table_instruction = [
            {
                "user": random.choice(parse_table_instruct),
                "assistant": str(example["table"]),
                "source": "robut_wikisql",
            }
        ]
        if random.random() > 0.5:
            example["texts"] = example["texts"] + parse_table_instruction
        else:
            example["texts"] = parse_table_instruction + example["texts"]
    return example


ds_robut_wikisql = ds_robut_wikisql.map(
    map_transform_ds_robut_wikisql, remove_columns=ds_robut_wikisql.column_names, features=FEATURES, num_proc=NUM_PROC
)
print(ds_robut_wikisql)
ds_robut_wikisql.save_to_disk("/fsx/hugo/fine_tuning_datasets/robut_wikisql", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- Robut wtqa ---------------------------------------
# -------------------------------------------------------------------------------

parse_table_instruct = [
    "Can you parse all the data within this table?",
    "Parse the table in full.",
    "Parse the full table.",
    "Could you parse the entire table?",
    "Would you be able to parse every entry in this table?",
    "Help me parse the entirety of this table.",
    "Would you mind parsing the complete table?",
    "Write the full table.",
    "Can you give me this table as a dict?",
    "Give me the full table as a dictionary.",
    "Could you parse the entire table as a dict?",
    "I'm looking to parse the entire table for insights. Could you assist me with that?",
    "Could you help me parse every detail presented in this table?",
]

ds_robut_wtq = datasets.load_dataset("HuggingFaceM4/ROBUT-wtq-rendered-tables", split="train")


def map_transform_ds_robut_wtq(example):
    example["images"] = [example["image"]]
    assistant_texts = ", ".join(example["answers"])
    example["texts"] = [
        {
            "user": correct_casing(example["question"], is_question=True),
            "assistant": correct_casing(assistant_texts, is_question=False),
            "source": "robut_wtq",
        }
    ]
    if random.random() < 0.15:
        parse_table_instruction = [
            {"user": random.choice(parse_table_instruct), "assistant": str(example["table"]), "source": "robut_wtq"}
        ]
        if random.random() > 0.5:
            example["texts"] = example["texts"] + parse_table_instruction
        else:
            example["texts"] = parse_table_instruction + example["texts"]
    return example


ds_robut_wtq = ds_robut_wtq.map(
    map_transform_ds_robut_wtq, remove_columns=ds_robut_wtq.column_names, features=FEATURES, num_proc=NUM_PROC
)
print(ds_robut_wtq)
ds_robut_wtq.save_to_disk("/fsx/hugo/fine_tuning_datasets/robut_wtq", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- FinQA ---------------------------------------
# -------------------------------------------------------------------------------


def translate_nested_expression(expr):
    symbols = {"divide": "/", "multiply": "*", "add": "+", "subtract": "-", "greater": ">", "less": "<", "exp": "**"}
    expr = re.sub(r"const_(\d+)", r"\1", expr)

    def parse(e):
        depth = 0
        for i, char in enumerate(e):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 1:
                op_end = e.find("(")
                op = e[:op_end]
                return f"({parse(e[op_end+1:i].strip())} {symbols[op]} {parse(e[i+1:-1].strip())})"
        return e

    for op in symbols:
        if op in expr:
            return parse(expr)
    return expr


# only happens with finqa and not sure we want to apply to other datasets, so I'm making this just for this one
def correct_casing_finqa(text, is_question=False, is_context=False):
    if text and text[0].islower():
        text = text.capitalize()
    if not is_context:
        if not text.endswith(".") and not is_question:
            text += "."
        if not text.endswith("?") and is_question:
            text += "?"
    return text


ds_finqa = datasets.load_dataset("HuggingFaceM4/FINQA-rendered-tables", split="train")


def map_transform_ds_finqa(example):
    example["images"] = [example["image"]]
    example["pre_text"] = [
        correct_casing_finqa(pre_text, is_question=False, is_context=True).replace("following table", "table")
        for pre_text in example["pre_text"]
        if len(pre_text) > 3
    ]
    example["post_text"] = [
        correct_casing_finqa(post_text, is_question=False, is_context=True)
        for post_text in example["post_text"]
        if len(post_text) > 3
    ]
    context = "\n".join(example["pre_text"]) + "\n" + "\n".join(example["post_text"])
    user_text = context + "\n" + correct_casing_finqa(example["question"], is_question=True)

    rationale = ""
    if len(example["explanation"]) > 0:
        rationale = "\nRationale: " + example["explanation"]

    if len(example["program_re"]) > 0:
        # edge case where the program is not in nested format
        if "table" not in example["program_re"]:
            translated_program = translate_nested_expression(example["program_re"])
        else:
            translated_program = example["program_re"]
    assistant_text = rationale + f"\nComputations: {translated_program}" + f"\nAnswer: {example['answer']}"

    example["texts"] = [
        {
            "user": correct_casing_finqa(user_text, is_question=True),
            "assistant": correct_casing_finqa(assistant_text, is_question=False),
            "source": "finqa",
        }
    ]
    return example


ds_finqa = ds_finqa.map(
    map_transform_ds_finqa, remove_columns=ds_finqa.column_names, features=FEATURES, num_proc=NUM_PROC
)
print(ds_finqa)
ds_finqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/finqa", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- MultiHiertt ---------------------------------------
# -------------------------------------------------------------------------------


ds_multihiertt = datasets.load_dataset("HuggingFaceM4/MultiHiertt-rendered-tables", split="train")


def map_transform_ds_multihiertt(example):
    user_text = example["question"]
    assistant_text = ""
    if len(example["translated_program"]) > 0:
        assistant_text += f"\nComputations: {example['translated_program']}"
    assistant_text += f"\nAnswer: {example['answer']}"
    example["texts"] = [
        {"user": user_text, "assistant": correct_casing(assistant_text, is_question=False), "source": "multihiertt"}
    ]
    return example


# don't take off images column
columns_to_remove = ds_multihiertt.column_names[1:]
ds_multihiertt = ds_multihiertt.map(
    map_transform_ds_multihiertt, remove_columns=columns_to_remove, features=FEATURES, num_proc=10
)
print(ds_multihiertt)
ds_multihiertt.save_to_disk("/fsx/hugo/fine_tuning_datasets/multihiertt", num_proc=10)


# -------------------------------------------------------------------------------
# --------------------------- HiTab ---------------------------------------
# -------------------------------------------------------------------------------

parse_table_instruct = [
    "Can you parse all the data within this table?",
    "Parse the table in full.",
    "Parse the full table.",
    "Could you parse the entire table?",
    "Would you be able to parse every entry in this table?",
    "Help me parse the entirety of this table.",
    "Would you mind parsing the complete table?",
    "Write the full table.",
    "Can you give me this table as a dict?",
    "Give me the full table as a dictionary.",
    "Could you parse the entire table as a dict?",
    "I'm looking to parse the entire table for insights. Could you assist me with that?",
    "Could you help me parse every detail presented in this table?",
]

ds_hitab = datasets.load_dataset("HuggingFaceM4/HiTab-rendered-tables", split="train")


def map_transform_ds_finqa(example):
    example["images"] = [example["image"]]
    user_texts = [user_text for user_text in example["questions"]]
    assistant_texts = [assistant_text for assistant_text in example["answers"]]
    example["texts"] = [
        {
            "user": correct_casing(user_text, is_question=True),
            "assistant": correct_casing(assistant_text, is_question=False),
            "source": "hitab",
        }
        for user_text, assistant_text in zip(user_texts, assistant_texts)
    ]

    if random.random() < 0.15:
        parse_table_instruction = [
            {"user": random.choice(parse_table_instruct), "assistant": str(example["table"]), "source": "hitab"}
        ]
        if random.random() > 0.5:
            example["texts"] = example["texts"] + parse_table_instruction
        else:
            example["texts"] = parse_table_instruction + example["texts"]
    return example


ds_hitab = ds_hitab.map(
    map_transform_ds_finqa, remove_columns=ds_hitab.column_names, features=FEATURES, num_proc=NUM_PROC
)
print(ds_hitab)
ds_hitab.save_to_disk("/fsx/hugo/fine_tuning_datasets/hitab", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- Chart2Text ----------------------------------------
# -------------------------------------------------------------------------------

chart2text_instruct = [
    "Explain what this graph is communicating.",
    "Can you elaborate on the message conveyed by this graph?",
    "Please clarify the meaning conveyed by this graph.",
    "What is the main idea being communicated through this graph?",
    "Can you break down the data visualization and explain its message?",
    "Could you shed some light on the insights conveyed by this graph?",
    "What conclusions can be drawn from the information depicted in this graph?",
    "Please describe the key points or trends indicated by this graph.",
    "I'd like to understand the message this graph is trying to highlight.",
]

ds_chart2text_statista = datasets.load_dataset("heegyu/chart2text_statista", split="train")
ds_chart2text_pew = datasets.load_dataset("heegyu/chart2text_pew", split="train")


def map_transform_chart2text(example, config):
    example["images"] = [example["image"]]
    example["texts"] = [
        {
            "user": random.choice(chart2text_instruct),
            "assistant": correct_casing(
                example["first_caption"].strip() if config == "statista" else example["caption"].strip(),
                is_question=False,
            ),
            "source": f"chart2text-{config}",
        }
    ]
    return example


ds_chart2text_statista = ds_chart2text_statista.map(
    partial(map_transform_chart2text, config="statista"),
    remove_columns=ds_chart2text_statista.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)
ds_chart2text_pew = ds_chart2text_pew.map(
    partial(map_transform_chart2text, config="pew"),
    remove_columns=ds_chart2text_pew.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)
ds_chart2text = concatenate_datasets([ds_chart2text_statista, ds_chart2text_pew])
ds_chart2text = ds_chart2text.filter(lambda example: "â€" not in example["texts"][0]["assistant"], num_proc=NUM_PROC)
print(ds_chart2text)
ds_chart2text.save_to_disk("/fsx/hugo/fine_tuning_datasets/chart2text", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- Robut sqa ---------------------------------------
# -------------------------------------------------------------------------------

parse_table_instruct = [
    "Can you parse all the data within this table?",
    "Parse the table in full.",
    "Parse the full table.",
    "Parse the full table in json format.",
    "Could you parse the entire table?",
    "Would you be able to parse every entry in this table?",
    "Help me parse the entirety of this table.",
    "Would you mind parsing the complete table?",
    "Write the full table.",
    "Can you give me this table as a dict?",
    "Can you give me this table in json format?",
    "Give me the full table as a dictionary.",
    "Could you parse the entire table as a dict?",
    "I'm looking to parse the entire table for insights. Could you assist me with that?",
    "Could you help me parse every detail presented in this table?",
]

ds_robut_sqa = datasets.load_dataset(
    "HuggingFaceM4/ROBUT-sqa-rendered-tables", download_mode="force_redownload", split="train"
)


def map_transform_ds_robut_sqa(example):
    example["images"] = [example["image"]]
    example["texts"] = [
        {
            "user": correct_casing(user_text, is_question=True),
            "assistant": correct_casing(assistant_text, is_question=False),
            "source": "robut_sqa",
        }
        for user_text, assistant_text in zip(example["questions"], example["answers"])
    ]
    if random.random() < 0.10:
        parse_table_instruction = {
            "user": random.choice(parse_table_instruct),
            "assistant": str(example["table"]),
            "source": "robut_sqa",
        }
        random_index = random.randint(0, len(example["texts"]))
        example["texts"].insert(random_index, parse_table_instruction)

    return example


ds_robut_sqa = ds_robut_sqa.map(
    map_transform_ds_robut_sqa, remove_columns=ds_robut_sqa.column_names, features=FEATURES, num_proc=NUM_PROC
)
print(ds_robut_sqa)
ds_robut_sqa.save_to_disk("/fsx/hugo/fine_tuning_datasets/robut_sqa", num_proc=NUM_PROC)

# -------------------------------------------------------------------------------
# --------------------------- ny_cc_explanation ---------------------------------------
# -------------------------------------------------------------------------------


ask_for_captionning_instruct = [
    "How would you encapsulate the essence of this scene in a few words?",
    "What words spring to mind when you gaze upon this image?",
    "In what way does this image speak to you, and how would you convey that in a caption?",
    "Imagine you're telling a friend about this image without showing it to them. What would you say?",
    "How would you frame this scene with your words?",
    "Can you weave a brief narrative or description for this picture?",
    "Describe this image in a caption.",
    "Write a caption that details what you see.",
    "Caption this picture with its visible elements.",
    "Provide a caption describing this scene.",
    "Sum up this image in a descriptive caption.",
    "Give a detailed caption of this cartoon.",
    "Describe the scene depicted here in a caption.",
    "How would you caption this to describe its contents?",
    "Create a caption that tells us what's in this image.",
    "Caption this image, focusing on its details.",
    "Write a caption for this cartoon.",
    "How would you caption this?",
    "Can you come up with a caption here?",
    "What caption fits this cartoon best?",
    "Caption this cartoon, please.",
    "Need a caption. Any ideas?",
    "What's your caption for this?",
    "Give this cartoon a caption.",
    "Be the cartoon's voice: Craft a proper caption for it.",
]
ask_for_description_of_uncannyness = [
    "What makes this image uncanny in a few words?",
    "Spot the uncanny detail. Can you describe it briefly?",
    "In a sentence, how is this image unsettling?",
    "Quickly, what's the eerie part of this picture?",
    "Quick take: why is this image offbeat?",
    "Describe the surreal element quickly.",
    "What gives this a peculiar feel?",
    "Spot the anomaly. What is it?",
    "Briefly, what makes this not quite right?",
    "Describe the odd vibe in a few words.",
    "What seems most unusual to you in this image?",
    "Can you sum up the image's weirdness?",
    "In short, why does this feel otherworldly?",
    "What's the quick uncanny summary of this image?",
]

ask_for_lengthy_explanation = [
    "Can you unravel the layers of humor in this cartoon and explain what makes it so funny in detail?",
    (
        "Take a moment to dissect the joke presented in this image. What elements contribute to its comedic effect,"
        " and why?"
    ),
    (
        "What's the underlying joke in this cartoon, and how do the visual and textual elements come together to make"
        " it humorous? Please explain in depth."
    ),
    (
        "Could you provide a detailed analysis of the humor in this image? Consider the characters, setting, and any"
        " societal or cultural references."
    ),
    "How does this cartoon play with expectations to create humor? Please elaborate on the mechanics of its joke.",
    (
        "In your view, what makes this image comically effective? Dive into the nuances that might not be immediately"
        " obvious."
    ),
    (
        "Can you explain the joke in this cartoon, considering both the immediate laugh and any deeper, more subtle"
        " layers of humor?"
    ),
    "What comedic techniques are employed in this image? Analyze how these elements work together to produce humor.",
    (
        "This cartoon seems to be telling a joke on multiple levels. Can you break down these layers and discuss how"
        " they interact?"
    ),
    (
        "There's a lot going on in this humorous image. Could you give a comprehensive explanation of the joke,"
        " including its context and delivery?"
    ),
    "Explain the joke here in detail, please.",
    "Why is this funny? Give a full explanation.",
    "Detail the humor in this cartoon.",
    "What makes this humorous? Elaborate.",
    "Dissect the joke's layers here.",
    "How does this achieve its humor? Explain.",
    "Unpack the comedy in this image, please.",
    "Analyze the humor's construction here.",
    "Why does this image elicit laughter? Detail.",
]

ask_for_location_of_scene_in_image = {
    "Please provide the specific location of the scene depicted in the image, using the template: Location of scene: {}.": (
        "Location of scene: "
    ),
    "Identify the setting where the scene unfolds and answer using the template: Scene takes place in: {}.": (
        "Scene takes place in: "
    ),
    "What is the setting of this image? Answer using the template: Setting: {}.": "Setting: ",
    "Pinpoint the exact location shown in this image and use the template: Identify the location: {} for your answer.": (
        "Identify the location: "
    ),
    "Describe the specific place where this scene is set, using the format: Where is this scene set? Location: {}.": (
        "Where is this scene set? Location: "
    ),
    "What is the backdrop of this image? Please answer with the template: Backdrop of this image: {}.": (
        "Backdrop of this image: "
    ),
    "Specify the location of the scene, answering with: Scene location: {}.": "Scene location: ",
    "Where is the scene in this image taking place? Use the template: This image's setting is in: {} for your response.": (
        "This image's setting is in: "
    ),
    "What place is being shown? Respond using the template: Place of action: {}.": "Place of action: ",
    "Give the setting or location shown in the image using the template: Image setting: {} for your reply.": (
        "Image setting: "
    ),
}
ask_for_funny_caption = [
    "Craft a witty caption for this scene.",
    "What's a hilarious one-liner that sums up this cartoon?",
    "Give this image a caption that would make someone chuckle.",
    "Can you concoct a funny caption that's brief yet brilliant?",
    "What punchline best captures the humor of this scene?",
    "Sum up this cartoon with a humorous caption. Keep it short!",
    "What's a clever, comedic caption for this image?",
    "Inject some humor into this picture with a short caption.",
    "How would you give this scene a comedic twist with words?",
    "What quick, funny caption would you give this cartoon?",
    "In a sentence, how would you add a comic spin to this scene?",
    "Dream up a caption that’s as funny as it is succinct.",
    "With minimal words, how can you turn this image into a joke?",
    "What witty remark instantly comes to mind for this cartoon?",
    "Create a caption that delivers a laugh in under five words.",
    "What’s the funniest observation you can make about this image?",
    "How would you caption this for maximum comedic effect?",
    "What cheeky caption fits this scene perfectly?",
    "Can you sum up the humor here with a clever quip?",
    "What’s a short and snappy caption that would make this cartoon pop?",
    "Be the cartoon's voice: Craft a funny caption for it.",
]
ask_for_questions = [
    "What question(s) could you ask to uncover the story behind this picture?",
    "What question(s) could help decipher the humor or message in this cartoon?",
    "What question(s) might you ask to understand the context better?",
    "Based on this scene, what question(s) could lead to insights about the underlying joke or satire?",
    "What question(s) do you think this image begs the viewer to consider?",
    "What question(s) do you have after seeing this cartoon?",
    "Looking at this image, what do you want to ask about it?",
    "What comes to mind that you'd question about this scene?",
    "What are you curious to ask when you see this picture?",
    "Seeing this, what question(s) would you pose?",
    "What inquiries does this image provoke in you?",
    "After viewing this cartoon, what would you like to know?",
    "What question(s) does this picture raise for you?",
    "Upon observing this image, what do you find yourself questioning?",
]

ds_ny_cc_explanation = datasets.load_dataset("jmhessel/newyorker_caption_contest", "explanation", split="train")


def map_transform_ds_ny_cc_explanation(example):
    example["images"] = [example["image"]]
    funny_caption = (random.choice(ask_for_funny_caption), example["caption_choices"])
    serious_caption = (random.choice(ask_for_captionning_instruct), example["image_description"])
    description_of_uncannyness = (
        random.choice(ask_for_description_of_uncannyness),
        example["image_uncanny_description"],
    )
    explanation = (random.choice(ask_for_lengthy_explanation), example["label"])
    questions = (random.choice(ask_for_questions), " ".join(example["questions"]))
    location = example["image_location"] + "."
    random_location_key = random.choice(list(ask_for_location_of_scene_in_image.keys()))
    location = (random_location_key, ask_for_location_of_scene_in_image[random_location_key] + location)

    all_texts = [funny_caption, serious_caption, description_of_uncannyness, questions, location, explanation]
    random.shuffle(all_texts)
    example["texts"] = [
        {"user": user_text, "assistant": assistant_text, "source": "ny_cc_explanation"}
        for (user_text, assistant_text) in all_texts
    ]

    return example


ds_ny_cc_explanation = ds_ny_cc_explanation.map(
    map_transform_ds_ny_cc_explanation,
    remove_columns=ds_ny_cc_explanation.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)
print(ds_ny_cc_explanation)
ds_ny_cc_explanation.save_to_disk("/fsx/hugo/fine_tuning_datasets/ny_cc_explanation", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- ny_cc_matching ---------------------------------------
# -------------------------------------------------------------------------------

ask_for_captionning_instruct = [
    "How would you encapsulate the essence of this scene in a few words?",
    "What words spring to mind when you gaze upon this image?",
    "In what way does this image speak to you, and how would you convey that in a caption?",
    "Imagine you're telling a friend about this image without showing it to them. What would you say?",
    "How would you frame this scene with your words?",
    "Can you weave a brief narrative or description for this picture?",
    "Describe this image in a caption.",
    "Write a caption that details what you see.",
    "Caption this picture with its visible elements.",
    "Provide a caption describing this scene.",
    "Sum up this image in a descriptive caption.",
    "Give a detailed caption of this cartoon.",
    "Describe the scene depicted here in a caption.",
    "How would you caption this to describe its contents?",
    "Create a caption that tells us what's in this image.",
    "Caption this image, focusing on its details.",
    "Write a caption for this cartoon.",
    "How would you caption this?",
    "Can you come up with a caption here?",
    "What caption fits this cartoon best?",
    "Caption this cartoon, please.",
    "Need a caption. Any ideas?",
    "What's your caption for this?",
    "Give this cartoon a caption.",
    "Be the cartoon's voice: Craft a proper caption for it.",
]
ask_for_description_of_uncannyness = [
    "What makes this image uncanny in a few words?",
    "Spot the uncanny detail. Can you describe it briefly?",
    "In a sentence, how is this image unsettling?",
    "Quickly, what's the eerie part of this picture?",
    "Quick take: why is this image offbeat?",
    "Describe the surreal element quickly.",
    "What gives this a peculiar feel?",
    "Spot the anomaly. What is it?",
    "Briefly, what makes this not quite right?",
    "Describe the odd vibe in a few words.",
    "What seems most unusual to you in this image?",
    "Can you sum up the image's weirdness?",
    "In short, why does this feel otherworldly?",
    "What's the quick uncanny summary of this image?",
]

ask_for_location_of_scene_in_image = {
    "Please provide the specific location of the scene depicted in the image, using the template: Location of scene: {}.": (
        "Location of scene: "
    ),
    "Identify the setting where the scene unfolds and answer using the template: Scene takes place in: {}.": (
        "Scene takes place in: "
    ),
    "What is the setting of this image? Answer using the template: Setting: {}.": "Setting: ",
    "Pinpoint the exact location shown in this image and use the template: Identify the location: {} for your answer.": (
        "Identify the location: "
    ),
    "Describe the specific place where this scene is set, using the format: Where is this scene set? Location: {}.": (
        "Where is this scene set? Location: "
    ),
    "What is the backdrop of this image? Please answer with the template: Backdrop of this image: {}.": (
        "Backdrop of this image: "
    ),
    "Specify the location of the scene, answering with: Scene location: {}.": "Scene location: ",
    "Where is the scene in this image taking place? Use the template: This image's setting is in: {} for your response.": (
        "This image's setting is in: "
    ),
    "What place is being shown? Respond using the template: Place of action: {}.": "Place of action: ",
    "Give the setting or location shown in the image using the template: Image setting: {} for your reply.": (
        "Image setting: "
    ),
}
ask_which_caption_fits_the_image = [
    "Which caption accurately fits this image?",
    "Out of the options, which caption is right for this cartoon?",
    "Which one of these captions matches the image correctly?",
    "Select the caption that actually describes this scene.",
    "From the given captions, which one belongs to this image?",
    "Identify the correct caption for this picture.",
    "Which caption is the true fit for this image?",
    "Choose the caption that properly fits this cartoon.",
    "Out of the captions, which one is the appropriate one?",
    "Determine the fitting caption for this image.",
    "Which caption captures the unique humor of this scene accurately?",
    "Given the quirky details in this cartoon, which caption truly fits?",
    "Considering the twist in this image, which caption is the right one?",
    "Looking at the comedic elements, which caption aligns with the scene?",
    "Assessing the irony in this picture, which caption actually describes it?",
    "In light of the character's predicament, which caption is spot-on?",
    "Reflecting on the absurdity showcased, which caption suits best?",
    "Given the punchline hidden in this image, which caption reveals it correctly?",
    "Observing the joke unfolding, which caption does justice to this cartoon?",
    "Considering the unexpected turn in this scene, which caption fits perfectly?",
]

ask_for_questions = [
    "What question(s) could you ask to uncover the story behind this picture?",
    "What question(s) could help decipher the humor or message in this cartoon?",
    "What question(s) might you ask to understand the context better?",
    "Based on this scene, what question(s) could lead to insights about the underlying joke or satire?",
    "What question(s) do you think this image begs the viewer to consider?",
    "What question(s) do you have after seeing this cartoon?",
    "Looking at this image, what do you want to ask about it?",
    "What comes to mind that you'd question about this scene?",
    "What are you curious to ask when you see this picture?",
    "Seeing this, what question(s) would you pose?",
    "What inquiries does this image provoke in you?",
    "After viewing this cartoon, what would you like to know?",
    "What question(s) does this picture raise for you?",
    "Upon observing this image, what do you find yourself questioning?",
]

ask_answer_with_letter = [
    "Please indicate your choice by answering with the letter corresponding to the correct option.",
    "Select the most appropriate option and respond with its letter only.",
    "Which option is correct? Reply with the letter of your answer.",
    "Identify your selection by providing the letter that represents it.",
    "For your answer, use the letter that matches your choice.",
    "What's your pick? Answer with the letter that applies.",
    "Choose the best match and reply with its associated letter.",
    "Can you pinpoint the right choice by letter?",
    "Which letter best represents your answer to this question?",
    "Respond with the letter that aligns with your selection.",
    "Answer with the letter.",
]
ds_ny_cc_matching = datasets.load_dataset("jmhessel/newyorker_caption_contest", "matching", split="train")


def map_transform_ds_ny_cc_matching(example):
    example["images"] = [example["image"]]
    serious_caption = (random.choice(ask_for_captionning_instruct), example["image_description"])
    description_of_uncannyness = (
        random.choice(ask_for_description_of_uncannyness),
        example["image_uncanny_description"],
    )
    questions = (random.choice(ask_for_questions), " ".join(example["questions"]))
    location = example["image_location"] + "."
    random_location_key = random.choice(list(ask_for_location_of_scene_in_image.keys()))
    location = (random_location_key, ask_for_location_of_scene_in_image[random_location_key] + location)
    choices = ["A", "B", "C", "D", "E"]
    punctuation_choice = random.choice([".", ")", ":", ""])
    caption_choices_question = random.choice(ask_which_caption_fits_the_image)
    caption_choices = [(choice, choice_caption) for choice, choice_caption in zip(choices, example["caption_choices"])]
    for choice, caption in caption_choices:
        caption_choices_question += f"\n{choice}{punctuation_choice} {caption}"
    caption_choices_question += f"\n{random.choice(ask_answer_with_letter)}"
    if random.random() < 0.5:
        caption_choice = (caption_choices_question, "Answer: " + example["label"])
    else:
        caption_choice = (caption_choices_question, example["label"])

    all_texts = [serious_caption, description_of_uncannyness, questions, location, caption_choice]
    random.shuffle(all_texts)
    example["texts"] = [
        {"user": user_text, "assistant": assistant_text, "source": "ny_cc_matching"}
        for (user_text, assistant_text) in all_texts
    ]

    return example


ds_ny_cc_matching = ds_ny_cc_matching.map(
    map_transform_ds_ny_cc_matching,
    remove_columns=ds_ny_cc_matching.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)
print(ds_ny_cc_matching)
ds_ny_cc_matching.save_to_disk("/fsx/hugo/fine_tuning_datasets/ny_cc_matching", num_proc=NUM_PROC)


# -------------------------------------------------------------------------------
# --------------------------- ny_cc_ranking ---------------------------------------
# -------------------------------------------------------------------------------


ask_for_captionning_instruct = [
    "How would you encapsulate the essence of this scene in a few words?",
    "What words spring to mind when you gaze upon this image?",
    "In what way does this image speak to you, and how would you convey that in a caption?",
    "Imagine you're telling a friend about this image without showing it to them. What would you say?",
    "How would you frame this scene with your words?",
    "Can you weave a brief narrative or description for this picture?",
    "Describe this image in a caption.",
    "Write a caption that details what you see.",
    "Caption this picture with its visible elements.",
    "Provide a caption describing this scene.",
    "Sum up this image in a descriptive caption.",
    "Give a detailed caption of this cartoon.",
    "Describe the scene depicted here in a caption.",
    "How would you caption this to describe its contents?",
    "Create a caption that tells us what's in this image.",
    "Caption this image, focusing on its details.",
    "Write a caption for this cartoon.",
    "How would you caption this?",
    "Can you come up with a caption here?",
    "What caption fits this cartoon best?",
    "Caption this cartoon, please.",
    "Need a caption. Any ideas?",
    "What's your caption for this?",
    "Give this cartoon a caption.",
    "Be the cartoon's voice: Craft a proper caption for it.",
]
ask_for_description_of_uncannyness = [
    "What makes this image uncanny in a few words?",
    "Spot the uncanny detail. Can you describe it briefly?",
    "In a sentence, how is this image unsettling?",
    "Quickly, what's the eerie part of this picture?",
    "Quick take: why is this image offbeat?",
    "Describe the surreal element quickly.",
    "What gives this a peculiar feel?",
    "Spot the anomaly. What is it?",
    "Briefly, what makes this not quite right?",
    "Describe the odd vibe in a few words.",
    "What seems most unusual to you in this image?",
    "Can you sum up the image's weirdness?",
    "In short, why does this feel otherworldly?",
    "What's the quick uncanny summary of this image?",
]

ask_for_location_of_scene_in_image = {
    "Please provide the specific location of the scene depicted in the image, using the template: Location of scene: {}.": (
        "Location of scene: "
    ),
    "Identify the setting where the scene unfolds and answer using the template: Scene takes place in: {}.": (
        "Scene takes place in: "
    ),
    "What is the setting of this image? Answer using the template: Setting: {}.": "Setting: ",
    "Pinpoint the exact location shown in this image and use the template: Identify the location: {} for your answer.": (
        "Identify the location: "
    ),
    "Describe the specific place where this scene is set, using the format: Where is this scene set? Location: {}.": (
        "Where is this scene set? Location: "
    ),
    "What is the backdrop of this image? Please answer with the template: Backdrop of this image: {}.": (
        "Backdrop of this image: "
    ),
    "Specify the location of the scene, answering with: Scene location: {}.": "Scene location: ",
    "Where is the scene in this image taking place? Use the template: This image's setting is in: {} for your response.": (
        "This image's setting is in: "
    ),
    "What place is being shown? Respond using the template: Place of action: {}.": "Place of action: ",
    "Give the setting or location shown in the image using the template: Image setting: {} for your reply.": (
        "Image setting: "
    ),
}


ask_for_questions = [
    "What question(s) could you ask to uncover the story behind this picture?",
    "What question(s) could help decipher the humor or message in this cartoon?",
    "What question(s) might you ask to understand the context better?",
    "Based on this scene, what question(s) could lead to insights about the underlying joke or satire?",
    "What question(s) do you think this image begs the viewer to consider?",
    "What question(s) do you have after seeing this cartoon?",
    "Looking at this image, what do you want to ask about it?",
    "What comes to mind that you'd question about this scene?",
    "What are you curious to ask when you see this picture?",
    "Seeing this, what question(s) would you pose?",
    "What inquiries does this image provoke in you?",
    "After viewing this cartoon, what would you like to know?",
    "What question(s) does this picture raise for you?",
    "Upon observing this image, what do you find yourself questioning?",
]

ask_for_2_captions = [
    "Create two witty captions that capture the essence of this cartoon's humor.",
    "Provide two clever captions that play off the irony in this image.",
    "Generate two humorous captions that reflect the characters' predicament in this scene.",
    "Conjure up two funny captions that make light of the unexpected twist depicted here.",
    "Invent two witty captions that could accompany this cartoon, highlighting the absurdity.",
    "Formulate two clever captions that would suit the comedic tone of this image.",
    "Devise two humorous captions that comment on the peculiar situation in this cartoon.",
    "Craft two witty captions that could serve as punchlines to the joke presented in this image.",
    "Compose two funny captions that encapsulate the quirky details of this scene.",
    "Write two captions that bring a humorous perspective to this image.",
    "Come up with two captions that add a witty twist to this scene.",
    "Draft two funny captions that could describe what’s happening here.",
    "Think of two clever captions that fit the theme of this cartoon.",
    "Suggest two humorous captions that play on the visual elements present.",
    "How might you create two captions that inject humor into this scenario?",
    "What two witty captions could reflect on the scene's context?",
    "What are two funny captions that might go well with this image?",
    "How would you construct two clever captions to match the mood of this cartoon?",
    "Can you develop two humorous captions that comment on the depicted actions?",
]

ask_answer_with_letter = [
    "Please indicate your choice by answering with the letter corresponding to the correct option.",
    "Select the most appropriate option and respond with its letter only.",
    "Reply with the letter of your answer.",
    "Identify your selection by providing the letter that represents it.",
    "For your answer, use the letter that matches your choice.",
    "Answer with the letter that applies.",
    "Choose the best match and reply with its associated letter.",
    "Can you pinpoint the right choice by letter?",
    "Which letter best represents your answer to this question?",
    "Respond with the letter that aligns with your selection.",
    "Answer with the letter.",
]
choose_the_best = [
    "Which of the two captions do you think is the best fit for this image?",
    "Out of the two, which caption do you personally prefer?",
    "Considering both options, which caption feels more fitting to you?",
    "Between these captions, which one do you believe is the better choice?",
    "Of the two, which one stands out to you as the most appropriate?",
    "Which caption resonates more with you for this scene?",
    "Given the two options, which do you find to be the superior caption?",
    "Between the two, which caption captures your preference?",
    "Of these captions, which one would you choose as the best?",
    "Considering the image, which caption do you feel is the most fitting?",
    "Which of these captions speaks to you more as the right fit for the image?",
    "Out of the two captions, which one do you see as the most compelling?",
    "Of the two options, which caption draws your favor more strongly?",
    "Which caption do you find more suitable for the image, based on your intuition?",
    "Given the two captions, which one resonates with you as the more fitting description?",
    "Between the two, which caption do you believe does justice to the image?",
    "Of these two choices, which caption do you prefer for its connection to the scene?",
    "Considering the image, which caption do you find to be the most apt?",
]

ds_ny_cc_ranking = datasets.load_dataset("jmhessel/newyorker_caption_contest", "ranking", split="train")


def map_transform_ds_ny_cc_ranking(example):
    example["images"] = [example["image"]]
    serious_caption = (random.choice(ask_for_captionning_instruct), example["image_description"])
    description_of_uncannyness = (
        random.choice(ask_for_description_of_uncannyness),
        example["image_uncanny_description"],
    )
    questions = (random.choice(ask_for_questions), " ".join(example["questions"]))
    location = example["image_location"] + "."
    random_location_key = random.choice(list(ask_for_location_of_scene_in_image.keys()))
    location = (random_location_key, ask_for_location_of_scene_in_image[random_location_key] + location)
    choices = ["A", "B"]
    generate_captions_question = random.choice(ask_for_2_captions)
    captions_to_generate = [
        (choice, choice_caption) for choice, choice_caption in zip(choices, example["caption_choices"])
    ]
    str_captions_to_generate = ""
    for choice, caption in captions_to_generate:
        str_captions_to_generate += f"\n{choice}. {caption}"
    choose_better_caption_question = random.choice(choose_the_best) + "\n" + random.choice(ask_answer_with_letter)

    generate_captions = (generate_captions_question, str_captions_to_generate)
    caption_choice = (choose_better_caption_question, example["label"])

    all_texts = [serious_caption, description_of_uncannyness, questions, location]
    random.shuffle(all_texts)
    insert_position_for_tuples = random.randint(0, len(all_texts))
    all_texts = (
        all_texts[:insert_position_for_tuples]
        + [generate_captions, caption_choice]
        + all_texts[insert_position_for_tuples:]
    )
    example["texts"] = [
        {"user": user_text, "assistant": assistant_text, "source": "ny_cc_ranking"}
        for (user_text, assistant_text) in all_texts
    ]

    return example


ds_ny_cc_ranking = ds_ny_cc_ranking.map(
    map_transform_ds_ny_cc_ranking,
    remove_columns=ds_ny_cc_ranking.column_names,
    features=FEATURES,
    num_proc=NUM_PROC,
)
print(ds_ny_cc_ranking)
ds_ny_cc_ranking.save_to_disk("/fsx/hugo/fine_tuning_datasets/ny_cc_ranking", num_proc=NUM_PROC)


# --------------------------- Localized Narratives ------------------------------
# -------------------------------------------------------------------------------

ds_ln = load_dataset("HuggingFaceM4/LocalizedNarratives", split="train")

prompts_ln = [
    "Describe this image in one or two sentences.",
    "Can you describe this image briefly?",
    "How would you summarize this image in a sentence or two?",
    "Please provide a concise description of this image.",
    "Could you give a brief overview of what you see in this image?",
    "In one or two sentences, can you explain what this image depicts?",
]


def map_transform_ln(example):
    example["images"] = [example["image"]]
    example["texts"] = [
        {
            "user": random.choice(prompts_ln),
            "assistant": example["caption"],
            "source": "localized_narratives",
        }
    ]
    return example


ds_ln = ds_ln.map(map_transform_ln, remove_columns=ds_ln.column_names, features=FEATURES, num_proc=NUM_PROC)
print(ds_ln)
ds_ln.save_to_disk("/fsx/hugo/fine_tuning_datasets/localized_narratives", num_proc=NUM_PROC)
