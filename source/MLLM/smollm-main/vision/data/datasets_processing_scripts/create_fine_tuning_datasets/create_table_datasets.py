import json
import re
from io import BytesIO
from multiprocessing import Pool
from random import random

import datasets
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageChops, ImageFile
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

NUM_PROC = 48


def trim(image):
    im = image.convert("RGB")
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -20)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)


def process_list_of_strings(input_list):
    processed_list = []

    for item in input_list:
        if isinstance(item, list):
            # Process list represented as string
            processed_item = ",".join(eval(item))
            processed_list.append(processed_item)
        else:
            # Process single string
            processed_list.append(item)
    return processed_list


# -------------------------------------------------------------------------------
# --------------------------- Robut sqa ---------------------------------------
# -------------------------------------------------------------------------------

FEATURES_DATASET = datasets.Features(
    {
        "image": datasets.Image(decode=True),
        "table": {
            "header": datasets.Sequence(datasets.Value("string")),
            "rows": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
        },
        "id": datasets.Value("string"),
        "original_id": datasets.Value("string"),
        "questions": datasets.Sequence(datasets.Value("string")),
        "perturbation_type": datasets.Value("string"),
        "answers": datasets.Sequence(datasets.Value("string")),
        "num_rows": datasets.Value("int32"),
    }
)


colors = [
    "aliceblue",
    "aqua",
    "aquamarine",
    "azure",
    "beige",
    "lavenderblush",
    "lawngreen",
    "lemonchiffon",
    "lightblue",
    "lightcoral",
    "lightcyan",
    "lightgoldenrodyellow",
    "lightgray",
    "lightgrey",
    "lightgreen",
    "lightpink",
    "lightsalmon",
    "lightseagreen",
    "peachpuff",
    "peru",
    "pink",
    "plum",
    "powderblue",
    "rosybrown",
]
font_families = [
    "Arial",
    "Balto",
    "Courier New",
    "Droid Sans",
    "Droid Serif",
    "Droid Sans Mono",
    "Gravitas One",
    "Old Standard TT",
    "Open Sans",
    "Overpass",
    "PT Sans Narrow",
    "Raleway",
    "Times New Roman",
]

ds_robut_sqa = datasets.load_dataset("yilunzhao/robut", split="sqa")


def map_transform_ds_robut_sqa(example):
    table = example["table"]
    df = pd.DataFrame(table["rows"], columns=table["header"])
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=table["header"],
                    fill_color=random.choice(colors) if random.random() > 0.75 else None,
                    font_family=random.choice(font_families),
                ),
                cells=dict(
                    values=[list(column_data) for _, column_data in df.items()],
                    fill_color=random.choice(colors) if random.random() > 0.75 else None,
                    font_family=random.choice(font_families),
                ),
            ),
        ]
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )
    num_rows = len(fig.data[0].cells.values[0])
    height = num_rows * 130 + 150
    image = fig.to_image(format="png", scale=1, height=height, width=980, engine="kaleido")
    image = Image.open(BytesIO(image))
    example["image"] = trim(image)
    example["num_rows"] = num_rows
    example["answers"] = [", ".join(eval(answer_list)) for answer_list in example["answers"]]
    example["questions"] = eval(example["question"])
    return example


ds_robut_sqa = ds_robut_sqa.map(
    map_transform_ds_robut_sqa, remove_columns=["question"], features=FEATURES_DATASET, num_proc=20
)
ds_robut_sqa.push_to_hub("HuggingFaceM4/ROBUT-sqa-rendered-tables", private=True)


# -------------------------------------------------------------------------------
# --------------------------- TaTQA ---------------------------------------
# -------------------------------------------------------------------------------


FEATURES_DATASET = datasets.Features(
    {
        "image": datasets.Image(decode=True),
        "table": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
        "id": datasets.Value("string"),
        "paragraphs": datasets.Sequence(datasets.Value("string")),
        "questions": datasets.Sequence(datasets.Value("string")),
        "derivations": datasets.Sequence(datasets.Value("string")),
        "scales": datasets.Sequence(datasets.Value("string")),
        "answers": datasets.Sequence(datasets.Value("string")),
        "num_rows": datasets.Value("int32"),
    }
)

_ANNOTATIONS_PATH = "/fsx/leo/repos/TAT-QA/dataset_raw/tatqa_dataset_train.json"
colors = [
    None,
    "aliceblue",
    "aqua",
    "aquamarine",
    "azure",
    "beige",
    "lavenderblush",
    "lawngreen",
    "lemonchiffon",
    "lightblue",
    "lightcoral",
    "lightcyan",
    "lightgoldenrodyellow",
    "lightgray",
    "lightgrey",
    "lightgreen",
    "lightpink",
    "lightsalmon",
    "lightseagreen",
    "peachpuff",
    "peru",
    "pink",
    "plum",
    "powderblue",
    "rosybrown",
]
font_families = [
    "Arial",
    "Balto",
    "Courier New",
    "Droid Sans",
    "Droid Serif",
    "Droid Sans Mono",
    "Gravitas One",
    "Old Standard TT",
    "Open Sans",
    "Overpass",
    "PT Sans Narrow",
    "Raleway",
    "Times New Roman",
]


def process_annotation(annotation):
    try:
        table = {"header": annotation["table"]["table"][0], "rows": annotation["table"]["table"][1:]}
        df = pd.DataFrame(table["rows"], columns=table["header"])
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=table["header"],
                        fill_color=random.choice(colors) if random.random() > 0.75 else None,
                        font_family=random.choice(font_families),
                    ),
                    cells=dict(
                        values=[list(column_data) for _, column_data in df.items()],
                        fill_color=random.choice(colors) if random.random() > 0.75 else None,
                        font_family=random.choice(font_families),
                    ),
                ),
            ]
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
        )
        num_rows = len(fig.data[0].cells.values[0])
        height = num_rows * 130 + 150
        image = fig.to_image(format="png", scale=1, height=height, engine="kaleido")
        image = Image.open(BytesIO(image))
        image = trim(image)
        questions = [ex["question"] for ex in annotation["questions"]]
        derivations = [ex["derivation"] for ex in annotation["questions"]]
        answers_unprocessed = [ex["answer"] for ex in annotation["questions"]]
        answers = []
        for item in answers_unprocessed:
            if isinstance(item, list):
                answer = ", ".join(item).lower()
                answers.append(answer)
            else:
                answers.append(str(item))
        scales = [ex["scale"] for ex in annotation["questions"]]
        paragraphs = [paragraph["text"] for paragraph in annotation["paragraphs"]]
        return {
            "image": image,
            "table": annotation["table"]["table"],
            "id": annotation["table"]["uid"],
            "paragraphs": paragraphs,
            "questions": questions,
            "derivations": derivations,
            "scales": scales,
            "answers": answers,
            "num_rows": num_rows,
        }
    except Exception as e:
        print(f"Exception  {e} for annotation: {annotation}")
        pass


def load_annotations(_ANNOTATIONS_PATH):
    with open(_ANNOTATIONS_PATH, "r", encoding="utf-8") as file:
        annotations = json.load(file)
    return annotations


annotations = load_annotations(_ANNOTATIONS_PATH)
with Pool(10) as pool:
    ds_tatqa_list = list(tqdm(pool.imap(process_annotation, annotations), total=len(annotations)))
ds_tatqa_dict = {key: [item[key] for item in ds_tatqa_list] for key in ds_tatqa_list[0]}


full_dataset = datasets.Dataset.from_dict(ds_tatqa_dict, features=FEATURES_DATASET)
full_dataset.push_to_hub("HuggingFaceM4/TATQA-rendered-tables", private=True)


# -------------------------------------------------------------------------------
# --------------------------- Robut wikisql ---------------------------------------
# -------------------------------------------------------------------------------

FEATURES_DATASET = datasets.Features(
    {
        "image": datasets.Image(decode=True),
        "table": {
            "header": datasets.Sequence(datasets.Value("string")),
            "rows": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
        },
        "id": datasets.Value("string"),
        "original_id": datasets.Value("string"),
        "question": datasets.Value("string"),
        "perturbation_type": datasets.Value("string"),
        "answers": datasets.Sequence(datasets.Value("string")),
        "num_rows": datasets.Value("int32"),
    }
)

colors = [
    None,
    "aliceblue",
    "aqua",
    "aquamarine",
    "azure",
    "beige",
    "lavenderblush",
    "lawngreen",
    "lemonchiffon",
    "lightblue",
    "lightcoral",
    "lightcyan",
    "lightgoldenrodyellow",
    "lightgray",
    "lightgrey",
    "lightgreen",
    "lightpink",
    "lightsalmon",
    "lightseagreen",
    "peachpuff",
    "peru",
    "pink",
    "plum",
    "powderblue",
    "rosybrown",
]
font_families = [
    "Arial",
    "Balto",
    "Courier New",
    "Droid Sans",
    "Droid Serif",
    "Droid Sans Mono",
    "Gravitas One",
    "Old Standard TT",
    "Open Sans",
    "Overpass",
    "PT Sans Narrow",
    "Raleway",
    "Times New Roman",
]

ds_robut_wikisql = datasets.load_dataset("yilunzhao/robut", split="wikisql")


def map_transform_ds_robut_wikisql(example):
    table = example["table"]
    df = pd.DataFrame(table["rows"], columns=table["header"])
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=df.columns, fill_color=random.choice(colors), font_family=random.choice(font_families)
                ),
                cells=dict(
                    values=[df[col] for col in df.columns],
                    fill_color=random.choice(colors),
                    font_family=random.choice(font_families),
                ),
            ),
        ]
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )
    num_rows = len(fig.data[0].cells.values)
    height = num_rows * 180 + 150
    image = fig.to_image(format="jpg", scale=1, height=height, engine="kaleido")
    image = Image.open(BytesIO(image))
    example["image"] = trim(image)
    example["num_rows"] = num_rows
    return example


ds_robut_wikisql = ds_robut_wikisql.map(map_transform_ds_robut_wikisql, features=FEATURES_DATASET, num_proc=NUM_PROC)
ds_robut_wikisql.push_to_hub("HuggingFaceM4/ROBUT-wikisql-rendered-tables", private=True)


# -------------------------------------------------------------------------------
# --------------------------- Robut wtqa ---------------------------------------
# -------------------------------------------------------------------------------


FEATURES_DATASET = datasets.Features(
    {
        "image": datasets.Image(decode=True),
        "table": {
            "header": datasets.Sequence(datasets.Value("string")),
            "rows": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
        },
        "id": datasets.Value("string"),
        "original_id": datasets.Value("string"),
        "question": datasets.Value("string"),
        "perturbation_type": datasets.Value("string"),
        "answers": datasets.Sequence(datasets.Value("string")),
        "num_rows": datasets.Value("int32"),
    }
)

colors = [
    None,
    "aliceblue",
    "aqua",
    "aquamarine",
    "azure",
    "beige",
    "lavenderblush",
    "lawngreen",
    "lemonchiffon",
    "lightblue",
    "lightcoral",
    "lightcyan",
    "lightgoldenrodyellow",
    "lightgray",
    "lightgrey",
    "lightgreen",
    "lightpink",
    "lightsalmon",
    "lightseagreen",
    "peachpuff",
    "peru",
    "pink",
    "plum",
    "powderblue",
    "rosybrown",
]
font_families = [
    "Arial",
    "Balto",
    "Courier New",
    "Droid Sans",
    "Droid Serif",
    "Droid Sans Mono",
    "Gravitas One",
    "Old Standard TT",
    "Open Sans",
    "Overpass",
    "PT Sans Narrow",
    "Raleway",
    "Times New Roman",
]

ds_robut_wtq = datasets.load_dataset("yilunzhao/robut", split="wtq")


def map_transform_ds_robut_wtq(example):
    table = example["table"]
    df = pd.DataFrame(table["rows"], columns=table["header"])
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=df.columns, fill_color=random.choice(colors), font_family=random.choice(font_families)
                ),
                cells=dict(
                    values=[df[col] for col in df.columns],
                    fill_color=random.choice(colors),
                    font_family=random.choice(font_families),
                ),
            ),
        ]
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )
    num_rows = len(fig.data[0].cells.values)
    height = num_rows * 190 + 150
    image = fig.to_image(format="jpg", scale=1, height=height, width=1000, engine="kaleido")
    image = Image.open(BytesIO(image))
    example["image"] = trim(image)
    example["num_rows"] = num_rows
    return example


ds_robut_wtq = ds_robut_wtq.map(map_transform_ds_robut_wtq, features=FEATURES_DATASET, num_proc=90)
ds_robut_wtq.push_to_hub("HuggingFaceM4/ROBUT-wtq-rendered-tables", private=True)


# -------------------------------------------------------------------------------
# --------------------------- FinQA ---------------------------------------
# -------------------------------------------------------------------------------
FEATURES_DATASET = datasets.Features(
    {
        "image": datasets.Image(decode=True),
        "table": {
            "header": datasets.Sequence(datasets.Value("string")),
            "rows": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
        },
        "id": datasets.Value("string"),
        "pre_text": datasets.Sequence(datasets.Value("string")),
        "post_text": datasets.Sequence(datasets.Value("string")),
        "question": datasets.Value("string"),  # the question;
        "program": datasets.Value("string"),  # the reasoning program;
        "gold_inds": datasets.Value("string"),  # the gold supporting facts;
        "answer": datasets.Value("string"),  # the gold execution result;
        "program_re": datasets.Value("string"),  # the reasoning program in nested format;
        "explanation": datasets.Value("string"),  # explanation, not alweays there, not sure from where it comes
    }
)


_ANNOTATIONS_PATH = "/fsx/leo/FinQA/dataset/train.json"
colors = [
    "aliceblue",
    "aqua",
    "aquamarine",
    "azure",
    "beige",
    "lavenderblush",
    "lawngreen",
    "lemonchiffon",
    "lightblue",
    "lightcoral",
    "lightcyan",
    "lightgoldenrodyellow",
    "lightgray",
    "lightgrey",
    "lightgreen",
    "lightpink",
    "lightsalmon",
    "lightseagreen",
    "peachpuff",
    "peru",
    "pink",
    "plum",
    "powderblue",
    "rosybrown",
]
font_families = [
    "Arial",
    "Balto",
    "Courier New",
    "Droid Sans",
    "Droid Serif",
    "Droid Sans Mono",
    "Gravitas One",
    "Old Standard TT",
    "Open Sans",
    "Overpass",
    "PT Sans Narrow",
    "Raleway",
    "Times New Roman",
]


def process_annotation(annotation):
    try:
        table = {"header": annotation["table"][0], "rows": annotation["table"][1:]}
        df = pd.DataFrame(table["rows"], columns=table["header"])
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=table["header"],
                        fill_color=random.choice(colors) if random.random() > 0.75 else None,
                        font_family=random.choice(font_families),
                    ),
                    cells=dict(
                        values=[list(column_data) for _, column_data in df.items()],
                        fill_color=random.choice(colors) if random.random() > 0.75 else None,
                        font_family=random.choice(font_families),
                    ),
                ),
            ]
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
        )
        num_rows = len(fig.data[0].cells.values)
        height = num_rows * 130 + 150
        image = fig.to_image(format="png", scale=1, height=height, engine="kaleido")
        image = Image.open(BytesIO(image))
        image = trim(image)

        return {
            "image": image,
            "table": table,
            "id": annotation["id"],
            "pre_text": annotation["pre_text"],
            "post_text": annotation["post_text"],
            "question": annotation["qa"]["question"],  # the question;
            "program": annotation["qa"]["program"],  # the reasoning program;
            "gold_inds": str(annotation["qa"]["gold_inds"]),  # the gold supporting facts;
            "answer": annotation["qa"]["exe_ans"],  # the gold execution result;
            "program_re": annotation["qa"]["program_re"],  # the reasoning program in nested format;
            "explanation": annotation["qa"]["explanation"],
        }
    except Exception as e:
        print(f"Exception  {e} for annotation: {annotation}")
        pass


def load_annotations(_ANNOTATIONS_PATH):
    with open(_ANNOTATIONS_PATH, "r", encoding="utf-8") as file:
        annotations = json.load(file)
    return annotations


annotations = load_annotations(_ANNOTATIONS_PATH)
with Pool(NUM_PROC) as pool:
    ds_finqa_list = list(tqdm(pool.imap(process_annotation, annotations), total=len(annotations)))
ds_finqa_dict = {key: [item[key] for item in ds_finqa_list] for key in ds_finqa_list[0]}

full_dataset = datasets.Dataset.from_dict(ds_finqa_dict, features=FEATURES_DATASET)
full_dataset.push_to_hub("HuggingFaceM4/FINQA-rendered-tables", private=True)

# -------------------------------------------------------------------------------
# --------------------------- MultiHiertt ---------------------------------------
# -------------------------------------------------------------------------------


FEATURES_DATASET = datasets.Features(
    {
        "images": datasets.Sequence(datasets.Image(decode=True)),
        "tables": datasets.Sequence(datasets.Value("string")),  # the list of tables in HTML format in the document;
        "id": datasets.Value("string"),  # unique example id;
        "paragraphs": datasets.Sequence(datasets.Value("string")),  # the list of sentences in the document;
        "table_description": datasets.Value(
            "string"
        ),  # the list of table descriptions for each data cell in tables. Generated by the pre-processing script;
        "question": datasets.Value("string"),  # the question;
        "answer": datasets.Value("string"),  # the answer;
        "program": datasets.Value("string"),  # the reasoning program;
        "translated_program": datasets.Value("string"),  # the translated reasoning program;
        "text_evidence": datasets.Sequence(
            datasets.Value("string")
        ),  # the list of indices of gold supporting text facts;
        "table_evidence": datasets.Sequence(
            datasets.Value("string")
        ),  # the list of indices of gold supporting table facts;
    }
)


_ANNOTATIONS_PATH = "/fsx/leo/datasets/multi_hiertt/train.json"
colors = [
    "aliceblue",
    "aqua",
    "aquamarine",
    "azure",
    "beige",
    "lavenderblush",
    "lawngreen",
    "lemonchiffon",
    "lightblue",
    "lightcoral",
    "lightcyan",
    "lightgoldenrodyellow",
    "lightgray",
    "lightgrey",
    "lightgreen",
    "lightpink",
    "lightsalmon",
    "lightseagreen",
    "peachpuff",
    "peru",
    "pink",
    "plum",
    "powderblue",
    "rosybrown",
]


def translate_sequential_expressions(input_str):
    operation_symbols = {
        "divide": "/",
        "multiply": "*",
        "add": "+",
        "subtract": "-",
    }

    # Replace "const_X" with "X"
    input_str = re.sub(r"const_(\d+)", r"\1", input_str)

    # Preprocess to replace references and split expressions
    expressions = input_str.split(", ")
    results = []

    for expr in expressions:
        # Identify and replace references with previous results
        def replace_ref(match):
            index = int(match.group(1))
            return results[index]

        expr = re.sub(r"#(\d+)", replace_ref, expr)

        # Translate operations to symbols
        for op, symbol in operation_symbols.items():
            if op in expr:
                start_idx = expr.find("(")
                end_idx = expr.rfind(")")
                args_str = expr[start_idx + 1 : end_idx]
                args = [arg.strip() for arg in args_str.split(",")]
                expr = f"({args[0]} {symbol} {args[1]})"
                break  # Assuming only one operation per expression

        results.append(expr)

    # Joining all expressions for the final output
    return results[-1]


def process_annotation(annotation):
    try:
        images = []
        tables = []
        for i in range(len(annotation["tables"])):
            table = pd.read_html(annotation["tables"][i])
            rows_as_lists = [
                [str(el) if str(el) != "nan" not in str(el) else "" for el in row.tolist()]
                for _, row in table[0].iterrows()
            ]
            table_dict = {"header": rows_as_lists[0], "rows": rows_as_lists[1:]}

            df = pd.DataFrame(table_dict["rows"], columns=table_dict["header"])
            fig = go.Figure(
                data=[
                    go.Table(
                        header=dict(
                            values=table_dict["header"],
                            fill_color=random.choice(colors) if random.random() > 0.75 else None,
                        ),
                        cells=dict(
                            values=[list(column_data) for _, column_data in df.items()],
                            fill_color=random.choice(colors) if random.random() > 0.75 else None,
                        ),
                    ),
                ]
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
            )
            num_rows = len(fig.data[0].cells.values)
            height = num_rows * 150 + 150
            image = fig.to_image(format="png", scale=1, height=height, engine="kaleido")
            image = Image.open(BytesIO(image))
            image = trim(image)
            images.append(image)
            tables.append(str(table_dict))

        return {
            "images": images,
            "tables": tables,
            "id": annotation["uid"],
            "paragraphs": annotation["paragraphs"],
            "table_description": str(annotation["table_description"]),
            "question": str(annotation["qa"]["question"]),
            "answer": str(annotation["qa"]["answer"]),
            "program": str(annotation["qa"]["program"]),
            "translated_program": translate_sequential_expressions(str(annotation["qa"]["program"])),
            "text_evidence": [str(evidence) for evidence in annotation["qa"]["text_evidence"]],
            "table_evidence": [str(evidence) for evidence in annotation["qa"]["table_evidence"]],
        }
    except Exception as e:
        print(f"Exception  {e} for annotation: {annotation}")
        pass


def load_annotations(_ANNOTATIONS_PATH):
    with open(_ANNOTATIONS_PATH, "r", encoding="utf-8") as file:
        annotations = json.load(file)
    return annotations


annotations = load_annotations(_ANNOTATIONS_PATH)
with Pool(60) as pool:
    ds_multihiertt_list = list(tqdm(pool.imap(process_annotation, annotations), total=len(annotations)))
ds_multihiertt_dict = {key: [item[key] for item in ds_multihiertt_list] for key in ds_multihiertt_list[0]}

full_dataset = datasets.Dataset.from_dict(ds_multihiertt_dict, features=FEATURES_DATASET)
full_dataset.push_to_hub("HuggingFaceM4/MultiHiertt-rendered-tables", private=True)


# -------------------------------------------------------------------------------
# --------------------------- HiTab ---------------------------------------
# -------------------------------------------------------------------------------

FEATURES_DATASET = datasets.Features(
    {
        "image": datasets.Image(decode=True),
        "table_id": datasets.Value("string"),
        "table": {
            "header": datasets.Sequence(datasets.Value("string")),
            "rows": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
        },
        "qids": datasets.Sequence(datasets.Value("string")),
        "questions": datasets.Sequence(datasets.Value("string")),
        "answers": datasets.Sequence(datasets.Value("string")),
    }
)


ds_hitab_not_rendered = datasets.load_dataset("zhoujun/hitab", split="train")
hitab_not_rendered_table_id_index = {key: idx for idx, key in enumerate(ds_hitab_not_rendered["table_id"])}

print(ds_hitab_not_rendered)
dedup_dict_hitab_no_table = {}
for i, example in enumerate(tqdm(ds_hitab_not_rendered)):
    table_id = example["table_id"]
    curr_example = [
        {
            "qids": example["id"],
            "questions": example["question"],
            "answers": " ".join(example["answer"]),
        }
    ]
    dedup_dict_hitab_no_table[table_id] = dedup_dict_hitab_no_table.get(table_id, []) + curr_example

colors = [
    "aliceblue",
    "aqua",
    "aquamarine",
    "azure",
    "beige",
    "lavenderblush",
    "lawngreen",
    "lemonchiffon",
    "lightblue",
    "lightcoral",
    "lightcyan",
    "lightgoldenrodyellow",
    "lightgray",
    "lightgrey",
    "lightgreen",
    "lightpink",
    "lightsalmon",
    "lightseagreen",
    "peachpuff",
    "peru",
    "pink",
    "plum",
    "powderblue",
    "rosybrown",
]
font_families = [
    "Arial",
    "Balto",
    "Courier New",
    "Droid Sans",
    "Droid Serif",
    "Droid Sans Mono",
    "Gravitas One",
    "Old Standard TT",
    "Open Sans",
    "Overpass",
    "PT Sans Narrow",
    "Raleway",
    "Times New Roman",
]


def process_annotation(table_id):
    try:
        annotation = {
            key: [item[key] for item in dedup_dict_hitab_no_table[table_id]]
            for key in dedup_dict_hitab_no_table[table_id][0]
        }

        table = ds_hitab_not_rendered[hitab_not_rendered_table_id_index[table_id]]["table"]
        table = {"header": table["cells"][0], "rows": table["cells"][1:]}
        df = pd.DataFrame(table["rows"], columns=table["header"])
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=table["header"],
                        fill_color=random.choice(colors) if random.random() > 0.75 else None,
                        font_family=random.choice(font_families),
                    ),
                    cells=dict(
                        values=[list(column_data) for _, column_data in df.items()],
                        fill_color=random.choice(colors) if random.random() > 0.75 else None,
                        font_family=random.choice(font_families),
                    ),
                ),
            ]
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
        )
        num_rows = len(fig.data[0].cells.values)
        height = num_rows * 130 + 150
        image = fig.to_image(format="png", scale=1, height=height, width=980, engine="kaleido")
        image = Image.open(BytesIO(image))
        annotation["image"] = trim(image)
        annotation["table_id"] = table_id
        annotation["table"] = table
        return annotation
    except Exception as e:
        print(f"Exception  {e} for: {table_id}")
        pass


with Pool(30) as pool:
    ds_hitab_list = list(
        tqdm(pool.imap(process_annotation, dedup_dict_hitab_no_table), total=len(dedup_dict_hitab_no_table))
    )
ds_hitab_dict = {key: [item[key] for item in ds_hitab_list] for key in ds_hitab_list[0]}
full_dataset = datasets.Dataset.from_dict(ds_hitab_dict, features=FEATURES_DATASET)
print(full_dataset)
full_dataset.push_to_hub("HuggingFaceM4/HiTab-rendered-tables", private=True)
