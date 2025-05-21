"""
To deploy: streamlit run viz_tool.py
If on HFC, probably will need to do port forwarding to get it in your local browser.
"""

import json
import re

import streamlit as st
from datasets import load_dataset


dataset_names = ["HuggingFaceM4/ChartQA"]
datasets_splits = ["val"]
# Check id_column_name given in evaluations. If None, set to None.
id_column_names = [None]
paths_to_generation_data = [
    "/fsx/m4/experiments/local_experiment_dir/evals/results/saved_generations_results/final_sfts_chartqa_evaluations.jsonl"
]


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = {}
    for item in list_of_dicts:
        for key, value in item.items():
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            dict_of_lists[key].append(value)
    return dict_of_lists


def load_annotations(path_to_generation_data):
    annotations = []
    with open(path_to_generation_data, "r", encoding="utf-8") as file:
        for line in file:
            json_line = json.loads(line)
            annotations.append(json_line)
    generation_data = {"model_name": [], "data": [], "opt_step": []}
    for i in range(len(annotations)):
        model_name = ": ".join(annotations[i]["model_name_or_path"].split("/")[-3:-1])
        generation_data["opt_step"].append(int(re.findall(r"opt_step-(\d+)", model_name)[0]))
        generation_data["model_name"].append(model_name)
        generation_data["data"].append(eval(annotations[i]["score"])["server_results"])

    # Get the indices that would sort the 'col3' list
    sorted_data = sorted(
        zip(generation_data["model_name"], generation_data["data"], generation_data["opt_step"]),
        key=lambda x: x[2],
        reverse=True,
    )

    sorted_generation_data = {
        "model_name": [x[0] for x in sorted_data],
        "data": [x[1] for x in sorted_data],
        "opt_step": [x[2] for x in sorted_data],
    }

    # Sort all lists based on the sorted indices
    return sorted_generation_data


ds = load_annotations(paths_to_generation_data[0])


def display_examples(dataset, generation_data, dataset_question_id_index, multiple_images, num_examples=300):
    st.header(f"Sample Examples - First {num_examples}")

    for idx in range(num_examples):
        st.subheader(f"Example {idx}")

        question_id = generation_data["data"][0][idx]["question_id"]
        st.write(f"Q id: {question_id}")

        idx_dataset = dataset_question_id_index[question_id]
        st.subheader(f"Dataset idx  {idx_dataset}")

        example = dataset[idx_dataset]
        if multiple_images:
            images = [im for im in example["images"] if im is not None]
        else:
            images = [example["image"]]
        for im in images:
            st.image(im, width=250, caption=f"Image dimension: {im.size}")

        if "question" in example:
            question = example["question"]
        elif "query" in example:
            question = example["query"]
        else:
            raise ValueError("Dataset must contain a column question or query")
        st.markdown("<pre><strong>Question:<strong><pre>", unsafe_allow_html=True)
        # Might look strange, but it's a more accurate display of the question because it
        # preserves the newlines and doesn't affect <image> tags
        st.write([question])
        if "answers" in example:
            answers = example["answers"]
        elif "answer" in example:
            answers = example["answer"]
        elif "label" in example:
            answers = example["label"]
        else:
            raise ValueError("Dataset must contain a column answers or answer")
        if isinstance(answers, str):
            answers = [answers]

        for i, answer in enumerate(answers):
            display_text = f"<strong>Answer {i}:</strong> {answer}\n".replace("\n", "<br>")
            st.markdown(f"<pre>{display_text}</pre>", unsafe_allow_html=True)

        dominant_answer = [
            answer for answer in set(answers) if answers.count(answer) == max(map(answers.count, answers))
        ][0].lower()
        for model_name, data in zip(generation_data["model_name"], generation_data["data"]):
            generated_answer = data[idx]["answer"]
            if dominant_answer in generated_answer.lower():
                display_text = (
                    f'<strong>{model_name} Answer:</strong> <span style="color:green">{generated_answer}\n</span>'
                    .replace("\n", "<br>")
                )
            else:
                display_text = (
                    f'<strong>{model_name} Answer:</strong> <span style="color:red">{generated_answer}\n</span>'
                    .replace("\n", "<br>")
                )

            st.markdown(f"<pre>{display_text}</pre>", unsafe_allow_html=True)

        st.divider()


def main():
    st.set_page_config(page_title="Prompted set viewer", layout="wide")

    def load_datasets_and_generations(dataset_idx, id_column_name=None):
        try:
            dataset = load_dataset(dataset_names[dataset_idx], split=datasets_splits[dataset_idx])

            generation_data = load_annotations(paths_to_generation_data[dataset_idx])
            if id_column_name is not None:
                dataset_question_id_index = {str(key): idx for idx, key in enumerate(dataset[id_column_name])}
            else:
                dataset_question_id_index = {str(idx): idx for idx in range(len(dataset))}
            return dataset, generation_data, dataset_question_id_index
        except Exception:
            return None

    st.sidebar.header("Prompted set viewer")
    selected_dataset = st.sidebar.selectbox(
        label="Select a dataset",
        options=dataset_names,
        index=0,
    )
    dataset_idx = dataset_names.index(selected_dataset)
    dataset, generation_data, dataset_question_id_index = load_datasets_and_generations(
        dataset_idx, id_column_name=id_column_names[dataset_idx]
    )
    if "image" in dataset.features:
        multiple_images = False
    elif "images" in dataset.features:
        multiple_images = True
    else:
        raise ValueError("Dataset must contain a column image or images")

    if dataset is not None:
        num_examples = st.sidebar.slider(
            label="Number of examples to display", value=200, min_value=100, max_value=2000
        )
        st.sidebar.write(f"Dataset length: `{len(dataset)}`")
        display_examples(
            dataset, generation_data, dataset_question_id_index, multiple_images, num_examples=num_examples
        )


if __name__ == "__main__":
    main()
