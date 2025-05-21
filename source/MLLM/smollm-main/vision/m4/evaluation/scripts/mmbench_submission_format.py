import json
import re

import pandas as pd
from datasets import load_dataset


dataset_name = "HuggingFaceM4/MMBench"
dataset_split = "test"
# Check id_column_name given in evaluations. If None, set to None.
id_column_name = "index"
path_to_test_eval_data = "/fsx/m4/experiments/local_experiment_dir/evals/results/tr_288_cinco_final_sft_sphinx_test_server_one_image_evaluations.jsonl"
path_to_eval_annotations = "/fsx/hugo/mmbench/mmbench/mmbench_test_en_20231003.tsv"
output_path = "mmbench_one_image_eval_test.xlsx"


LABEL_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


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
        if annotations[i]["task"] == "MMBenchChatbotVMistralClassificationInContextAccWithKLAndEntropy":
            model_name = ": ".join(annotations[i]["model_name_or_path"].split("/")[-3:-1])
            generation_data["opt_step"].append(int(re.findall(r"opt_step-(\d+)", model_name)[0]))
            generation_data["model_name"].append(model_name)
            generation_data["data"].append(eval(annotations[i]["score"])["server_results"])
            break

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


def main():
    def load_datasets_and_generations(dataset_name, path_to_test_eval_data, id_column_name=None):
        try:
            dataset = load_dataset(dataset_name, split=dataset_split)

            generation_data = load_annotations(path_to_test_eval_data)
            if id_column_name is not None:
                dataset_question_id_index = {str(key): idx for idx, key in enumerate(dataset[id_column_name])}
            else:
                dataset_question_id_index = {str(idx): idx for idx in range(len(dataset))}
            return dataset, generation_data, dataset_question_id_index
        except Exception:
            return None

    dataset, generation_data, dataset_question_id_index = load_datasets_and_generations(
        dataset_name, path_to_test_eval_data, id_column_name=id_column_name
    )

    # They have a tsv version on their website: mmbench_test_en_20231003.tsv so 2023/10/03 https://mmbench.opencompass.org.cn/home which
    # is more recent than the one on their github repo: mmbench_test_20230712.tsv so 2023/07/12
    # This means it is normal that we have only 6666 examples while "lmms-lab/MMBench_EN" has 6718 examples.
    sumbission_list = []
    for gen_ex in generation_data["data"][0]:
        ex_dict = {}
        index = gen_ex["id"]
        idx_dataset = dataset_question_id_index[index]

        example = dataset[idx_dataset]

        ex_dict["index"] = index
        ex_dict["A"] = example["A"]
        ex_dict["B"] = example["B"]
        ex_dict["C"] = example["C"]
        ex_dict["D"] = example["D"]
        ex_dict["question"] = example["question"]
        ex_dict["prediction"] = LABEL_TO_LETTER[gen_ex["label"]]
        sumbission_list.append(ex_dict)
    df = pd.DataFrame(sumbission_list)
    # Need to have pip installed openpyxl. Not in shared env atm
    df.to_excel(output_path, index=False)


if __name__ == "__main__":
    main()
