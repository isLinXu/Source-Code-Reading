import json


path_to_test_eval_data = "/fsx/m4/experiments/local_experiment_dir/evals/results/tr_288_cinco_final_sft_sphinx_test_server_one_image_evaluations.jsonl"
output_path = "mmmu_test_results.json"

LABEL_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J"}


def list_of_dicts_to_single_dict(list_of_dicts):
    dict_of_lists = {}
    for item in list_of_dicts:
        for key, value in item.items():
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            dict_of_lists[key] = value
    return dict_of_lists


def load_annotations(path_to_generation_data, target_task):
    annotations = []
    with open(path_to_generation_data, "r", encoding="utf-8") as file:
        for line in file:
            json_line = json.loads(line)
            annotations.append(json_line)
    server_results = {}
    for i in range(len(annotations)):
        if annotations[i]["task"] == target_task:
            server_results["task"] = annotations[i]["task"]
            server_results["data"] = eval(annotations[i]["score"])["server_results"]
            break

    return server_results


def main():
    mcq_server_results = load_annotations(
        path_to_test_eval_data, "MMMUMCQChatbotVMistralClassificationInContextAccWithKLAndEntropy"
    )
    mcq_server_results = [
        {id_label["id"]: LABEL_TO_LETTER[id_label["label"]]} for id_label in mcq_server_results["data"]
    ]
    open_ended_server_results = load_annotations(
        path_to_test_eval_data, "MMMUOpenEndedChatbotVMistralOpenEndedVQAInContextAcc"
    )
    open_ended_server_results = [
        {id_label["question_id"]: id_label["answer"]} for id_label in open_ended_server_results["data"]
    ]
    mmmu_server_results = mcq_server_results + open_ended_server_results
    mmmu_server_results = list_of_dicts_to_single_dict(mmmu_server_results)
    with open(output_path, "w") as f:
        json.dump(mmmu_server_results, f)


if __name__ == "__main__":
    main()
