import json


PATH_JSONL = "/Users/hugolaurencon/Desktop/tr_209_ift_mixture_test_final_evaluations.jsonl"


BANNED_KEYS = [
    "kl_distribution",
    "entropy_distribution",
    "kl_mean",
    "Bleu_1",
    "Bleu_1_all",
    "Bleu_2",
    "Bleu_2_all",
    "Bleu_3",
    "Bleu_3_all",
    "Bleu_4",
    "Bleu_4_all",
    "METEOR",
    "METEOR_all",
    "CIDEr_all",
    "ROUGE_L",
    "ROUGE_L_all",
    "per_bucket_accuracy",
    "std_per_bucket_accuracy",
    "entropy_mean",
]


jsonl_data = []
with open(PATH_JSONL, "r") as file:
    for line in file:
        json_data = json.loads(line)
        jsonl_data.append(json_data)


for idx, data in enumerate(jsonl_data):
    if "score" in data:
        if type(data["score"]) == str:
            data["score"] = json.loads(data["score"].replace("'", '"'))
        for banned_key in BANNED_KEYS:
            if banned_key in data["score"]:
                data["score"].pop(banned_key)
    jsonl_data[idx] = data


with open(PATH_JSONL, "w") as file:
    for item in jsonl_data:
        item_json = json.dumps(item)
        file.write(item_json + "\n")
