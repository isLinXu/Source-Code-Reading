# Run evaluation

- create log folder before running the slurm script

<!-- # Push the metrics to Wandb

Currently, this step is done manually. The purpose of this section is to describe the process used.

Here are the steps to follow:
1. Run a slurm script that will evaluate all the checkpoints saved for a training on a single task (e.g. see for example the slurm script [`tr_18`](experiments/evaluation/vloom/tr_18/tr_18.slurm)). Be careful to put `%x_%A_%a` in the title of the log files,
2. Note the common job id `JOB_ARRAY_COMMON_ID` for the whole jobarray which correspond to `%A` in the log file name,
3. Go to the folder containing the produced log files and run `grep "'Evaluate the checkpoint: \|<TASK_NAME>Vgpt2ZeroShoter<METRIC> <JOB_NAME>_<JOB_ARRAY_COMMON_ID>*` - where `<TASK_NAME>`, , `<METRIC>`, `<JOB_NAME>`, and `<JOB_ARRAY_COMMON_ID>` should be replaced accordingly - then copy the result
4. Use the [push_results_to_wandb.py](/home/lucile_huggingface_co/repos/m4/experiments/evaluation/vloom/utils/push_results_to_wandb.py) script to push the results to Wandb by changing the values of variables `run_name` and `content`. -->

# Evaluation to be submitted to an outside server

Many test set evaluations do not have ground truth and therefore require a file to be sent to an external server specific to each task.

These submissions are very often limited in number (per day, per month and per year) and require the creation of an account and a team.

Where possible, it's better to start by submitting the file obtained on the split `server_check` in order to check that the format of the result file is correct and that the figure calculated with our tools corresponds to the figure calculated by the server.

To retrieve the file to be submitted, for the moment you need to perform a few manual steps to:

1. extract the results subpart from the jsonl results file and
2. post-process the results file.

A template to perform those steps is provided bellow:

```python
from pathlib import Path
import json

result_file = Path("/fsx/m4/experiments/local_experiment_dir/evals/results/tr_190_01_64n_check_server_evaluations.jsonl")

with open(result_file, 'r', encoding="ISO-8859-1") as file:
    json_data = file.read()

json_entities = json_data.strip().split('\n')
parsed_json_objects = []
for entity in json_entities:
    parsed_json = json.loads(entity)
    parsed_json_objects.append(parsed_json)

parsed_json_object = # Code to select the result item we want to extract

task = parsed_json_object["task"]
num_shots = parsed_json_object["in_context_params"]["num_shots"]
scores = eval(parsed_json_object["score"])
for metric, score in scores.items():
    if "server_results" in metric:
        prompt_id = parsed_json_object["prompt_template_id"]
        num_shots = parsed_json_object["in_context_params"]["num_shots"]
        max_new_tokens = parsed_json_object["text_generation_params"]["max_new_tokens"]
        checkpoint_id = parsed_json_object["model_name_or_path"].split("/")[-2].split("-")[-1]


        server_results = scores["server_results"]
        # Custom code to format server results, for example for VQAv2:
        # server_results = [{"question_id": int(server_result["question_id"]), "answer": server_result["answer"]} for server_result in server_results]

        output_file_path = result_file.parent / f"{task_name}_test_server_results" / f"CHANGEME_{checkpoint_id}_num_shots_{num_shots}_promptid_{prompt_id}_max_new_toks_{max_new_tokens}_{task_name}_result.json"
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"output_file_path: {output_file_path}")


        with open(output_file_path, 'w') as output_file:
            output_file.write(json.dumps(server_results))
```

To date, the tasks that requires a submission to an outside server are:
- VQAv2, format:
```
results = [result]

result{
"question_id": int,
"answer": str
}
```
- VizWiz, format:
```
results = [result]

result = {
    "image": string, # e.g., 'VizWiz_test_00020000.jpg'
    "answer": string
}
```
- TextCaps, format:
```
results = [result]

result = {
    "image_id": string,
    "caption": string
}
```
- NoCaps, format:
```
results = [result]

result = {
    "image_id": int,
    "caption": string
}
```
