# Webdataset

WebDataset is a Python library that provides a convenient way to work with large datasets that are stored in a remote location, such as an Amazon S3 bucket or a Google Cloud Storage bucket. The library allows to stream data from these remote locations on the fly during the training.

To work on HFC, we need to use this type of solution because we don't have the space to store our data on disk.

## Setup
Incidentally, to stream data from s3 you need:
- install s5cmd with `conda install -c conda-forge s5cmd`
- configure aws with yours credentials
- put a specific command instead of the local paths that were used before, such as `pipe:bash ${WORKING_DIR}/experiments/pretraining/vloom/common/webdataset_get_file.sh ${tar_file_path_on_s3}`
- set in the configuration use_webdataset to true

## Philosophy of webdataset

The idea of Webdataset is to define a sequence of operations, each of which will be applied to the iterable resulting from the previous operation.

Here is an outline of the sequence of operations carried out:
- Input: a list or an iterable of commands or local paths. There are two identified use cases for m4. Either you want to open tar files stored locally, in which case the list should simply contain the local paths to the shards files. Or you want to open tar files stored on s3 and in this case you need to pass a command - `pipe:bash <PATH_TO_M4_CLONE>/experiments/pretraining/vloom/common/webdataset_get_file.sh <S3_URI>` - filled with the uri of the tar file.
- Sharding by node: Each gpu in the training will receive a subset of the iterable of input files. If this list has no duplicates, this means that each gpu will have unique shards.
- Sharding by worker: Each dataloader on each GPU can have several workers. If this is the case, the list of files assigned to a GPU will again be divided between the workers.
- Conversion of the tar file into samples: This step involves re-grouping the tar files that make up a single example. Webdataset is currently not designed for webdocuments that are made up of several texts and images, which is why we have a customised method `group_by_keys_interleaved`. This method also ensures that each example is complete, as piping can result in tar files being cut off in the middle.
- Collation of the samples into instances: This method involves changing the format of the samples to get closer to the format expected by our pipeline.
- Decoding of the images and the texts: So far, the elements in the instances are still bytes. This step converts them to their final python objects (PIL image and string)
- Batching of the examples: To finish the pipeline, we batch `map_batch_size` examples together

## Shuffling

To shuffle the examples, we can either:
1. Pseudo shuffle on the fly
2. Shuffle upstream

### Pseudo shuffle on the fly

When the tar files are not shuffled, or when several epochs are carried out on the same tar files, this is the method available to us.

The idea of on-the-fly pseudo-shuffling is to add buffers between targeted pipeline operations in order to randomly select the examples in this buffer. We have configuration variables to define the length of each buffer, but unfortunately the larger the buffer, the more RAM it consumes. In addition, webdataset offers to add a wrap-up phase during which you can start to draw from the buffer without having to wait until the buffer is completely full. By default, there is no shuffling on the fly, the following variables need to be adjusted for each dataset:
```python
shuffle_initial_urls_list: bool
shuffle_before_split_by_node_buffer_size: Optional[int]
shuffle_before_split_by_worker_buffer_size: Optional[int]
shuffle_after_tarfile_to_samples_buffer_size: Optional[int]
shuffle_after_batching_buffer_size: Optional[int]
```

### Shuffle upstream

The idea here is to store into the tar files the samples in a random order so that there is not pseudo-shuffling to do on the fly. In that case, the configuration parameters that need to be set to None/False are:
```python
shuffle_initial_urls_list: bool = False
shuffle_before_split_by_node_buffer_size: Optional[int] = None
shuffle_before_split_by_worker_buffer_size: Optional[int] = None
shuffle_after_tarfile_to_samples_buffer_size: Optional[int] = None
shuffle_after_batching_buffer_size: Optional[int] = None
```

## Resume training

Currently, we don't have a feature to resume a training where from where it left on the previous run in term of data (see "Potential improvements" section).

## Hyper-parameters tuning

On-the-fly streaming of examples from s3 adds hyper-parameters that have to be tuned for almost every scale of experiment. The parameters that will have the greatest influence on each other are: `max_num_images`, `max_seq_len`, `map_batch_size`, `max_num_samples_per_document`, `shuffle_before_split_by_node_buffer_size`, `shuffle_before_split_by_worker_buffer_size`, `shuffle_after_tarfile_to_samples_buffer_size`, `shuffle_after_batching_buffer_size`, `batch_size_per_gpu`, `num_workers` and the time of the forward + backward + opt step.

## Potential improvements

### S3 pipe script

Currently, the s3 piping works by downloading the shard to the node's NVME drive and then piping the file. This solution appears to be sub-optimal because there is an extra write and read on the NVME. However, without this trick, we don't have the full tar files in the pipe, we never get to the end. The current hypothesis is that the internal retry system of libs such as `s5cmd` or `aws s3` do not work with the pipe.

### Disallow list

To have good control over the non-repetition of data in a training that is split into several jobs, a disallow list system should be implemented and used in conjunction with upstream shuffling. It's not a simple feature, especially if you want a perfect implementation. Nevertheless, if you accept losing the end of a few shards, the solution shown in the PR #1307 should provide a good basis.

## Create tar files

All the files used to create the tar files are inside `datasets_processing_scripts/01_tar_datasets_with_jpeg`. For future processing, particular attention should be paid to the order of the files in the tar.

## Debug tips

Currently, if there is a bug in the webdataset pipeline, it will not cause the training to crash. The error will simply be logged and the code will move on. For the future or for debugging, the following change should be considered: `handler=log_and_continue` -> `handler=wds.reraise_exception`.

# Checkpoint extraction

At the end of the training the normal model weights file isn't in the checkpoint and requires a manual extraction, which is done offline and the script now has the luxury of using the whole node's CPU RAM, e.g.

```

cd /fsx/m4/experiments/local_experiment_dir/tr_171-save-load/opt_step-50/accelerator_state/
./zero_to_fp32.py . output.bin
```

The `zero_to_fp32.py` script is already copied into the checkpoint upon checkpoint saving.

We aren't gathering the full model on every save because it's slow and there might not be enough memory to perform that. Therefore we use `stage3_gather_16bit_weights_on_model_save: False` to only having each gpu save its own shards.

# Monitoring a training

What does it mean to monitor a training? The most important things to look at:
- Most importantly, the loss should go down (on average) consistenly. If it diverges, intervention is required (see next section)
- Looking at the colab metrics (with parameters/gradients/activations) is a useful indicator. Weird behaviors (as in explosion) usually precedes a divergence of the loss.
- Is the training still in the slurm queue? If not, intervention is required (see next section)
- Nodes failures (it will most likely make the training crash). You can do a `grep 'srun: error: Node failure'` on the text logs. Reporting them on #science-cluster-support is a good idea.

# How to intervene when a training diverges

In case of rewinding, I recommend starting another plot on WB by setting the `wandb_run_id` inside `resume_run_infos.json` to the empty string. This will create a new run on WB and you can compare the two runs, and hopefully the loss for that new run will not diverge.

Try in order of complexity:
- Rewind and restart (that is essentially reshuffling the data)
- Rewind, decrease the LR and restart
- Rewind, reset the optimizer, set lr=0, restart, train with lr=0 for a bit, then restart again now with restored lr

## How to reset the wandb run on rollback

empty the `wandb_run_id` string in the `resume_run_infos.json`

## How do I rewind to a previous training?

Change the path of the resuming checkpoint in `$SAVE_DIR/latest_opt_step_dir`.

Additionally, if you already started to evaluate the previous checkpoints that you are now discarding, you might want to backup your previous results and roll back your evaluations results json file to the same opt step.

Here's an example script to do it:
```python

from pathlib import Path
import json
import shutil

# ------------ Fill in these variables ------------
roll_back_number = XX # e.g. 4
roll_back_to_opt_step = XX  # e.g. 16000
run_name = "XXX" # e.g. "tr_190_01_64n"
# -------------------------------------------------

main_eval_path = Path(f"/fsx/m4/experiments/local_experiment_dir/evals/results/{run_name}_evaluations.jsonl")
archive_eval_path = main_eval_path.parent / f"{run_name}_evaluations_archive_exp_{roll_back_number}.jsonl"

# First we start by backing up the current evals file
shutil.move(main_eval_path, archive_eval_path)


# Then we select the evals we want to keep

# 1. Load the evals from the archive
parsed_json_objects = []
try:
    with open(archive_eval_path, 'r', encoding="ISO-8859-1") as file:
        json_data = file.read()

    # Split the JSON data into separate entities
    json_entities = json_data.strip().split('\n')

    for entity in json_entities:
        try:
            parsed_json = json.loads(entity)
            parsed_json_objects.append(parsed_json)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON entity:", e)

    print("JSON entities parsing succeeded.")
except IOError as e:
    print("Error reading the file:", e)

# 2. Select the evals we want to keep
cleaned_parsed_json_objects = []
for parsed_json in parsed_json_objects:
    curr_opt_step = int(parsed_json["model_name_or_path"].split("/")[-2].split("-")[-1])
    if curr_opt_step <= roll_back_to_opt_step:
        cleaned_parsed_json_objects.append(parsed_json)

print(len(cleaned_parsed_json_objects), len(parsed_json_objects))

with open(main_eval_path, 'w') as output_file:
    for r in cleaned_parsed_json_objects:
        output_file.write(json.dumps(r) + "\n")
    print("Parsed JSON data saved to", main_eval_path)

```

## How do I decrease/change the LR mid-training?

Tweak the states (there are two fields for the LR) in `opt_step-xxx/accelerator_state/custom_checkpoint.pkl`.

e.g. to reduce lr by 25% do:
```
python -c "import sys, torch; sd=torch.load(sys.argv[1]); \
print(sd['base_lrs']); sd['base_lrs'] = [x*0.75 for x in sd['base_lrs']]; print(sd['base_lrs']); \
print(sd['_last_lr']); sd['_last_lr'] = [x*0.75 for x in sd['_last_lr']]; print(sd['_last_lr']); \
torch.save(sd, sys.argv[1])" opt_step-14500/accelerator_state/custom_checkpoint_0.pkl
```

## How do I reset the optimizer?

Set `load_optimizer_states` to False.

# I detected a bug in the code, and it will require some time to fix.

If you think that fixing a bug might take more than 1 day and fixing it doesn't require the full capacity (in terms of node) that we are training on, then put the job array on hold (`scontrol hold <job_id>`), let people know #science-cluster-planning that they should use the idle GPUs with some other runs. Right now bigcode almost have some smaller jobs to run on the side, so they can squeeze in some jobs and halt them when we are ready to relaunch.
