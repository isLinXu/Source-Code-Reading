SLURM driven cronjobs to manage asynchronously various tasks around checkpoints

1. a slurm cronjob to convert new checkpoints to hf
2. a slurm cronjob to launch multiple evals when it finds a new hf checkpoint
3. a slurm cronjob to launch s3 sync to clear disc space (checkpoints and other files)
4. a slurm cronjob to delete checkpoints that got eval'ed and synced already (to clear disc space)

All made to work with potentially overlapping slurm jobs and time-based recovery from aborted jobs - requires tuning for estimated run-time of each job for fastest recovery.

The jobs are self-replicating - they will re-schedule themselves before doing the actual work. Each job defines its repetition frequency inside its slurm file. The recommendation for a good frequency is at about the same speed as the frequency of saving checkpoints.

To launch them all:

```
sbatch experiments/pretraining/vloom/slurm_scripts_templates/hfc_with_launcher/cleanup-checkpoints.slurm
sbatch experiments/pretraining/vloom/slurm_scripts_templates/hfc_with_launcher/convert-checkpoints.slurm
sbatch experiments/pretraining/vloom/slurm_scripts_templates/hfc_with_launcher/s3-upload-checkpoints.slurm
sbatch experiments/pretraining/vloom/slurm_scripts_templates/hfc_with_launcher/schedule-evals.slurm
```

To run these manually instead, do:

```
m4/scripts/cleanup-checkpoints.py   /fsx/m4/experiments/local_experiment_dir/tr-XXX/
m4/scripts/convert-checkpoints.py   /fsx/m4/experiments/local_experiment_dir/tr-XXX/
m4/scripts/s3-upload-checkpoints.py /fsx/m4/experiments/local_experiment_dir/tr-XXX/
m4/scripts/schedule-evals.py        /fsx/m4/experiments/local_experiment_dir/tr-XXX/
```

The jobs can recover from aborted jobs. They rely on pre-configured heuristics of the longest time each job could run. If a new job detects the previous job hasn't finished in that pre-configured time it assumes it fails and will start it again. Since we have jobs running on different nodes we can't rely on PIDs and instead use special files on the shared file system and check the staleness of these file using `mtime` to tell how long again some job has started.

If you don't want to wait for the safety period to elapse and want to force re-processing, almost all scripts come with `-f` option which will ignore the heuristics that make it safe to have overlapping jobs and force a re-run. Only `cleanup-checkpoints.slurm` doesn't have it since we never should force deletion without solid heuristics check.
