# Generation Process:

- find one or more opt-step checkpoints to make generations with
- create folder in code/m4/experiments/generations
- add a config.yaml and a [gen_folder_name]_generate.slurm folder
- fill the config file according to desired hyperparameters: prompt/num_beams/ngram_repeats etc..
- run sbatch [m4_repo_name]/experiments/generation/[gen_folder_name]/[gen_folder_name]_generate.slurm
- check wandb and make sure your column shows up. If it doesn't, click on "columns" at the bottom right of the generation table and slide the missing generation to the "Displayed columns" side
