## Locally
Run the `run_document_ngrams_extraction.sh` script.

## On JZ
On JZ:
- Add to your `~/.bashrc` the following line (custom installation of `jq` and `parallel`):
```bash
export PATH=$PATH:/gpfswork/rech/six/commun/lib/jq-1.5/bin/:/gpfswork/rech/six/commun/lib/parallel/bin/
```

Then, run the slurm script (`sbatch pipe.slurm`).
