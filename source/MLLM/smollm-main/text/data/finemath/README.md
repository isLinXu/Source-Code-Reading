# 📚 FineWeb-Edu pipeline

<center>
    <img src="https://cdn-uploads.huggingface.co/production/uploads/61c141342aac764ce1654e43/wwRnEQydH9qdRtFofIE-A.png" alt="FineWeb-Edu: The finest collection of educational content the web has to offer">
</center>


Here you can find the pipeline for training [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/)'s [classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) and running the annotation on FineWeb.

### 1. Finetune a model for educational value regression

* edit `train_edu_bert.slurm`
```bash
--base_model_name="Snowflake/snowflake-arctic-embed-m" \  # BERT-like base model
--dataset_name="HuggingFaceFW/fineweb-edu-llama3-annotations" \  # Llama3-annotated eduational value dataset
--target_column="score" 
```
* run the training script on a SLURM cluster:
```bash
sbatch train_edu_bert.slurm
```

### 2. Annotate a dataset with the educational scores predicted by the model
    
```bash
sbatch run_edu_bert.slurm
```