from dataset import get_dataset
from distilabel.llms import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import TextGeneration

with Pipeline(name="smol-rewrite").ray() as pipeline:
    TextGeneration(
        llm=vLLM(
            model="Qwen/Qwen2.5-72B-Instruct",
            tokenizer="Qwen/Qwen2.5-72B-Instruct",
            generation_kwargs={
                "temperature": 0.2,
                "max_new_tokens": 1024,
                "top_p": 0.95,
            },
            extra_kwargs={
                "tensor_parallel_size": 8,
                "max_model_len": 4096,
                "enable_prefix_caching": True,
            },
        ),
        input_batch_size=1000,
        resources=StepResources(replicas=4),
    )


if __name__ == "__main__":
    dataset = get_dataset()
    distiset = pipeline.run(dataset=dataset, dataset_batch_size=10000, use_cache=True)
    distiset.push_to_hub("HuggingFaceTB/smollm-v2-rewriting")
