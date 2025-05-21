import argparse
import os


parser = argparse.ArgumentParser("Quickly launch thom's style of tokenization.")

# python /fsx/loubna/projects/datatrove/examples/edu_fw.py hf://datasets/bigcode/stackoverflow-clean stackoverflow --n_tasks 50 --tokenizer HuggingFaceTB/cosmo2-tokenizer


# parser.add_argument("data_path", type=str, help="Path to the data to tokenize.")
parser.add_argument("output_name", type=str, help="Output name.")
parser.add_argument("--n_tasks", type=int, help="nb of tokenization tasks", default=100)
parser.add_argument("--max_toks", type=int, help="max tokens per file", default=1e9)
parser.add_argument("--tokenizer", type=str, help="tokenizer to use", default="meta-llama/Llama-3.2-1B")
parser.add_argument("--text_key", type=str, default="text")


if __name__ == "__main__":
    args = parser.parse_args()
    from datatrove.executor import SlurmPipelineExecutor
    from datatrove.pipeline.filters import SamplerFilter
    from datatrove.pipeline.readers import ParquetReader, JsonlReader
    from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
    from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
    
    dic = {
        "finemath-3-plus":"hf://datasets/HuggingFaceTB/finemath/finemath-3plus",
        "finemath-4-plus":"hf://datasets/HuggingFaceTB/finemath/finemath-4plus",
        "infiwebmath-3-plus":"hf://datasets/HuggingFaceTB/finemath/infiwebmath-3plus",
        "infiwebmath-4-plus":"hf://datasets/HuggingFaceTB/finemath/infiwebmath-4plus",
        # "fw-edu-dedup": "hf://datasets/HuggingFaceTB/smollm-corpus/fineweb-edu-dedup",
        # "infiwebmath-ablation-new": "hf://datasets/Infi-MM/InfiMM-WebMath-40B",
        # "owm-ablation": "hf://datasets/open-web-math/open-web-math/data",
    }
    for name, path in dic.items():
        dist_executor = SlurmPipelineExecutor(
            job_name=f"tok-{name}",
            pipeline=[
                ParquetReader(
                    path, # read directly from huggingface
                    glob_pattern="*.parquet", # "**/*.parquet", 
                    text_key=args.text_key,
                ),
                #SamplerFilter(rate=0.5),
                DocumentTokenizer(
                    output_folder=f"/fsx/elie_bakouch/data/{name}",
                    tokenizer_name_or_path=args.tokenizer,
                    batch_size=10000,
                    max_tokens_per_file=args.max_toks,
                    shuffle=True,
                ),
            ],
            tasks=args.n_tasks,
            time="20:00:00",
            partition="hopper-cpu",
            logging_dir=f"/fsx/elie_bakouch/tokenize_logs/fw-edu-classico/{name}",
            cpus_per_task=32,
            mem_per_cpu_gb=2,
            qos="high",
            mail_user=args.email,
        )
        dist_executor.run()