import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("opt_step_dir", type=str, help="opt_step_dir with unwrapped_model")
    return parser.parse_args()




def main():
    args = get_args()
    source_dir = Path(args.opt_step_dir) / "unwrapped_model"

    # with open(source_dir / "model.safetensors.index.json", "r") as f:
    #     index = json.load(f)

    # _ = index["metadata"].pop("total_size")
    # with open(source_dir / "model.safetensors.index.json", "w") as json_file:
    #     json.dump(index, json_file, indent=4)

    tensor_files_mapping = {}
    for k in [
        "additional_fc.weight",
        "lm_head.weight",
        "model.embed_tokens.additional_embedding.weight",
        "model.embed_tokens.weight",
    ]:
        tensor_files_mapping[k] = "model.safetensors"

    def load(weight_name):
        tensor_file = tensor_files_mapping[weight_name]
        with safe_open(source_dir / tensor_file, framework="pt") as f:
            return f.get_tensor(weight_name)

    def init_new(matrix, matrix_additional):
        return torch.vstack([matrix, matrix_additional]).mean(dim=0, keepdim=True)

    def pad_to_multiple_of_128(tensor, embedding_matrix):
        current_size = tensor.size(0)
        target_size = ((current_size + 127) // 128) * 128  # Round up to next multiple of 128
        padding_needed = target_size - current_size
        
        if padding_needed == 0:
            return tensor
            
        # Create padding rows by averaging existing embeddings
        padding = init_new(embedding_matrix, tensor).repeat(padding_needed, 1)
        return torch.vstack([tensor, padding])

    def save_modified(weight_name, new_weight):
        tensor_file = tensor_files_mapping[weight_name]
        all_tensors = {}
        with safe_open(source_dir / tensor_file, framework="pt") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)
        all_tensors[weight_name] = new_weight
        save_file(all_tensors, source_dir / tensor_file, metadata={"format": "pt"})

    embed = load("model.embed_tokens.weight")
    embed_additional = load("model.embed_tokens.additional_embedding.weight")
    embed_additional = pad_to_multiple_of_128(embed_additional, embed)
    save_modified("model.embed_tokens.additional_embedding.weight", embed_additional)
    print(f"Resized embedding matrix to {embed_additional.size(0)} rows")

    lm_head = load("lm_head.weight")
    lm_head_additional = load("additional_fc.weight")
    lm_head_additional = pad_to_multiple_of_128(lm_head_additional)
    save_modified("additional_fc.weight", lm_head_additional)
    print(f"Resized lm head matrix to {lm_head_additional.size(0)} rows")


if __name__ == "__main__":
    main()
