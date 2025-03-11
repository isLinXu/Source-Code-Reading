# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import copy
import torch.distributed
from torch.distributed import init_device_mesh
from verl.utils.distributed import initialize_global_process_group
from verl.utils.model import create_random_mask, compute_position_id_with_mask
from verl.utils.torch_functional import masked_mean, log_probs_from_logits_all_rmpad, logprobs_from_logits
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, set_ulysses_sequence_parallel_group
from verl.workers.sharding_manager import FSDPUlyssesShardingManager
from verl.models.transformers.llama import llama_flash_attn_forward
from verl.models.transformers.qwen2 import qwen2_flash_attn_forward
from verl.protocol import DataProto
from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis, rearrange

from transformers import LlamaConfig, MistralConfig, GemmaConfig, Qwen2Config
from transformers.models.llama.modeling_llama import LlamaFlashAttention2
from transformers.models.qwen2.modeling_qwen2 import Qwen2FlashAttention2
from transformers import AutoModelForCausalLM, AutoModelForTokenClassification
# TODO(sgm): add more models for test
# we only need one scale for each model
test_configs = {
    'llama': (LlamaConfig(num_hidden_layers=2), LlamaFlashAttention2),
    'qwen2': (Qwen2Config(num_hidden_layers=2), Qwen2FlashAttention2)
}

patches = {'llama': llama_flash_attn_forward, 'qwen2': qwen2_flash_attn_forward}


def sync_model_parameters_global(layer):
    # synchronize weights
    for p in layer.parameters():
        torch.distributed.broadcast(tensor=p.data, src=0)


def test_hf_casual_fwd():
    assert torch.cuda.device_count() >= 2, "need at least 2 gpus for test"

    sp_size = 8
    dp_size = 1
    ulysses_device_mesh = init_device_mesh(device_type='cuda',
                                           mesh_shape=(dp_size, sp_size),
                                           mesh_dim_names=('dp', 'sp'))
    sharding_manager = FSDPUlyssesShardingManager(ulysses_device_mesh)

    batch_size = 1
    seqlen = 128
    response_length = 127

    for model_name, (config, attn) in test_configs.items():
        # patch before load
        attn.forward = patches[model_name]
        with torch.device('cuda'):
            model = AutoModelForCausalLM.from_config(config=config,
                                                     torch_dtype=torch.bfloat16,
                                                     attn_implementation='flash_attention_2')
            model = model.to(device='cuda')
            sync_model_parameters_global(model)

        # different rank will generate different input_ids following fsdp
        input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seqlen), device='cuda')
        attention_mask = create_random_mask(input_ids=input_ids,
                                            max_ratio_of_left_padding=0,
                                            max_ratio_of_valid_token=0.9,
                                            min_ratio_of_valid_token=0.8)
        position_ids = compute_position_id_with_mask(
            attention_mask)  # TODO(sgm): we can construct the position_ids_rmpad here

        model_inputs = {
            'input_ids': input_ids.cuda(),
            'attention_mask': attention_mask.cuda(),
            'position_ids': position_ids.int().cuda()
        }

        model_inputs = DataProto.from_dict(model_inputs)

        # 1. perform ulysses forward
        with sharding_manager:
            model_inputs = sharding_manager.preprocess_data(model_inputs)
            input_ids = model_inputs.batch['input_ids']
            attention_mask = model_inputs.batch['attention_mask']
            position_ids = model_inputs.batch['position_ids']
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                       attention_mask)  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
            # unpad the position_ids to align the rotary
            position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                  indices).transpose(0, 1)

            # slice input tensor for ulysses
            # input_ids are padded and sliced
            # postition_ids are only padded but not sliced
            input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())

            # input with input_ids_rmpad and postition_ids to enable flash attention varlen
            logits_split_in_seq = model(input_ids_rmpad_sliced, position_ids=position_ids_rmpad_padded,
                                        use_cache=False).logits  # (1, total_nnz/n, vocab_size)

            # all_gather output
            logits_full = gather_outpus_and_unpad(logits_split_in_seq, gather_dim=1, unpad_dim=1, padding_size=pad_size)

        # 2. perform normal forward
        set_ulysses_sequence_parallel_group(None)
        logits_rmpad_local = model(input_ids_rmpad, position_ids=position_ids_rmpad,
                                   use_cache=False).logits  # (1, total_nnz, vocab_size)

        mean_local = logits_rmpad_local.mean()
        mean_full = logits_full.mean()
        torch.testing.assert_close(mean_local, mean_full, rtol=1e-2, atol=1e-5)
    print(f'Fwd Check pass')


def test_hf_casual_fwd_bwd():
    assert torch.cuda.device_count() >= 2, "need at least 2 gpus for test"

    sp_size = 8
    dp_size = 1
    ulysses_device_mesh = init_device_mesh(device_type='cuda',
                                           mesh_shape=(dp_size, sp_size),
                                           mesh_dim_names=('dp', 'sp'))
    sharding_manager = FSDPUlyssesShardingManager(ulysses_device_mesh)

    batch_size = 1
    seqlen = 128
    response_length = 127

    for model_name, (config, attn) in test_configs.items():
        # patch before load
        attn.forward = patches[model_name]
        with torch.device('cuda'):
            model = AutoModelForCausalLM.from_config(config=config,
                                                     torch_dtype=torch.bfloat16,
                                                     attn_implementation='flash_attention_2')
            model = model.to(device='cuda')
            sync_model_parameters_global(model)

        # different rank will generate different input_ids following fsdp
        input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seqlen), device='cuda')
        attention_mask = create_random_mask(input_ids=input_ids,
                                            max_ratio_of_left_padding=0,
                                            max_ratio_of_valid_token=0.9,
                                            min_ratio_of_valid_token=0.8)
        position_ids = compute_position_id_with_mask(
            attention_mask)  # TODO(sgm): we can construct the position_ids_rmpad here

        model_inputs = {
            'input_ids': input_ids.cuda(),
            'attention_mask': attention_mask.cuda(),
            'position_ids': position_ids.int().cuda()
        }

        model_inputs = DataProto.from_dict(model_inputs)

        # 1. perform ulysses forward
        with sharding_manager:
            model_inputs = sharding_manager.preprocess_data(model_inputs)
            input_ids = model_inputs.batch['input_ids']
            attention_mask = model_inputs.batch['attention_mask']
            position_ids = model_inputs.batch['position_ids']
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                       attention_mask)  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
            # unpad the position_ids to align the rotary
            position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                  indices).transpose(0, 1)

            # slice input tensor for ulysses
            # input_ids are padded and sliced
            # postition_ids are only padded but not sliced
            input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())

            # input with input_ids_rmpad and postition_ids to enable flash attention varlen
            logits_split_in_seq = model(input_ids_rmpad_sliced, position_ids=position_ids_rmpad_padded,
                                        use_cache=False).logits  # (1, total_nnz/n, vocab_size)

            # all_gather output
            logits_full = gather_outpus_and_unpad(logits_split_in_seq, gather_dim=1, unpad_dim=1, padding_size=pad_size)

        # 2. perform normal forward
        set_ulysses_sequence_parallel_group(None)
        input_ids_full = copy.deepcopy(input_ids_rmpad)
        position_ids_full = copy.deepcopy(position_ids_rmpad)
        model_no_sp = copy.deepcopy(model)
        logits_rmpad_local = model_no_sp(input_ids_full, position_ids=position_ids_full,
                                         use_cache=False).logits  # (1, total_nnz, vocab_size)

        mean_local = logits_rmpad_local.mean()
        mean_full = logits_full.mean()

        mean_full.backward()
        mean_local.backward()

        # 3. check the gradients
        grad = model.model.layers[0].self_attn.q_proj.weight.grad
        grad_full = model_no_sp.model.layers[0].self_attn.q_proj.weight.grad
        torch.testing.assert_close(grad, grad_full, atol=1e-2, rtol=1e-5)

    print(f'Fwd + BWD Check pass')


if __name__ == '__main__':
    local_rank, rank, world_size = initialize_global_process_group()
    test_hf_casual_fwd()
    test_hf_casual_fwd_bwd()
