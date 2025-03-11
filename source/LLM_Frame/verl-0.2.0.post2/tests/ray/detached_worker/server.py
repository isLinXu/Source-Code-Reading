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
"""
Server starts a Trainer. Client sends data to the server to train.
"""

import os

os.environ['MEGATRON_USE_CUDA_TIMER'] = '0'
os.environ['MEGATRON_START_PROCESS_TIMER'] = 'False'
os.environ['NCCL_DEBUG'] = 'WARN'

import torch
from torch import nn

import ray
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool
from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
from verl.single_controller.base.megatron.worker import MegatronWorker
from verl.single_controller.base.decorator import register, Dispatch
from verl import DataProto
from verl.models.llama.megatron import ParallelLlamaForCausalLMRmPadPP

from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_model import ModelType
from megatron.core import tensor_parallel
from verl.utils.megatron_utils import get_model, init_megatron_optim_config, init_model_parallel_config
from verl.utils.megatron.optimizer import get_megatron_optimizer

from transformers import LlamaConfig

from omegaconf import OmegaConf

from tensordict import TensorDict


@ray.remote
class Trainer(MegatronWorker):

    def __init__(self):
        super().__init__()

        if not torch.distributed.is_initialized():
            rank = int(os.environ['LOCAL_RANK'])
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(rank)

            os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=2,
                pipeline_model_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=1,
                expert_model_parallel_size=1,
                nccl_communicator_config_path=None,
            )
            tensor_parallel.model_parallel_cuda_manual_seed(10)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        actor_model_config = LlamaConfig(vocab_size=256,
                                         hidden_size=2048,
                                         intermediate_size=5504,
                                         num_hidden_layers=24,
                                         num_attention_heads=16,
                                         num_key_value_heads=16)

        megatron_config = OmegaConf.create({
            'sequence_parallel_enabled': True,
            'param_dtype': 'bf16',
            'pipeline_model_parallel_rank': mpu.get_pipeline_model_parallel_rank(),
            'pipeline_model_parallel_size': mpu.get_pipeline_model_parallel_world_size(),
            'virtual_pipeline_model_parallel_rank': mpu.get_virtual_pipeline_model_parallel_rank(),
            'virtual_pipeline_model_parallel_size': mpu.get_virtual_pipeline_model_parallel_world_size()
        })

        megatron_config = init_model_parallel_config(megatron_config)
        self.megatron_config = megatron_config

        def megatron_actor_model_provider(pre_process, post_process):
            # vpp is not supported yet because it will hang for some reason. Need debugging
            vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()  # this will be set inside get_model
            # this_megatron_config = copy.deepcopy(megatron_config)
            # this_megatron_config.virtual_pipeline_model_parallel_rank = vpp_rank
            parallel_model = ParallelLlamaForCausalLMRmPadPP(config=actor_model_config,
                                                             megatron_config=megatron_config,
                                                             pre_process=pre_process,
                                                             post_process=post_process)
            parallel_model.cuda()
            return parallel_model

        actor_module = get_model(model_provider_func=megatron_actor_model_provider,
                                 model_type=ModelType.encoder_or_decoder,
                                 wrap_with_ddp=True)
        actor_module = nn.ModuleList(actor_module)

        optim_config = OmegaConf.create({'lr': 1e-6, 'clip_grad': 1.0})

        optim_config = init_megatron_optim_config(optim_config)
        self.optimizer_config = optim_config
        actor_optimizer = get_megatron_optimizer(model=actor_module, config=optim_config)

        self.model = actor_module[0]
        self.optimizer = actor_optimizer

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    def train_model(self, data: DataProto) -> DataProto:
        input_ids = data.batch['input_ids']
        attention_mask = data.batch['attention_mask']
        position_ids = data.batch['position_ids']

        self.optimizer.zero_grad()
        self.model.zero_grad_buffer(
            zero_buffer=(not self.optimizer_config.use_distributed_optimizer
                        ))  # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
        # update for 1 iteration
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids).logits
        output.mean().backward()

        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step(self.megatron_config,
                                                                              self.megatron_config.timers)

        return DataProto(batch=TensorDict({'loss': output.detach()}, batch_size=output.shape[0]))


if __name__ == '__main__':
    ray.init(address='auto', namespace='verl')

    resource_pool = RayResourcePool(process_on_nodes=[2], detached=True)
    cls_with_init_args = RayClassWithInitArgs(cls=Trainer)
    worker_group = NVMegatronRayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=cls_with_init_args,
        name_prefix='trainer',
        detached=True,
    )

    worker_group.init_model()

    worker_names = worker_group.worker_names
    print(worker_names)
