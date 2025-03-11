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

import ray

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.single_controller.ray.base import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup, create_colocated_worker_cls

from verl import DataProto


@ray.remote
class Actor(Worker):

    def __init__(self) -> None:
        super().__init__()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def add(self, data: DataProto):
        data.batch['a'] += self.rank
        return data


@ray.remote
class Critic(Worker):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def sub(self, data: DataProto):
        data.batch['a'] -= self.config['b']
        return data


def test_colocated_workers():
    ray.init()

    import torch
    data = DataProto.from_dict({'a': torch.zeros(10)})
    # create separate workers on the same resource pool
    actor_cls = RayClassWithInitArgs(cls=Actor)
    critic_cls = RayClassWithInitArgs(cls=Critic, config={'b': 10})
    resource_pool = RayResourcePool(process_on_nodes=[2])

    actor_wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=actor_cls)
    critic_wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=critic_cls)

    expected_actor_output = actor_wg.add(data)
    expected_critic_output = critic_wg.sub(data)

    # create colocated workers
    cls_dict = {'actor': actor_cls, 'critic': critic_cls}
    ray_cls_with_init = create_colocated_worker_cls(cls_dict)
    wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    spawn_wg = wg_dict.spawn(prefix_set=cls_dict.keys())

    colocated_actor_wg = spawn_wg['actor']
    colocated_critic_wg = spawn_wg['critic']

    actor_output = colocated_actor_wg.add(data)
    critic_output = colocated_critic_wg.sub(data)

    torch.testing.assert_close(expected_actor_output.batch, actor_output.batch, atol=0, rtol=0)
    torch.testing.assert_close(expected_critic_output.batch, critic_output.batch, atol=0, rtol=0)

    ray.shutdown()
