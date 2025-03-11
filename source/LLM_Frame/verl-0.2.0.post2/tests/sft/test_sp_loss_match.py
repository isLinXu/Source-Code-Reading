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
import torch.distributed
from tensordict import TensorDict
from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
from torch.distributed.device_mesh import init_device_mesh
from verl.utils.distributed import initialize_global_process_group


def test_trainer_forward_consistency(trainer: FSDPSFTTrainer, total_steps: int = 4):
    """Test consistency between original forward pass and SP+rmpad forward passes.
    
    Args:
        trainer: The FSDPSFTTrainer instance to test
        total_steps: Number of steps to test (default: 4)
    """
    if trainer.device_mesh.get_rank() == 0:
        print("\nStarting debug comparison between original and SP+rmpad forward passes...")
        print(f"Sequence parallel size: {trainer.config.ulysses_sequence_parallel_size}")
        print(f"Remove padding: {trainer.use_remove_padding}\n")

    steps_remaining = total_steps

    for epoch in range(1):  # Just one epoch for testing
        trainer.train_sampler.set_epoch(epoch=epoch)
        for data in trainer.train_dataloader:
            data = TensorDict(data, batch_size=trainer.config.data.train_batch_size).cuda()
            trainer.fsdp_model.train()
            micro_batches = data.split(trainer.config.data.micro_batch_size)

            for idx, micro_batch in enumerate(micro_batches):
                if trainer.device_mesh.get_rank() == 0:
                    print(f"\nProcessing micro batch {idx + 1}/{len(micro_batches)}")

                # Compute losses using both methods
                # Disable SP and rmpad
                trainer.use_remove_padding = False
                old_sp = trainer.config.ulysses_sequence_parallel_size
                trainer.config.ulysses_sequence_parallel_size = 1
                loss_ref = trainer._compute_loss_and_backward(micro_batch.copy(), do_backward=False)

                # Do SP and rmpad
                trainer.config.ulysses_sequence_parallel_size = old_sp
                trainer.use_remove_padding = True
                loss_sp = trainer._compute_loss_and_backward(micro_batch.copy(), do_backward=False)

                # Collect losses across all ranks
                loss_ref_all = loss_ref.clone()
                loss_sp_all = loss_sp.clone()
                torch.distributed.all_reduce(loss_ref_all, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(loss_sp_all, op=torch.distributed.ReduceOp.AVG)

                # Calculate relative difference of averaged losses
                rel_diff = torch.abs(loss_ref_all - loss_sp_all) / (torch.abs(loss_ref_all) + 1e-8)

                if trainer.device_mesh.get_rank() == 0:
                    print("\nComparison Results (Averaged across ranks):")
                    print(f"Reference Loss: {loss_ref_all.item():.6f}")
                    print(f"SP+rmpad Loss: {loss_sp_all.item():.6f}")
                    print(f"Relative Difference: {rel_diff.item():.6f}")

                    assert rel_diff.item() < 1e-2, "Significant difference detected between averaged losses!"
                    print("Loss difference is within the acceptable range.")

                steps_remaining -= 1
                if steps_remaining == 0:
                    break
            if steps_remaining == 0:
                break
        break

    if trainer.device_mesh.get_rank() == 0:
        print("\nDebug comparison completed successfully.")


def create_trainer(config):
    """Create and initialize a trainer instance with the given config.
    
    Args:
        config: Configuration object with training parameters
        
    Returns:
        FSDPSFTTrainer: Initialized trainer instance
    """
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type='cuda', mesh_shape=(world_size,), mesh_dim_names=('fsdp',))

    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(device_type='cuda',
                                           mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
                                           mesh_dim_names=('dp', 'sp'))

    return FSDPSFTTrainer(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh)


def main(config):
    """Main function to run trainer tests.
    
    Args:
        config: Configuration object with training parameters
    """
    trainer = create_trainer(config)
    test_trainer_forward_consistency(trainer)


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="../../verl/trainer/config", config_name="sft_trainer")
    def hydra_entry(cfg: DictConfig) -> None:
        main(cfg)

    hydra_entry()
