set -x

# prepare pre-trained model ckpt
huggingface-cli download deepseek-ai/deepseek-llm-7b-chat --local-dir $HOME/models/deepseek-llm-7b-chat

# ``actor_rollout_ref.rollout.tensor_model_parallel_size`` in theory could be different from
# ``**.megatron.tensor_model_parallel_size``

# the config file used: verl/trainer/main_ppo/config/ppo_megatron_trainer.yaml

python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.val_batch_size=1312 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=$HOME/models/deepseek-llm-7b-chat \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
    critic.optim.lr=2e-5 \
    critic.model.path=$HOME/models/deepseek-llm-7b-chat \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.megatron.tensor_model_parallel_size=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_megatron_gsm8k_examples' \
    trainer.experiment_name='deepseek_llm_7b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=15 \
    +trainer.val_before_train=False $@
