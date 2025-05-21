#!/bin/bash
#SBATCH --job-name=ifeval-like-data
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=3
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err
#SBATCH --time=5-00:00:00

set -ex

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

source .venv/bin/activate

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# Get the IP address of the head node
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Start Ray head node
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# Generate a unique Ray tmp dir for the head node
head_tmp_dir="/tmp/ray_tmp_${SLURM_JOB_ID}_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --temp-dir="$head_tmp_dir" \
    --block &

# Give some time to head node to start...
sleep 10

# Start Ray worker nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))

# Start from 1 (0 is head node)
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    worker_tmp_dir="/tmp/ray_tmp_${SLURM_JOB_ID}_worker_$i"
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --temp-dir="$worker_tmp_dir" \
        --block &
    sleep 5
done

# Give some time to the Ray cluster to gather info
sleep 60

# Finally submit the job to the cluster
RAY_ADDRESS="http://$head_node_ip:8265" ray job submit --working-dir pipeline -- python -u pipeline.py
