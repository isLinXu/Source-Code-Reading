#!/bin/bash

set -ex

nohup python3 -m vllm.entrypoints.openai.api_server --host ${LOCAL_IP} --port 8020 \
--model ${MODEL_PATH} \
--trust-remote-code \
--tensor-parallel-size 8 \
--pipeline-parallel-size 1 \
--max-num-seqs 8 \
--quantization experts_int8 \
2>&1 | tee log_server_int8.txt &

