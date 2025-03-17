curl http://${LOCAL_IP}:8020/v1/chat/completions -H 'Content-Type: application/json' -d '{
"model": "${MODEL_PATH}",
"messages": [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "请按面积大小对四大洋进行排序，并给出面积最小的洋是哪一个？直接输出结果。"
    }
],
"max_tokens": 2048,
"temperature":0.7,
"top_p": 0.6,
"top_k": 20,
"repetition_penalty": 1.05,
"stop_token_ids": [127960]
}'