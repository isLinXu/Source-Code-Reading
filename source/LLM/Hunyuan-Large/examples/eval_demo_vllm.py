# pip install zhipuai
import os
import traceback
import json
import tqdm
import requests
from multiprocessing import Pool


task_file = "car_test.json"
model_path = "/checkpoint-80"
model_flag = model_path.split("/")[-1]
urls = ["http://127.0.0.1:8020/v1/completions"]
save_file = f"results/{model_flag}"

def get_input_text(data):
    if 'input' in data:
        input_text = data['input']
    elif 'prompt' in data:
        input_text = data['prompt']
    elif 'question' in data:
        input_text = data['question']
    elif 'query' in data:
        input_text = data['query']
    return input_text

def process_message(data, url):
    input_text = get_input_text(data)
    # if 'prompt' not in data:
    #     input_text = data['input']
    # else:
    #     input_text = data["prompt"]
    if isinstance(input_text, str):
        input_text = input_text.split("[CHAT_SEP]")
    assert isinstance(input_text, list)
    output_list = []
    history_list = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }]
    try:
        for input_str in input_text:
            history_list.append({"role": "user", "content": input_str})
            response = requests.post(url, data=json.dumps({"model": model_path,
                # "messages": history_list,
                "prompt": input_str,
                "max_tokens": 50,
                "temperature": 0.7,
                "top_p": 0.6,
                "top_k": 20,
                "repetition_penalty": 1.05,
                "stop": ["<|eos|>"]
                }))
            # print(response.json())
            # output = response.json()["choices"][0]["message"]["content"]
            output = response.json()["choices"][0]["text"]
            output_list.append(output)
            history_list.append({"role": "assistant", "content": output})
            print(f"Input: {input_str}\nOutput: {output}\n--------\n")
        if output_list:
            if len(output_list) == 1:
                output_list = output_list[0]
            data["output"] = output_list
            return data
    except:
        traceback.print_exc()
        return {}

already_set  = set()
if os.path.exists(save_file):
    with open(save_file, encoding="utf-8") as f:
        annos = f.readlines()
        for anno in annos:
            anno = json.loads(anno)
            try:
                input_str = get_input_text(anno)
                already_set.add(json.dumps(input_str, ensure_ascii=False))
            except:
                pass


writer = open(save_file, "a+", encoding="utf-8")
def my_callback(anno):
    if anno:
        writer.writelines(json.dumps(anno, ensure_ascii=False)+"\n")

annos = open(task_file, encoding="utf-8").readlines()
with Pool(4) as pool:
    for i, anno in enumerate(tqdm.tqdm(annos)):
        anno = json.loads(anno.split("\t")[0])
        try:
            input_str = get_input_text(anno)
            if json.dumps(input_str, ensure_ascii=False) in already_set:
                print(f"Already process: {input_str}")
                continue
        except:
            continue
        pool.apply_async(process_message, (anno,urls[i%len(urls)]), callback=my_callback)
        # output_list = process_message(anno)
        # if output_list:
        #     anno["output"] = output_list
        #     anno = json.dumps(anno, ensure_ascii=False) + "\n"
        # else:
        #     print("Failed!!!")
    pool.close()
    pool.join()
writer.close()
