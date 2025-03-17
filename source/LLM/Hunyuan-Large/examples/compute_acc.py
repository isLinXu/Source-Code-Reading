import json
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 待评测文件
json_file = sys.argv[1]

def compute_car(json_file):
    with open(json_file, encoding="utf-8") as f:
        annos = f.readlines()
    label_names = set()
    input2pred = {}
    input2label = {}
    for anno in annos:
        anno = json.loads(anno)
        # 多个预测结果按；分割
        input2pred[anno["input"]] = anno["output"][0].split("；")
        for name in anno["output"][0].split("；"):
            label_names.add(name)
        input2label[anno["input"]] = anno["label"][0].split("；")
        for name in anno["label"][0].split("；"):
            label_names.add(name)
    # 收集labels
    label_names = list(label_names)
    labels = []
    predictions = []
    for input, label in input2label.items():
        # one-hot padding
        label_zero = [0] * len(label_names)
        for _ in label:
            label_zero[label_names.index(_)] = 1
        labels.append(label_zero)  
        pred_zero = [0] * len(label_names)
        for _ in input2pred[input]:
            pred_zero[label_names.index(_)] = 1
        predictions.append(pred_zero)  
    # print(labels[0])
    # print(predictions[0])
    # 打印统计指标
    print("acc:", accuracy_score(labels, predictions))
    print("p:", precision_score(labels, predictions, average='micro'))
    print("r:", recall_score(labels, predictions, average='micro'))
    print("f1:", f1_score(labels, predictions, average='micro'))

if __name__ == "__main__":
    compute_car(json_file)
