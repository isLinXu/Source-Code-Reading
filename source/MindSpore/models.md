# Models

---

该GitHub项目为MindSpore官方提供的模型仓库，其中包含了各种人工智能领域的模型，用户可以在此基础上进行二次开发。

该仓库的项目目录结构如下：

```
models
├── README.md
├── _config.yml
├── vision
│   ├── README.md
│   ├── classification
│   │   ├── lenet
│   │   ├── googlenet
│   │   ├── resnet
│   │   ├── scripts
│   │   ├── vgg
│   ├── detection
│   │   ├── faster_rcnn
│   │   ├── mask_rcnn
│   │   ├── rfcn
│   ├── segmentation
│   │   ├── unet
│   │   ├── deeplabv3
│   ├── scripts
│   │   ├──_prepare_dataset.sh
│   │   ├──run_standalone_train.sh
│   ├── mindspore_hub_conf.md
├── nlp
    ├── README.md
    ├── bert
    ├── scripts
    ├── mindspore_hub_conf.md
```

现对各个文件（夹）进行介绍：

- [README.md](http://readme.md/)：项目的说明文档。
- _config.yml：该文件用于配置GitHub Pages相关信息。
- vision：该文件夹下包含了各种计算机视觉领域的模型。
- nlp：该文件夹下包含了各种自然语言处理领域的模型。
- vision/README.md：计算机视觉领域介绍文档。
- vision/classification：该文件夹下包含了各种图像分类领域的模型，如LeNet、GoogLeNet、ResNet和VGG等。
- vision/detection：该文件夹下包含了各种目标检测领域的模型，如Faster-RCNN、Mask-RCNN和RFCN等。
- vision/segmentation：该文件夹下包含了各种语义分割领域的模型，如Unet和DeepLabV3等。
- vision/scripts：该文件夹下包含了各个模型的训练和评估脚本。
- nlp/README.md：自然语言处理领域介绍文档。
- nlp/bert：该文件夹下包含了BERT预训练模型和Fine-tuning模型。
- nlp/scripts：该文件夹下包含了BERT模型的训练和评估脚本。

该仓库中的MindSpore模型也都支持训练、推理和转换为ONNX、TensorFlow等格式。