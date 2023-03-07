# MMSegmentation

---

这个代码仓库的主要目的是提供分割（segmentation）领域的深度学习方法。

mmsegmentation 代码仓库目录下的所有文件介绍:

- configs/

  目录: 包含了各种各样的用于训练和测试图像分割的配置文件，这些文件包括训练时的所有超参数以及网络结构。

  - `cascade_mask_rcnn_r50_fpn_1x_coco.py`: 基于cascade mask rcnn的分割模型的配置文件。
  - `deeplabv3/`目录: 包含了使用 DeepLabV3 网络的分割模型的配置文件。
  - `encnet/`目录: 包含了使用 Efficient Context Encoding 网络的分割模型的配置文件。
  - `fcn/`目录: 包含了使用 Fully Convolutional Networks 网络的分割模型的配置文件。
  - `gcn/`目录: 包含了使用 GCN 网络的分割模型的配置文件。
  - `hrnet/`目录: 包含了使用 HRNet 网络的分割模型的配置文件。
  - `ocrnet/`目录: 包含了使用 OCRNet 网络的分割模型的配置文件。
  - `pspnet/`目录: 包含了使用 PSPNet 网络的分割模型的配置文件。
  - `segformer/`目录: 包含了使用 SegFormer 网络的分割模型的配置文件。
  - `unet/`目录: 包含了使用 U-Net 网络的分割模型的配置文件。

- mmseg/

  目录: 包含了实现分割模型的所有代码，其中包括数据处理，训练，推理等功能的实现。

  - `apis/`目录: 包含了与模型交互的顶级API。
  - `core/`目录: 包含了模型训练和推理的核心函数实现。
  - `datasets/`目录: 包含了用于训练和测试分割模型的数据集的处理函数和预处理方法。
  - `models/`目录: 包含了多种分割模型的实现。
  - `utils/`目录: 包含了各种用于描述、测试、评估和可视化这些分割模型的实用函数。
  - `version.py`: 包含mmsegmentation安装的版本信息。

- tools/

  目录: 包含了各种各样的用于模型训练，模型测试和模型转换的脚本。

  - `convert_datasets_to_coco.py`: 将自定义数据集转换为COCO格式。
  - `train.py`: 用于启动训练的脚本。
  - `test.py`: 用于测试模型性能的脚本。
  - `inference.py`: 用于进行推理的脚本。

- `configs.py`: 定义了一些默认的超参数和其他一些全局配置项，迁移学习权重下载链接，预训练好的权重文件位置以及一些默认的数据集配置。

- `.gitignore`: 列出不需要Git跟踪的文件或文件夹。

- `.flake8`: 定义了flake8代码检查工具的规则和配置。

- `.pre-commit-config.yaml`: 定义了在提交之前需要运行的代码格式化程序和代码检查工具。

- `LICENSE`: 授予使用mmsegmentation的开源许可证。

- `README.md`: 项目说明文档，向用户简要说明了项目的内容和使用方法。

- `requirements.txt`: 所需的Python包和版本。

- `setup.cfg` 和 `setup.py`: 包含了安装和打包信息。