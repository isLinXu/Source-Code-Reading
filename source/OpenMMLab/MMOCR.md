# MMOCR

---

这个代码仓库的主要目的是提供OCR（Optical Character Recognition）领域的深度学习方法。

mmocr代码仓库目录下的所有文件介绍:

- configs/目录: 包含了各种类型、规模的OCR模型的配置文件，这些配置文件包括训练、调参、模型结构等。
  - `cascade_mask_rcnn_r50_fpn_ocr_toy_dataset.py`: 基于cascade mask rcnn的ocr模型的示例配置文件。
  - `grrnet/`目录: 包含了使用一种名为Gated Recurrent Residual Network的网络结构的OCR模型的配置文件。
  - `ocrnet/`目录: 包含了使用一种名为OCRNet的网络结构的OCR模型的配置文件。
  - `sarnet/`目录: 包含了使用一种名为SARNet的网络结构的OCR模型的配置文件。
  - `textsnake/`目录: 包含了使用一种名为TextSnake的网络结构的OCR模型的配置文件。
  - `.cache/`目录: 缓存的模型和数据集。
- mmocr/目录: 包含了实现OCR模型的所有代码，其中包括数据处理、训练、推理等功能的实现。
  - `core/`目录: 包含了模型、评估、评价指标等核心功能的实现。
  - `datasets/`目录: 包含了多种OCR数据集的处理函数和预处理方法。
  - `models/`目录: 包含了多种OCR模型的实现。
  - `ops/`目录: 包含了一些特定算法的操作函数实现。
  - `utils/`目录: 包含了各种用于描述、测试、评估和可视化OCR模型的实用函数。
- tools/目录: 包含了用于训练、测试和推理OCR模型的代码。
  - `demo/`目录: 包含了一个快速上手的简单例子和演示环境。
  - `tests/`目录: 包含了对代码及模型实现的测试用例。
  - `train.py`: 用于训练OCR模型的脚本。
  - `test.py`: 用于测试OCR模型性能的脚本。
  - `infer.py`: 用于进行OCR模型推理的脚本。
- `docs/`目录: 包含了用于文档描述、帮助文档、使用手册的文档。
- `tests/`目录: 包含了对代码及模型实现的测试用例。
- `.gitignore`: 列出了Git不应跟踪的文件或文件类型。
- `.pre-commit-config.yaml`: 定义了在提交之前需要运行的代码格式化程序和代码检查工具。
- `LICENSE`: mmocr的开源许可证。
- `README.md`: mmocr的项目说明文档，向用户简要说明了项目的内容和使用方法。
- `requirements.txt`: 所需的Python包和版本。
- `setup.cfg`和`setup.py`: 包含了用于打包、安装和发布mmocr的设置和元数据。