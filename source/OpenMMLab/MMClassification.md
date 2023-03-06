# MMClassification

---

该代码仓库的目录结构如下：

```
mmclassification/
├── configs/
│   ├── backbones/
│   ├── datasets/
│   ├── models/
│   ├── schedules/
│   ├── __init__.py
│   └── config.py
├── mmcls/
│   ├── apis/
│   ├── core/
│   ├── datasets/
│   ├── models/
│   ├── optimizers/
│   ├── runner/
│   ├── utils/
│   ├── __init__.py
│   └── version.py
├── tests/
│   ├── data/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_inference.py
│   ├── test_models.py
│   ├── test_train.py
│   └── test_utils.py
├── tools/
│   ├── data/
│   ├── __init__.py
│   ├── compress.py
│   ├── convert_datasets.py
│   ├── convert_models.py
│   ├── extract_feature.py
│   ├── inference.py
│   ├── prepare_data.py
│   ├── test.py
│   ├── train.py
│   └── utils/
├── .gitignore
├── .flake8
├── .pre-commit-config.yaml
├── .travis.yml
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

其中，每个文件/目录的详细介绍如下：

- configs/: 配置文件目录，包含了各种模型的配置文件，如网络结构、超参数、数据集等信息。
  - backbones/: 存放各种网络的配置文件。
  - datasets/: 存放各种数据集的配置文件。
  - models/: 存放各种模型的配置文件。
  - schedules/: 存放各种学习率调度器的配置文件。
  - **init**.py: 初始化文件。
  - config.py: 定义了Config类，用于解析和保存配置文件。
- mmcls/: 框架核心代码目录，包含了数据读取、模型定义、训练、测试、推理等功能。
  - apis/: 存放了一些高层级的API，如训练、测试、推理等。
  - core/: 存放了框架的核心代码，如模型、数据读取、损失函数、评价指标等。
  - datasets/: 存放了各种数据集的代码实现。
  - models/: 存放了各种模型的代码实现。
  - optimizers/: 存放了各种优化器的代码实现。
  - runner/: 存放了训练和测试的代码实现。
  - utils/: 存放了一些常用的工具函数。
  - **init**.py: 初始化文件。
  - version.py: 定义了框架的版本号。
- tests/: 单元测试目录，包含了对框架各个组件的单元测试。
  - data/: 存放了测试数据。
  - **init**.py: 初始化文件。
  - test_data.py: 测试数据读取和预处理的功能。
  - test_inference.py: 测试推理的功能。
  - test_models.py: 测试模型的功能。
  - test_train.py: 测试训练的功能。
  - test_utils.py: 测试工具函数的功能。
- tools/: 工具目录，包含了一些辅助工具，如模型转换、模型融合、数据预处理等。
  - data/: 存放了一些工具需要的数据。
  - **init**.py: 初始化文件。
  - compress.py: 模型压缩工具。
  - convert_datasets.py: 数据集转换工具。
  - convert_models.py: 模型转换工具。
  - extract_feature.py: 特征提取工具。
  - inference.py: 推理工具。
  - prepare_data.py: 数据预处理工具。
  - test.py: 测试工具。
  - train.py: 训练工具。
  - utils/: 存放了一些常用的工具函数。
- .gitignore: Git忽略文件列表。
- .flake8: Flake8代码风格检查配置文件。
- .pre-commit-config.yaml: pre-commit配置文件。
- .travis.yml: Travis CI配置文件。
- LICENSE: 开源协议文件。
- README.md: 仓库的说明文件，介绍了框架的概述、安装、使用等信息。
- requirements.txt: 依赖包列表。
- setup.py: 安装脚本，用于安装框架及其依赖。