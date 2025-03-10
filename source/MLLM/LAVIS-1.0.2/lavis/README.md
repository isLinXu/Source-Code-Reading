## 项目架构与模块解析

### 整体架构图
```mermaid
graph TD

subgraph 核心模块
    A[模型基类] --> B[ALBEF系列]
    A --> C[BLIP系列]
    A --> D[BLIP-2系列]
    A --> E[ALPRO系列]
    
    B --> F[视觉编码器]
    B --> G[文本编码器]
    C --> F
    C --> G
    D --> H[Q-Former]
    D --> I[大语言模型]
    
    J[数据加载] --> K[预处理]
    K --> L[训练器]
    L --> M[评估器]
    
    F --> N[多模态融合]
    G --> N
    N --> O[任务头]
end

subgraph 支持系统
    P[配置管理] --> Q[YAML配置]
    R[分布式训练] --> S[数据并行]
    T[注册中心] --> U[模块注册]
    V[工具类] --> W[日志/指标/下载]
end
```

### 目录结构详解
```text
lavis/
├── configs/                 # 配置文件目录
│   ├── datasets/            # 数据集配置
│   │   ├── coco/            # COCO数据集配置
│   │   ├── gqa/             # GQA视觉问答配置
│   │   └── ...              # 其他数据集配置
│   └── models/              # 模型配置

├── common/                  # 公共组件
│   ├── dist_utils.py         # 分布式训练工具
│   ├── registry.py           # 模块注册中心
│   ├── utils.py              # 通用工具函数
│   └── vqa_tools/            # VQA评估工具

├── datasets/                 # 数据集处理
│   ├── builders/            # 数据集构建器
│   └── data_utils/          # 数据预处理工具

├── models/                  # 模型实现
│   ├── albef_models/         # ALBEF系列模型
│   ├── blip_models/          # BLIP一代模型
│   ├── blip2_models/         # BLIP-2二代模型
│   └── alpro_models/         # ALPRO视频模型

├── processors/              # 数据处理器
│   ├── image_processor.py    # 图像预处理
│   └── text_processor.py     # 文本预处理

├── tasks/                   # 训练任务
│   ├── caption_task.py       # 图像描述任务
│   └── vqa_task.py          # 视觉问答任务

└── trainers/                # 训练框架
    ├── base_trainer.py       # 基础训练器
    └── lavis_trainer.py      # 定制训练流程
```

### 核心模块说明

#### 1. 模型架构体系
```mermaid
classDiagram
    class BaseModel{
        <<Abstract>>
        +from_pretrained()
        +from_config()
        +forward()
    }
    
    BaseModel <|-- AlbefBase
    BaseModel <|-- BlipBase
    BaseModel <|-- Blip2Base
    
    class AlbefBase{
        +momentum_update()
        +compute_sim_matrix()
        -vision_proj
        -text_proj
    }
    
    class Blip2Base{
        +generate()
        +compute_itc()
        +qformer: QFormer
        +opt_model: OPT
    }
```

#### 2. 数据处理流程
```python
# 典型数据加载流程
dataset = build_dataset("coco_caption")
dataloader = create_loader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True
)

for batch in dataloader:
    images = batch["image"].cuda()
    texts = tokenizer(batch["text"])
    
    # 模型前向
    outputs = model(images, texts)
    
    # 计算多任务损失
    loss = 0.5*outputs.itc_loss + 0.3*outputs.itm_loss + 0.2*outputs.mlm_loss
```

#### 3. 配置管理系统
```yaml
# 典型模型配置 (configs/models/albef_retrieval.yaml)
model:
  arch: albef_retrieval
  model_type: base
  image_encoder:
    name: vit_base_patch16_224
    pretrained: true
  text_encoder:
    name: bert-base-uncased
  embed_dim: 256
  queue_size: 65536
```

### 扩展开发指南

#### 添加新数据集
1. 在`datasets/builders`下创建构建器
2. 实现数据加载逻辑
```python
@registry.register_builder("new_dataset")
class NewDatasetBuilder(BaseDatasetBuilder):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        
    def build(self):
        # 实现数据集构建逻辑
```

#### 自定义模型
1. 继承基类并注册
```python
@registry.register_model("custom_model")
class CustomModel(AlbefBase):
    def __init__(self, image_encoder, text_encoder, custom_param):
        super().__init__()
        
    def forward(self, samples):
        # 实现自定义前向逻辑
```

### 训练流程示例
```mermaid
sequenceDiagram
    participant Trainer
    participant Model
    participant Dataloader
    
    Trainer->>Dataloader: 加载数据
    loop 每个epoch
        Trainer->>Model: 训练模式
        Dataloader->>Model: 输入批次数据
        Model-->>Trainer: 返回损失
        Trainer->>Optimizer: 参数更新
    end
    
    Trainer->>Model: 评估模式
    Dataloader->>Model: 输入验证数据
    Model-->>Trainer: 返回指标
```

该文档补充了：
1. 完整的系统架构图
2. 详细的目录结构解析
3. 核心模块的类图表示
4. 典型数据流和训练流程
5. 扩展开发的具体示例