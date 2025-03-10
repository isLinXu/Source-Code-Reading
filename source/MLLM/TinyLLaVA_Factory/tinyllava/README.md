❤️ Community efforts

* Our codebase is built upon the [LLaVA](https://github.com/haotian-liu/LLaVA) project. Great work!
* Our project uses data from the [ShareGPT4V](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V) project. Great work!

```mermaid
%% TinyLLaVA Factory 核心架构图
graph TD
    A[输入模态] --> B[视觉输入]
    A --> C[音频输入]
    A --> D[文本输入]
    
    B --> E[Vision Tower]
    C --> F[Audio Tower]
    D --> G[Tokenizer]
    
    subgraph 特征处理层
        E --> H[Connector]
        F --> I[Audio Projector]
    end
    
    subgraph 多模态融合
        H --> J[LLM Backbone]
        I --> J
        G --> J
    end
    
    J --> K[文本生成输出]
    
    classDef input fill:#f9f,stroke:#333;
    classDef tower fill:#9cf,stroke:#333;
    classDef processor fill:#cfc,stroke:#333;
    classDef fusion fill:#f96,stroke:#333;
    classDef output fill:#fc9,stroke:#333;
    
    class A,B,C,D input
    class E,F tower
    class H,I processor
    class J fusion
    class K output
```

### 架构说明

#### 数据流

1. **输入处理**：
   
   - 图像通过视觉编码器(CLIP/SigLIP)提取特征
   - 文本通过分词器(Tokenizer)转换为token
   - 特征融合模块将多模态特征对齐
   
2. **模型训练**：
   ```mermaid
   graph LR
   PT[预训练] --> |图像-文本匹配| FT[微调]
   FT --> |多任务学习| EVAL[评估]
   PT --> |Zero3优化| DS[DeepSpeed]
   FT --> |LoRA/QLoRA| PE[参数高效微调]
   ```

3. **核心模块**：
   ```mermaid
   graph TB
   Model[模型架构] --> VT[视觉塔]
   Model --> LLM[语言模型]
   Model --> CN[连接器]
   Config[配置中心] --> Model
   Factory[工厂模式] --> Model
   ```

#### 关键目录说明
| 目录/文件                | 功能描述                                                                 |
|-------------------------|--------------------------------------------------------------------------|
| `scripts/train/`        | 训练脚本，支持全参数微调/LoRA/QLoRA                                     |
| `scripts/eval/`         | 评估脚本，覆盖6大主流多模态基准                                         |
| `tinyllava/model/`      | 模型实现，包含视觉塔/语言模型/连接器                                     |
| `tinyllava/data/`       | 数据处理，支持图像/文本预处理和数据集加载                               |
| `tinyllava/train/`      | 训练核心逻辑，包含DeepSpeed优化和梯度检查点                             |
| `tinyllava/eval/`       | 评估指标实现，包含VQA/TextVQA/MMMU等评估器                              |

#### 典型训练流程

```mermaid
sequenceDiagram
    用户->>+训练脚本: 启动训练
    训练脚本->>+数据加载: 加载预处理数据
    数据加载->>+模型构建: 工厂模式创建模型
    模型构建->>+优化器: 配置训练参数
    优化器->>+训练循环: 前向/反向传播
    训练循环-->>+评估模块: 定期验证
    评估模块-->>+ 用户: 输出评估结果
```

#### 模块交互关系

```mermaid
flowchart TB
    subgraph 输入层
    A[原始图像] --> B[图像预处理]
    C[原始文本] --> D[文本预处理]
    end
    
    subgraph 模型层
    B --> E[视觉编码器]
    D --> F[文本编码器]
    E --> G[跨模态融合]
    F --> G
    G --> H[解码器]
    end
    
    subgraph 输出层
    H --> I[预测结果]
    I --> J[评估指标]
    end
```

建议结合代码文件理解实现细节：
- 模型架构：`modeling_tinyllava.py`
- 视觉编码：`vision_tower/clip.py` 和 `vision_tower/siglip.py`
- 训练流程：`train/train.py`
- 评估逻辑：`eval/model_vqa*.py` 系列文件



```mermaid
pie title 训练优化
"DeepSpeed Zero3" : 45
"LoRA微调" : 30
"梯度检查点" : 15
"混合精度" : 10
```



```mermaid
%% 特征处理流程
sequenceDiagram
    图像->>Vision Tower: 原始像素
    Vision Tower->>Connector: 视觉特征
    音频->>Audio Tower: 波形数据
    Audio Tower->>Audio Projector: 音频特征
    Connector->>LLM: 投影后视觉特征
    Audio Projector->>LLM: 投影后音频特征
    Text->>Tokenizer: 原始文本
    Tokenizer->>LLM: Token嵌入
    LLM->>输出: 多模态融合生成
```







```mermaid
%% 模块实现细节
flowchart LR
    subgraph Vision_Tower[视觉塔]
        VT1[CLIP] --> VT2[CLIPVisionModel]
        VT3[SigLIP] --> VT4[SiglipVisionModel]
    end
    
    subgraph Audio_Tower[音频塔]
        AT1[Whisper] --> AT2[WhisperModel]
    end
    
    subgraph Connector[连接器]
        C1[Linear] --> C2[nn.Linear]
        C3[MLP] --> C4[nn.Sequential]
        C5[Q-Former] --> C6[BertModel]
        C7[Resampler] --> C8[PerceiverResampler]
    end
    
    subgraph LLM[语言模型]
        L1[TinyLlama] --> L2[LlamaForCausalLM]
        L3[Phi-2] --> L4[PhiForCausalLM]
        L5[StableLM] --> L6[StableLmForCausalLM]
    end
    
    Vision_Tower --> Connector
    Audio_Tower --> Audio_Projector[AudioProjector]
    Connector --> LLM
    Audio_Projector --> LLM
```

训练流程示意图：

```mermaid
%% 训练流程
sequenceDiagram
    participant User
    participant Trainer
    participant DataLoader
    participant Model
    participant Optimizer
    
    User->>+Trainer: 启动训练
    Trainer->>+DataLoader: 加载多模态数据
    DataLoader->>+Model: 生成批次数据
    Model->>+Optimizer: 前向传播
    Optimizer->>+Model: 反向传播
    Model-->>-Trainer: 返回损失
    Trainer->>+Model: 参数更新
    loop 每N步验证
        Trainer->>+DataLoader: 加载验证集
        DataLoader->>+Model: 生成验证数据
        Model-->>-Trainer: 返回评估指标
    end
    Trainer-->>-User: 输出最终模型
```

该架构图反映了以下代码设计特点：

1. 模块化设计：

```python
# 示例：Vision Tower工厂模式
def VisionTowerFactory(name):
    for model in VISION_TOWER_FACTORY:
        if model in name.lower():
            return VISION_TOWER_FACTORY[model]()
```

2.动态特征处理：

```python
# Connector模块选择
class LinearConnector(Connector):
    def __init__(self, config):
        self._connector = nn.Linear(...)
```

3.多模态融合：

```python
# modeling_tinyllava.py
def forward(...):
    visual_features = self.connector(vision_output)
    audio_features = self.audio_projector(audio_output)
    inputs_embeds = torch.cat([text_embeds, visual_features, audio_features], dim=1)
```

建议结合以下代码文件理解架构：

modeling_tinyllava.py: 主模型架构

vision_tower/: 视觉特征提取

connector/: 跨模态特征投影

llm/: 语言模型实现

audio_tower/: 音频处理模块

```

### 评估流程

```mermaid
graph TD
    E[评估数据] --> F[MMMU评估集]
    F --> G[多学科测试]
    G --> H[人工评估]
    G --> I[自动评估]
    H --> J[生成报告]
    I --> J
```


