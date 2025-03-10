# MGM 架构解析

```mermaid
graph TD
    A[核心模块] --> B[模型架构]
    A --> C[数据处理]
    A --> D[训练流程]
    A --> E[评估体系]
    A --> F[服务部署]
    
    subgraph 模型架构
    B1[双视觉编码器] -->|低分辨率特征| B2[Patch信息挖掘]
    B1 -->|高分辨率候选| B2
    B2 --> B3[LLM融合模块]
    B3 --> B4[文本生成]
    end

    subgraph 数据处理
    C1[图像预处理] --> C2[多模态对齐]
    C2 --> C3[指令数据集]
    end

    subgraph 训练流程
    D1[预训练阶段] --> D2[阶段1数据]
    D2 --> D3[图像-文本对]
    D1 --> D4[微调阶段]
    D4 --> D5[阶段2数据]
    D5 --> D6[复杂指令数据]
    end

    subgraph 评估体系
    E1[MMMU评估] --> E2[多学科理解]
    E1 --> E3[复杂推理]
    E2 --> E4[900验证样本]
    E3 --> E5[10500测试样本]
    end

    subgraph 服务部署
    F1[CLI接口] --> F2[单图推理]
    F1 --> F3[OCR增强]
    F1 --> F4[生成模式]
    F5[Gradio服务] --> F6[控制器]
    F5 --> F7[模型工作节点]
    F5 --> F8[Web界面]
    end

    B --> D
    C --> D
    D --> E
    D --> F
```

## 核心模块说明

### 1. 模型架构
- **双视觉编码器**：CLIP-L作为基础视觉编码器，同时处理低分辨率(336px)和高分辨率(672px)输入
- **Patch信息挖掘**：通过交叉注意力机制实现高低分辨率特征融合
- **LLM融合模块**：支持Gemma/Vicuna/LLaMA-3/Mixtral等多种LLM架构

### 2. 训练流程
- **两阶段训练**：
  - 阶段1：图像-文本对预训练（MGM-Pretrain）
  - 阶段2：复杂指令微调（MGM-Instruct）
- **混合精度训练**：使用Flash Attention优化显存效率

### 3. 评估体系
- **MMMU基准**：覆盖6大核心学科30个子领域
- **评估维度**：
  - 图像理解（Hi-Resolution Understanding）
  - 推理生成（Generation with Reasoning）
  - 多学科综合能力

### 4. 服务部署
- **灵活推理模式**：
  ```bash
  python -m mgm.serve.cli --model-path [MODEL] --image-file [IMG]
  ```
- **分布式服务**：
  - 控制器(controller)管理多个模型工作节点
  - 支持4-bit/8-bit量化推理
  - 多GPU负载均衡

## 典型工作流程

```mermaid
sequenceDiagram
    用户->>+Gradio界面: 上传图片/输入问题
    Gradio界面->>+控制器: 请求分配工作节点
    控制器->>+模型工作节点: 分配推理任务
    模型工作节点->>+视觉编码器: 提取特征
    视觉编码器-->>-模型工作节点: 多分辨率特征
    模型工作节点->>+LLM: 多模态融合推理
    LLM-->>-模型工作节点: 生成结果
    模型工作节点-->>-控制器: 返回响应
    控制器-->>-用户: 显示最终结果
```

## 关键配置文件

```bash
configs/
├── train.yaml          # 训练超参数配置
├── model/
│   ├── gemma-2b.yaml   # Gemma模型配置
│   └── llama3.yaml    # LLaMA-3配置
└── eval/
    └── mmmu.yaml      # MMMU评估配置
``` 