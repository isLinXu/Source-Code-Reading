# MGM 多模态模型架构解析

```mermaid
graph TD
    A[核心模块] --> B[视觉编码器]
    A --> C[语言模型]
    A --> D[多模态融合]
    A --> E[训练框架]
    A --> F[推理服务]
    
    subgraph 视觉编码器
    B1[CLIP-L 低分辨率编码] -->|336x336| B2[特征金字塔]
    B1 -->|672x672 高分辨率| B3[候选区域提取]
    B3 --> B4[Patch信息挖掘]
    end

    subgraph 语言模型
    C1[LLaMA-3] --> C2[文本理解]
    C1 --> C3[指令跟随]
    C1 --> C4[生成控制]
    end

    subgraph 多模态融合
    D1[交叉注意力] --> D2[特征对齐]
    D2 --> D3[层次化融合]
    D3 --> D4[多尺度推理]
    end

    subgraph 训练框架
    E1[两阶段训练] --> E2[预训练阶段]
    E2 --> E3[图像-文本对比学习]
    E1 --> E4[微调阶段]
    E4 --> E5[复杂指令优化]
    E5 --> E6[混合精度训练]
    end

    subgraph 推理服务
    F1[Gradio接口] --> F2[多节点管理]
    F2 --> F3[负载均衡]
    F3 --> F4[量化推理]
    F4 --> F5[4-bit/8-bit支持]
    end

    B --> D
    C --> D
    D --> E
    E --> F
```

## 关键代码结构

```bash
MGM/
├── configs/                 # 训练配置
│   ├── train_vicuna.yaml    # Vicuna训练参数
│   └── model/               # 模型架构配置
├── mgm/
│   ├── model/               # 核心模型实现
│   │   └── modeling_llama.py # 多模态融合逻辑
│   ├── train/               # 训练流程
│   │   ├── train_mgm.py     # 两阶段训练主逻辑
│   │   └── datasets/        # 数据加载
│   ├── eval/                # 评估模块
│   │   └── MMMU/            # 多学科评估套件
│   └── serve/               # 服务部署
│       ├── gradio_web_server.py # Web服务
│       └── model_worker.py  # 分布式工作节点
```

## 典型数据流

```mermaid
sequenceDiagram
    用户->>+输入处理: 上传图片/文本
    输入处理->>+视觉编码器: 多尺度图像编码
    视觉编码器-->>-多模态融合: 特征金字塔输出
    输入处理->>+语言模型: 指令解析
    语言模型-->>-多模态融合: 文本嵌入
    多模态融合->>+推理引擎: 联合表征
    推理引擎->>+输出生成: 多模态推理
    输出生成-->>-用户: 生成结果
```

## 核心实现细节

### 1. 多尺度视觉编码 (`mgm/model/vision_encoder.py`)
```python
class DualVisionEncoder(nn.Module):
    def __init__(self):
        self.low_res = CLIPEncoder(resolution=336)
        self.high_res = PatchMiningEncoder(resolution=672)
        
    def forward(self, x):
        low_feat = self.low_res(x)  # 低分辨率全局特征
        high_feat = self.high_res(x)  # 高分辨率局部特征
        return cross_attention_fusion(low_feat, high_feat)
```

### 2. 指令微调流程 (`mgm/train/train_mgm.py`)
```python
def train_phase2():
    # 加载预训练权重
    model.load_phase1_ckpt()  
    
    # 混合精度训练
    with autocast():
        for batch in phase2_data:
            loss = model(
                images=batch['images'],
                texts=batch['instructions']
            )
            scaler.scale(loss).backward()
            
    # 保存微调模型
    save_checkpoint()
```

### 3. 服务部署架构 (`mgm/serve/model_worker.py`)
```python
class ModelWorker:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.tokenizer = load_tokenizer()
        
    def stream_generate(self, params):
        # 多模态推理流水线
        image = process_image(params['image'])
        text = self.tokenizer.encode(params['text'])
        
        # 多GPU并行支持
        with distributed_ctx():
            outputs = self.model.generate(
                image_inputs=image,
                text_inputs=text
            )
            
        yield from streaming_output(outputs)
```

## 性能优化策略

1. **混合精度训练** - 使用Flash Attention优化显存
2. **动态批处理** - 根据分辨率自动分组样本
3. **梯度检查点** - 降低显存消耗达30%
4. **模型并行** - 支持多GPU分布式推理