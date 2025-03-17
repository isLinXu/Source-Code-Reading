# MS-SWIFT 架构分析

## 整体架构流程

```mermaid
graph TD
    A[开发者] --> B(代码提交)
    B --> C{触发事件}
    C -->|Push/PR| D[GitHub Actions]
    D --> E[执行CI流程]
    E --> F[代码检查]
    E --> G[单元测试]
    E --> H[构建Docker]
    
    subgraph 核心功能
        I[数据准备] --> J[训练流程]
        J --> K{训练类型}
        K -->|SFT| L[监督微调]
        K -->|RLHF| M[人类对齐]
        K -->|PT| N[预训练]
        J --> O[分布式训练]
        O -->|DDP| P[数据并行]
        O -->|DeepSpeed| Q[ZeRO优化]
        J --> R[模型导出]
        R --> S[模型部署]
        S -->|vLLM| T[推理加速]
        S -->|LMDeploy| U[服务化]
    end
    
    subgraph 工具链
        V[命令行工具] --> W[训练]
        V --> X[推理]
        V --> Y[评估]
        V --> Z[量化]
        AA[Web UI] --> AB[交互式训练]
        AA --> AC[实时推理]
    end
    
    H --> AD[生成容器镜像]
    G --> AE[测试报告]
    F --> AF[代码规范检查]
    R --> AG[模型仓库]
```

## 关键模块解析

### 1. CI/CD 流程
```mermaid
graph LR
    A[代码提交] --> B{触发条件}
    B -->|Push到master| C[执行全量测试]
    B -->|PR提交| D[执行差异测试]
    C --> E[多GPU测试]
    D --> F[单GPU测试]
    E --> G[生成测试报告]
    F --> G
    G --> H[结果通知]
    
    subgraph Docker构建
        I[基础镜像] --> J[安装依赖]
        J --> K[构建生产镜像]
        K --> L[推送镜像仓库]
    end
```

### 2. 训练系统架构
```mermaid
graph TB
    A[训练脚本] --> B[参数解析]
    B --> C{训练类型}
    C -->|SFT| D[监督微调]
    C -->|DPO| E[偏好优化]
    C -->|PPO| F[策略优化]
    
    D --> G[数据加载]
    E --> G
    F --> G
    
    G --> H[模型准备]
    H --> I[分布式配置]
    I -->|DeepSpeed| J[ZeRO优化]
    I -->|FSDP| K[全分片数据并行]
    
    J --> L[训练循环]
    K --> L
    L --> M[检查点保存]
    M --> N[模型评估]
```

### 3. 推理部署流程
```mermaid
graph LR
    A[训练模型] --> B[模型导出]
    B --> C{部署方式}
    C -->|vLLM| D[高性能推理]
    C -->|LMDeploy| E[服务化部署]
    
    D --> F[API服务]
    E --> F
    F --> G[客户端调用]
    
    subgraph 量化支持
        H[原始模型] --> I[AWQ量化]
        H --> J[GPTQ量化]
        H --> K[BNB量化]
        I --> L[量化模型]
        J --> L
        K --> L
    end
```

## 核心脚本关系

```mermaid
graph TD
    A[dockerci.sh] --> B[容器测试]
    B --> C[CI流程]
    
    D[build_docs.sh] --> E[文档生成]
    E --> F[API文档]
    E --> G[用户手册]
    
    H[ci_container_test.sh] --> I[依赖安装]
    I --> J[代码规范检查]
    J --> K[测试执行]
    
    L[web-ui] --> M[训练界面]
    L --> N[推理界面]
    
    O[swift CLI] --> P[训练管理]
    O --> Q[模型部署]
    O --> R[量化转换]
```

## 关键特性实现

### 1. 分布式训练配置
```python
# .dev_scripts/ci_container_test.sh
if [ "$MODELSCOPE_SDK_DEBUG" == "True" ]; then
    pip install -r requirements/framework.txt -U
    pip install diffusers decord einops -U
    pip install autoawq -U --no-deps
fi
```

### 2. 模型加速支持
```python
# dockerci.sh
--gpus='"'"device=$gpu"'"' \
-v $MODELSCOPE_CACHE:$MODELSCOPE_CACHE_DIR_IN_CONTAINER \
-e CUDA_VISIBLE_DEVICES=$gpu \
--cpuset-cpus=${cpu_sets_arr[$idx]} \
```

### 3. 多模态训练流程
```mermaid
graph TB
    A[多模态数据] --> B[特征提取]
    B --> C[模态融合]
    C --> D{任务类型}
    D -->|VQA| E[视觉问答]
    D -->|Caption| F[图像描述]
    D -->|OCR| G[文字识别]
    E --> H[联合训练]
    F --> H
    G --> H
    H --> I[多模态输出]
```

## 性能优化策略

1. **混合精度训练**：
```shell
# 训练脚本参数
--torch_dtype bfloat16 \
--mixed_precision fp16 \
```

2. **内存优化**：
```python
# 使用DeepSpeed Zero优化
--deepspeed zero2 \
--gradient_accumulation_steps 16 \
```

3. **并行处理**：
```shell
# 数据加载优化
--dataloader_num_workers 8 \
--preprocessing_num_workers 16 \
```

4. **缓存优化**：
```python
# 模型缓存配置
-v $MODELSCOPE_CACHE:$MODELSCOPE_CACHE_DIR_IN_CONTAINER \
-e HF_HOME=/modelscope_cache \
``` 