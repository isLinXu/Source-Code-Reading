# YOLO 检测模型架构流程图

```mermaid
%% YOLO 检测模型架构流程图
graph TD
    A[输入图像] --> B[数据加载与增强]
    B --> C[马赛克增强]
    C --> D[多尺度训练]
    
    subgraph 训练流程
        D --> E[模型构建]
        E --> F[前向传播]
        F --> G[损失计算]
        G --> H[反向传播]
        H --> I[参数更新]
        I --> J[模型保存]
    end
    
    subgraph 验证流程
        B --> K[模型加载]
        K --> L[推理预测]
        L --> M[非极大值抑制]
        M --> N[指标计算]
        N --> O[mAP50/mAP50-95]
        O --> P[混淆矩阵]
    end
    
    subgraph 预测流程
        B --> Q[图像预处理]
        Q --> R[模型推理]
        R --> S[边界框缩放]
        S --> T[结果输出]
    end
    
    J --> K
    P -->|模型调优| E
    T -->|部署应用| U[目标检测API]
    
    style A fill:#f9f,stroke:#333
    style U fill:#bbf,stroke:#f66

```

