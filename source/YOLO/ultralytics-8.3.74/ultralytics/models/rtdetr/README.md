%% RT-DETR实时检测架构流程图
graph TD
    A[输入图像] --> B[图像预处理]
    B --> C[Hybrid Encoder]
    C --> D[IoU感知查询选择]
    
    subgraph 核心处理流程
        D --> E[Transformer解码器]
        E --> F[动态匹配]
        F --> G[多尺度特征融合]
    end
    
    subgraph 训练流程
        G --> H[损失计算]
        H --> I[GIOU损失]
        H --> J[分类损失]
        H --> K[L1回归损失]
        I & J & K --> L[反向传播更新]
    end
    
    subgraph 预测流程
        G --> M[查询去噪]
        M --> N[后处理]
        N --> O[非极大抑制]
        O --> P[坐标转换]
    end
    
    subgraph 扩展功能
        P --> Q[视频流处理]
        Q --> R[时序一致性优化]
        R --> S[实时跟踪]
    end
    
    L -->|模型保存| C
    S -->|部署应用| T[实时检测API]
    
    style A fill:#f9f,stroke:#333
    style T fill:#bbf,stroke:#f66 