%% 实例分割模型架构流程图
graph TD
    A[输入图像] --> B[数据加载与增强]
    B --> C[掩码敏感增强]
    C --> D[多尺度分割训练]
    
    subgraph 训练流程
        D --> E[分割模型构建]
        E --> F[前向传播]
        F --> G[分割损失计算]
        G --> H[反向传播]
        H --> I[参数更新]
        I --> J[模型保存]
    end
    
    subgraph 验证流程
        B --> K[模型加载]
        K --> L[分割推理]
        L --> M[掩码后处理]
        M --> N[掩码IoU计算]
        N --> O[分割mAP]
        O --> P[COCO格式转换]
    end
    
    subgraph 预测流程
        B --> Q[分割敏感预处理]
        Q --> R[模型推理]
        R --> S[原型解码]
        S --> T[实例分割可视化]
    end
    
    J --> K
    P -->|模型调优| E
    T -->|部署应用| U[智能抠图API]
    
    style A fill:#f9f,stroke:#333
    style U fill:#bbf,stroke:#f66 