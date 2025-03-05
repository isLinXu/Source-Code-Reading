# OBB旋转框检测模型架构流程图
```mermaid
%% OBB旋转框检测模型架构流程图
graph TD
    A[输入图像] --> B[数据加载与增强]
    B --> C[旋转马赛克增强]
    C --> D[角度敏感多尺度训练]
    
    subgraph 训练流程
        D --> E[旋转框模型构建]
        E --> F[前向传播]
        F --> G[角度回归损失计算]
        G --> H[反向传播]
        H --> I[参数更新]
        I --> J[模型保存]
    end
    
    subgraph 验证流程
        B --> K[模型加载]
        K --> L[旋转框推理]
        L --> M[旋转NMS处理]
        M --> N[旋转IoU计算]
        N --> O[mAP50/mAP50-95]
        O --> P[DOTA格式转换]
    end
    
    subgraph 预测流程
        B --> Q[角度敏感预处理]
        Q --> R[模型推理]
        R --> S[旋转框后处理]
        S --> T[多边形坐标转换]
    end
    
    J --> K
    P -->|模型调优| E
    T -->|部署应用| U[航拍检测API]
    
    style A fill:#f9f,stroke:#333
    style U fill:#bbf,stroke:#f66 
```

