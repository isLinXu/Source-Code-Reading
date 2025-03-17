%% Swift 框架架构图

graph TD
    A[swift/cli] --> B[核心入口]
    A --> C[功能子命令]
    B --> D[main.py]
    C --> E[app/deploy/eval/export/infer...]
    
    F[swift/hub] --> G[模型仓库交互]
    G --> H[HFHub]
    G --> I[MSHub]
    
    J[swift/llm] --> K[核心功能模块]
    K --> L[训练/推理/评估]
    K --> M[模型/模板/数据集]
    K --> N[参数配置]
    
    subgraph CLI模块
        D --> |路由分发| E
        E --> |调用| K
    end
    
    subgraph Hub适配层
        H --> |HuggingFace操作| P[创建/上传/下载]
        I --> |ModelScope操作| P
    end
    
    subgraph LLM核心
        L --> Q[训练流程sft/pt/rlhf]
        L --> R[推理服务infer/deploy]
        L --> S[评估eval]
        M --> T[模型架构model]
        M --> U[提示模板template]
        M --> V[数据集处理dataset]
        N --> W[参数类Arguments]
    end
    
    subgraph WebUI
        app.py --> X[Gradio界面]
        X --> |调用| R
    end
    
    style A fill:#f9f,stroke:#333
    style F fill:#bbf,stroke:#333
    style J fill:#9f9,stroke:#333 

%% 项目目录树
```
swift/
├── cli/                     # 命令行入口
│   ├── app.py               # Web应用服务
│   ├── deploy.py            # 模型部署
│   ├── eval.py              # 模型评估  
│   ├── export.py            # 模型导出
│   ├── infer.py             # 本地推理
│   ├── main.py              # 主路由控制
│   ├── merge_lora.py        # LoRA权重合并
│   ├── pt.py                # P-tuning训练
│   └── sft.py               # SFT微调
│
├── hub/                     # 模型仓库管理
│   ├── hub.py               # 多平台适配器（HF/MS）
│   └── constant.py          # 仓库配置常量
│
└── llm/                     # 大模型核心模块
    ├── app/                 # 应用层
    │   ├── app.py           # 应用主逻辑
    │   └── build_ui.py      # Gradio界面构建
    │
    ├── argument/            # 参数配置系统
    │   ├── app_args.py      # 应用参数
    │   ├── deploy_args.py   # 部署参数
    │   └── train_args.py    # 训练参数
    │
    ├── dataset/             # 数据集处理
    │   └── preprocess.py    # 数据预处理
    │
    ├── model/               # 模型架构
    │   ├── adapter.py       # 适配器实现
    │   └── lora.py          # LoRA实现
    │
    ├── template/            # 提示模板
    │   └── chatglm.py       # 模型特定模板
    │
    └── utils/               # 工具模块
        └── inference.py     # 推理工具类
```
