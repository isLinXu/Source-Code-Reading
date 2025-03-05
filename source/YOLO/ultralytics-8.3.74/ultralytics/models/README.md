```mermaid
graph LR
    A[YOLO11 主系统] --> B[核心模型]
    A --> C[训练流程]
    A --> D[验证评估]
    A --> E[预测推理]
    A --> F[模型导出]
    A --> G[工具集成]
    B --> B1[检测模型 Detect]
    B --> B2[分割模型 Segment]
    B --> B3[分类模型 Classify]
    B --> B4[姿态估计 Pose]
    B --> B5[旋转框 OBB]
    B --> B6[FastSAM]
    B --> B7[RT-DETR]
    B --> B8[SAM/SAM2]
    C --> C1[数据加载]
C1 --> C11[Dataset]
C1 --> C12[DataLoader]
C --> C2[模型初始化]
C2 --> C21[YOLO]
C2 --> C22[SAM]
C2 --> C23[RT-DETR]
C --> C3[优化器]
C3 --> C31[SGD]
C3 --> C32[Adam]
C --> C4[训练循环]
C4 --> C41[前向传播]
C4 --> C42[损失计算]
C4 --> C43[反向传播]

D --> D1[验证集评估]
D1 --> D11[mAP计算]
D1 --> D12[混淆矩阵]
D --> D2[测试集评估]
D2 --> D21[指标可视化]

E --> E1[输入预处理]
E1 --> E11[图像缩放]
E1 --> E12[归一化]
E --> E2[模型推理]
E2 --> E21[目标检测]
E2 --> E22[实例分割]
E --> E3[后处理]
E3 --> E31[NMS]
E3 --> E32[坐标转换]

F --> F1[格式转换]
F1 --> F11[ONNX]
F1 --> F12[TensorRT]
F --> F2[量化优化]
F2 --> F21[FP16]
F2 --> F22[INT8]

G --> G1[可视化工具]
G1 --> G11[结果绘制]
G1 --> G12[特征图可视化]
G --> G2[性能分析]
G2 --> G21[内存分析]
G2 --> G22[速度测试]
G --> G3[部署工具]
G3 --> G31[OpenVINO]
G3 --> G32[CoreML]
```


​    

### 2. 训练工作流（以YOLO为例）

```mermaid
flowchart TD
    Start[训练入口 train.py] --> LoadConfig[加载配置]
    LoadConfig --> InitModel[初始化模型]
    InitModel --> |ultralytics/engine/trainer.py| BuildDataLoader
    BuildDataLoader --> |ultralytics/data/datasets.py| ApplyAug[应用Mosaic增强]
    ApplyAug --> BuildOptim[构建优化器]
    
    BuildOptim --> TrainingLoop{训练循环}
    TrainingLoop --> |每个batch| Forward[前向传播]
    Forward --> |ultralytics/nn/modules/loss.py| ComputeLoss[计算损失]
    ComputeLoss --> Backward[反向传播]
    Backward --> Update[参数更新]
    
    TrainingLoop --> |每epoch| Validate[验证]
    Validate --> |ultralytics/engine/validator.py| Eval[计算mAP]
    Eval --> SaveCKPT[保存检查点]
    
    SaveCKPT --> Export[导出部署格式]
    Export --> |ultralytics/engine/exporter.py| Convert[转换ONNX/TensorRT]
```



```mermaid
sequenceDiagram
    participant User
    participant ModelFactory
    participant YOLO
    participant SAM
    participant RTDETR
```


​    User->>ModelFactory: 指定模型类型(yolo/sam/rtdetr)
​    ModelFactory->>YOLO: 加载YOLO模型
​    YOLO->>ultralytics/models/yolo/model.py: 构建YOLO网络
​    YOLO-->>User: 返回YOLO实例
​    
​    ModelFactory->>SAM: 加载SAM模型
​    SAM->>ultralytics/models/sam/build.py: 构建ViT编码器
​    SAM->>ultralytics/models/sam/model.py: 初始化提示编码器
​    SAM-->>User: 返回SAM实例
​    
​    ModelFactory->>RTDETR: 加载RT-DETR
​    RTDETR->>ultralytics/models/rtdetr/model.py: 构建混合编码器
​    RTDETR-->>User: 返回RT-DETR实例



### 关键脚本功能映射表：

| 脚本路径 | 主要功能 | 输入 | 输出 |
|---------|--------|------|------|
| models/yolo/model.py | YOLO模型定义 | 配置参数 | 初始化后的YOLO网络 |
| models/sam/predict.py | SAM提示推理 | 图像+提示 | 分割掩码 |
| models/rtdetr/val.py | RT-DETR验证 | 验证数据集 | mAP指标 |
| engine/trainer.py | 训练流程控制 | 训练配置 | 训练好的模型 |
| engine/validator.py | 指标计算 | 验证数据 | 评估报告 |
| nn/modules/block.py | 基础网络模块 | 输入张量 | 特征图 |
| data/datasets.py | 数据加载 | 原始图像 | 增强后的批次数据 |
| utils/ops.py | 张量操作 | 原始输出 | 后处理结果 |



### 代码阅读建议路径：

模型定义入口：

YOLO: models/yolo/model.py ➔ class DetectionModel

SAM: models/sam/build.py ➔ build_sam()

RT-DETR: models/rtdetr/model.py ➔ class RTDETRDetectionModel



```python
# ultralytics/engine/trainer.py
class BaseTrainer:
	def train(self):
		# 初始化模型
		self.model = self.get_model(...)
           
		# 数据加载
		self.train_loader = self.get_dataloader(...)
           
		# 优化器设置
		self.optimizer = self.build_optimizer(...)
           
		# 训练循环
		for epoch in range(self.epochs):
			self.train_one_epoch()
			self.validate()
			self.save_checkpoint()
```



推理后处理：

```python
   # ultralytics/models/yolo/detect/predict.py
   class DetectionPredictor:
       def postprocess(self, preds, img, orig_imgs):
           # 应用NMS
           preds = non_max_suppression(preds, ...)
           
           # 缩放坐标到原始图像尺寸
           for i, pred in enumerate(preds):
               pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_imgs[i].shape)
               
           return preds
```




    

```mermaid
flowchart LR
    Export[导出命令] --> CheckFormat[检查目标格式]
    CheckFormat --> |ONNX| Torch2ONNX[调用torch.onnx.export]
    CheckFormat --> |TensorRT| BuildEngine[构建TRT引擎]
    Torch2ONNX --> Optimize[优化计算图]
    Optimize --> Simplify[应用onnx-simplifier]
    Simplify --> Save[保存ONNX模型]

    BuildEngine --> Parse[解析模型结构]
    Parse --> Calibrate[INT8校准]
    Calibrate --> Serialize[序列化引擎]
    Serialize --> SaveTRT[保存.engine文件]

    Save --> Deploy[部署到推理框架]
    SaveTRT --> Deploy
```
