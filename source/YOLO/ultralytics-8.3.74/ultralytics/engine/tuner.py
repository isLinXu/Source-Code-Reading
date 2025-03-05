# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Module provides functionalities for hyperparameter tuning of the Ultralytics YOLO models for object detection, instance
segmentation, image classification, pose estimation, and multi-object tracking.

Hyperparameter tuning is the process of systematically searching for the optimal set of hyperparameters
that yield the best model performance. This is particularly crucial in deep learning models like YOLO,
where small changes in hyperparameters can lead to significant differences in model accuracy and efficiency.

Example:
    Tune hyperparameters for YOLO11n on COCO8 at imgsz=640 and epochs=30 for 300 tuning iterations.
    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    model.tune(data="coco8.yaml", epochs=10, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)
    ```
"""

import random  # 导入随机数生成模块
import shutil  # 导入文件操作模块
import subprocess  # 导入子进程模块
import time  # 导入时间模块

import numpy as np  # 导入NumPy库
import torch  # 导入PyTorch库

from ultralytics.cfg import get_cfg, get_save_dir  # 导入配置相关函数
from ultralytics.utils import DEFAULT_CFG, LOGGER, callbacks, colorstr, remove_colorstr, yaml_print, yaml_save  # 导入工具函数
from ultralytics.utils.plotting import plot_tune_results  # 导入绘图函数

class Tuner:
    """
    Class responsible for hyperparameter tuning of YOLO models.

    The class evolves YOLO model hyperparameters over a given number of iterations
    by mutating them according to the search space and retraining the model to evaluate their performance.

    Attributes:
        space (dict): Hyperparameter search space containing bounds and scaling factors for mutation.
        tune_dir (Path): Directory where evolution logs and results will be saved.
        tune_csv (Path): Path to the CSV file where evolution logs are saved.

    Methods:
        _mutate(hyp: dict) -> dict:
            Mutates the given hyperparameters within the bounds specified in `self.space`.

        __call__():
            Executes the hyperparameter evolution across multiple iterations.

    Example:
        Tune hyperparameters for YOLO11n on COCO8 at imgsz=640 and epochs=30 for 300 tuning iterations.
        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        model.tune(data="coco8.yaml", epochs=10, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)
        ```

        Tune with custom search space.
        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        model.tune(space={key1: val1, key2: val2})  # custom search space dictionary
        ```
    """

    def __init__(self, args=DEFAULT_CFG, _callbacks=None):
        """
        Initialize the Tuner with configurations.

        Args:
            args (dict, optional): Configuration for hyperparameter evolution.
        """
        # 使用配置初始化调优器。
        self.space = args.pop("space", None) or {  # key: (min, max, gain(optional))
            # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
            "lr0": (1e-5, 1e-1),  # 初始学习率（例如SGD=1E-2，Adam=1E-3）
            "lrf": (0.0001, 0.1),  # 最终OneCycleLR学习率（lr0 * lrf）
            "momentum": (0.7, 0.98, 0.3),  # SGD动量/Adam beta1
            "weight_decay": (0.0, 0.001),  # 优化器权重衰减5e-4
            "warmup_epochs": (0.0, 5.0),  # 预热epoch（允许小数）
            "warmup_momentum": (0.0, 0.95),  # 预热初始动量
            "box": (1.0, 20.0),  # box损失增益
            "cls": (0.2, 4.0),  # cls损失增益（与像素缩放）
            "dfl": (0.4, 6.0),  # dfl损失增益
            "hsv_h": (0.0, 0.1),  # 图像HSV-Hue增强（比例）
            "hsv_s": (0.0, 0.9),  # 图像HSV-Saturation增强（比例）
            "hsv_v": (0.0, 0.9),  # 图像HSV-Value增强（比例）
            "degrees": (0.0, 45.0),  # 图像旋转（+/-度）
            "translate": (0.0, 0.9),  # 图像平移（+/-比例）
            "scale": (0.0, 0.95),  # 图像缩放（+/-增益）
            "shear": (0.0, 10.0),  # 图像剪切（+/-度）
            "perspective": (0.0, 0.001),  # 图像透视（+/-比例），范围0-0.001
            "flipud": (0.0, 1.0),  # 图像上下翻转（概率）
            "fliplr": (0.0, 1.0),  # 图像左右翻转（概率）
            "bgr": (0.0, 1.0),  # 图像通道bgr（概率）
            "mosaic": (0.0, 1.0),  # 图像混合（概率）
            "mixup": (0.0, 1.0),  # 图像混合（概率）
            "copy_paste": (0.0, 1.0),  # 片段复制粘贴（概率）
        }
        self.args = get_cfg(overrides=args)  # 获取配置
        self.tune_dir = get_save_dir(self.args, name=self.args.name or "tune")  # 获取保存目录
        self.args.name = None  # 重置以不影响训练目录
        self.tune_csv = self.tune_dir / "tune_results.csv"  # 设置CSV文件路径
        self.callbacks = _callbacks or callbacks.get_default_callbacks()  # 获取回调函数
        self.prefix = colorstr("Tuner: ")  # 设置前缀
        callbacks.add_integration_callbacks(self)  # 添加集成回调
        LOGGER.info(
            f"{self.prefix}Initialized Tuner instance with 'tune_dir={self.tune_dir}'\n"
            f"{self.prefix}💡 Learn about tuning at https://docs.ultralytics.com/guides/hyperparameter-tuning"
        )  # 记录初始化信息

    def _mutate(self, parent="single", n=5, mutation=0.8, sigma=0.2):
        """
        Mutates the hyperparameters based on bounds and scaling factors specified in `self.space`.

        Args:
            parent (str): Parent selection method: 'single' or 'weighted'.
            n (int): Number of parents to consider.
            mutation (float): Probability of a parameter mutation in any given iteration.
            sigma (float): Standard deviation for Gaussian random number generator.

        Returns:
            (dict): A dictionary containing mutated hyperparameters.
        """
        # 根据指定的边界和缩放因子突变超参数。
        if self.tune_csv.exists():  # 如果CSV文件存在：选择最佳超参数并突变
            # Select parent(s)
            x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)  # 从CSV文件加载数据
            fitness = x[:, 0]  # 第一列为适应度
            n = min(n, len(x))  # 考虑的历史结果数量
            x = x[np.argsort(-fitness)][:n]  # 选择适应度最高的n个结果
            w = x[:, 0] - x[:, 0].min() + 1e-6  # 权重（确保和大于0）
            if parent == "single" or len(x) == 1:
                # x = x[random.randint(0, n - 1)]  # 随机选择
                x = x[random.choices(range(n), weights=w)[0]]  # 加权选择
            elif parent == "weighted":
                x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # 加权组合

            # Mutate
            r = np.random  # 随机数方法
            r.seed(int(time.time()))  # 设置随机种子
            g = np.array([v[2] if len(v) == 3 else 1.0 for v in self.space.values()])  # 获取增益
            ng = len(self.space)  # 超参数数量
            v = np.ones(ng)  # 初始化增益
            while all(v == 1):  # 突变直到发生变化（防止重复）
                v = (g * (r.random(ng) < mutation) * r.randn(ng) * r.random() * sigma + 1).clip(0.3, 3.0)  # 应用突变
            hyp = {k: float(x[i + 1] * v[i]) for i, k in enumerate(self.space.keys())}  # 生成突变后的超参数
        else:
            hyp = {k: getattr(self.args, k) for k in self.space.keys()}  # 如果CSV文件不存在，使用默认超参数

        # Constrain to limits
        for k, v in self.space.items():
            hyp[k] = max(hyp[k], v[0])  # 下限约束
            hyp[k] = min(hyp[k], v[1])  # 上限约束
            hyp[k] = round(hyp[k], 5)  # 保留5位有效数字

        return hyp  # 返回突变后的超参数

    def __call__(self, model=None, iterations=10, cleanup=True):
        """
        Executes the hyperparameter evolution process when the Tuner instance is called.

        This method iterates through the number of iterations, performing the following steps in each iteration:
        1. Load the existing hyperparameters or initialize new ones.
        2. Mutate the hyperparameters using the `mutate` method.
        3. Train a YOLO model with the mutated hyperparameters.
        4. Log the fitness score and mutated hyperparameters to a CSV file.

        Args:
           model (Model): A pre-initialized YOLO model to be used for training.
           iterations (int): The number of generations to run the evolution for.
           cleanup (bool): Whether to delete iteration weights to reduce storage space used during tuning.

        Note:
           The method utilizes the `self.tune_csv` Path object to read and log hyperparameters and fitness scores.
           Ensure this path is set correctly in the Tuner instance.
        """
        # 当调用Tuner实例时执行超参数进化过程。
        t0 = time.time()  # 记录开始时间
        best_save_dir, best_metrics = None, None  # 初始化最佳保存目录和指标
        (self.tune_dir / "weights").mkdir(parents=True, exist_ok=True)  # 创建权重保存目录
        for i in range(iterations):  # 迭代指定次数
            # Mutate hyperparameters
            mutated_hyp = self._mutate()  # 突变超参数
            LOGGER.info(f"{self.prefix}Starting iteration {i + 1}/{iterations} with hyperparameters: {mutated_hyp}")  # 记录当前迭代信息

            metrics = {}  # 初始化指标
            train_args = {**vars(self.args), **mutated_hyp}  # 合并参数
            save_dir = get_save_dir(get_cfg(train_args))  # 获取保存目录
            weights_dir = save_dir / "weights"  # 权重保存目录
            try:
                # Train YOLO model with mutated hyperparameters (run in subprocess to avoid dataloader hang)
                # 使用突变后的超参数训练YOLO模型（在子进程中运行以避免数据加载器挂起）
                cmd = ["yolo", "train", *(f"{k}={v}" for k, v in train_args.items())]  # 构建命令
                return_code = subprocess.run(" ".join(cmd), check=True, shell=True).returncode  # 执行命令
                ckpt_file = weights_dir / ("best.pt" if (weights_dir / "best.pt").exists() else "last.pt")  # 获取检查点文件
                metrics = torch.load(ckpt_file)["train_metrics"]  # 加载训练指标
                assert return_code == 0, "training failed"  # 检查训练是否成功

            except Exception as e:
                LOGGER.warning(f"WARNING ❌️ training failure for hyperparameter tuning iteration {i + 1}\n{e}")  # 记录训练失败的警告

            # Save results and mutated_hyp to CSV
            fitness = metrics.get("fitness", 0.0)  # 获取适应度
            log_row = [round(fitness, 5)] + [mutated_hyp[k] for k in self.space.keys()]  # 记录日志行
            headers = "" if self.tune_csv.exists() else (",".join(["fitness"] + list(self.space.keys())) + "\n")  # 表头
            with open(self.tune_csv, "a") as f:
                f.write(headers + ",".join(map(str, log_row)) + "\n")  # 写入CSV文件

            # Get best results
            x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=",", skiprows=1)  # 从CSV文件加载数据
            fitness = x[:, 0]  # 第一列为适应度
            best_idx = fitness.argmax()  # 获取最佳适应度的索引
            best_is_current = best_idx == i  # 判断当前是否为最佳结果
            if best_is_current:
                best_save_dir = save_dir  # 更新最佳保存目录
                best_metrics = {k: round(v, 5) for k, v in metrics.items()}  # 更新最佳指标
                for ckpt in weights_dir.glob("*.pt"):  # 遍历权重目录中的检查点文件
                    shutil.copy2(ckpt, self.tune_dir / "weights")  # 复制检查点到保存目录
            elif cleanup:
                shutil.rmtree(weights_dir, ignore_errors=True)  # 删除迭代权重目录以减少存储空间

            # Plot tune results
            plot_tune_results(self.tune_csv)  # 绘制调优结果

            # Save and print tune results
            header = (
                f"{self.prefix}{i + 1}/{iterations} iterations complete ✅ ({time.time() - t0:.2f}s)\n"
                f"{self.prefix}Results saved to {colorstr('bold', self.tune_dir)}\n"
                f"{self.prefix}Best fitness={fitness[best_idx]} observed at iteration {best_idx + 1}\n"
                f"{self.prefix}Best fitness metrics are {best_metrics}\n"
                f"{self.prefix}Best fitness model is {best_save_dir}\n"
                f"{self.prefix}Best fitness hyperparameters are printed below.\n"
            )
            LOGGER.info("\n" + header)  # 记录调优结果
            data = {k: float(x[best_idx, i + 1]) for i, k in enumerate(self.space.keys())}  # 获取最佳超参数
            yaml_save(
                self.tune_dir / "best_hyperparameters.yaml",
                data=data,
                header=remove_colorstr(header.replace(self.prefix, "# ")) + "\n",
            )  # 保存最佳超参数到YAML文件
            yaml_print(self.tune_dir / "best_hyperparameters.yaml")  # 打印最佳超参数