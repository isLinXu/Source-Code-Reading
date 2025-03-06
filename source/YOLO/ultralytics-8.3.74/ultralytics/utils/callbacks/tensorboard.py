# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING, colorstr  # 从 ultralytics.utils 导入 LOGGER、SETTINGS、TESTS_RUNNING 和 colorstr

try:
    # WARNING: do not move SummaryWriter import due to protobuf bug https://github.com/ultralytics/ultralytics/pull/4674
    from torch.utils.tensorboard import SummaryWriter  # 从 torch.utils.tensorboard 导入 SummaryWriter

    assert not TESTS_RUNNING  # do not log pytest  # 不记录 pytest
    assert SETTINGS["tensorboard"] is True  # verify integration is enabled  # 验证集成是否启用
    WRITER = None  # TensorBoard SummaryWriter instance  # TensorBoard SummaryWriter 实例
    PREFIX = colorstr("TensorBoard: ")  # 设置前缀为 "TensorBoard: "

    # Imports below only required if TensorBoard enabled
    import warnings  # 导入 warnings 模块
    from copy import deepcopy  # 从 copy 导入 deepcopy

    from ultralytics.utils.torch_utils import de_parallel, torch  # 从 ultralytics.utils.torch_utils 导入 de_parallel 和 torch

except (ImportError, AssertionError, TypeError, AttributeError):  # 捕获导入错误、断言错误、类型错误和属性错误
    # TypeError for handling 'Descriptors cannot not be created directly.' protobuf errors in Windows  # 处理 Windows 中 'Descriptors cannot not be created directly.' 的 protobuf 错误
    # AttributeError: module 'tensorflow' has no attribute 'io' if 'tensorflow' not installed  # 如果未安装 'tensorflow'，则会引发 AttributeError: module 'tensorflow' has no attribute 'io'
    SummaryWriter = None  # 如果导入失败，则将 SummaryWriter 设置为 None


def _log_scalars(scalars, step=0):
    """Logs scalar values to TensorBoard.  # 将标量值记录到 TensorBoard。"""
    if WRITER:  # 如果 WRITER 实例存在
        for k, v in scalars.items():  # 遍历标量字典
            WRITER.add_scalar(k, v, step)  # 记录标量值和步骤


def _log_tensorboard_graph(trainer):
    """Log model graph to TensorBoard.  # 将模型图记录到 TensorBoard。"""
    # Input image  # 输入图像
    imgsz = trainer.args.imgsz  # 获取图像大小
    imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz  # 如果是整数，则转换为元组
    p = next(trainer.model.parameters())  # for device, type  # 获取模型参数以确定设备和类型
    im = torch.zeros((1, 3, *imgsz), device=p.device, dtype=p.dtype)  # input image (must be zeros, not empty)  # 输入图像（必须为零，不可为空）

    with warnings.catch_warnings():  # 捕获警告
        warnings.simplefilter("ignore", category=UserWarning)  # suppress jit trace warning  # 抑制 jit 跟踪警告
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)  # suppress jit trace warning  # 抑制 jit 跟踪警告

        # Try simple method first (YOLO)  # 首先尝试简单方法（YOLO）
        try:
            trainer.model.eval()  # place in .eval() mode to avoid BatchNorm statistics changes  # 置于 .eval() 模式以避免 BatchNorm 统计变化
            WRITER.add_graph(torch.jit.trace(de_parallel(trainer.model), im, strict=False), [])  # 记录模型图
            LOGGER.info(f"{PREFIX}model graph visualization added ✅")  # 记录模型图可视化已添加
            return  # 返回

        except Exception:  # 捕获异常
            # Fallback to TorchScript export steps (RTDETR)  # 回退到 TorchScript 导出步骤（RTDETR）
            try:
                model = deepcopy(de_parallel(trainer.model))  # 深拷贝模型
                model.eval()  # 置于评估模式
                model = model.fuse(verbose=False)  # 融合模型
                for m in model.modules():  # 遍历模型的所有模块
                    if hasattr(m, "export"):  # Detect, RTDETRDecoder (Segment and Pose use Detect base class)  # 检测 RTDETRDecoder（Segment 和 Pose 使用 Detect 基类）
                        m.export = True  # 设置为可导出
                        m.format = "torchscript"  # 设置格式为 torchscript
                model(im)  # dry run  # 干运行
                WRITER.add_graph(torch.jit.trace(model, im, strict=False), [])  # 记录模型图
                LOGGER.info(f"{PREFIX}model graph visualization added ✅")  # 记录模型图可视化已添加
            except Exception as e:  # 捕获异常
                LOGGER.warning(f"{PREFIX}WARNING ⚠️ TensorBoard graph visualization failure {e}")  # 记录警告信息


def on_pretrain_routine_start(trainer):
    """Initialize TensorBoard logging with SummaryWriter.  # 使用 SummaryWriter 初始化 TensorBoard 日志记录。"""
    if SummaryWriter:  # 如果 SummaryWriter 可用
        try:
            global WRITER  # 声明全局变量 WRITER
            WRITER = SummaryWriter(str(trainer.save_dir))  # 初始化 SummaryWriter
            LOGGER.info(f"{PREFIX}Start with 'tensorboard --logdir {trainer.save_dir}', view at http://localhost:6006/")  # 记录启动信息
        except Exception as e:  # 捕获异常
            LOGGER.warning(f"{PREFIX}WARNING ⚠️ TensorBoard not initialized correctly, not logging this run. {e}")  # 记录警告信息


def on_train_start(trainer):
    """Log TensorBoard graph.  # 记录 TensorBoard 图。"""
    if WRITER:  # 如果 WRITER 实例存在
        _log_tensorboard_graph(trainer)  # 记录模型图


def on_train_epoch_end(trainer):
    """Logs scalar statistics at the end of a training epoch.  # 在训练周期结束时记录标量统计。"""
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1)  # 记录训练损失项
    _log_scalars(trainer.lr, trainer.epoch + 1)  # 记录学习率


def on_fit_epoch_end(trainer):
    """Logs epoch metrics at end of training epoch.  # 在训练周期结束时记录周期指标。"""
    _log_scalars(trainer.metrics, trainer.epoch + 1)  # 记录当前指标


callbacks = (  # 定义回调函数
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,  # 预训练例程开始时的回调
        "on_train_start": on_train_start,  # 训练开始时的回调
        "on_fit_epoch_end": on_fit_epoch_end,  # 拟合周期结束时的回调
        "on_train_epoch_end": on_train_epoch_end,  # 训练周期结束时的回调
    }
    if SummaryWriter  # 如果 SummaryWriter 可用
    else {}
)