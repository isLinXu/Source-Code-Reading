# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path  # 从 pathlib 导入 Path 类

from ultralytics import SAM, YOLO  # 从 ultralytics 导入 SAM 和 YOLO 类


def auto_annotate(  # 定义 auto_annotate 函数
    data,  # 要注释的数据
    det_model="yolo11x.pt",  # 检测模型的路径或名称，默认为 yolo11x.pt
    sam_model="sam_b.pt",  # 分割模型的路径或名称，默认为 sam_b.pt
    device="",  # 运行模型的设备（例如 'cpu', 'cuda', '0'）
    conf=0.25,  # 检测模型的置信度阈值，默认为 0.25
    iou=0.45,  # IoU 阈值，用于过滤检测结果中的重叠框，默认为 0.45
    imgsz=640,  # 输入图像的调整尺寸，默认为 640
    max_det=300,  # 每张图像的最大检测数量
    classes=None,  # 过滤预测到的类 ID，返回相关的检测结果
    output_dir=None,  # 注释结果保存的目录，默认为 None
):
    """
    Automatically annotates images using a YOLO object detection model and a SAM segmentation model.  # 使用 YOLO 目标检测模型和 SAM 分割模型自动注释图像

    This function processes images in a specified directory, detects objects using a YOLO model, and then generates  # 该函数处理指定目录中的图像，使用 YOLO 模型检测对象，然后生成
    segmentation masks using a SAM model. The resulting annotations are saved as text files.  # 使用 SAM 模型生成分割掩膜，结果注释保存为文本文件

    Args:  # 参数说明
        data (str): Path to a folder containing images to be annotated.  # 包含待注释图像的文件夹路径
        det_model (str): Path or name of the pre-trained YOLO detection model.  # 预训练 YOLO 检测模型的路径或名称
        sam_model (str): Path or name of the pre-trained SAM segmentation model.  # 预训练 SAM 分割模型的路径或名称
        device (str): Device to run the models on (e.g., 'cpu', 'cuda', '0').  # 运行模型的设备（例如 'cpu', 'cuda', '0'）
        conf (float): Confidence threshold for detection model; default is 0.25.  # 检测模型的置信度阈值，默认为 0.25
        iou (float): IoU threshold for filtering overlapping boxes in detection results; default is 0.45.  # 检测结果中过滤重叠框的 IoU 阈值，默认为 0.45
        imgsz (int): Input image resize dimension; default is 640.  # 输入图像的调整尺寸，默认为 640
        max_det (int): Limits detections per image to control outputs in dense scenes.  # 每张图像的最大检测数量
        classes (list): Filters predictions to specified class IDs, returning only relevant detections.  # 过滤预测到的类 ID，返回相关的检测结果
        output_dir (str | None): Directory to save the annotated results. If None, a default directory is created.  # 注释结果保存的目录，如果为 None，则创建默认目录

    Examples:  # 示例
        >>> from ultralytics.data.annotator import auto_annotate  # 从 ultralytics.data.annotator 导入 auto_annotate
        >>> auto_annotate(data="ultralytics/assets", det_model="yolo11n.pt", sam_model="mobile_sam.pt")  # 调用 auto_annotate 函数

    Notes:  # 注意事项
        - The function creates a new directory for output if not specified.  # 如果未指定，则该函数会创建一个新的输出目录
        - Annotation results are saved as text files with the same names as the input images.  # 注释结果保存为与输入图像同名的文本文件
        - Each line in the output text file represents a detected object with its class ID and segmentation points.  # 输出文本文件中的每一行表示一个检测到的对象及其类 ID 和分割点
    """
    det_model = YOLO(det_model)  # 初始化 YOLO 检测模型
    sam_model = SAM(sam_model)  # 初始化 SAM 分割模型

    data = Path(data)  # 将数据路径转换为 Path 对象
    if not output_dir:  # 如果未指定输出目录
        output_dir = data.parent / f"{data.stem}_auto_annotate_labels"  # 创建默认输出目录
    Path(output_dir).mkdir(exist_ok=True, parents=True)  # 创建输出目录（如果不存在）

    det_results = det_model(  # 使用 YOLO 模型进行检测
        data, stream=True, device=device, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, classes=classes  # 传递参数进行检测
    )

    for result in det_results:  # 遍历检测结果
        class_ids = result.boxes.cls.int().tolist()  # 获取检测到的类 ID
        if len(class_ids):  # 如果有检测到的类
            boxes = result.boxes.xyxy  # 获取边界框
            sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=device)  # 使用 SAM 模型生成分割结果
            segments = sam_results[0].masks.xyn  # 获取分割掩膜

            with open(f"{Path(output_dir) / Path(result.path).stem}.txt", "w") as f:  # 打开输出文件
                for i in range(len(segments)):  # 遍历分割结果
                    s = segments[i]  # 获取分割结果
                    if len(s) == 0:  # 如果分割结果为空
                        continue  # 跳过
                    segment = map(str, segments[i].reshape(-1).tolist())  # 将分割结果转换为字符串
                    f.write(f"{class_ids[i]} " + " ".join(segment) + "\n")  # 写入输出文件