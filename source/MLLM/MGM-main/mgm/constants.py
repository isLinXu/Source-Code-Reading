# 控制器心跳过期时间（秒），超过该时间未收到心跳则认为控制器离线
CONTROLLER_HEART_BEAT_EXPIRATION = 30
# 工作节点发送心跳的间隔时间（秒）
WORKER_HEART_BEAT_INTERVAL = 15

# 日志目录配置
LOGDIR = "."  # 当前目录存储日志

# 模型相关常量
IGNORE_INDEX = -100  # 用于损失计算时忽略的索引（如填充位置）
IMAGE_TOKEN_INDEX = -200  # 图像 token 在序列中的索引位置
PREDICT_TOKEN_INDEX = -300  # 预测 token 的索引位置
DEFAULT_IMAGE_TOKEN = "<image>"  # 默认图像占位符 token
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"  # 图像分块 token
DEFAULT_IM_START_TOKEN = "<im_start>"  # 图像序列开始标记
DEFAULT_IM_END_TOKEN = "<im_end>"  # 图像序列结束标记
IMAGE_PLACEHOLDER = "<image-placeholder>"  # 图像占位符（兼容旧版）
DEFAULT_PREDICT_TOKEN = "<predict>"  # 预测任务起始标记

# 图像描述生成提示词列表（用于视觉问答/图像描述任务）
DESCRIPT_PROMPT = [
    "Describe this image thoroughly.",  # 全面描述这张图片
    "Provide a detailed description in this picture.",  # 提供这张图片的详细描述
    "Detail every aspect of what's in this picture.",  # 详述图片中的每个细节
    "Explain this image with precision and detail.",  # 精确详细地解释这张图片
    "Give a comprehensive description of this visual.",  # 给出这个视觉内容的全面描述
    "Elaborate on the specifics within this image.",  # 详细阐述图片中的具体内容
    "Offer a detailed account of this picture's contents.",  # 详细说明这张图片的内容
    "Describe in detail what this image portrays.",  # 详细描述这张图片描绘的内容
    "Break down this image into detailed descriptions.",  # 将图片分解为详细描述
    "Provide a thorough description of the elements in this image."  # 全面描述图片中的元素
]