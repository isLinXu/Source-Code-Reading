# 控制器心跳过期时间，单位为秒
CONTROLLER_HEART_BEAT_EXPIRATION = 30

# 工作节点心跳间隔时间，单位为秒
WORKER_HEART_BEAT_INTERVAL = 15

# 日志目录，默认为当前目录
LOGDIR = "."

# Model Constants
# 模型常量
# 忽略的索引值，用于标记不需要处理的位置
IGNORE_INDEX = -100
# 图片标记索引值，用于标记图片位置
IMAGE_TOKEN_INDEX = -200
# 音频标记索引值，用于标记音频位置
AUDIO_TOKEN_INDEX = -300

# 默认图片标记，用于替换图片位置
DEFAULT_IMAGE_TOKEN = "<image>"
# 默认图片块标记，用于替换图片块位置
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
# 图片开始标记
DEFAULT_IM_START_TOKEN = "<im_start>"
# 图片结束标记
DEFAULT_IM_END_TOKEN = "<im_end>"
# 图片占位符
IMAGE_PLACEHOLDER = "<image-placeholder>"