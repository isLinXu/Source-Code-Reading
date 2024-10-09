# 定义控制器心跳超时时间（秒），用于检测控制器的活跃状态
CONTROLLER_HEART_BEAT_EXPIRATION = 30

# 定义工作者心跳间隔时间（秒），用于检测工作者的活跃状态
WORKER_HEART_BEAT_INTERVAL = 15

# 定义日志目录，用于记录程序运行过程中的日志信息
LOGDIR = "."

# 模型常量定义
# 忽略索引，用于标识需要忽略的序列位置
IGNORE_INDEX = -100

# 语音标记索引，用于标识语音片段的特殊位置
SPEECH_TOKEN_INDEX = -200

# 默认语音标记，用于在序列中表示语音片段的特殊标记
DEFAULT_SPEECH_TOKEN = "<speech>"
