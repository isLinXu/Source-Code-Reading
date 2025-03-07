# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import shutil  # 导入文件操作模块
import threading  # 导入线程模块
import time  # 导入时间模块
from http import HTTPStatus  # 导入 HTTP 状态码
from pathlib import Path  # 导入路径处理模块
from urllib.parse import parse_qs, urlparse  # 导入 URL 解析模块

import requests  # 导入请求模块

from ultralytics.hub.utils import HELP_MSG, HUB_WEB_ROOT, PREFIX, TQDM  # 从 ultralytics.hub.utils 导入相关工具
from ultralytics.utils import IS_COLAB, LOGGER, SETTINGS, __version__, checks, emojis  # 从 ultralytics.utils 导入工具和设置
from ultralytics.utils.errors import HUBModelError  # 从错误模块导入 HUBModelError

AGENT_NAME = f"python-{__version__}-colab" if IS_COLAB else f"python-{__version__}-local"  # 定义代理名称，根据环境设置

class HUBTrainingSession:
    """
    HUB training session for Ultralytics HUB YOLO models. Handles model initialization, heartbeats, and checkpointing.  # Ultralytics HUB YOLO 模型的训练会话。处理模型初始化、心跳和检查点。

    Attributes:
        model_id (str): Identifier for the YOLO model being trained.  # 正在训练的 YOLO 模型的标识符。
        model_url (str): URL for the model in Ultralytics HUB.  # Ultralytics HUB 中模型的 URL。
        rate_limits (dict): Rate limits for different API calls (in seconds).  # 不同 API 调用的速率限制（以秒为单位）。
        timers (dict): Timers for rate limiting.  # 用于速率限制的计时器。
        metrics_queue (dict): Queue for the model's metrics.  # 模型指标的队列。
        model (dict): Model data fetched from Ultralytics HUB.  # 从 Ultralytics HUB 获取的模型数据。
    """

    def __init__(self, identifier):
        """
        Initialize the HUBTrainingSession with the provided model identifier.  # 使用提供的模型标识符初始化 HUBTrainingSession。

        Args:
            identifier (str): Model identifier used to initialize the HUB training session.  # 用于初始化 HUB 训练会话的模型标识符。
                It can be a URL string or a model key with specific format.  # 可以是 URL 字符串或具有特定格式的模型密钥。

        Raises:
            ValueError: If the provided model identifier is invalid.  # 如果提供的模型标识符无效，则引发 ValueError。
            ConnectionError: If connecting with global API key is not supported.  # 如果不支持使用全局 API 密钥连接，则引发 ConnectionError。
            ModuleNotFoundError: If hub-sdk package is not installed.  # 如果未安装 hub-sdk 包，则引发 ModuleNotFoundError。
        """
        from hub_sdk import HUBClient  # 导入 HUBClient

        self.rate_limits = {"metrics": 3, "ckpt": 900, "heartbeat": 300}  # rate limits (seconds)  # 速率限制（秒）
        self.metrics_queue = {}  # holds metrics for each epoch until upload  # 保存每个 epoch 的指标，直到上传
        self.metrics_upload_failed_queue = {}  # holds metrics for each epoch if upload failed  # 如果上传失败，保存每个 epoch 的指标
        self.timers = {}  # holds timers in ultralytics/utils/callbacks/hub.py  # 在 ultralytics/utils/callbacks/hub.py 中保存计时器
        self.model = None  # 初始化模型为 None
        self.model_url = None  # 初始化模型 URL 为 None
        self.model_file = None  # 初始化模型文件为 None
        self.train_args = None  # 初始化训练参数为 None

        # Parse input  # 解析输入
        api_key, model_id, self.filename = self._parse_identifier(identifier)  # 解析标识符

        # Get credentials  # 获取凭证
        active_key = api_key or SETTINGS.get("api_key")  # 设置凭证
        credentials = {"api_key": active_key} if active_key else None  # 设置凭证

        # Initialize client  # 初始化客户端
        self.client = HUBClient(credentials)  # 创建 HUBClient 实例

        # Load models  # 加载模型
        try:
            if model_id:  # 如果有模型 ID
                self.load_model(model_id)  # load existing model  # 加载现有模型
            else:
                self.model = self.client.model()  # load empty model  # 加载空模型
        except Exception:  # 捕获异常
            if identifier.startswith(f"{HUB_WEB_ROOT}/models/") and not self.client.authenticated:  # 如果是模型 URL 且未认证
                LOGGER.warning(  # 记录警告信息
                    f"{PREFIX}WARNING ⚠️ Please log in using 'yolo login API_KEY'. "
                    "You can find your API Key at: https://hub.ultralytics.com/settings?tab=api+keys."
                )

    @classmethod
    def create_session(cls, identifier, args=None):
        """Class method to create an authenticated HUBTrainingSession or return None.  # 类方法创建经过身份验证的 HUBTrainingSession 或返回 None。"""
        try:
            session = cls(identifier)  # 创建 HUBTrainingSession 实例
            if args and not identifier.startswith(f"{HUB_WEB_ROOT}/models/"):  # not a HUB model URL  # 不是 HUB 模型 URL
                session.create_model(args)  # 创建模型
                assert session.model.id, "HUB model not loaded correctly"  # 确保模型已正确加载
            return session  # 返回会话
        # PermissionError and ModuleNotFoundError indicate hub-sdk not installed  # PermissionError 和 ModuleNotFoundError 表示未安装 hub-sdk
        except (PermissionError, ModuleNotFoundError, AssertionError):
            return None  # 返回 None

    def load_model(self, model_id):
        """Loads an existing model from Ultralytics HUB using the provided model identifier.  # 使用提供的模型标识符从 Ultralytics HUB 加载现有模型。"""
        self.model = self.client.model(model_id)  # 从客户端加载模型
        if not self.model.data:  # then model does not exist  # 如果模型数据不存在
            raise ValueError(emojis("❌ The specified HUB model does not exist"))  # TODO: improve error handling  # 引发 ValueError

        self.model_url = f"{HUB_WEB_ROOT}/models/{self.model.id}"  # 设置模型 URL
        if self.model.is_trained():  # 如果模型已训练
            print(emojis(f"Loading trained HUB model {self.model_url} 🚀"))  # 打印加载模型信息
            url = self.model.get_weights_url("best")  # download URL with auth  # 获取带认证的下载 URL
            self.model_file = checks.check_file(url, download_dir=Path(SETTINGS["weights_dir"]) / "hub" / self.model.id)  # 检查文件
            return  # 返回

        # Set training args and start heartbeats for HUB to monitor agent  # 设置训练参数并启动心跳以监控代理
        self._set_train_args()  # 设置训练参数
        self.model.start_heartbeat(self.rate_limits["heartbeat"])  # 启动心跳
        LOGGER.info(f"{PREFIX}View model at {self.model_url} 🚀")  # 记录模型信息

    def create_model(self, model_args):
        """Initializes a HUB training session with the specified model identifier.  # 使用指定的模型标识符初始化 HUB 训练会话。"""
        payload = {  # 创建负载
            "config": {
                "batchSize": model_args.get("batch", -1),  # 批量大小
                "epochs": model_args.get("epochs", 300),  # 训练周期
                "imageSize": model_args.get("imgsz", 640),  # 图像大小
                "patience": model_args.get("patience", 100),  # 耐心值
                "device": str(model_args.get("device", "")),  # convert None to string  # 将 None 转换为字符串
                "cache": str(model_args.get("cache", "ram")),  # convert True, False, None to string  # 将 True、False、None 转换为字符串
            },
            "dataset": {"name": model_args.get("data")},  # 数据集名称
            "lineage": {
                "architecture": {"name": self.filename.replace(".pt", "").replace(".yaml", "")},  # 构建架构名称
                "parent": {},  # 父级为空
            },
            "meta": {"name": self.filename},  # 元数据名称
        }

        if self.filename.endswith(".pt"):  # 如果文件名以 .pt 结尾
            payload["lineage"]["parent"]["name"] = self.filename  # 设置父级名称

        self.model.create_model(payload)  # 创建模型

        # Model could not be created  # 模型无法创建
        # TODO: improve error handling  # TODO: 改进错误处理
        if not self.model.id:  # 如果模型 ID 不存在
            return None  # 返回 None

        self.model_url = f"{HUB_WEB_ROOT}/models/{self.model.id}"  # 设置模型 URL

        # Start heartbeats for HUB to monitor agent  # 启动心跳以监控代理
        self.model.start_heartbeat(self.rate_limits["heartbeat"])  # 启动心跳

        LOGGER.info(f"{PREFIX}View model at {self.model_url} 🚀")  # 记录模型信息

    @staticmethod
    def _parse_identifier(identifier):
        """
        Parses the given identifier to determine the type of identifier and extract relevant components.  # 解析给定的标识符以确定标识符类型并提取相关组件。

        The method supports different identifier formats:  # 该方法支持不同的标识符格式：
            - A HUB model URL https://hub.ultralytics.com/models/MODEL  # HUB 模型 URL
            - A HUB model URL with API Key https://hub.ultralytics.com/models/MODEL?api_key=APIKEY  # 带 API 密钥的 HUB 模型 URL
            - A local filename that ends with '.pt' or '.yaml'  # 以 .pt 或 .yaml 结尾的本地文件名

        Args:
            identifier (str): The identifier string to be parsed.  # 要解析的标识符字符串。

        Returns:
            (tuple): A tuple containing the API key, model ID, and filename as applicable.  # 返回包含 API 密钥、模型 ID 和文件名的元组（如适用）。

        Raises:
            HUBModelError: If the identifier format is not recognized.  # 如果标识符格式无法识别，则引发 HUBModelError。
        """
        api_key, model_id, filename = None, None, None  # 初始化 API 密钥、模型 ID 和文件名
        if Path(identifier).suffix in {".pt", ".yaml"}:  # 如果标识符是文件名
            filename = identifier  # 设置文件名
        elif identifier.startswith(f"{HUB_WEB_ROOT}/models/"):  # 如果是 HUB 模型 URL
            parsed_url = urlparse(identifier)  # 解析 URL
            model_id = Path(parsed_url.path).stem  # 获取模型 ID
            query_params = parse_qs(parsed_url.query)  # 解析查询参数
            api_key = query_params.get("api_key", [None])[0]  # 获取 API 密钥
        else:
            raise HUBModelError(f"model='{identifier} invalid, correct format is {HUB_WEB_ROOT}/models/MODEL_ID")  # 引发 HUBModelError
        return api_key, model_id, filename  # 返回 API 密钥、模型 ID 和文件名

    def _set_train_args(self):
        """
        Initializes training arguments and creates a model entry on the Ultralytics HUB.  # 初始化训练参数并在 Ultralytics HUB 上创建模型条目。

        This method sets up training arguments based on the model's state and updates them with any additional  # 该方法根据模型状态设置训练参数，并使用任何附加参数更新它们。
        arguments provided. It handles different states of the model, such as whether it's resumable, pretrained,  # 它处理模型的不同状态，例如是否可恢复、是否预训练
        or requires specific file setup.

        Raises:
            ValueError: If the model is already trained, if required dataset information is missing, or if there are  # 如果模型已经训练、所需数据集信息缺失或提供的训练参数存在问题，则引发 ValueError。
                issues with the provided training arguments.
        """
        if self.model.is_resumable():  # 如果模型可恢复
            # Model has saved weights  # 模型有保存的权重
            self.train_args = {"data": self.model.get_dataset_url(), "resume": True}  # 设置训练参数
            self.model_file = self.model.get_weights_url("last")  # 获取最后的权重文件 URL
        else:
            # Model has no saved weights  # 模型没有保存的权重
            self.train_args = self.model.data.get("train_args")  # 获取训练参数

            # Set the model file as either a *.pt or *.yaml file  # 将模型文件设置为 *.pt 或 *.yaml 文件
            self.model_file = (
                self.model.get_weights_url("parent") if self.model.is_pretrained() else self.model.get_architecture()  # 获取权重文件 URL
            )

        if "data" not in self.train_args:  # 如果训练参数中没有数据
            # RF bug - datasets are sometimes not exported  # RF bug - 数据集有时未导出
            raise ValueError("Dataset may still be processing. Please wait a minute and try again.")  # 引发 ValueError

        self.model_file = checks.check_yolov5u_filename(self.model_file, verbose=False)  # YOLOv5->YOLOv5u
        self.model_id = self.model.id  # 设置模型 ID

    def request_queue(  # 请求队列
        self,
        request_func,  # 请求函数
        retry=3,  # 重试次数
        timeout=30,  # 超时时间
        thread=True,  # 是否使用线程
        verbose=True,  # 是否详细
        progress_total=None,  # 总进度
        stream_response=None,  # 流响应
        *args,  # 其他参数
        **kwargs,  # 关键字参数
    ):
        """Attempts to execute `request_func` with retries, timeout handling, optional threading, and progress.  # 尝试执行 `request_func`，带重试、超时处理、可选线程和进度。"""

        def retry_request():  # 重试请求
            """Attempts to call `request_func` with retries, timeout, and optional threading.  # 尝试使用重试、超时和可选线程调用 `request_func`。"""
            t0 = time.time()  # Record the start time for the timeout  # 记录开始时间
            response = None  # 初始化响应
            for i in range(retry + 1):  # 重试次数循环
                if (time.time() - t0) > timeout:  # 如果超时
                    LOGGER.warning(f"{PREFIX}Timeout for request reached. {HELP_MSG}")  # 记录超时警告
                    break  # 超时，退出循环

                response = request_func(*args, **kwargs)  # 调用请求函数
                if response is None:  # 如果没有响应
                    LOGGER.warning(f"{PREFIX}Received no response from the request. {HELP_MSG}")  # 记录未收到响应的警告
                    time.sleep(2**i)  # 指数退避
                    continue  # 跳过后续处理并重试

                if progress_total:  # 如果有进度总数
                    self._show_upload_progress(progress_total, response)  # 显示上传进度
                elif stream_response:  # 如果是流响应
                    self._iterate_content(response)  # 处理内容

                if HTTPStatus.OK <= response.status_code < HTTPStatus.MULTIPLE_CHOICES:  # 如果请求成功
                    # if request related to metrics upload  # 如果请求与指标上传相关
                    if kwargs.get("metrics"):  # 如果有指标
                        self.metrics_upload_failed_queue = {}  # 清空失败队列
                    return response  # 成功，无需重试

                if i == 0:  # 如果是第一次尝试
                    # Initial attempt, check status code and provide messages  # 初始尝试，检查状态码并提供消息
                    message = self._get_failure_message(response, retry, timeout)  # 获取失败消息

                    if verbose:  # 如果需要详细信息
                        LOGGER.warning(f"{PREFIX}{message} {HELP_MSG} ({response.status_code})")  # 记录警告信息

                if not self._should_retry(response.status_code):  # 如果不应该重试
                    LOGGER.warning(f"{PREFIX}Request failed. {HELP_MSG} ({response.status_code}")  # 记录请求失败的警告
                    break  # 不是应该重试的错误，退出循环

                time.sleep(2**i)  # 指数退避重试

            # if request related to metrics upload and exceed retries  # 如果请求与指标上传相关且超过重试次数
            if response is None and kwargs.get("metrics"):  # 如果没有响应并且有指标
                self.metrics_upload_failed_queue.update(kwargs.get("metrics"))  # 更新失败队列

            return response  # 返回响应

        if thread:  # 如果使用线程
            # Start a new thread to run the retry_request function  # 启动新线程运行重试请求函数
            threading.Thread(target=retry_request, daemon=True).start()  # 启动线程
        else:  # 如果在主线程中
            # If running in the main thread, call retry_request directly  # 直接调用重试请求函数
            return retry_request()  # 返回重试请求的结果

    @staticmethod
    def _should_retry(status_code):
        """Determines if a request should be retried based on the HTTP status code.  # 根据 HTTP 状态码确定请求是否应该重试。"""
        retry_codes = {  # 定义重试状态码
            HTTPStatus.REQUEST_TIMEOUT,  # 请求超时
            HTTPStatus.BAD_GATEWAY,  # 错误网关
            HTTPStatus.GATEWAY_TIMEOUT,  # 网关超时
        }
        return status_code in retry_codes  # 返回状态码是否在重试列表中

    def _get_failure_message(self, response: requests.Response, retry: int, timeout: int):
        """
        Generate a retry message based on the response status code.  # 根据响应状态码生成重试消息。

        Args:
            response: The HTTP response object.  # HTTP 响应对象。
            retry: The number of retry attempts allowed.  # 允许的重试次数。
            timeout: The maximum timeout duration.  # 最大超时时间。

        Returns:
            (str): The retry message.  # 返回重试消息。
        """
        if self._should_retry(response.status_code):  # 如果应该重试
            return f"Retrying {retry}x for {timeout}s." if retry else ""  # 返回重试消息
        elif response.status_code == HTTPStatus.TOO_MANY_REQUESTS:  # rate limit  # 超过请求限制
            headers = response.headers  # 获取响应头
            return (  # 返回限速消息
                f"Rate limit reached ({headers['X-RateLimit-Remaining']}/{headers['X-RateLimit-Limit']}). "
                f"Please retry after {headers['Retry-After']}s."
            )
        else:
            try:
                return response.json().get("message", "No JSON message.")  # 尝试获取 JSON 消息
            except AttributeError:
                return "Unable to read JSON."  # 无法读取 JSON。

    def upload_metrics(self):
        """Upload model metrics to Ultralytics HUB.  # 上传模型指标到 Ultralytics HUB。"""
        return self.request_queue(self.model.upload_metrics, metrics=self.metrics_queue.copy(), thread=True)  # 请求上传指标

    def upload_model(  # 上传模型
        self,
        epoch: int,  # 当前训练周期
        weights: str,  # 模型权重文件路径
        is_best: bool = False,  # 当前模型是否为最佳模型
        map: float = 0.0,  # 模型的平均精度
        final: bool = False,  # 模型是否为最终模型
    ) -> None:
        """
        Upload a model checkpoint to Ultralytics HUB.  # 上传模型检查点到 Ultralytics HUB。

        Args:
            epoch (int): The current training epoch.  # 当前训练周期。
            weights (str): Path to the model weights file.  # 模型权重文件路径。
            is_best (bool): Indicates if the current model is the best one so far.  # 当前模型是否为最佳模型。
            map (float): Mean average precision of the model.  # 模型的平均精度。
            final (bool): Indicates if the model is the final model after training.  # 模型是否为最终模型。
        """
        weights = Path(weights)  # 转换权重路径为 Path 对象
        if not weights.is_file():  # 如果权重文件不存在
            last = weights.with_name(f"last{weights.suffix}")  # 获取最后的权重文件路径
            if final and last.is_file():  # 如果是最终模型且最后的权重文件存在
                LOGGER.warning(  # 记录警告信息
                    f"{PREFIX} WARNING ⚠️ Model 'best.pt' not found, copying 'last.pt' to 'best.pt' and uploading. "
                    "This often happens when resuming training in transient environments like Google Colab. "
                    "For more reliable training, consider using Ultralytics HUB Cloud. "
                    "Learn more at https://docs.ultralytics.com/hub/cloud-training."
                )
                shutil.copy(last, weights)  # 将 last.pt 复制到 best.pt
            else:
                LOGGER.warning(f"{PREFIX} WARNING ⚠️ Model upload issue. Missing model {weights}.")  # 记录模型上传问题的警告
                return  # 返回

        self.request_queue(  # 请求上传模型
            self.model.upload_model,
            epoch=epoch,  # 当前周期
            weights=str(weights),  # 权重路径
            is_best=is_best,  # 是否最佳模型
            map=map,  # 平均精度
            final=final,  # 是否最终模型
            retry=10,  # 重试次数
            timeout=3600,  # 超时时间
            thread=not final,  # 如果不是最终模型则使用线程
            progress_total=weights.stat().st_size if final else None,  # 仅在最终模型时显示进度
            stream_response=True,  # 流响应
        )

    @staticmethod
    def _show_upload_progress(content_length: int, response: requests.Response) -> None:
        """
        Display a progress bar to track the upload progress of a file download.  # 显示进度条以跟踪文件下载的上传进度。

        Args:
            content_length (int): The total size of the content to be downloaded in bytes.  # 要下载的内容的总大小（以字节为单位）。
            response (requests.Response): The response object from the file download request.  # 文件下载请求的响应对象。

        Returns:
            None  # 无返回值
        """
        with TQDM(total=content_length, unit="B", unit_scale=True, unit_divisor=1024) as pbar:  # 初始化进度条
            for data in response.iter_content(chunk_size=1024):  # 逐块处理响应内容
                pbar.update(len(data))  # 更新进度条

    @staticmethod
    def _iterate_content(response: requests.Response) -> None:
        """
        Process the streamed HTTP response data.  # 处理流式 HTTP 响应数据。

        Args:
            response (requests.Response): The response object from the file download request.  # 文件下载请求的响应对象。

        Returns:
            None  # 无返回值
        """
        for _ in response.iter_content(chunk_size=1024):  # 逐块处理响应内容
            pass  # Do nothing with data chunks  # 不对数据块做任何处理