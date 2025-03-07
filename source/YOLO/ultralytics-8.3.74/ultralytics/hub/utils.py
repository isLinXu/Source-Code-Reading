# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import platform
import random
import threading
import time
from pathlib import Path

import requests

from ultralytics.utils import (
    ARGV,
    ENVIRONMENT,
    IS_COLAB,
    IS_GIT_DIR,
    IS_PIP_PACKAGE,
    LOGGER,
    ONLINE,
    RANK,
    SETTINGS,
    TESTS_RUNNING,
    TQDM,
    TryExcept,
    __version__,
    colorstr,
    get_git_origin_url,
)
from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

HUB_API_ROOT = os.environ.get("ULTRALYTICS_HUB_API", "https://api.ultralytics.com")
HUB_WEB_ROOT = os.environ.get("ULTRALYTICS_HUB_WEB", "https://hub.ultralytics.com")

PREFIX = colorstr("Ultralytics HUB: ")
HELP_MSG = "If this issue persists please visit https://github.com/ultralytics/hub/issues for assistance."


def request_with_credentials(url: str) -> any:
    """
    Make an AJAX request with cookies attached in a Google Colab environment.

    Args:
        url (str): The URL to make the request to.

    Returns:
        (any): The response data from the AJAX request.

    Raises:
        OSError: If the function is not run in a Google Colab environment.
    """
    if not IS_COLAB:
        raise OSError("request_with_credentials() must run in a Colab environment")
    from google.colab import output  # noqa
    from IPython import display  # noqa

    display.display(
        display.Javascript(
            f"""
            window._hub_tmp = new Promise((resolve, reject) => {{
                const timeout = setTimeout(() => reject("Failed authenticating existing browser session"), 5000)
                fetch("{url}", {{
                    method: 'POST',
                    credentials: 'include'
                }})
                    .then((response) => resolve(response.json()))
                    .then((json) => {{
                    clearTimeout(timeout);
                    }}).catch((err) => {{
                    clearTimeout(timeout);
                    reject(err);
                }});
            }});
            """
        )
    )
    return output.eval_js("_hub_tmp")


def requests_with_progress(method, url, **kwargs):
    """
    Make an HTTP request using the specified method and URL, with an optional progress bar.

    Args:
        method (str): The HTTP method to use (e.g. 'GET', 'POST').
        url (str): The URL to send the request to.
        **kwargs (any): Additional keyword arguments to pass to the underlying `requests.request` function.

    Returns:
        (requests.Response): The response object from the HTTP request.

    Note:
        - If 'progress' is set to True, the progress bar will display the download progress for responses with a known
        content length.
        - If 'progress' is a number then progress bar will display assuming content length = progress.
    """
    progress = kwargs.pop("progress", False)
    if not progress:
        return requests.request(method, url, **kwargs)
    response = requests.request(method, url, stream=True, **kwargs)
    total = int(response.headers.get("content-length", 0) if isinstance(progress, bool) else progress)  # total size
    try:
        pbar = TQDM(total=total, unit="B", unit_scale=True, unit_divisor=1024)
        for data in response.iter_content(chunk_size=1024):
            pbar.update(len(data))
        pbar.close()
    except requests.exceptions.ChunkedEncodingError:  # avoid 'Connection broken: IncompleteRead' warnings
        response.close()
    return response


def smart_request(method, url, retry=3, timeout=30, thread=True, code=-1, verbose=True, progress=False, **kwargs):
    """
    Makes an HTTP request using the 'requests' library, with exponential backoff retries up to a specified timeout.

    Args:
        method (str): The HTTP method to use for the request. Choices are 'post' and 'get'.
        url (str): The URL to make the request to.
        retry (int, optional): Number of retries to attempt before giving up. Default is 3.
        timeout (int, optional): Timeout in seconds after which the function will give up retrying. Default is 30.
        thread (bool, optional): Whether to execute the request in a separate daemon thread. Default is True.
        code (int, optional): An identifier for the request, used for logging purposes. Default is -1.
        verbose (bool, optional): A flag to determine whether to print out to console or not. Default is True.
        progress (bool, optional): Whether to show a progress bar during the request. Default is False.
        **kwargs (any): Keyword arguments to be passed to the requests function specified in method.

    Returns:
        (requests.Response): The HTTP response object. If the request is executed in a separate thread, returns None.
    """
    retry_codes = (408, 500)  # retry only these codes

    @TryExcept(verbose=verbose)
    def func(func_method, func_url, **func_kwargs):
        """Make HTTP requests with retries and timeouts, with optional progress tracking."""
        r = None  # response
        t0 = time.time()  # initial time for timer
        for i in range(retry + 1):
            if (time.time() - t0) > timeout:
                break
            r = requests_with_progress(func_method, func_url, **func_kwargs)  # i.e. get(url, data, json, files)
            if r.status_code < 300:  # return codes in the 2xx range are generally considered "good" or "successful"
                break
            try:
                m = r.json().get("message", "No JSON message.")
            except AttributeError:
                m = "Unable to read JSON."
            if i == 0:
                if r.status_code in retry_codes:
                    m += f" Retrying {retry}x for {timeout}s." if retry else ""
                elif r.status_code == 429:  # rate limit
                    h = r.headers  # response headers
                    m = (
                        f"Rate limit reached ({h['X-RateLimit-Remaining']}/{h['X-RateLimit-Limit']}). "
                        f"Please retry after {h['Retry-After']}s."
                    )
                if verbose:
                    LOGGER.warning(f"{PREFIX}{m} {HELP_MSG} ({r.status_code} #{code})")
                if r.status_code not in retry_codes:
                    return r
            time.sleep(2**i)  # exponential standoff
        return r

    args = method, url
    kwargs["progress"] = progress
    if thread:
        threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True).start()
    else:
        return func(*args, **kwargs)


class Events:  # 定义Events类，用于收集匿名事件分析
    """
    A class for collecting anonymous event analytics. Event analytics are enabled when sync=True in settings and
    disabled when sync=False. Run 'yolo settings' to see and update settings.  # 用于收集匿名事件分析的类。当设置sync=True时启用事件分析，sync=False时禁用。运行'yolo settings'查看和更新设置。

    Attributes:
        url (str): The URL to send anonymous events.  # 发送匿名事件的URL
        rate_limit (float): The rate limit in seconds for sending events.  # 发送事件的速率限制（秒）
        metadata (dict): A dictionary containing metadata about the environment.  # 包含环境元数据的字典
        enabled (bool): A flag to enable or disable Events based on certain conditions.  # 根据特定条件启用或禁用事件的标志
    """

    url = "https://www.google-analytics.com/mp/collect?measurement_id=G-X8NCJYTQXM&api_secret=QLQrATrNSwGRFRLE-cbHJw"  # 事件收集的URL

    def __init__(self):  # 初始化Events对象
        """Initializes the Events object with default values for events, rate_limit, and metadata.  # 用默认值初始化Events对象的事件、速率限制和元数据。"""
        self.events = []  # events list  # 事件列表
        self.rate_limit = 30.0  # rate limit (seconds)  # 速率限制（秒）
        self.t = 0.0  # rate limit timer (seconds)  # 速率限制计时器（秒）
        self.metadata = {  # 元数据字典
            "cli": Path(ARGV[0]).name == "yolo",  # CLI名称是否为'yolo'
            "install": "git" if IS_GIT_DIR else "pip" if IS_PIP_PACKAGE else "other",  # 安装方式
            "python": ".".join(platform.python_version_tuple()[:2]),  # Python版本，例如3.10
            "version": __version__,  # 当前版本
            "env": ENVIRONMENT,  # 环境
            "session_id": round(random.random() * 1e15),  # 会话ID
            "engagement_time_msec": 1000,  # 参与时间（毫秒）
        }
        self.enabled = (  # 根据条件启用或禁用事件
            SETTINGS["sync"]  # 是否同步
            and RANK in {-1, 0}  # 排名条件
            and not TESTS_RUNNING  # 不是在测试运行中
            and ONLINE  # 在线状态
            and (IS_PIP_PACKAGE or get_git_origin_url() == "https://github.com/ultralytics/ultralytics.git")  # 安装来源
        )

    def __call__(self, cfg):  # 定义可调用方法
        """
        Attempts to add a new event to the events list and send events if the rate limit is reached.  # 尝试向事件列表添加新事件，并在达到速率限制时发送事件。

        Args:
            cfg (IterableSimpleNamespace): The configuration object containing mode and task information.  # 包含模式和任务信息的配置对象。
        """
        if not self.enabled:  # 如果事件未启用
            # Events disabled, do nothing  # 事件被禁用，不执行任何操作
            return

        # Attempt to add to events
        if len(self.events) < 25:  # 事件列表限制为25个事件（丢弃超过此限制的事件）
            params = {  # 事件参数
                **self.metadata,  # 合并元数据
                "task": cfg.task,  # 任务信息
                "model": cfg.model if cfg.model in GITHUB_ASSETS_NAMES else "custom",  # 模型信息
            }
            if cfg.mode == "export":  # 如果模式为'export'
                params["format"] = cfg.format  # 添加格式参数
            self.events.append({"name": cfg.mode, "params": params})  # 将事件添加到列表

        # Check rate limit
        t = time.time()  # 获取当前时间
        if (t - self.t) < self.rate_limit:  # 如果时间在速率限制内
            # Time is under rate limiter, wait to send  # 时间在速率限制内，等待发送
            return

        # Time is over rate limiter, send now
        data = {"client_id": SETTINGS["uuid"], "events": self.events}  # SHA-256匿名UUID哈希和事件列表

        # POST equivalent to requests.post(self.url, json=data)  # 发送POST请求
        smart_request("post", self.url, json=data, retry=0, verbose=False)  # 发送事件数据

        # Reset events and rate limit timer
        self.events = []  # 重置事件列表
        self.t = t  # 重置速率限制计时器

# Run below code on hub/utils init -------------------------------------------------------------------------------------
events = Events()  # 初始化事件对象
