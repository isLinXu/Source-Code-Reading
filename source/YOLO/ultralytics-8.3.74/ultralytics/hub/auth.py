# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import requests

from ultralytics.hub.utils import HUB_API_ROOT, HUB_WEB_ROOT, PREFIX, request_with_credentials
from ultralytics.utils import IS_COLAB, LOGGER, SETTINGS, emojis

API_KEY_URL = f"{HUB_WEB_ROOT}/settings?tab=api+keys"


class Auth:
    """
    Manages authentication processes including API key handling, cookie-based authentication, and header generation.

    The class supports different methods of authentication:
    1. Directly using an API key.
    2. Authenticating using browser cookies (specifically in Google Colab).
    3. Prompting the user to enter an API key.

    Attributes:
        id_token (str or bool): Token used for identity verification, initialized as False.
        api_key (str or bool): API key for authentication, initialized as False.
        model_key (bool): Placeholder for model key, initialized as False.
    """

    id_token = api_key = model_key = False
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import requests  # 导入请求模块

from ultralytics.hub.utils import HUB_API_ROOT, HUB_WEB_ROOT, PREFIX, request_with_credentials  # 从 ultralytics.hub.utils 导入相关工具
from ultralytics.utils import IS_COLAB, LOGGER, SETTINGS, emojis  # 从 ultralytics.utils 导入工具和设置

API_KEY_URL = f"{HUB_WEB_ROOT}/settings?tab=api+keys"  # 定义 API 密钥的 URL


class Auth:
    """
    Manages authentication processes including API key handling, cookie-based authentication, and header generation.  # 管理身份验证过程，包括 API 密钥处理、基于 cookie 的身份验证和头部生成。

    The class supports different methods of authentication:  # 该类支持不同的身份验证方法：
    1. Directly using an API key.  # 直接使用 API 密钥。
    2. Authenticating using browser cookies (specifically in Google Colab).  # 使用浏览器 cookie 进行身份验证（特别是在 Google Colab 中）。
    3. Prompting the user to enter an API key.  # 提示用户输入 API 密钥。

    Attributes:
        id_token (str or bool): Token used for identity verification, initialized as False.  # 用于身份验证的令牌，初始化为 False。
        api_key (str or bool): API key for authentication, initialized as False.  # 用于身份验证的 API 密钥，初始化为 False。
        model_key (bool): Placeholder for model key, initialized as False.  # 模型密钥的占位符，初始化为 False。
    """

    id_token = api_key = model_key = False  # 初始化 id_token、api_key 和 model_key 为 False

    def __init__(self, api_key="", verbose=False):
        """
        Initialize Auth class and authenticate user.  # 初始化 Auth 类并进行用户身份验证。

        Handles API key validation, Google Colab authentication, and new key requests. Updates SETTINGS upon successful  # 处理 API 密钥验证、Google Colab 身份验证和新密钥请求。成功后更新 SETTINGS。
        authentication.

        Args:
            api_key (str): API key or combined key_id format.  # API 密钥或组合的 key_id 格式。
            verbose (bool): Enable verbose logging.  # 启用详细日志记录。
        """
        # Split the input API key in case it contains a combined key_model and keep only the API key part  # 分割输入的 API 密钥，以防它包含组合的 key_model，仅保留 API 密钥部分
        api_key = api_key.split("_")[0]  # 获取 API 密钥部分

        # Set API key attribute as value passed or SETTINGS API key if none passed  # 如果未提供，则将 API 密钥属性设置为传递的值或 SETTINGS 中的 API 密钥
        self.api_key = api_key or SETTINGS.get("api_key", "")  # 设置 API 密钥

        # If an API key is provided  # 如果提供了 API 密钥
        if self.api_key:
            # If the provided API key matches the API key in the SETTINGS  # 如果提供的 API 密钥与 SETTINGS 中的 API 密钥匹配
            if self.api_key == SETTINGS.get("api_key"):
                # Log that the user is already logged in  # 记录用户已登录的信息
                if verbose:
                    LOGGER.info(f"{PREFIX}Authenticated ✅")  # 记录已认证信息
                return  # 返回
            else:
                # Attempt to authenticate with the provided API key  # 尝试使用提供的 API 密钥进行身份验证
                success = self.authenticate()  # 进行身份验证
        # If the API key is not provided and the environment is a Google Colab notebook  # 如果未提供 API 密钥并且环境是 Google Colab 笔记本
        elif IS_COLAB:
            # Attempt to authenticate using browser cookies  # 尝试使用浏览器 cookie 进行身份验证
            success = self.auth_with_cookies()  # 使用 cookie 进行身份验证
        else:
            # Request an API key  # 请求 API 密钥
            success = self.request_api_key()  # 请求 API 密钥

        # Update SETTINGS with the new API key after successful authentication  # 成功身份验证后，用新 API 密钥更新 SETTINGS
        if success:
            SETTINGS.update({"api_key": self.api_key})  # 更新 SETTINGS
            # Log that the new login was successful  # 记录新的登录成功信息
            if verbose:
                LOGGER.info(f"{PREFIX}New authentication successful ✅")  # 记录新认证成功信息
        elif verbose:
            LOGGER.info(f"{PREFIX}Get API key from {API_KEY_URL} and then run 'yolo login API_KEY'")  # 记录获取 API 密钥的信息

    def request_api_key(self, max_attempts=3):
        """
        Prompt the user to input their API key.  # 提示用户输入他们的 API 密钥。

        Returns the model ID.  # 返回模型 ID。
        """
        import getpass  # 导入 getpass 模块

        for attempts in range(max_attempts):  # 尝试次数循环
            LOGGER.info(f"{PREFIX}Login. Attempt {attempts + 1} of {max_attempts}")  # 记录登录尝试信息
            input_key = getpass.getpass(f"Enter API key from {API_KEY_URL} ")  # 提示用户输入 API 密钥
            self.api_key = input_key.split("_")[0]  # remove model id if present  # 如果存在，移除模型 ID
            if self.authenticate():  # 尝试进行身份验证
                return True  # 返回成功
        raise ConnectionError(emojis(f"{PREFIX}Failed to authenticate ❌"))  # 引发连接错误

    def authenticate(self) -> bool:
        """
        Attempt to authenticate with the server using either id_token or API key.  # 尝试使用 id_token 或 API 密钥进行服务器身份验证。

        Returns:
            (bool): True if authentication is successful, False otherwise.  # 如果身份验证成功，返回 True，否则返回 False。
        """
        try:
            if header := self.get_auth_header():  # 获取身份验证头部
                r = requests.post(f"{HUB_API_ROOT}/v1/auth", headers=header)  # 发送身份验证请求
                if not r.json().get("success", False):  # 检查响应是否成功
                    raise ConnectionError("Unable to authenticate.")  # 引发连接错误
                return True  # 返回成功
            raise ConnectionError("User has not authenticated locally.")  # 引发连接错误
        except ConnectionError:
            self.id_token = self.api_key = False  # reset invalid  # 重置无效
            LOGGER.warning(f"{PREFIX}Invalid API key ⚠️")  # 记录无效 API 密钥警告
            return False  # 返回失败

    def auth_with_cookies(self) -> bool:
        """
        Attempt to fetch authentication via cookies and set id_token.  # 尝试通过 cookie 获取身份验证并设置 id_token。
        User must be logged in to HUB and running in a supported browser.  # 用户必须登录 HUB 并在支持的浏览器中运行。

        Returns:
            (bool): True if authentication is successful, False otherwise.  # 如果身份验证成功，返回 True，否则返回 False。
        """
        if not IS_COLAB:  # 如果不是 Google Colab
            return False  # 目前仅在 Colab 中有效
        try:
            authn = request_with_credentials(f"{HUB_API_ROOT}/v1/auth/auto")  # 尝试获取身份验证
            if authn.get("success", False):  # 如果获取成功
                self.id_token = authn.get("data", {}).get("idToken", None)  # 设置 id_token
                self.authenticate()  # 进行身份验证
                return True  # 返回成功
            raise ConnectionError("Unable to fetch browser authentication details.")  # 引发连接错误
        except ConnectionError:
            self.id_token = False  # reset invalid  # 重置无效
            return False  # 返回失败

    def get_auth_header(self):
        """
        Get the authentication header for making API requests.  # 获取进行 API 请求的身份验证头部。

        Returns:
            (dict): The authentication header if id_token or API key is set, None otherwise.  # 如果设置了 id_token 或 API 密钥，则返回身份验证头部，否则返回 None。
        """
        if self.id_token:  # 如果存在 id_token
            return {"authorization": f"Bearer {self.id_token}"}  # 返回 Bearer 令牌
        elif self.api_key:  # 如果存在 API 密钥
            return {"x-api-key": self.api_key}  # 返回 API 密钥头部
        # else returns None  # 否则返回 None
