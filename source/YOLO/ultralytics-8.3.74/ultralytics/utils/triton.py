# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import List
from urllib.parse import urlsplit

import numpy as np


class TritonRemoteModel:  # 定义 TritonRemoteModel 类
    """
    Client for interacting with a remote Triton Inference Server model.  # 与远程 Triton 推理服务器模型交互的客户端

    Attributes:  # 属性说明
        endpoint (str): The name of the model on the Triton server.  # 模型在 Triton 服务器上的名称
        url (str): The URL of the Triton server.  # Triton 服务器的 URL
        triton_client: The Triton client (either HTTP or gRPC).  # Triton 客户端（HTTP 或 gRPC）
        InferInput: The input class for the Triton client.  # Triton 客户端的输入类
        InferRequestedOutput: The output request class for the Triton client.  # Triton 客户端的输出请求类
        input_formats (List[str]): The data types of the model inputs.  # 模型输入的数据类型
        np_input_formats (List[type]): The numpy data types of the model inputs.  # 模型输入的 numpy 数据类型
        input_names (List[str]): The names of the model inputs.  # 模型输入的名称
        output_names (List[str]): The names of the model outputs.  # 模型输出的名称
    """

    def __init__(self, url: str, endpoint: str = "", scheme: str = ""):  # 初始化方法
        """
        Initialize the TritonRemoteModel.  # 初始化 TritonRemoteModel

        Arguments may be provided individually or parsed from a collective 'url' argument of the form  # 参数可以单独提供或从一个集体的 'url' 参数解析
            <scheme>://<netloc>/<endpoint>/<task_name>  # URL 格式

        Args:  # 参数说明
            url (str): The URL of the Triton server.  # Triton 服务器的 URL
            endpoint (str): The name of the model on the Triton server.  # 模型在 Triton 服务器上的名称
            scheme (str): The communication scheme ('http' or 'grpc').  # 通信方案（'http' 或 'grpc'）
        """
        if not endpoint and not scheme:  # 如果没有提供 endpoint 和 scheme，则从 URL 字符串解析所有参数
            splits = urlsplit(url)  # 解析 URL
            endpoint = splits.path.strip("/").split("/")[0]  # 获取 endpoint
            scheme = splits.scheme  # 获取 scheme
            url = splits.netloc  # 获取 URL 的网络位置

        self.endpoint = endpoint  # 设置实例的 endpoint 属性
        self.url = url  # 设置实例的 url 属性

        # Choose the Triton client based on the communication scheme  # 根据通信方案选择 Triton 客户端
        if scheme == "http":  # 如果方案是 HTTP
            import tritonclient.http as client  # noqa  # 导入 HTTP 客户端

            self.triton_client = client.InferenceServerClient(url=self.url, verbose=False, ssl=False)  # 创建 Triton 客户端实例
            config = self.triton_client.get_model_config(endpoint)  # 获取模型配置
        else:  # 如果方案不是 HTTP
            import tritonclient.grpc as client  # noqa  # 导入 gRPC 客户端

            self.triton_client = client.InferenceServerClient(url=self.url, verbose=False, ssl=False)  # 创建 Triton 客户端实例
            config = self.triton_client.get_model_config(endpoint, as_json=True)["config"]  # 获取模型配置

        # Sort output names alphabetically, i.e. 'output0', 'output1', etc.  # 按字母顺序排序输出名称，例如 'output0', 'output1' 等
        config["output"] = sorted(config["output"], key=lambda x: x.get("name"))  # 对输出进行排序

        # Define model attributes  # 定义模型属性
        type_map = {"TYPE_FP32": np.float32, "TYPE_FP16": np.float16, "TYPE_UINT8": np.uint8}  # 数据类型映射
        self.InferRequestedOutput = client.InferRequestedOutput  # 设置 InferRequestedOutput 属性
        self.InferInput = client.InferInput  # 设置 InferInput 属性
        self.input_formats = [x["data_type"] for x in config["input"]]  # 获取输入格式
        self.np_input_formats = [type_map[x] for x in self.input_formats]  # 获取 numpy 输入格式
        self.input_names = [x["name"] for x in config["input"]]  # 获取输入名称
        self.output_names = [x["name"] for x in config["output"]]  # 获取输出名称
        self.metadata = eval(config.get("parameters", {}).get("metadata", {}).get("string_value", "None"))  # 获取元数据

    def __call__(self, *inputs: np.ndarray) -> List[np.ndarray]:  # 定义调用方法
        """
        Call the model with the given inputs.  # 使用给定的输入调用模型

        Args:  # 参数说明
            *inputs (List[np.ndarray]): Input data to the model.  # 输入数据

        Returns:  # 返回值说明
            (List[np.ndarray]): Model outputs.  # 模型输出
        """
        infer_inputs = []  # 初始化推理输入列表
        input_format = inputs[0].dtype  # 获取输入数据类型
        for i, x in enumerate(inputs):  # 遍历输入数据
            if x.dtype != self.np_input_formats[i]:  # 如果数据类型不匹配
                x = x.astype(self.np_input_formats[i])  # 转换数据类型
            infer_input = self.InferInput(self.input_names[i], [*x.shape], self.input_formats[i].replace("TYPE_", ""))  # 创建推理输入
            infer_input.set_data_from_numpy(x)  # 设置推理输入数据
            infer_inputs.append(infer_input)  # 添加到推理输入列表

        infer_outputs = [self.InferRequestedOutput(output_name) for output_name in self.output_names]  # 创建推理输出列表
        outputs = self.triton_client.infer(model_name=self.endpoint, inputs=infer_inputs, outputs=infer_outputs)  # 执行推理

        return [outputs.as_numpy(output_name).astype(input_format) for output_name in self.output_names]  # 返回模型输出