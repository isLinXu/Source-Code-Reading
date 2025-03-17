# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# API服务入口文件

import os  # 导入操作系统模块用于环境变量读取

import uvicorn  # 导入ASGI服务器用于运行FastAPI应用

from llamafactory.api.app import create_app  # 从项目模块导入FastAPI应用创建函数
from llamafactory.chat import ChatModel  # 导入聊天模型类


def main():
    """主函数，负责初始化并启动API服务"""
    chat_model = ChatModel()  # 实例化聊天模型（加载预训练模型）
    app = create_app(chat_model)  # 创建FastAPI应用实例并注入模型
    
    # 从环境变量读取配置，设置默认值
    api_host = os.getenv("API_HOST", "0.0.0.0")  # 服务绑定地址，默认监听所有接口
    api_port = int(os.getenv("API_PORT", "8000"))  # 服务端口号，默认8000
    
    # 打印访问文档提示信息（Swagger UI）
    print(f"Visit http://localhost:{api_port}/docs for API document.")
    
    # 启动UVicorn服务器
    uvicorn.run(
        app,  # FastAPI应用实例
        host=api_host,  # 服务绑定地址
        port=api_port  # 服务监听端口
    )


if __name__ == "__main__":
    main()  # 直接运行时的入口点
