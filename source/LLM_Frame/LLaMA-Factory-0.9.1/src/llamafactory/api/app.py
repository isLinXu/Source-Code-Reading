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

# 导入异步和Web框架相关模块
import asyncio
import os
from contextlib import asynccontextmanager
from functools import partial
from typing import Optional

from typing_extensions import Annotated

# 导入项目内部模块
from ..chat import ChatModel
from ..extras.misc import torch_gc  # GPU内存清理工具
from ..extras.packages import is_fastapi_available, is_starlette_available, is_uvicorn_available  # 依赖检查

# 导入API响应生成函数
from .chat import (
    create_chat_completion_response,
    create_score_evaluation_response,
    create_stream_chat_completion_response,
)

# 导入数据协议模型
from .protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelCard,
    ModelList,
    ScoreEvaluationRequest,
    ScoreEvaluationResponse,
)

# 条件导入FastAPI组件
if is_fastapi_available():
    from fastapi import Depends, FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware  # CORS中间件
    from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer  # 认证组件

# 条件导入SSE支持
if is_starlette_available():
    from sse_starlette import EventSourceResponse  # 服务器发送事件响应

# 条件导入服务器组件
if is_uvicorn_available():
    import uvicorn


async def sweeper() -> None:
    """定时清理GPU内存的后台任务"""
    while True:
        torch_gc()  # 执行GPU内存清理
        await asyncio.sleep(300)  # 每5分钟执行一次


@asynccontextmanager
async def lifespan(app: "FastAPI", chat_model: "ChatModel"):  # collects GPU memory
    """应用生命周期管理（HuggingFace引擎专用）"""
    if chat_model.engine_type == "huggingface":
        asyncio.create_task(sweeper())  # 创建定时清理任务

    yield  # 应用运行期间保持
    torch_gc()  # 应用关闭时最后清理


def create_app(chat_model: "ChatModel") -> "FastAPI":
    """创建并配置FastAPI应用实例"""
    root_path = os.getenv("FASTAPI_ROOT_PATH", "")  # 获取根路径配置
    app = FastAPI(lifespan=partial(lifespan, chat_model=chat_model), root_path=root_path)  # 注入生命周期管理
    
    # 添加CORS中间件配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有来源
        allow_credentials=True,  # 允许携带凭证
        allow_methods=["*"],  # 允许所有HTTP方法
        allow_headers=["*"],  # 允许所有请求头
    )
    
    api_key = os.getenv("API_KEY")  # 从环境变量获取API密钥
    security = HTTPBearer(auto_error=False)  # 创建Bearer认证组件

    async def verify_api_key(auth: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)]):
        """API密钥验证依赖项"""
        if api_key and (auth is None or auth.credentials != api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")

    @app.get(
        "/v1/models",
        response_model=ModelList,
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_api_key)],  # 添加认证依赖
    )
    async def list_models():
        """模型列表接口（兼容OpenAI格式）"""
        model_card = ModelCard(id=os.getenv("API_MODEL_NAME", "gpt-3.5-turbo"))  # 创建模型卡片
        return ModelList(data=[model_card])  # 返回包装后的响应

    @app.post(
        "/v1/chat/completions",
        response_model=ChatCompletionResponse,
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_api_key)],
    )
    async def create_chat_completion(request: ChatCompletionRequest):
        """聊天补全接口（兼容OpenAI格式）"""
        if not chat_model.engine.can_generate:  # 检查模型是否支持生成
            raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Not allowed")

        if request.stream:  # 处理流式响应
            generate = create_stream_chat_completion_response(request, chat_model)
            return EventSourceResponse(generate, media_type="text/event-stream")  # SSE流式响应
        else:  # 处理普通响应
            return await create_chat_completion_response(request, chat_model)

    @app.post(
        "/v1/score/evaluation",
        response_model=ScoreEvaluationResponse,
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_api_key)],
    )
    async def create_score_evaluation(request: ScoreEvaluationRequest):
        """评分评估接口（自定义功能）"""
        if chat_model.engine.can_generate:  # 检查模型是否支持评估
            raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Not allowed")

        return await create_score_evaluation_response(request, chat_model)

    return app


def run_api() -> None:
    """启动API服务的入口函数"""
    chat_model = ChatModel()  # 初始化聊天模型
    app = create_app(chat_model)  # 创建FastAPI应用
    
    # 从环境变量获取配置参数
    api_host = os.getenv("API_HOST", "0.0.0.0")  # 监听地址
    api_port = int(os.getenv("API_PORT", "8000"))  # 监听端口
    
    print(f"Visit http://localhost:{api_port}/docs for API document.")  # 打印文档地址
    uvicorn.run(app, host=api_host, port=api_port)  # 启动UVicorn服务器
