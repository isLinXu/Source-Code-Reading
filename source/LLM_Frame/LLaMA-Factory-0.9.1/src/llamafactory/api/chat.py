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

# 导入基础模块和类型声明
import base64  # Base64编解码
import io  # 字节流处理
import json  # JSON处理
import os  # 文件系统操作
import re  # 正则表达式
import uuid  # UUID生成
from typing import TYPE_CHECKING, AsyncGenerator, Dict, List, Optional, Tuple

# 导入项目内部模块
from ..data import Role as DataRole  # 数据角色枚举
from ..extras import logging  # 日志模块
from ..extras.packages import is_fastapi_available, is_pillow_available, is_requests_available  # 依赖检查
from .common import dictify, jsonify  # 数据转换工具
from .protocol import (  # 协议数据结构
    ChatCompletionMessage,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseUsage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    Finish,
    Function,
    FunctionCall,
    Role,
    ScoreEvaluationResponse,
)

# 条件导入FastAPI组件
if is_fastapi_available():
    from fastapi import HTTPException, status  # HTTP异常处理

# 条件导入图像处理模块
if is_pillow_available():
    from PIL import Image  # 图像处理

# 条件导入HTTP请求模块
if is_requests_available():
    import requests  # HTTP客户端

# 类型提示相关导入
if TYPE_CHECKING:
    from ..chat import ChatModel  # 聊天模型类型提示
    from ..data.mm_plugin import ImageInput  # 图像输入类型提示
    from .protocol import ChatCompletionRequest, ScoreEvaluationRequest  # 请求类型提示

logger = logging.get_logger(__name__)

# 角色映射表（将API角色转换为内部角色）
ROLE_MAPPING = {
    Role.USER: DataRole.USER.value,
    Role.ASSISTANT: DataRole.ASSISTANT.value,
    Role.SYSTEM: DataRole.SYSTEM.value,
    Role.FUNCTION: DataRole.FUNCTION.value,
    Role.TOOL: DataRole.OBSERVATION.value,
}


def _process_request(
    request: "ChatCompletionRequest",
) -> Tuple[List[Dict[str, str]], Optional[str], Optional[str], Optional[List["ImageInput"]]]:
    """处理聊天请求的核心逻辑"""
    logger.info_rank0(f"==== request ====\n{json.dumps(dictify(request), indent=2, ensure_ascii=False)}")

    # 消息长度校验
    if len(request.messages) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid length")

    # 提取系统消息
    if request.messages[0].role == Role.SYSTEM:
        system = request.messages.pop(0).content
    else:
        system = None

    # 消息顺序校验（必须为user/assistant交替）
    if len(request.messages) % 2 == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only supports u/a/u/a/u...")

    input_messages = []  # 格式化后的消息列表
    images = []  # 图像输入列表
    for i, message in enumerate(request.messages):
        # 角色顺序校验
        if i % 2 == 0 and message.role not in [Role.USER, Role.TOOL]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role")
        elif i % 2 == 1 and message.role not in [Role.ASSISTANT, Role.FUNCTION]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role")

        # 处理工具调用结果
        if message.role == Role.ASSISTANT and isinstance(message.tool_calls, list) and len(message.tool_calls):
            tool_calls = [
                {"name": tool_call.function.name, "arguments": tool_call.function.arguments}
                for tool_call in message.tool_calls
            ]
            content = json.dumps(tool_calls, ensure_ascii=False)
            input_messages.append({"role": ROLE_MAPPING[Role.FUNCTION], "content": content})
        # 处理多模态输入
        elif isinstance(message.content, list):
            for input_item in message.content:
                if input_item.type == "text":  # 文本内容
                    input_messages.append({"role": ROLE_MAPPING[message.role], "content": input_item.text})
                else:  # 图像内容
                    image_url = input_item.image_url.url
                    # 处理Base64编码图像
                    if re.match(r"^data:image\/(png|jpg|jpeg|gif|bmp);base64,(.+)$", image_url):
                        image_stream = io.BytesIO(base64.b64decode(image_url.split(",", maxsplit=1)[1]))
                    # 处理本地文件图像
                    elif os.path.isfile(image_url):
                        image_stream = open(image_url, "rb")
                    # 处理网络URL图像
                    else:
                        image_stream = requests.get(image_url, stream=True).raw

                    images.append(Image.open(image_stream).convert("RGB"))
        else:  # 普通文本消息
            input_messages.append({"role": ROLE_MAPPING[message.role], "content": message.content})

    # 处理工具参数
    tool_list = request.tools
    if isinstance(tool_list, list) and len(tool_list):
        try:
            tools = json.dumps([dictify(tool.function) for tool in tool_list], ensure_ascii=False)
        except json.JSONDecodeError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid tools")
    else:
        tools = None

    return input_messages, system, tools, images or None


def _create_stream_chat_completion_chunk(
    completion_id: str,
    model: str,
    delta: "ChatCompletionMessage",
    index: Optional[int] = 0,
    finish_reason: Optional["Finish"] = None,
) -> str:
    """构建流式响应数据块"""
    choice_data = ChatCompletionStreamResponseChoice(index=index, delta=delta, finish_reason=finish_reason)
    chunk = ChatCompletionStreamResponse(id=completion_id, model=model, choices=[choice_data])
    return jsonify(chunk)  # 序列化为JSON字符串


async def create_chat_completion_response(
    request: "ChatCompletionRequest", chat_model: "ChatModel"
) -> "ChatCompletionResponse":
    """生成普通聊天响应"""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"  # 生成唯一ID
    input_messages, system, tools, images = _process_request(request)
    responses = await chat_model.achat(  # 异步调用模型聊天
        input_messages,
        system,
        tools,
        images,
        do_sample=request.do_sample,
        temperature=request.temperature,
        top_p=request.top_p,
        max_new_tokens=request.max_tokens,
        num_return_sequences=request.n,
        stop=request.stop,
    )

    prompt_length, response_length = 0, 0
    choices = []
    for i, response in enumerate(responses):
        # 提取工具调用结果
        if tools:
            result = chat_model.engine.template.extract_tool(response.response_text)
        else:
            result = response.response_text

        # 构建响应消息
        if isinstance(result, list):  # 工具调用
            tool_calls = []
            for tool in result:
                function = Function(name=tool[0], arguments=tool[1])
                tool_calls.append(FunctionCall(id=f"call_{uuid.uuid4().hex}", function=function))

            response_message = ChatCompletionMessage(role=Role.ASSISTANT, tool_calls=tool_calls)
            finish_reason = Finish.TOOL
        else:  # 普通文本
            response_message = ChatCompletionMessage(role=Role.ASSISTANT, content=result)
            finish_reason = Finish.STOP if response.finish_reason == "stop" else Finish.LENGTH

        choices.append(ChatCompletionResponseChoice(index=i, message=response_message, finish_reason=finish_reason))
        prompt_length = response.prompt_length
        response_length += response.response_length

    # 构建使用量统计
    usage = ChatCompletionResponseUsage(
        prompt_tokens=prompt_length,
        completion_tokens=response_length,
        total_tokens=prompt_length + response_length,
    )

    return ChatCompletionResponse(id=completion_id, model=request.model, choices=choices, usage=usage)


async def create_stream_chat_completion_response(
    request: "ChatCompletionRequest", chat_model: "ChatModel"
) -> AsyncGenerator[str, None]:
    """生成流式聊天响应"""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    input_messages, system, tools, images = _process_request(request)
    
    # 流式传输限制检查
    if tools:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot stream function calls.")
    if request.n > 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot stream multiple responses.")

    # 发送初始空响应
    yield _create_stream_chat_completion_chunk(
        completion_id=completion_id, model=request.model, delta=ChatCompletionMessage(role=Role.ASSISTANT, content="")
    )
    
    # 流式生成令牌
    async for new_token in chat_model.astream_chat(
        input_messages,
        system,
        tools,
        images,
        do_sample=request.do_sample,
        temperature=request.temperature,
        top_p=request.top_p,
        max_new_tokens=request.max_tokens,
        stop=request.stop,
    ):
        if len(new_token) != 0:
            yield _create_stream_chat_completion_chunk(
                completion_id=completion_id, model=request.model, delta=ChatCompletionMessage(content=new_token)
            )

    # 发送结束标记
    yield _create_stream_chat_completion_chunk(
        completion_id=completion_id, model=request.model, delta=ChatCompletionMessage(), finish_reason=Finish.STOP
    )
    yield "[DONE]"


async def create_score_evaluation_response(
    request: "ScoreEvaluationRequest", chat_model: "ChatModel"
) -> "ScoreEvaluationResponse":
    """生成评分评估响应"""
    score_id = f"scoreval-{uuid.uuid4().hex}"  # 生成唯一评分ID
    if len(request.messages) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request")

    scores = await chat_model.aget_scores(request.messages, max_length=request.max_length)
    return ScoreEvaluationResponse(id=score_id, model=request.model, scores=scores)
