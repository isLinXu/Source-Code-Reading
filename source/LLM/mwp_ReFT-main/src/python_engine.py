# Copyright 2023 Bytedance Ltd.
# 
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
from typing import Any, Dict
# 从 typing 模块导入 Any 和 Dict 类型

from src.utils import timeout
# 从 src.utils 模块导入 timeout 函数

import time
# 导入 time 模块
from tqdm import tqdm
# 从 tqdm 模块导入 tqdm，用于显示进度条
import numpy as np
# 导入 numpy 模块并简写为 np
from pebble import ProcessPool
# 从 pebble 模块导入 ProcessPool，用于并行处理
import sympy
# 导入 sympy 模块，用于符号数学计算
import math
# 导入 math 模块，用于数学运算
import copy
# 导入 copy 模块，用于对象复制

# 创建一个全局限制字典，包含 'sympy' 和 'math' 库
global_restricted = {lib: globals()[lib] for lib in ['sympy', 'math']}
# del global_restricted['sympy'].init_session  # 注释掉的代码，可能用于删除 sympy 的初始化会话

local_restricted = {}
# 初始化本地限制字典

def exec_code(code_piece, _global_vars, _local_vars):
    # 定义执行代码的函数，接受代码片段、全局变量和局部变量
    exec(code_piece, _global_vars, _local_vars)
    # 使用 exec 函数执行代码片段

def eval_code(expr, _global_vars, _local_vars):
    # 定义评估表达式的函数，接受表达式、全局变量和局部变量
    return eval(expr, _global_vars, _local_vars)
    # 使用 eval 函数评估表达式并返回结果

def run(code_piece, expr):
    # 定义运行代码的函数，接受代码片段和表达式
    _global_vars, _local_vars = {}, {}
    # 初始化全局和局部变量字典
    for lib in ['sympy', 'math']:
        # 遍历需要导入的库
        _global_vars[lib] = global_restricted[lib]
        # 将全局限制字典中的库添加到全局变量中
        if lib in local_restricted:
            _local_vars[lib] = local_restricted[lib]
            # 如果库在本地限制字典中，将其添加到局部变量中
    exec(code_piece, _global_vars, _local_vars)
    # 执行代码片段
    result = eval(expr, _global_vars, _local_vars)
    # 评估表达式并获取结果
    return result
    # 返回结果

def process_code(code_gen, truncate_first_return=False):
    # 定义处理代码的函数，接受代码生成字符串和是否截断第一个返回值的标志
    ## deal with blacklist keyword
    if 'sys.exit' in code_gen:
        # 如果代码生成字符串中包含 sys.exit
        code_gen = code_gen.replace('sys.exit', 'print')
        # 将 sys.exit 替换为 print，避免程序退出
    snippet = code_gen.split('\n')
    # 将代码生成字符串按行分割成列表
    ## post process the code
    updated_code_snippet = ['import math', 'import sympy']
    # 初始化更新后的代码片段列表，包含必要的导入语句
    for snippet_line in snippet:
        # 遍历每一行代码
        if snippet_line.startswith('def solution'):
            # 如果当前行是函数定义
            updated_code_snippet.append(snippet_line)
            # 将函数定义添加到更新后的代码片段列表中
            continue
        if snippet_line.strip() == "":
            # 如果当前行为空行
            break
            # 退出循环
        if truncate_first_return:
            # 如果设置了截断第一个返回值
            if snippet_line == "    return result":
                break
                # 如果当前行是返回语句，则退出循环
        updated_code_snippet.append(snippet_line)
        # 将当前行代码添加到更新后的代码片段列表中
    updated_code_gen = '\n'.join(updated_code_snippet)
    # 将更新后的代码片段列表合并为字符串
    return updated_code_gen
    # 返回更新后的代码字符串

def run_python_code(programs, TIMEOUT: float, safe=True):
    # 定义运行 Python 代码的函数，接受代码列表、超时时间和安全标志
    is_single_program = False
    # 初始化标志，表示是否为单个程序
    if not isinstance(programs, list):
        # 如果 programs 不是列表
        is_single_program = True
        # 设置标志为 True，表示是单个程序
        programs = [programs]
        # 将单个程序包装成列表
    updated_programs = [process_code(code) for code in programs]
    # 处理每个程序代码，返回更新后的代码列表

    if safe:
        # Safer -- executed code can't affect main code (e.g numpy.random.seed(...))
        # 但速度较慢 ...
        # 如果安全标志为 True，表示以安全方式执行代码
        with ProcessPool(max_workers=8) as pool:
            # 创建一个进程池，最多使用 8 个工作进程
            futures = [pool.schedule(run, args=[code, 'solution()'], timeout=TIMEOUT) for code in updated_programs]
            # 调度每个更新后的程序代码，运行 [run](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/LLM/mwp_ReFT-main/src/python_engine.py:51:0-67:18) 函数并传入参数，设置超时
            results = []
            for i, f in tqdm(enumerate(futures), total=len(futures), disable=True):
                # 遍历每个未来对象，使用 tqdm 显示进度条
                try:
                    res = f.result()
                    # 获取执行结果
                except Exception as e:
                    print(str(e)) #, updated_programs[i])
                    # 捕获异常并打印错误信息
                    res = None
                    # 如果发生异常，结果设为 None
                results.append(res)
                # 将结果添加到结果列表中
    else:
        results = []
        # 如果安全标志为 False，初始化结果列表
        for code in tqdm(updated_programs, disable=True):
            # 遍历每个更新后的程序代码，使用 tqdm 显示进度条
            with timeout(seconds=int(TIMEOUT)):
                # 设置超时上下文管理器
                try:
                    res = run(code_piece=code, expr="solution()")
                    # 运行代码并评估表达式 "solution()"
                except Exception as e:
                    print(str(e), code)
                    # 捕获异常并打印错误信息和代码
                    res = None
                    # 如果发生异常，结果设为 None
                results.append(res)
                # 将结果添加到结果列表中

    if is_single_program:
        # 如果是单个程序
        assert len(results) == 1, len(results)
        # 确保结果列表中只有一个结果
        return results[0]
        # 返回结果

    return results
    # 返回结果列表


if __name__ == '__main__':
    # 如果当前模块是主模块
    code = '''
    # 定义一个多行字符串，包含一个解决方案函数
    def solution():
        """Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"""
        # Natalia在四月卖给48个朋友的夹子，然后在五月卖出一半的夹子。Natalia在四月和五月总共卖了多少夹子？
        import time
        # 导入 time 模块
        time.sleep(2)
        # 暂停 2 秒
        from sympy import init_session
        # 从 sympy 模块导入 init_session 函数
        init_session()
        # 初始化 sympy 会话
        # raise
        # raise 语句被注释掉，可能用于调试
        clips_april = 48
        # 四月卖出的夹子数量
        clips_may = clips_april / 2
        # 五月卖出的夹子数量（四月的一半）
        clips_total = clips_april + clips_may
        # 计算总共卖出的夹子数量
        result = clips_total
        # 将总数赋值给结果变量
        # import numpy as np
        # np.random.seed(42)
        # return np.random.randint(10)
        # np.random.seed(42)
        # 这些行被注释掉，可能用于生成随机数
        return result
        # 返回总共卖出的夹子数量
    '''.strip()
    # 去掉字符串前后的空白字符
    print(code)
    # 打印代码字符串
    s = time.time()
    # 记录当前时间
    for i in tqdm(range(1)):
        # 使用 tqdm 显示进度条，循环 1 次
        res = run_python_code([code]*10, 2.5, safe=True)
        # 运行代码 10 次，设置超时时间为 2.5 秒，安全模式为 True
        print(res)
        # 打印运行结果
    print(time.time()-s)
    # 打印运行代码所用的时间
    print(np.random.randint(10))
    # 打印一个随机整数（0 到 9 之间）
    sum([elem for elem in res if elem is not None])/len(res)
    # 计算结果列表中非 None 元素的平均值