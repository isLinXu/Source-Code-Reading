import torch  # 导入PyTorch库
import torch.fx as fx  # 导入PyTorch的FX模块并命名为fx

# An inverse mapping is one that takes a function f(x) and returns a function g  # 反向映射是一个将函数f(x)转换为函数g的映射
# such that f(g(x)) == x. For example,since log(exp(x)) == x, exp and log are  # 使得f(g(x)) == x。例如，由于log(exp(x)) == x，exp和log是
# inverses.  # 反函数。

invert_mapping = {}  # 初始化反向映射字典
def add_inverse(a, b):  # 定义添加反向映射的函数
    invert_mapping[a] = b  # 将a映射到b
    invert_mapping[b] = a  # 将b映射到a

# 定义反函数对
inverses = [  # 反函数列表
    (torch.sin, torch.arcsin),  # sin和arcsin
    (torch.cos, torch.arccos),  # cos和arccos
    (torch.tan, torch.arctan),  # tan和arctan
    (torch.exp, torch.log),  # exp和log
]

for a, b in inverses:  # 遍历反函数对
    add_inverse(a, b)  # 添加反向映射

# The general strategy is that we walk the graph backwards, transforming each  # 一般策略是我们反向遍历图，转换每个
# node into its inverse. To do so, we swap the outputs and inputs of the  # 节点为其反函数。为此，我们交换
# functions, and then we look up its inverse in `invert_mapping`. Note that  # 函数的输入和输出，然后在`invert_mapping`中查找其反向映射。请注意
# this transform assumes that all operations take in only one input and return  # 此转换假设所有操作仅接受一个输入并返回
# one output.  # 一个输出。

def invert(model: torch.nn.Module) -> torch.nn.Module:  # 定义反转函数，接受一个torch.nn.Module
    fx_model = fx.symbolic_trace(model)  # 对模型进行符号跟踪
    new_graph = fx.Graph()  # 创建一个新的图
    env = {}  # 初始化环境字典
    for node in reversed(fx_model.graph.nodes):  # 反向遍历跟踪图中的节点
        if node.op == 'call_function':  # 如果节点是函数调用
            # This creates a node in the new graph with the inverse function,  # 这将在新图中创建一个使用反向函数的节点，
            # and passes `env[node.name]` (i.e. the previous output node) as  # 并将`env[node.name]`（即前一个输出节点）作为输入。
            # input.  # 输入。
            new_node = new_graph.call_function(invert_mapping[node.target], (env[node.name],))  # 创建新节点
            env[node.args[0].name] = new_node  # 更新环境字典
        elif node.op == 'output':  # 如果节点是输出
            # We turn the output into an input placeholder  # 我们将输出转换为输入占位符
            new_node = new_graph.placeholder(node.name)  # 创建占位符节点
            env[node.args[0].name] = new_node  # 更新环境字典
        elif node.op == 'placeholder':  # 如果节点是占位符
            # We turn the input placeholder into an output  # 我们将输入占位符转换为输出
            new_graph.output(env[node.name])  # 设置新图的输出
        else:  # 如果操作不被支持
            raise RuntimeError("Not implemented")  # 引发运行时错误

    new_graph.lint()  # 检查新图的正确性
    return fx.GraphModule(fx_model, new_graph)  # 返回新的GraphModule


def f(x):  # 定义函数f
    return torch.exp(torch.tan(x))  # 返回tan(x)的指数值

res = invert(f)  # 反转函数f
print(res.code)  # 打印反转后的函数代码
"""
def forward(self, output):  # 定义前向传播函数
    log_1 = torch.log(output);  output = None  # 计算输出的对数并将输出置为None
    arctan_1 = torch.arctan(log_1);  log_1 = None  # 计算对数的反正切并将log_1置为None
    return arctan_1  # 返回反正切的结果
"""
print(f(res((torch.arange(5) + 1))))  # [1., 2., 3., 4, 5.]  # 打印反转函数的结果