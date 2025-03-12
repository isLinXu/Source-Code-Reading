from enum import Enum, auto  # 从enum模块导入Enum和auto

import torch  # 导入PyTorch库
from torch.fx import GraphModule, Node, Proxy, symbolic_trace  # 从torch.fx导入GraphModule、Node、Proxy和symbolic_trace

'''
Wrap Graph Output Dynamically  # 动态包装图输出

The following code demonstrates how change an existing Graph based on  # 以下代码演示了如何根据
parameters specified at runtime. We'll let the user specify an  # 在运行时指定的参数更改现有图。我们将让用户指定
activation function from a predefined Enum list, then we'll symbolically  # 从预定义的Enum列表中选择激活函数，然后我们将进行符号
trace it. Next, we'll create a Proxy from the last operation in the  # 跟踪。接下来，我们将从图中的最后一个操作创建一个Proxy。
Graph. We'll call our traced activation function with this Proxy and  # 我们将使用这个Proxy调用我们跟踪的激活函数，并
insert the ``output`` Node from that call into our Graph. (This final  # 将该调用的``output``节点插入到我们的图中。（最后这一步
step will automatically inline the entire traced function.)  # 将自动内联整个跟踪函数。）
'''

# Sample module  # 示例模块
class M(torch.nn.Module):  # 定义一个名为M的类，继承自torch.nn.Module
    def __init__(self):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法

    def forward(self, x, y):  # 定义前向传播方法，接受参数x和y
        y = torch.cat([x, y])  # 将x和y连接
        return y  # 返回连接后的结果

# Symbolically trace an instance of `M`  # 对模块`M`的一个实例进行符号跟踪
traced = symbolic_trace(M())  # 对M类进行符号跟踪

# Selected activation functions  # 选择的激活函数
class ActivationFunction(Enum):  # 定义激活函数的枚举类
    RELU = auto()  # ReLU激活函数
    LEAKY_RELU = auto()  # Leaky ReLU激活函数
    PRELU = auto()  # PReLU激活函数

# Map activation function names to their implementation  # 将激活函数名称映射到其实现
activation_functions = {  # 创建一个字典以映射激活函数
    ActivationFunction.RELU: torch.nn.ReLU(),  # ReLU函数的实现
    ActivationFunction.LEAKY_RELU: torch.nn.LeakyReLU(),  # Leaky ReLU函数的实现
    ActivationFunction.PRELU: torch.nn.PReLU(),  # PReLU函数的实现
}

def wrap_in_activation_function(m: GraphModule, fn: ActivationFunction) -> GraphModule:  # 定义包装函数，接受GraphModule和激活函数
    # Get output node  # 获取输出节点
    output_node: Optional[Node] = None  # 初始化输出节点为None
    for n in reversed(m.graph.nodes):  # 反向遍历图中的节点
        if n.op == "output":  # 如果节点操作是输出
            output_node = n  # 设置输出节点
            break  # 退出循环
    assert output_node  # 确保找到了输出节点

    # Get the actual output (the "input" of the output node). This is  # 获取实际输出（输出节点的“输入”）。这是
    # the Node we want to wrap in a user-specified activation function  # 我们要用用户指定的激活函数包装的节点
    assert len(output_node.all_input_nodes) == 1  # 确保输出节点只有一个输入节点
    wrap_node = output_node.all_input_nodes[0]  # 获取输入节点

    # Wrap the actual output in a Proxy  # 将实际输出包装在Proxy中
    wrap_proxy = Proxy(wrap_node)  # 创建wrap_node的Proxy

    # Get the implementation of the specified activation function and  # 获取指定激活函数的实现并
    # symbolically trace it  # 对其进行符号跟踪
    fn_impl = activation_functions[fn]  # 获取激活函数的实现
    fn_impl_traced = symbolic_trace(fn_impl)  # 对激活函数实现进行符号跟踪

    # Call the specified activation function using the Proxy wrapper for  # 使用Proxy包装器调用指定的激活函数
    # `output_op`. The result of this call is another Proxy, which we  # `output_op`。此调用的结果是另一个Proxy，我们
    # can hook into our existing Graph.  # 可以将其连接到现有图中。
    with traced.graph.inserting_after(wrap_node):  # 在wrap_node之后插入
        fn_impl_output_node = fn_impl_traced(wrap_proxy)  # 调用激活函数并传入Proxy
        new_args = (fn_impl_output_node.node,)  # 获取新参数
        output_node.args = new_args  # 更新输出节点的参数

    m.recompile()  # 重新编译图


# Example call  # 示例调用
x, y = torch.randn(5, 3), torch.randn(5, 3)  # 生成随机输入
orig_output = traced(x, y)  # 获取原始输出

wrap_in_activation_function(traced, ActivationFunction.LEAKY_RELU)  # 用Leaky ReLU包装traced
new_output = traced(x, y)  # 获取新的输出

torch.testing.assert_close(new_output, torch.nn.LeakyReLU()(orig_output))  # 验证新输出与原始输出的一致性