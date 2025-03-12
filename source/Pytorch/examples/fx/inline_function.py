import torch  # 导入PyTorch库
from torch.fx import Proxy, symbolic_trace  # 从torch.fx导入Proxy和符号跟踪功能
from torch.fx.node import map_arg  # 从torch.fx.node导入map_arg函数


'''
How to Inline a Function Into an Existing Graph  # 如何将函数内联到现有图中

One reason you might want to inline a function is to get around FX's  # 你可能想要将一个函数内联的一个原因是为了绕过FX的
default tracing behavior. For example, unless you've defined a custom  # 默认跟踪行为。例如，除非你定义了一个自定义
Tracer, the out-of-the-box implementation of ``symbolic_trace`` causes  # Tracer，否则开箱即用的``symbolic_trace``实现会导致
references to ``torch.nn`` module instances to appear as  # 对``torch.nn``模块实例的引用会显示为
``call_module`` calls rather than being traced through. Let's say this  # ``call_module``调用，而不是被跟踪。假设这种
behavior is almost what you need; the only problem is that there's a  # 行为几乎是你所需要的；唯一的问题是有一个
single module call that you want to replace with an inlined trace of the  # 单个模块调用，你想用该函数的内联跟踪替换
function. Creating a custom Tracer would be too much. Instead, you can  # 创建一个自定义Tracer会太复杂。相反，你可以
accomplish this using Proxies.  # 使用Proxy来实现这一点。

The following code demonstrates how to trace a module and inline it  # 以下代码演示了如何跟踪模块并将其内联到
into an existing Graph using Proxy. We'll trace our Graph, then iterate  # 现有图中。我们将跟踪我们的图，然后迭代
through its Nodes until we find the right place to swap out the  # 通过其节点，直到找到合适的位置将
``call_module`` Node with an inlined trace. At that point, we'll create  # ``call_module``节点替换为内联跟踪。此时，我们将创建
Proxies from the Node's args and kwargs. Finally, we'll call the  # 从节点的参数和关键字参数创建Proxy。最后，我们将调用
function we want to replace with those Proxies--which will, in essence,  # 我们想用这些Proxy替换的函数——这将本质上
"trace" that function. Finally, we'll insert the result of that call  # “跟踪”该函数。最后，我们将插入该调用的结果
into our Graph. (This last step will automatically inline the function.)  # 到我们的图中。（这最后一步将自动将函数内联。）
'''


# Sample module  # 示例模块
class M(torch.nn.Module):  # 定义模块M，继承自torch.nn.Module
    def __init__(self):  # 初始化函数
        super().__init__()  # 调用父类构造函数
        self.relu = torch.nn.ReLU()  # 创建ReLU激活函数的实例

    def forward(self, x):  # 前向传播函数
        return self.relu(x) + 1.0  # 返回ReLU的输出加1.0


# Symbolically trace an instance of `M`. After tracing, `self.relu` is  # 对模块M的实例进行符号跟踪。跟踪后，`self.relu`被
# represented as a `call_module` Node. The full operation in the  # 表示为`call_module`节点。生成的
# generated [forward](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/Pytorch/examples/fx/custom_tracer.py:94:4-95:48) function's code will appear as `self.relu(x)`  # [forward](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/Pytorch/examples/fx/custom_tracer.py:94:4-95:48)函数代码中的完整操作将显示为`self.relu(x)`
m = symbolic_trace(M())  # 对M进行符号跟踪

# Insert nodes from the ReLU graph in place of the original call to  # 在原始对`self.relu`的调用位置插入ReLU图中的节点
# `self.relu`  # `self.relu`
# create a graph-appending tracer pointing to the original graph  # 创建一个指向原始图的图附加跟踪器
tracer = torch.fx.proxy.GraphAppendingTracer(m.graph)  # 创建图附加跟踪器实例
for node in m.graph.nodes:  # 遍历图中的每个节点
    # Find `call_module` Node in `m` that corresponds to `self.relu`.  # 找到在`m`中与`self.relu`对应的`call_module`节点。
    # This is the Node we want to swap out for an inlined version of the  # 这是我们想要替换为内联版本的节点
    # same call  # 同一调用
    if (node.op, node.target) == ("call_module", "relu"):  # 如果节点的操作和目标是`call_module`和`relu`
        with m.graph.inserting_before(node):  # 在节点之前插入
            # Create a Proxy from each Node in the current Node's  # 从当前节点的每个节点创建Proxy
            # args/kwargs  # 参数/关键字参数
            proxy_args = map_arg(node.args, lambda n: Proxy(n, tracer))  # 将节点的参数映射为Proxy
            proxy_kwargs = map_arg(node.kwargs, lambda n: Proxy(n, tracer))  # 将节点的关键字参数映射为Proxy
            # Call `m.relu` with the newly-created Proxy arguments.  # 使用新创建的Proxy参数调用`m.relu`。
            # `m.relu` is the generic version of the function; by  # `m.relu`是函数的通用版本；通过
            # calling it with Proxies created from Nodes in `m`, we're  # 使用从`m`中的节点创建的Proxy调用它，我们
            # emitting Nodes that reference exiting values in the IR.  # 发出引用IR中现有值的节点。
            # The result of this call is another Proxy, which we can  # 此调用的结果是另一个Proxy，我们可以
            # hook into our existing Graph to complete the function  # 将其连接到现有图中以完成函数
            # inlining.  # 内联。
            proxy_output = m.relu(*proxy_args, **proxy_kwargs)  # 调用ReLU并获取输出
            # Replace the relu `call_module` node with the inlined  # 用内联版本的函数替换relu `call_module`节点
            # version of the function  # 函数的版本
            node.replace_all_uses_with(proxy_output.node)  # 替换所有使用该节点的地方
            # Make sure that the old relu Node is erased  # 确保旧的relu节点被删除
            m.graph.erase_node(node)  # 从图中删除旧节点