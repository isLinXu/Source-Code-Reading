import torch  # 导入PyTorch库
from torch.fx import symbolic_trace, Tracer, Graph, GraphModule, Node  # 从torch.fx导入符号跟踪、Tracer、Graph、GraphModule和Node
from typing import Any, Callable, Dict, Optional, Tuple, Union  # 导入类型提示所需的模块


"""
How to Create and Use Custom Tracers  # 如何创建和使用自定义跟踪器

`Tracer`--the class that implements the symbolic tracing functionality  # `Tracer`——实现符号跟踪功能的类
of `torch.fx.symbolic_trace`--can be subclassed to override various  # 可以被子类化以重写跟踪过程的各种行为
behaviors of the tracing process. In this tutorial, we'll demonstrate  # 在本教程中，我们将演示
how to customize the symbolic tracing process using some handwritten  # 如何使用一些手写的跟踪器自定义符号跟踪过程
Tracers. Each example will show that, by simply overriding a few methods  # 每个示例将显示，通过简单地重写Tracer类中的几个方法
in the `Tracer` class, you can alter the Graph produced by symbolic  # 您可以更改符号跟踪生成的图
tracing. For a complete description of the methods that can be changed,  # 有关可以更改的方法的完整描述，
refer to the docstrings of the methods in the Tracer class. Information  # 请参阅Tracer类中方法的文档字符串。信息
can be found at: https://pytorch.org/docs/master/fx.html#torch.fx.Tracer  # 可以在此处找到：https://pytorch.org/docs/master/fx.html#torch.fx.Tracer

If you want a real-world example of a custom tracer, check out FX's AST  # 如果您想要自定义跟踪器的实际示例，请查看FX的AST
Rewriter in `rewriter.py`. `RewritingTracer` inherits from Tracer but  # Rewriter在`rewriter.py`中。`RewritingTracer`继承自Tracer，但
overrides the `trace` function so that we can rewrite all calls to  # 重写`trace`函数，以便我们可以重写所有对
`assert` to the more FX-friendly `torch.assert`.  # `assert`的调用为更适合FX的`torch.assert`。

Note that a call to `symbolic_trace(m)` is equivalent to  # 请注意，对`symbolic_trace(m)`的调用等价于
`GraphModule(m, Tracer().trace(m))`. (`Tracer` is the default  # `GraphModule(m, Tracer().trace(m))`。（`Tracer`是默认的
implementation of Tracer as defined in `symbolic_trace.py`.)  # Tracer实现，如`symbolic_trace.py`中定义的那样。）
"""


"""
Custom Tracer #1: Trace Through All `torch.nn.ReLU` Submodules  # 自定义跟踪器#1：跟踪所有`torch.nn.ReLU`子模块

During symbolic tracing, some submodules are traced through and their  # 在符号跟踪期间，某些子模块被跟踪并记录其
constituent ops are recorded; other submodules appear as an  # 组成操作；其他子模块作为原子“call_module”节点出现在
atomic "call_module" Node in the IR. A module in this latter category  # IR中。在后一类中，模块称为“叶模块”。
is called a "leaf module". By default, all modules in the PyTorch  # 默认情况下，PyTorch标准库（`torch.nn`）中的所有模块都是叶模块。
standard library (`torch.nn`) are leaf modules. We can change this  # 我们可以通过创建自定义Tracer并重写`is_leaf_module`来更改此行为。
by creating a custom Tracer and overriding `is_leaf_module`. In this  # 在这种情况下，我们将保留对所有`torch.nn`模块的默认行为，除了`ReLU`。
case, we'll keep the default behavior for all `torch.nn` Modules except 
for `ReLU`.
"""

class M1(torch.nn.Module):  # 定义模块M1，继承自torch.nn.Module
    def __init__(self):  # 初始化函数
        super().__init__()  # 调用父类构造函数
        self.relu = torch.nn.ReLU()  # 创建ReLU激活函数的实例

    def forward(self, x):  # 前向传播函数
        return self.relu(x)  # 返回ReLU的输出


default_traced: GraphModule = symbolic_trace(M1())  # 使用默认跟踪器对M1进行符号跟踪
"""
Tracing with the default tracer and calling `print_tabular` produces:  # 使用默认跟踪器进行跟踪并调用`print_tabular`生成：
    
    opcode       name    target    args       kwargs  # 操作码       名称    目标    参数       关键字参数
    -----------  ------  --------  ---------  --------  # -----------  ------  --------  ---------  --------
    placeholder  x       x         ()         {}  # 占位符  x       x         ()         {}
    call_module  relu_1  relu      (x,)       {}  # 调用模块  relu_1  relu      (x,)       {}
    output       output  output    (relu_1,)  {}  # 输出       输出  output    (relu_1,)  {}

"""
default_traced.graph.print_tabular()  # 打印跟踪图的表格形式

class LowerReluTracer(Tracer):  # 定义自定义跟踪器LowerReluTracer，继承自Tracer
    def is_leaf_module(self, m: torch.nn.Module, qualname: str):  # 重写is_leaf_module方法
        if isinstance(m, torch.nn.ReLU):  # 如果模块是ReLU
            return False  # 返回False，表示不作为叶模块
        return super().is_leaf_module(m, qualname)  # 否则调用父类方法


"""
Tracing with our custom tracer and calling `print_tabular` produces:  # 使用我们的自定义跟踪器进行跟踪并调用`print_tabular`生成：
    
    opcode         name    target                             args       kwargs  # 操作码         名称    目标                             参数       关键字参数
    -------------  ------  ---------------------------------  ---------  ------------------  # -------------  ------  ---------------------------------  ---------  ------------------
    placeholder    x       x                                  ()         {}  # 占位符    x       x                                  ()         {}
    call_function  relu_1  <function relu at 0x7f66f7170b80>  (x,)       {'inplace': False}  # 调用函数  relu_1  <function relu at 0x7f66f7170b80>  (x,)       {'inplace': False}
    output         output  output                             (relu_1,)  {}  # 输出         输出  output                             (relu_1,)  {}

"""
lower_relu_tracer = LowerReluTracer()  # 创建LowerReluTracer实例
custom_traced_graph: Graph = lower_relu_tracer.trace(M1())  # 使用自定义跟踪器对M1进行跟踪
custom_traced_graph.print_tabular()  # 打印自定义跟踪图的表格形式


"""
Custom Tracer #2: Add an Extra Attribute to Each Node  # 自定义跟踪器#2：为每个节点添加额外属性

Here, we'll override `create_node` so that we can add a new attribute to  # 在这里，我们将重写`create_node`，以便在创建每个节点时添加新属性
each Node during its creation  # 在其创建期间
"""

class M2(torch.nn.Module):  # 定义模块M2，继承自torch.nn.Module
    def forward(self, a, b):  # 前向传播函数
        return a + b  # 返回两个输入的和

class TaggingTracer(Tracer):  # 定义自定义跟踪器TaggingTracer，继承自Tracer
    def create_node(self, kind: str, target: Union[str, Callable],  # 重写create_node方法
                    args: Tuple[Any], kwargs: Dict[str, Any], name: Optional[str] = None,
                    type_expr: Optional[Any] = None) -> Node:  # 定义create_node的参数
        n = super().create_node(kind, target, args, kwargs, name)  # 调用父类的create_node方法
        n.tag = "foo"  # 为节点添加新属性tag
        return n  # 返回节点

custom_traced_graph: Graph = TaggingTracer().trace(M2())  # 使用TaggingTracer对M2进行跟踪

def assert_all_nodes_have_tags(g: Graph) -> bool:  # 定义函数以检查所有节点是否具有标签
    for n in g.nodes:  # 遍历图中的所有节点
        if not hasattr(n, "tag") or not n.tag == "foo":  # 检查节点是否具有tag属性且值是否为"foo"
            return False  # 如果没有，则返回False
    return True  # 如果所有节点都有标签，则返回True

# Prints "True"  # 打印"True"
print(assert_all_nodes_have_tags(custom_traced_graph))  # 输出检查结果