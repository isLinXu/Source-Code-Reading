import torch  # 导入PyTorch库
from torch.fx import symbolic_trace, replace_pattern  # 从torch.fx导入symbolic_trace和replace_pattern


'''
How to Use the FX Subgraph Rewriter  # 如何使用FX子图重写器

For easy subgraph rewriting, FX exposes the utility function:  # 为了方便子图重写，FX提供了实用函数：

    replace_pattern(gm : GraphModule,  # replace_pattern(gm : GraphModule,
                    pattern : Callable,  # pattern : Callable,
                    replacement : Callable)  # replacement : Callable)
                    -> None  # -> None

`replace_pattern` matches all possible non-overlapping sets of operators  # `replace_pattern`匹配图中所有可能的不重叠操作集合
and their data dependencies (`pattern`) in the Graph of a GraphModule  # 及其数据依赖关系（`pattern`），
(`gm`), then replaces each of these matched subgraphs with another  # 然后用另一个子图替换每个匹配的子图
subgraph (`replacement).  # （`replacement）。

The docstring for `replace_pattern` (located in `subgraph_rewriter.py`)  # `replace_pattern`的文档字符串（位于`subgraph_rewriter.py`中）
gives an in-depth explanation as to how `pattern` and `replacement`  # 对如何指定`pattern`和`replacement`进行了深入解释，
should be specified, what happens during pattern matching, and other  # 以及模式匹配期间发生的事情和其他
important technical details. This tutorial, therefore, is only meant to  # 重要的技术细节。因此，本教程仅旨在
give an overview as to the FX Subgraph Rewriter's basic functionality.  # 概述FX子图重写器的基本功能。
Let's go rewrite a Graph!  # 让我们来重写一个图吧！
'''

# Sample module  # 示例模块
class M(torch.nn.Module):  # 定义一个名为M的类，继承自torch.nn.Module
    def __init__(self):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法

    def forward(self, x, w1, w2):  # 定义前向传播方法，接受参数x、w1和w2
        val1 = torch.neg(w1)  # 计算w1的负值
        m1 = torch.cat([val1, w2]).sum()  # 将val1和w2连接并求和
        val2 = torch.neg(w1)  # 再次计算w1的负值
        m2 = torch.cat([val2, w2]).sum()  # 将val2和w2连接并求和
        return x + torch.max(m1) + torch.max(m2)  # 返回x与m1和m2的最大值之和

# Symbolically trace an instance of `M`  # 对模块`M`的一个实例进行符号跟踪
traced = symbolic_trace(M())  # 对M类进行符号跟踪

# Define the pattern. The FX Subgraph Rewriter will match all  # 定义模式。FX子图重写器将匹配所有
# non-overlapping instances of the pattern in the larger graph.  # 在更大图中不重叠的模式实例。
# Note that Pattern-matching is done based on data dependencies,  # 注意，模式匹配是基于数据依赖关系进行的，
# not Node names. Even though we're operating on Nodes named `a1` and  # 而不是节点名称。即使我们操作的是名为`a1`和
# `a2` instead of `w1` and `w2`, the pattern is still a valid match  # `a2`的节点，而不是`w1`和`w2`，该模式仍然是有效匹配
# for the two instances of `torch.cat([w1, w2]).sum()` above. Only  # 上述两个`torch.cat([w1, w2]).sum()`实例。只有
# operations that contribute to the single output value of the pattern  # 贡献于模式单个输出值的操作
# are considered  # 被视为有效
def pattern(a1, a2):  # 定义匹配模式的函数，接受参数a1和a2
    val1 = torch.neg(a1)  # 计算a1的负值
    return torch.cat([val1, a2]).sum()  # 返回val1和a2连接后的和

# Define the replacement (same rules as the pattern)  # 定义替换（与模式相同的规则）
def replacement(w1, w2):  # 定义替换函数，接受参数w1和w2
    return torch.stack([w1, w2])  # 返回w1和w2的堆叠结果

# Replace `pattern` with `replacement` in `traced`  # 在`traced`中将`pattern`替换为`replacement`
replace_pattern(traced, pattern, replacement)  # 调用replace_pattern进行替换

# After calling `replace_pattern`, the generated code is:  # 调用`replace_pattern`后，生成的代码是：
'''
def forward(self, x, w1, w2):  # 前向传播方法
    stack = torch.stack([w1, w2])  # 堆叠w1和w2
    max_1 = torch.max(stack);  stack = None  # 计算stack的最大值并置为None
    add = x + max_1;  x = max_1 = None  # 返回x与max_1的和并置为None
    stack_1 = torch.stack([w1, w2]);  w1 = w2 = None  # 再次堆叠w1和w2并置为None
    max_2 = torch.max(stack_1);  stack_1 = None  # 计算stack_1的最大值并置为None
    add_1 = add + max_2;  add = max_2 = None  # 返回add与max_2的和并置为None
    return add_1  # 返回最终结果add_1
'''