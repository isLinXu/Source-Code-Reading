import torch  # 导入PyTorch库
from torch.fx import symbolic_trace  # 从torch.fx导入symbolic_trace
import operator  # 导入操作符模块

"""
How to Replace One Op With Another  # 如何用另一个操作替换一个操作

1. Iterate through all Nodes in your GraphModule's Graph.  # 1. 遍历GraphModule的图中的所有节点。
2. Determine if the current Node should be replaced. (Suggested: match  # 2. 确定当前节点是否应该被替换。（建议：匹配
on the Node's ``target`` attribute).  # 在节点的``target``属性上进行匹配）。
3. Create a replacement Node and add it to the Graph.  # 3. 创建一个替换节点并将其添加到图中。
4. Use the FX built-in ``replace_all_uses_with`` to replace all uses of  # 4. 使用FX内置的``replace_all_uses_with``替换当前节点的所有使用。
the current Node with the replacement.  # 将当前节点替换为替换节点。
5. Delete the old Node from the graph.  # 5. 从图中删除旧节点。
6. Call ``recompile`` on the GraphModule. This updates the generated  # 6. 在GraphModule上调用``recompile``。这将更新生成的
Python code to reflect the new Graph state.  # Python代码以反映新的图状态。

Currently, FX does not provide any way to guarantee that replaced  # 目前，FX没有提供任何方法来保证替换的
operators are syntactically valid. It's up to the user to confirm that  # 操作符在语法上是有效的。用户需要确认
any new operators will work with the existing operands.  # 任何新操作符都能与现有操作数配合使用。

The following code demonstrates an example of replacing any instance of  # 以下代码演示了如何替换任何加法实例
addition with a bitwise AND.  # 用按位与替换加法。

To examine how the Graph evolves during op replacement, add the  # 要检查操作替换期间图的演变，请添加
statement `print(traced.graph)` after the line you want to inspect.  # 语句`print(traced.graph)`在您想检查的行之后。
Alternatively, call `traced.graph.print_tabular()` to see the IR in a  # 或者，调用`traced.graph.print_tabular()`以表格格式查看IR。
tabular format.  # 
"""

# Sample module  # 示例模块
class M(torch.nn.Module):  # 定义一个名为M的类，继承自torch.nn.Module
    def forward(self, x, y):  # 定义前向传播方法，接受参数x和y
        return x + y, torch.add(x, y), x.add(y)  # 返回x和y的加法结果

# Symbolically trace an instance of the module  # 对模块的一个实例进行符号跟踪
traced = symbolic_trace(M())  # 对M类进行符号跟踪

# As demonstrated in the above example, there are several different ways  # 如上例所示，有几种不同的方式
# to denote addition. The possible cases are:  # 来表示加法。可能的情况有：
#     1. `x + y` - A `call_function` Node with target `operator.add`.  # 1. `x + y` - 一个目标为`operator.add`的`call_function`节点。
#         We can match for equality on that `operator.add` directly.  # 我们可以直接匹配该`operator.add`。
#     2. `torch.add(x, y)` - A `call_function` Node with target  # 2. `torch.add(x, y)` - 一个目标为
#         `torch.add`. Similarly, we can match this function directly.  # `torch.add`的`call_function`节点。同样，我们可以直接匹配此函数。
#     3. `x.add(y)` - The Tensor method call, whose target we can match  # 3. `x.add(y)` - 张量方法调用，我们可以匹配
#         as a string.  # 作为字符串的目标。

patterns = set([operator.add, torch.add, "add"])  # 定义一个包含加法操作的模式集合

# Go through all the nodes in the Graph  # 遍历图中的所有节点
for n in traced.graph.nodes:  # 对每个节点n进行遍历
    # If the target matches one of the patterns  # 如果目标匹配模式之一
    if any(n.target == pattern for pattern in patterns):  # 检查节点的目标是否在模式集合中
        # Set the insert point, add the new node, and replace all uses  # 设置插入点，添加新节点，并替换所有使用
        # of `n` with the new node  # 将`n`的所有使用替换为新节点
        with traced.graph.inserting_after(n):  # 在节点n之后插入
            new_node = traced.graph.call_function(torch.bitwise_and, n.args, n.kwargs)  # 创建新的按位与操作节点
            n.replace_all_uses_with(new_node)  # 替换所有使用n的地方
        # Remove the old node from the graph  # 从图中删除旧节点
        traced.graph.erase_node(n)  # 删除节点n

# Don't forget to recompile!  # 不要忘记重新编译！
traced.recompile()  # 重新编译图