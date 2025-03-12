import torch  # 导入PyTorch库
import torch.fx  # 导入PyTorch的FX模块
"""
In this example we are going do define a library of  # 在这个例子中，我们将定义一个库
"composite" operations. Composite operations are those  # “复合”操作。复合操作是指
that are defined as callable functions that are composed  # 被定义为可调用函数，由多个其他操作组成
of several other operations in their implementation.  # 在其实现中。

Composite operations allow you to choose at what level  # 复合操作允许您选择在何种抽象级别
of abstraction you want to interpret/manipulate the  # 解释/操作代码。
code. We show that we can provide a function to inline  # 我们展示了可以提供一个函数来内联
these functions as well as use a custom Tracer to auto-  # 这些函数，以及使用自定义跟踪器自动
matically inline such functions.  # 内联这些函数。

Composite operations can be useful for exposing higher-  # 复合操作可以用于向后端/转换暴露更高层次的上下文
level context to a backend/transform while still  # 同时仍然保持在更细粒度级别检查事物的能力。
maintaining the ability to examine things at a more  # 
fine-grained level.  # 
"""

def sigmoid_lowp(x: torch.Tensor):  # 定义sigmoid_lowp函数，接受一个torch.Tensor类型的参数x
    x = x.float()  # 将x转换为float类型
    x = x.sigmoid()  # 计算x的sigmoid值
    return x.half()  # 返回sigmoid值的半精度表示

# wrap() indicates that the passed-in function should always  # wrap()表示传入的函数应始终
# be recorded as a call_function node rather than being traced  # 作为call_function节点记录，而不是被跟踪
# through. Later, we will see how we can:  # 稍后，我们将看到如何：
# a. Inline the implementation of such a function and  # a. 内联此类函数的实现
# b. Define a tracer that automatically traces through such a function  # b. 定义一个跟踪器，自动跟踪此类函数
torch.fx.wrap(sigmoid_lowp)  # 使用wrap函数包装sigmoid_lowp

def add_lowp(a: torch.Tensor, b: torch.Tensor):  # 定义add_lowp函数，接受两个torch.Tensor类型的参数a和b
    a, b = a.float(), b.float()  # 将a和b转换为float类型
    c = a + b  # 计算a和b的和
    return c.half()  # 返回和的半精度表示

torch.fx.wrap(add_lowp)  # 使用wrap函数包装add_lowp

# Let's see what happens when we symbolically trace through some code  # 让我们看看当我们符号跟踪一些代码时会发生什么
# that uses these functions  # 使用这些函数的代码

class Foo(torch.nn.Module):  # 定义一个名为Foo的类，继承自torch.nn.Module
    def forward(self, x, y):  # 定义前向传播方法，接受两个参数x和y
        x = sigmoid_lowp(x)  # 计算x的sigmoid_lowp值
        y = sigmoid_lowp(y)  # 计算y的sigmoid_lowp值
        return add_lowp(x, y)  # 返回x和y的add_lowp值

traced = torch.fx.symbolic_trace(Foo())  # 对Foo类进行符号跟踪
print(traced.code)  # 打印跟踪后的代码
"""
def forward(self, x, y):  # 前向传播方法
    sigmoid_lowp = __main___sigmoid_lowp(x);  x = None  # 调用sigmoid_lowp并将x置为None
    sigmoid_lowp_1 = __main___sigmoid_lowp(y);  y = None  # 调用sigmoid_lowp并将y置为None
    add_lowp = __main___add_lowp(sigmoid_lowp, sigmoid_lowp_1);  sigmoid_lowp = sigmoid_lowp_1 = None  # 调用add_lowp并将sigmoid_lowp置为None
    return add_lowp  # 返回add_lowp的结果
"""

# Notice that the calls to `sigmoid_lowp` and `add_lowp`  # 请注意，对`sigmoid_lowp`和`add_lowp`的调用
# appear literally in the trace; they are not traced through  # 直接出现在跟踪中；它们没有被跟踪

# ***** Inlining calls *****  # ***** 内联调用 *****
# Now let's define a function that allows for inlining these calls  # 现在让我们定义一个函数，允许内联这些调用
# during graph manipulation.  # 在图形操作期间。

def inline_lowp_func(n: torch.fx.Node):  # 定义inline_lowp_func函数，接受一个torch.fx.Node类型的参数n
    # If we find a call to a function in our "lowp" module, inline it  # 如果我们找到对“lowp”模块中函数的调用，则内联它
    if n.op == 'call_function' and n.target.__module__ == inline_lowp_func.__module__:  # 检查节点是否为call_function类型且目标模块与当前模块相同
        # We want to insert the operations comprising the implementation of the  # 我们希望在函数本身之前插入组成实现的操作
        # function before the function itself. Then, we can swap the output value  # 然后，我们可以交换函数调用的输出值
        # of the function call with the output value for its implementation nodes  # 与其实现节点的输出值
        tracer = torch.fx.proxy.GraphAppendingTracer(n.graph)  # 创建一个GraphAppendingTracer实例
        with n.graph.inserting_before(n):  # 在节点n之前插入
            # We can inline code by using `fx.Proxy` instances.  # 我们可以通过使用`fx.Proxy`实例内联代码。
            # map_arg traverses all aggregate types and applies the given function  # map_arg遍历所有聚合类型并将给定函数应用于
            # to Node instances in the data structure. In this case, we are applying  # 数据结构中的节点实例。在这种情况下，我们应用
            # the fx.Proxy constructor.  # fx.Proxy构造函数。
            proxy_args = torch.fx.node.map_arg(n.args, lambda x: torch.fx.Proxy(x, tracer))  # 将节点参数转换为Proxy
            proxy_kwargs = torch.fx.node.map_arg(n.kwargs, lambda x: torch.fx.Proxy(x, tracer))  # 将节点关键字参数转换为Proxy
            # Call the function itself with proxy arguments. This will emit  # 使用代理参数调用函数本身。这将发出
            # nodes in the graph corresponding to the operations in the im-  # 图中对应于实现中的操作的节点
            # plementation of the function  # 
            output_proxy = n.target(*proxy_args, **proxy_kwargs)  # 调用目标函数并获取输出代理
            # Now replace the original node's uses with the output node of  # 现在用实现的输出节点替换原始节点的使用
            # the implementation.  # 
            node.replace_all_uses_with(output_proxy.node)  # 替换所有使用
            # Delete the old node  # 删除旧节点
            node.graph.erase_node(node)  # 从图中删除节点

for node in traced.graph.nodes:  # 遍历跟踪图中的每个节点
    if node.op == 'call_function' and node.target is sigmoid_lowp:  # 检查节点是否为sigmoid_lowp的调用
        inline_lowp_func(node)  # 调用内联函数

# Don't forget to recompile after graph manipulation  # 在图形操作后不要忘记重新编译
traced.recompile()  # 重新编译图

print(traced.code)  # 打印重新编译后的代码
"""
def forward(self, x, y):  # 前向传播方法
    float_1 = x.float();  x = None  # 将x转换为float并置为None
    sigmoid = float_1.sigmoid();  float_1 = None  # 计算sigmoid并将float_1置为None
    half = sigmoid.half();  sigmoid = None  # 计算半精度并将sigmoid置为None
    float_2 = y.float();  y = None  # 将y转换为float并置为None
    sigmoid_1 = float_2.sigmoid();  float_2 = None  # 计算sigmoid并将float_2置为None
    half_1 = sigmoid_1.half();  sigmoid_1 = None  # 计算半精度并将sigmoid_1置为None
    add_lowp = __main___add_lowp(half, half_1);  half = half_1 = None  # 调用add_lowp并将half和half_1置为None
    return add_lowp  # 返回add_lowp的结果
"""

# At this point, the implementation of `sigmoid_lowp` has been substituted  # 此时，`sigmoid_lowp`的实现已被替换
# in for all of the calls to that function.  # 为对该函数的所有调用。

# ***** Inlining calls during tracing *****  # ***** 在跟踪期间内联调用 *****
# Now we are going to define a custom tracer that can selectively inline  # 现在我们将定义一个自定义跟踪器，可以选择性地内联
# calls to certain composite operations on-the-fly.  # 对某些复合操作的调用。

# New instance of our module  # 新的模块实例
f = Foo()  # 创建Foo类的实例

class InliningTracer(torch.fx.Tracer):  # 定义InliningTracer类，继承自torch.fx.Tracer
    FNS_TO_INLINE = [add_lowp]  # 定义要内联的函数列表

    def create_node(self, kind, target, args, kwargs, name=None, type_expr=None):  # 重写create_node方法
        if kind == 'call_function' and target in self.FNS_TO_INLINE:  # 检查节点类型和目标
            tracer = torch.fx.proxy.GraphAppendingTracer(self.graph)  # 创建GraphAppendingTracer实例
            # Trace through the implementation of the function rather than  # 跟踪函数的实现而不是
            # create a node  # 创建一个节点
            proxy_args = torch.fx.node.map_arg(args, lambda x: torch.fx.Proxy(x, tracer))  # 将参数转换为Proxy
            proxy_kwargs = torch.fx.node.map_arg(kwargs, lambda x: torch.fx.Proxy(x, tracer))  # 将关键字参数转换为Proxy
            return target(*proxy_args, **proxy_kwargs).node  # 返回目标函数的节点
        else:
            return super().create_node(kind, target, args, kwargs, name, type_expr)  # 调用父类的create_node方法


tracer = InliningTracer()  # 创建InliningTracer实例
graph = tracer.trace(f)  # 对模块f进行跟踪
module = torch.fx.GraphModule(f, graph)  # 创建GraphModule实例
print(module.code)  # 打印模块代码
"""
def forward(self, x, y):  # 前向传播方法
    sigmoid_lowp = __main___sigmoid_lowp(x);  x = None  # 调用sigmoid_lowp并将x置为None
    sigmoid_lowp_1 = __main___sigmoid_lowp(y);  y = None  # 调用sigmoid_lowp并将y置为None
    float_1 = sigmoid_lowp.float();  sigmoid_lowp = None  # 将sigmoid_lowp转换为float并置为None
    float_2 = sigmoid_lowp_1.float();  sigmoid_lowp_1 = None  # 将sigmoid_lowp_1转换为float并置为None
    add = float_1 + float_2;  float_1 = float_2 = None  # 计算float_1和float_2的和并置为None
    half = add.half();  add = None  # 计算和的半精度并将add置为None
    return half  # 返回半精度结果
"""

# As you can see, the implementation for `add_lowp` has been  # 如您所见，`add_lowp`的实现已被
# inlined in the course of tracing with our InliningTracer.  # 在使用我们的InliningTracer进行跟踪时内联。
# Such functionality can be used to, for example, implement  # 这种功能可以用于，例如，实现
# a backend that wants to see the lowered form of some operations  # 一个希望查看某些操作的降级形式的后端
# but the high-level form of another.  # 但另一种形式则为高级形式。

# ***** Future direction *****  # ***** 未来方向 *****
#
# We may define an API, such as `Tracer.is_leaf_function`, that  # 我们可以定义一个API，例如`Tracer.is_leaf_function`，
# Tracer implementers can use to more easily specify the inlining  # 供跟踪器实现者更轻松地指定内联行为
# behavior implemented in InliningTracer. Such a method would return  # 在InliningTracer中实现。这样的方法将返回
# True by default, but a Tracer can override it and return `False` for  # 默认为True，但跟踪器可以重写它并返回`False`
# functions the Tracer wants to be traced through.  # 对于希望被跟踪的函数。