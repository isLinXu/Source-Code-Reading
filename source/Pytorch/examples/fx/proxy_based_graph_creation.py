import torch  # 导入PyTorch库
from torch.fx import Proxy, Graph, GraphModule  # 从torch.fx导入Proxy、Graph和GraphModule


'''
How to Create a Graph Using Proxy Objects Instead of Tracing  # 如何使用Proxy对象而不是跟踪创建图

It's possible to directly create a Proxy object around a raw Node. This  # 可以直接在原始节点周围创建Proxy对象。这
can be used to create a Graph independently of symbolic tracing.  # 可用于独立于符号跟踪创建图。

The following code demonstrates how to use Proxy with a raw Node to  # 以下代码演示了如何使用Proxy和原始节点
append operations to a fresh Graph. We'll create two parameters (``x``  # 将操作附加到新图。我们将创建两个参数（``x``
and ``y``), perform some operations on those parameters, then add  # 和``y``），对这些参数执行一些操作，然后将
everything we created to the new Graph. We'll then wrap that Graph in  # 我们创建的所有内容添加到新图。然后，我们将
a GraphModule. Doing so creates a runnable instance of ``nn.Module``  # 该图包装在GraphModule中。这样创建了一个可运行的
where previously-created operations are represented in the Module's  # ``nn.Module``实例，其中先前创建的操作在模块的
`[forward](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/Pytorch/examples/fx/primitive_library.py:43:4-46:58)` function.  # `[forward](cci:1://file:///Users/gatilin/MyWork/Source-Code-Reading1/source/Pytorch/examples/fx/primitive_library.py:43:4-46:58)`函数中表示。

By the end of the tutorial, we'll have added the following method to an  # 到本教程结束时，我们将向一个空的
empty ``nn.Module`` class.  # ``nn.Module``类添加以下方法。

.. code-block:: python  # .. 代码块:: python

    def forward(self, x, y):  # 前向传播方法
        cat_1 = torch.cat([x, y]);  x = y = None  # 将x和y连接并置为None
        tanh_1 = torch.tanh(cat_1);  cat_1 = None  # 计算cat_1的tanh并置为None
        neg_1 = torch.neg(tanh_1);  tanh_1 = None  # 计算tanh_1的负值并置为None
        return neg_1  # 返回负值
'''


# Create a graph independently of symbolic tracing  # 独立于符号跟踪创建图
graph = Graph()  # 创建一个新的Graph实例
tracer = torch.fx.proxy.GraphAppendingTracer(graph)  # 创建一个GraphAppendingTracer实例，用于向图中添加节点

# Create raw Nodes  # 创建原始节点
raw1 = graph.placeholder('x')  # 创建一个名为'x'的占位符节点
raw2 = graph.placeholder('y')  # 创建一个名为'y'的占位符节点

# Initialize Proxies using the raw Nodes and graph's default tracer  # 使用原始节点和图的默认跟踪器初始化Proxy
y = Proxy(raw1, tracer)  # 创建y的Proxy，使用raw1和tracer
z = Proxy(raw2, tracer)  # 创建z的Proxy，使用raw2和tracer
# y = Proxy(raw1)  # 也可以直接使用raw1创建Proxy
# z = Proxy(raw2)  # 也可以直接使用raw2创建Proxy

# Create other operations using the Proxies `y` and `z`  # 使用Proxy `y`和`z`创建其他操作
a = torch.cat([y, z])  # 将y和z连接
b = torch.tanh(a)  # 计算a的tanh值
c = torch.neg(b)  # 计算b的负值
# By using the graph's own appending tracer to create Proxies,  # 通过使用图自己的添加跟踪器创建Proxy，
# notice we can now use n-ary operators on operations without  # 请注意，我们现在可以在操作上使用n元运算符，而无需
# multiple tracers being created at run-time (line 52) which leads  # 在运行时创建多个跟踪器（第52行），这会导致
# to errors # To try this out for yourself, replace lines 42, 43  # 错误#要自己尝试，请将第42、43行替换为
# with 44, 45  # 与第44、45行
z = torch.add(b, c)  # 计算b和c的和

# Create a new output Node and add it to the Graph. By doing this, the  # 创建一个新的输出节点并将其添加到图中。通过这样做，
# Graph will contain all the Nodes we just created (since they're all  # 图将包含我们刚刚创建的所有节点（因为它们都
# linked to the output Node)  # 与输出节点相连）
graph.output(c.node)  # 将输出节点添加到图中

# Wrap our created Graph in a GraphModule to get a final, runnable  # 将我们创建的图包装在GraphModule中，以获得最终的可运行
# `nn.Module` instance  # `nn.Module`实例
mod = GraphModule(torch.nn.Module(), graph)  # 创建GraphModule实例