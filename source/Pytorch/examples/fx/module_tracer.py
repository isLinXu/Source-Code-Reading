"""
Recording Module Hierarchy With a Custom Tracer  # 使用自定义跟踪器记录模块层次结构

In this example, we are going to define a custom `fx.Tracer` instance that--  # 在这个例子中，我们将定义一个自定义的`fx.Tracer`实例，
for each recorded operation--also notes down the qualified name of the module  # 对于每个记录的操作，还记录该操作来源的模块的合格名称。
from which that operation originated. The _qualified name_ is the path to the  # 该_合格名称_是从根模块到模块的路径。
Module from the root module. More information about this concept can be  # 有关此概念的更多信息，请参阅`Module.get_submodule`的文档：
found in the documentation for `Module.get_submodule`:  # https://github.com/pytorch/pytorch/blob/9f2aea7b88f69fc74ad90b1418663802f80c1863/torch/nn/modules/module.py#L385
https://github.com/pytorch/pytorch/blob/9f2aea7b88f69fc74ad90b1418663802f80c1863/torch/nn/modules/module.py#L385  # 
"""
import torch  # 导入PyTorch库
import torch.fx  # 导入PyTorch的FX模块
from typing import Any, Callable, Dict, Optional, Tuple  # 导入类型提示所需的模块

class ModulePathTracer(torch.fx.Tracer):  # 定义ModulePathTracer类，继承自torch.fx.Tracer
    """
    ModulePathTracer is an FX tracer that--for each operation--also records  # ModulePathTracer是一个FX跟踪器，对于每个操作，还记录
    the qualified name of the Module from which the operation originated.  # 该操作来源的模块的合格名称。
    """

    # The current qualified name of the Module being traced. The top-level  # 当前被跟踪模块的合格名称。顶级模块用空字符串表示。
    # module is signified by empty string. This is updated when entering  # 进入call_module时更新，退出时恢复
    # call_module and restored when exiting call_module  # 
    current_module_qualified_name: str = ''  
    # A map from FX Node to the qualname of the Module from which it  # 从FX节点到其来源模块的合格名称的映射。
    # originated. This is recorded by `create_proxy` when recording an  # 这是在记录操作时通过`create_proxy`记录的
    # operation  # 
    node_to_originating_module: Dict[torch.fx.Node, str] = {}  # 初始化节点到来源模块的映射

    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any],  # 重写Tracer.call_module方法
                    args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:  # 定义call_module的参数
        """
        Override of Tracer.call_module (see  # 重写Tracer.call_module（见
        https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer.call_module).  # https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer.call_module）。

        This override:  # 此重写：
        1) Stores away the qualified name of the caller for restoration later  # 1）存储调用者的合格名称以便稍后恢复
        2) Installs the qualified name of the caller in `current_module_qualified_name`  # 2）将调用者的合格名称安装在`current_module_qualified_name`中
           for retrieval by `create_proxy`  # 以便`create_proxy`检索
        3) Delegates into the normal Tracer.call_module method  # 3）委托给正常的Tracer.call_module方法
        4) Restores the caller's qualified name into current_module_qualified_name  # 4）将调用者的合格名称恢复到current_module_qualified_name
        """
        old_qualname = self.current_module_qualified_name  # 保存当前合格名称
        try:
            self.current_module_qualified_name = self.path_of_module(m)  # 更新为当前模块的合格名称
            return super().call_module(m, forward, args, kwargs)  # 调用父类的call_module方法
        finally:
            self.current_module_qualified_name = old_qualname  # 恢复到旧的合格名称

    def create_proxy(self, kind: str, target: torch.fx.node.Target, args: Tuple[Any, ...],  # 重写`Tracer.create_proxy`方法
                     kwargs: Dict[str, Any], name: Optional[str] = None, type_expr: Optional[Any] = None) -> Node:  # 定义create_proxy的参数
        """
        Override of `Tracer.create_proxy`. This override intercepts the recording  # 重写`Tracer.create_proxy`。此重写拦截记录
        of every operation and stores away the current traced module's qualified  # 每个操作，并将当前被跟踪模块的合格名称存储在
        name in `node_to_originating_module`  # `node_to_originating_module`
        """
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)  # 调用父类的create_proxy方法
        self.node_to_originating_module[proxy.node] = self.current_module_qualified_name  # 将当前模块的合格名称存储在映射中
        return proxy  # 返回Proxy


# Testing: let's see how this works on a torchvision ResNet18 model  # 测试：让我们看看这在torchvision ResNet18模型上是如何工作的
import torchvision.models as models  # 导入torchvision的模型模块

# Model under test  # 测试中的模型
rn18 = models.resnet18()  # 创建ResNet18模型实例

# Instantiate our ModulePathTracer and use that to trace our ResNet18  # 实例化我们的ModulePathTracer并用它来跟踪ResNet18
tracer = ModulePathTracer()  # 创建ModulePathTracer实例
traced_rn18 = tracer.trace(rn18)  # 对ResNet18进行跟踪

# Print (node, module qualified name) for every node in the Graph  # 打印图中每个节点的（节点，模块合格名称）
for node in traced_rn18.nodes:  # 遍历跟踪图中的每个节点
    module_qualname = tracer.node_to_originating_module.get(node)  # 获取节点的合格名称
    print('Node', node, 'is from module', module_qualname)  # 打印节点和其来源模块的合格名称
"""
Node x is from module  # 节点x来自模块
Node conv1 is from module conv1  # 节点conv1来自模块conv1
Node bn1 is from module bn1  # 节点bn1来自模块bn1
Node relu is from module relu  # 节点relu来自模块relu
Node maxpool is from module maxpool  # 节点maxpool来自模块maxpool
Node layer1_0_conv1 is from module layer1.0.conv1  # 节点layer1_0_conv1来自模块layer1.0.conv1
Node layer1_0_bn1 is from module layer1.0.bn1  # 节点layer1_0_bn1来自模块layer1.0.bn1
Node layer1_0_relu is from module layer1.0.relu  # 节点layer1_0_relu来自模块layer1.0.relu
Node layer1_0_conv2 is from module layer1.0.conv2  # 节点layer1_0_conv2来自模块layer1.0.conv2
Node layer1_0_bn2 is from module layer1.0.bn2  # 节点layer1_0_bn2来自模块layer1.0.bn2
Node add is from module layer1.0  # 节点add来自模块layer1.0
Node layer1_0_relu_1 is from module layer1.0.relu  # 节点layer1_0_relu_1来自模块layer1.0.relu
Node layer1_1_conv1 is from module layer1.1.conv1  # 节点layer1_1_conv1来自模块layer1.1.conv1
Node layer1_1_bn1 is from module layer1.1.bn1  # 节点layer1_1_bn1来自模块layer1.1.bn1
Node layer1_1_relu is from module layer1.1.relu  # 节点layer1_1_relu来自模块layer1.1.relu
Node layer1_1_conv2 is from module layer1.1.conv2  # 节点layer1_1_conv2来自模块layer1.1.conv2
Node layer1_1_bn2 is from module layer1.1.bn2  # 节点layer1_1_bn2来自模块layer1.1.bn2
Node add_1 is from module layer1.1  # 节点add_1来自模块layer1.1
Node layer1_1_relu_1 is from module layer1.1.relu  # 节点layer1_1_relu_1来自模块layer1.1.relu
Node layer2_0_conv1 is from module layer2.0.conv1  # 节点layer2_0_conv1来自模块layer2.0.conv1
Node layer2_0_bn1 is from module layer2.0.bn1  # 节点layer2_0_bn1来自模块layer2.0.bn1
Node layer2_0_relu is from module layer2.0.relu  # 节点layer2_0_relu来自模块layer2.0.relu
Node layer2_0_conv2 is from module layer2.0.conv2  # 节点layer2_0_conv2来自模块layer2.0.conv2
Node layer2_0_bn2 is from module layer2.0.bn2  # 节点layer2_0_bn2来自模块layer2.0.bn2
Node layer2_0_downsample_0 is from module layer2.0.downsample.0  # 节点layer2_0_downsample_0来自模块layer2.0.downsample.0
Node layer2_0_downsample_1 is from module layer2.0.downsample.1  # 节点layer2_0_downsample_1来自模块layer2.0.downsample.1
Node add_2 is from module layer2.0  # 节点add_2来自模块layer2.0
Node layer2_0_relu_1 is from module layer2.0.relu  # 节点layer2_0_relu_1来自模块layer2.0.relu
Node layer2_1_conv1 is from module layer2.1.conv1  # 节点layer2_1_conv1来自模块layer2.1.conv1
Node layer2_1_bn1 is from module layer2.1.bn1  # 节点layer2_1_bn1来自模块layer2.1.bn1
Node layer2_1_relu is from module layer2.1.relu  # 节点layer2_1_relu来自模块layer2.1.relu
Node layer2_1_conv2 is from module layer2.1.conv2  # 节点layer2_1_conv2来自模块layer2.1.conv2
Node layer2_1_bn2 is from module layer2.1.bn2  # 节点layer2_1_bn2来自模块layer2.1.bn2
Node add_3 is from module layer2.1  # 节点add_3来自模块layer2.1
Node layer2_1_relu_1 is from module layer2.1.relu  # 节点layer2_1_relu_1来自模块layer2.1.relu
Node layer3_0_conv1 is from module layer3.0.conv1  # 节点layer3_0_conv1来自模块layer3.0.conv1
Node layer3_0_bn1 is from module layer3.0.bn1  # 节点layer3_0_bn1来自模块layer3.0.bn1
Node layer3_0_relu is from module layer3.0.relu  # 节点layer3_0_relu来自模块layer3.0.relu
Node layer3_0_conv2 is from module layer3.0.conv2  # 节点layer3_0_conv2来自模块layer3.0.conv2
Node layer3_0_bn2 is from module layer3.0.bn2  # 节点layer3_0_bn2来自模块layer3.0.bn2
Node layer3_0_downsample_0 is from module layer3.0.downsample.0  # 节点layer3_0_downsample_0来自模块layer3.0.downsample.0
Node layer3_0_downsample_1 is from module layer3.0.downsample.1  # 节点layer3_0_downsample_1来自模块layer3.0.downsample.1
Node add_4 is from module layer3.0  # 节点add_4来自模块layer3.0
Node layer3_0_relu_1 is from module layer3.0.relu  # 节点layer3_0_relu_1来自模块layer3.0.relu
Node layer3_1_conv1 is from module layer3.1.conv1  # 节点layer3_1_conv1来自模块layer3.1.conv1
Node layer3_1_bn1 is from module layer3.1.bn1  # 节点layer3_1_bn1来自模块layer3.1.bn1
Node layer3_1_relu is from module layer3.1.relu  # 节点layer3_1_relu来自模块layer3.1.relu
Node layer3_1_conv2 is from module layer3.1.conv2  # 节点layer3_1_conv2来自模块layer3.1.conv2
Node layer3_1_bn2 is from module layer3.1.bn2  # 节点layer3_1_bn2来自模块layer3.1.bn2
Node add_5 is from module layer3.1  # 节点add_5来自模块layer3.1
Node layer3_1_relu_1 is from module layer3.1.relu  # 节点layer3_1_relu_1来自模块layer3.1.relu
Node layer4_0_conv1 is from module layer4.0.conv1  # 节点layer4_0_conv1来自模块layer4.0.conv1
Node layer4_0_bn1 is from module layer4.0.bn1  # 节点layer4_0_bn1来自模块layer4.0.bn1
Node layer4_0_relu is from module layer4.0.relu  # 节点layer4_0_relu来自模块layer4.0.relu
Node layer4_0_conv2 is from module layer4.0.conv2  # 节点layer4_0_conv2来自模块layer4.0.conv2
Node layer4_0_bn2 is from module layer4.0.bn2  # 节点layer4_0_bn2来自模块layer4.0.bn2
Node layer4_0_downsample_0 is from module layer4.0.downsample.0  # 节点layer4_0_downsample_0来自模块layer4.0.downsample.0
Node layer4_0_downsample_1 is from module layer4.0.downsample.1  # 节点layer4_0_downsample_1来自模块layer4.0.downsample.1
Node add_6 is from module layer4.0  # 节点add_6来自模块layer4.0
Node layer4_0_relu_1 is from module layer4.0.relu  # 节点layer4_0_relu_1来自模块layer4.0.relu
Node layer4_1_conv1 is from module layer4.1.conv1  # 节点layer4_1_conv1来自模块layer4.1.conv1
Node layer4_1_bn1 is from module layer4.1.bn1  # 节点layer4_1_bn1来自模块layer4.1.bn1
Node layer4_1_relu is from module layer4.1.relu  # 节点layer4_1_relu来自模块layer4.1.relu
Node layer4_1_conv2 is from module layer4.1.conv2  # 节点layer4_1_conv2来自模块layer4.1.conv2
Node layer4_1_bn2 is from module layer4.1.bn2  # 节点layer4_1_bn2来自模块layer4.1.bn2
Node add_7 is from module layer4.1  # 节点add_7来自模块layer4.1
Node layer4_1_relu_1 is from module layer4.1.relu  # 节点layer4_1_relu_1来自模块layer4.1.relu
Node avgpool is from module avgpool  # 节点avgpool来自模块avgpool
Node flatten is from module  # 节点flatten来自模块
Node fc is from module fc  # 节点fc来自模块fc
Node output is from module None  # 节点output来自模块None
"""