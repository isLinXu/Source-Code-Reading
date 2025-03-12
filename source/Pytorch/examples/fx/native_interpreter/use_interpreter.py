import torch  # 导入PyTorch库
import torch.fx  # 导入PyTorch的FX模块，用于符号化和图形化
import operator  # 导入操作符模块


# Does this path not exist? Check that you've done the following:  # 该路径不存在吗？检查您是否完成以下操作：
# 1) Read README.md and follow the instructions to build libinterpreter.  # 1）阅读README.md并按照说明构建libinterpreter。
# 2) If this file still does not exist after you've followed those instructions,  # 2）如果在遵循这些说明后该文件仍然不存在，
#    check if it is under a different extension (e.g. `dylib` on mac or `dll` on  # 检查它是否具有不同的扩展名（例如，mac上的`dylib`或windows上的`dll`）。
#    windows).  
torch.classes.load_library('build/libinterpreter.so')  # 加载libinterpreter库


# This is what a lowering pass should look like: a function that takes  # 这就是降低转换的样子：一个接受有效nn.Module的函数，
# a valid nn.Module, symbolically traces it, lowers the Module to some  # 符号化跟踪它，将模块降低到某种表示，
# representation, and wraps that representation up into another  # 并将该表示包装到另一个
# nn.Module instance that handles dispatch to the compiled/lowered code.  # nn.Module实例中，该实例处理对已编译/降低代码的调度。
# This will ensure that this lowering transformation still fits into the  # 这将确保降低转换仍然适合
# PyTorch programming model and enables features like composing with other  # PyTorch编程模型，并启用与其他
# transformations and TorchScript compilation.  # 转换和TorchScript编译的功能。
def lower_to_elementwise_interpreter(orig_mod: torch.nn.Module) -> torch.nn.Module:  # 定义将模块降低为逐元素解释器的函数
    # ===== Stage 1: Symbolic trace the module =====  # ===== 阶段1：符号跟踪模块 =====
    mod = torch.fx.symbolic_trace(orig_mod)  # 符号化跟踪原始模块

    # ===== Stage 2: Lower GraphModule representation to the C++  # ===== 阶段2：将GraphModule表示降低到C++
    #       interpreter's instruction format ======  #       解释器的指令格式 ======
    instructions = []  # 初始化指令列表
    constant_idx = 0  # 初始化常量索引
    constants = {}  # 初始化常量字典
    fn_input_names = []  # 初始化函数输入名称列表

    target_to_name = {  # 定义操作符到名称的映射
        operator.add: "add",  # 加法操作符映射到"add"
        operator.mul: "mul"  # 乘法操作符映射到"mul"
    }

    output_node: Optional[torch.fx.Node] = None  # 初始化输出节点
    # For each instruction, create a triple  # 对于每个指令，创建一个三元组
    # (instruction_name : str, inputs : List[str], output : str)  # （指令名称：字符串，输入：字符串列表，输出：字符串）
    # to feed into the C++ interpreter  # 以便传递给C++解释器
    for n in mod.graph.nodes:  # 遍历图中的每个节点
        target, args, out_name = n.target, n.args, n.name  # 获取节点的目标、参数和输出名称
        assert len(n.kwargs) == 0, "kwargs currently not supported"  # 断言不支持关键字参数

        if n.op == 'placeholder':  # 如果操作是占位符
            # Placeholders specify function argument names. Save these  # 占位符指定函数参数名称。保存这些
            # for later when we generate the wrapper GraphModule  # 以便在稍后生成包装GraphModule时使用
            fn_input_names.append(target)  # 将占位符的目标添加到输入名称列表
        elif n.op == 'call_function':  # 如果操作是调用函数
            assert target in target_to_name, "Unsupported call target " + target  # 断言目标在映射中
            arg_names = []  # 初始化参数名称列表
            for arg in args:  # 遍历参数
                if not isinstance(arg, torch.fx.Node):  # 如果参数不是节点
                    # Pull out constants. These constants will later be  # 提取常量。这些常量稍后将被
                    # fed to the interpreter C++ object via add_constant()  # 通过add_constant()传递给解释器C++对象
                    arg_name = f'constant_{constant_idx}'  # 创建常量名称
                    constants[arg_name] = torch.Tensor(  # 将常量添加到字典
                        [arg] if isinstance(arg, numbers.Number) else arg)  # 如果参数是数字，则创建张量
                    arg_names.append(arg_name)  # 将常量名称添加到参数名称列表
                    constant_idx += 1  # 增加常量索引
                else:
                    arg_names.append(arg.name)  # 将节点名称添加到参数名称列表
            instructions.append((target_to_name[target], arg_names, out_name))  # 将指令添加到指令列表
        elif n.op == 'output':  # 如果操作是输出
            if output_node is not None:  # 如果已经存在输出节点
                raise RuntimeError('Multiple output nodes!')  # 引发运行时错误
            output_node = n  # 设置当前节点为输出节点
        else:  # 如果操作不受支持
            raise RuntimeError('Unsupported opcode ' + n.op)  # 引发运行时错误

    interpreter = torch.classes.NativeInterpretation.ElementwiseInterpreter()  # 创建元素级解释器实例
    # Load constants  # 加载常量
    for k, v in constants.items():  # 遍历常量字典
        interpreter.add_constant(k, v)  # 将常量添加到解释器
    # Specify names for positional input arguments  # 指定位置输入参数的名称
    interpreter.set_input_names(fn_input_names)  # 设置输入名称
    # Load instructions  # 加载指令
    interpreter.set_instructions(instructions)  # 设置指令
    # Specify name for single output  # 指定单个输出的名称
    assert isinstance(output_node.args[0], torch.fx.Node)  # 断言输出节点的参数是节点
    interpreter.set_output_name(output_node.args[0].name)  # 设置输出名称

    # ===== Stage 3: Create a wrapper GraphModule around the interpreter =====  # ===== 阶段3：在解释器周围创建包装GraphModule =====
    class WrapperModule(torch.nn.Module):  # 定义包装模块类
        def __init__(self, interpreter):  # 初始化函数，接受解释器
            super().__init__()  # 调用父类构造函数
            self.interpreter = interpreter  # 保存解释器

    wrapper = WrapperModule(interpreter)  # 创建包装模块实例

    # Create a forward() function that is compatible with TorchScript compilation.  # 创建与TorchScript编译兼容的forward()函数。
    # Create a graph that: 1) Takes function arguments  # 创建一个图：1）接受函数参数
    # 2) Invokes the interpreter  # 2）调用解释器
    # 3) Returns the specified return value  # 3）返回指定的返回值

    graph = torch.fx.Graph()  # 创建FX图
    # Add placeholders for fn inputs  # 为函数输入添加占位符
    placeholder_nodes = []  # 初始化占位符节点列表
    for name in fn_input_names:  # 遍历输入名称
        placeholder_nodes.append(graph.create_node('placeholder', name))  # 创建占位符节点并添加到图中

    # Get the interpreter object  # 获取解释器对象
    interpreter_node = graph.create_node('get_attr', 'interpreter')  # 创建获取解释器属性的节点

    # Add a node to call the interpreter instance  # 添加节点以调用解释器实例
    output_node = graph.create_node(
        op='call_method', target='__call__', args=(interpreter_node, placeholder_nodes))  # 创建调用解释器的节点

    # Register output  # 注册输出
    graph.output(output_node)  # 设置图的输出

    graph.lint(wrapper)  # 检查图的正确性

    # Return final GraphModule!!!  # 返回最终的GraphModule!!!
    return torch.fx.GraphModule(wrapper, graph)  # 返回包装的GraphModule

class MyElementwiseModule(torch.nn.Module):  # 定义自定义逐元素模块类
    def forward(self, x, y):  # 前向传播函数
        return x * y + y  # 返回逐元素乘法和加法的结果

mem = MyElementwiseModule()  # 创建自定义模块实例
lowered = lower_to_elementwise_interpreter(mem)  # 将模块降低为逐元素解释器
print(lowered.code)  # 打印降低后的模块代码
# The lowered module can also be compiled into TorchScript  # 降低后的模块也可以编译为TorchScript
scripted = torch.jit.script(lowered)  # 编译为TorchScript
print(scripted.graph)  # 打印TorchScript图

# Stress test correctness  # 压力测试正确性
for _ in range(50):  # 进行50次测试
    x, y = torch.randn(10, 20, 30), torch.randn(10, 20, 30)  # 生成随机输入
    torch.testing.assert_allclose(lowered(x, y), mem(x, y))  # 检查降低后的模块输出与原始模块输出是否相等
    torch.testing.assert_allclose(scripted(x, y), mem(x, y))  # 检查TorchScript输出与原始模块输出是否相等