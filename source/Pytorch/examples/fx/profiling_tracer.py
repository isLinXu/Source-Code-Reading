"""
This file demonstrates using a custom FX Tracer to override  # 该文件演示了如何使用自定义FX跟踪器来重写
the behavior of `torch.autograd.profiler.record_function` and  # `torch.autograd.profiler.record_function`的行为，并
make profiler ranges appear in FX-traced code. This is done  # 使分析器范围出现在FX跟踪的代码中。这是通过
with Python dynamic patching magic, allowing us to explicitly  # Python动态补丁魔法实现的，允许我们显式
emit calls to  # 发出调用
`torch.ops.profiler._record_function_enter/_record_function_exit`.  # `torch.ops.profiler._record_function_enter/_record_function_exit`。
 
Please note that before https://github.com/pytorch/pytorch/pull/65180 lands,  # 请注意，在https://github.com/pytorch/pytorch/pull/65180合并之前，
these ranges may be eliminated by `Graph.eliminate_dead_code`  # 这些范围可能会被`Graph.eliminate_dead_code`消除
"""
import torch  # 导入PyTorch库
import torch.fx  # 导入PyTorch的FX模块

# Setup: a module with `record_function`  # 设置：一个带有`record_function`的模块
class Foo(torch.nn.Module):  # 定义一个名为Foo的类，继承自torch.nn.Module
  def forward(self, x):  # 定义前向传播方法，接受参数x
    with torch.profiler.record_function('foo'):  # 使用torch.profiler.record_function记录名为'foo'的范围
      return torch.relu(x)  # 返回x的ReLU值

f = Foo()  # 创建Foo类的实例
x = torch.randn(5, 3, 2)  # 生成一个随机的5x3x2张量
with torch.autograd.profiler.profile() as prof:  # 使用torch.autograd.profiler进行性能分析
  f(x)  # 调用Foo实例的前向传播方法

print(prof)  # 打印性能分析结果
# "foo" range is correctly recorded with normal execution  # "foo"范围在正常执行中被正确记录
"""
-------------------  ------------  ------------  ------------  ------------  ------------  ------------  
               Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------  ------------  ------------  ------------  ------------  ------------  ------------  
        aten::zeros         6.10%      10.298us        10.04%      16.943us      16.943us             1  
        aten::empty         2.88%       4.857us         2.88%       4.857us       4.857us             1  
        aten::zero_         1.06%       1.788us         1.06%       1.788us       1.788us             1  
                foo        21.28%      35.925us        89.96%     151.888us     151.888us             1  
        aten::empty        11.59%      19.572us        11.59%      19.572us      19.572us             1  
         aten::relu        23.81%      40.203us        57.09%      96.391us      96.391us             1  
    aten::clamp_min         3.87%       6.539us        33.28%      56.188us      56.188us             1  
        aten::empty         1.09%       1.847us         1.09%       1.847us       1.847us             1  
    aten::clamp_min        28.31%      47.802us        28.31%      47.802us      47.802us             1  
-------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 168.831us  # 自身CPU时间总计：168.831微秒
"""

traced = torch.fx.symbolic_trace(f)  # 对Foo类进行符号跟踪
with torch.autograd.profiler.profile() as prof:  # 使用torch.autograd.profiler进行性能分析
  traced(x)  # 调用跟踪后的Foo实例

print(prof)  # 打印性能分析结果
# "foo" range is not recorded with FX tracing  # "foo"范围在FX跟踪中未被记录
"""
-------------------  ------------  ------------  ------------  ------------  ------------  ------------  
               Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------  ------------  ------------  ------------  ------------  ------------  ------------  
         aten::relu        23.50%      10.618us       100.00%      45.186us      45.186us             1  
    aten::clamp_min        18.05%       8.154us        76.50%      34.568us      34.568us             1  
        aten::empty        11.77%       5.317us        11.77%       5.317us       5.317us             1  
    aten::clamp_min        46.69%      21.097us        46.69%      21.097us      21.097us             1  
-------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 45.186us  # 自身CPU时间总计：45.186微秒
"""

class ProfilerTracer(torch.fx.Tracer):  # 定义ProfilerTracer类，继承自torch.fx.Tracer
  def trace(self, root, concrete_args=None):  # 重写trace方法
    orig_record_function_enter = torch.autograd.profiler.record_function.__enter__  # 保存原始的__enter__方法
    orig_record_function_exit = torch.autograd.profiler.record_function.__exit__  # 保存原始的__exit__方法

    def fake_profiler_enter(_self):  # 定义伪分析器进入方法
      nonlocal self  # 使用外部的self
      handle_proxy = self.create_proxy(  # 创建代理
          kind='call_function',  # 节点类型为call_function
          target=torch.ops.profiler._record_function_enter,  # 目标为记录函数进入的操作
          args=(_self.name,),  # 传递函数名称作为参数
          kwargs={})  # 关键字参数为空
      
      assert getattr(_self, '_fx_profiler_ctx', None) is None  # 确保_fx_profiler_ctx属性为None
      setattr(_self, '_fx_profiler_ctx', handle_proxy)  # 设置_fx_profiler_ctx属性为handle_proxy
      return handle_proxy  # 返回代理

    def fake_profiler_exit(_self, exc_type, exc_value, traceback):  # 定义伪分析器退出方法
      assert hasattr(_self, '_fx_profiler_ctx')  # 确保_fx_profiler_ctx属性存在
      handle_proxy = _self._fx_profiler_ctx  # 获取代理
      torch.ops.profiler._record_function_exit(handle_proxy)  # 调用记录函数退出的操作
      setattr(_self, '_fx_profiler_ctx', None)  # 将_fx_profiler_ctx属性设置为None

    torch.autograd.profiler.record_function.__enter__ = fake_profiler_enter  # 替换__enter__方法
    torch.autograd.profiler.record_function.__exit__ = fake_profiler_exit  # 替换__exit__方法

    try:
      return super().trace(root, concrete_args)  # 调用父类的trace方法
    finally:
      torch.autograd.profiler.record_function.__enter__ = orig_record_function_enter  # 恢复原始__enter__方法
      torch.autograd.profiler.record_function.__exit__ = orig_record_function_exit  # 恢复原始__exit__方法

pt = ProfilerTracer()  # 创建ProfilerTracer实例

graph_with_profiler = pt.trace(f)  # 对模块f进行跟踪
traced_with_profiler = torch.fx.GraphModule(pt.root, graph_with_profiler)  # 创建GraphModule实例

with torch.autograd.profiler.profile() as prof:  # 使用torch.autograd.profiler进行性能分析
  traced_with_profiler(x)  # 调用带有分析器的跟踪模块

print(prof)  # 打印性能分析结果
# "foo" range is recorded with special tracer behavior  # "foo"范围在特殊跟踪器行为下被记录
"""
-------------------  ------------  ------------  ------------  ------------  ------------  ------------  
               Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                foo        19.76%      39.928us       100.00%     202.055us     202.055us             1  
        aten::empty         3.93%       7.950us         3.93%       7.950us       7.950us             1  
         aten::relu        33.79%      68.282us        76.30%     154.177us     154.177us             1  
    aten::clamp_min        27.32%      55.198us        42.51%      85.895us      85.895us             1  
        aten::empty         1.28%       2.585us         1.28%       2.585us       2.585us             1  
    aten::clamp_min        13.91%      28.112us        13.91%      28.112us      28.112us             1  
-------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 202.055us  # 自身CPU时间总计：202.055微秒
"""