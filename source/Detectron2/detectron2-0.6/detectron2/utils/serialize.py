# Copyright (c) Facebook, Inc. and its affiliates.
import cloudpickle  # 导入cloudpickle库，它是pickle的增强版，可以序列化更多类型的对象


class PicklableWrapper(object):  # 定义一个可序列化包装器类
    """
    Wrap an object to make it more picklable, note that it uses
    heavy weight serialization libraries that are slower than pickle.
    It's best to use it only on closures (which are usually not picklable).

    This is a simplified version of
    https://github.com/joblib/joblib/blob/master/joblib/externals/loky/cloudpickle_wrapper.py
    """  # 包装一个对象使其更易于pickle序列化，注意它使用了比pickle更慢的重量级序列化库。
        # 最好只在闭包函数（通常不可直接pickle）上使用它。
        #
        # 这是以下代码的简化版本：
        # https://github.com/joblib/joblib/blob/master/joblib/externals/loky/cloudpickle_wrapper.py

    def __init__(self, obj):  # 初始化方法，接受一个需要包装的对象
        while isinstance(obj, PicklableWrapper):  # 如果传入的对象已经是PicklableWrapper实例
            # Wrapping an object twice is no-op  # 包装一个对象两次是无操作的
            obj = obj._obj  # 获取内部的原始对象
        self._obj = obj  # 存储原始对象

    def __reduce__(self):  # 定义__reduce__方法，使类可以被pickle序列化
        s = cloudpickle.dumps(self._obj)  # 使用cloudpickle序列化内部对象
        return cloudpickle.loads, (s,)  # 返回一个二元组，包含反序列化函数和序列化数据

    def __call__(self, *args, **kwargs):  # 使实例可调用，如果原始对象是函数或可调用对象
        return self._obj(*args, **kwargs)  # 调用原始对象并返回结果

    def __getattr__(self, attr):  # 处理属性访问
        # Ensure that the wrapped object can be used seamlessly as the previous object.  # 确保包装的对象可以无缝地像以前的对象一样使用
        if attr not in ["_obj"]:  # 如果不是访问_obj属性
            return getattr(self._obj, attr)  # 从原始对象获取属性
        return getattr(self, attr)  # 如果是访问_obj属性，直接从self获取
