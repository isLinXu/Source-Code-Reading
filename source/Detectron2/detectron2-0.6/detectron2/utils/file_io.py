# Copyright (c) Facebook, Inc. and its affiliates. 
from iopath.common.file_io import HTTPURLHandler, OneDrivePathHandler, PathHandler  # 从iopath库导入处理不同路径类型的处理器
from iopath.common.file_io import PathManager as PathManagerBase  # 导入基础路径管理器并重命名为PathManagerBase

__all__ = ["PathManager", "PathHandler"]  # 定义模块公开的API，只有这两个类会被暴露给外部


PathManager = PathManagerBase()  # 创建一个PathManagerBase实例，作为detectron2专用的路径管理器
"""
This is a detectron2 project-specific PathManager.
We try to stay away from global PathManager in fvcore as it
introduces potential conflicts among other libraries.
"""  # 这是detectron2项目特定的PathManager。我们尽量避免使用fvcore中的全局PathManager，因为它可能会与其他库产生潜在冲突。


class Detectron2Handler(PathHandler):  # 定义Detectron2处理器类，继承自PathHandler
    """
    Resolve anything that's hosted under detectron2's namespace.
    """  # 解析托管在detectron2命名空间下的任何资源。

    PREFIX = "detectron2://"  # 定义detectron2资源的URL前缀
    S3_DETECTRON2_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"  # 定义detectron2在S3存储的实际URL前缀

    def _get_supported_prefixes(self):  # 实现获取支持的前缀的方法
        return [self.PREFIX]  # 返回此处理器支持的前缀列表，即detectron2://

    def _get_local_path(self, path, **kwargs):  # 实现获取本地路径的方法
        name = path[len(self.PREFIX) :]  # 从路径中移除前缀，获取资源的实际名称
        return PathManager.get_local_path(self.S3_DETECTRON2_PREFIX + name, **kwargs)  # 通过PathManager获取S3上对应资源的本地路径

    def _open(self, path, mode="r", **kwargs):  # 实现打开文件的方法
        return PathManager.open(self._get_local_path(path), mode, **kwargs)  # 通过获取到的本地路径打开文件


PathManager.register_handler(HTTPURLHandler())  # 注册HTTP URL处理器，用于处理http://和https://开头的URL
PathManager.register_handler(OneDrivePathHandler())  # 注册OneDrive路径处理器，用于处理OneDrive上的文件
PathManager.register_handler(Detectron2Handler())  # 注册上面定义的Detectron2处理器，用于处理detectron2://开头的资源
