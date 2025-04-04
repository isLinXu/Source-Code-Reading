# Copyright (c) Facebook, Inc. and its affiliates.
import ast  # 用于解析Python抽象语法树
import builtins  # 用于访问Python内建函数
import importlib  # 用于动态导入模块
import inspect  # 用于检查对象
import logging  # 用于日志记录
import os  # 用于操作系统功能
import uuid  # 用于生成唯一标识符
from collections import abc  # 用于访问抽象基类
from contextlib import contextmanager  # 用于创建上下文管理器
from copy import deepcopy  # 用于深度复制对象
from dataclasses import is_dataclass  # 用于检查对象是否为数据类
from typing import List, Tuple, Union  # 用于类型提示
import cloudpickle  # 用于序列化Python对象
import yaml  # 用于解析YAML文件
from omegaconf import DictConfig, ListConfig, OmegaConf  # 用于配置管理

from detectron2.utils.file_io import PathManager  # 用于文件IO操作
from detectron2.utils.registry import _convert_target_to_string  # 用于转换目标为字符串

__all__ = ["LazyCall", "LazyConfig"]  # 指定公开的API


class LazyCall:
    """
    Wrap a callable so that when it's called, the call will not be executed,
    but returns a dict that describes the call.
    包装一个可调用对象，当它被调用时，调用不会被执行，
    而是返回一个描述该调用的字典。

    LazyCall object has to be called with only keyword arguments. Positional
    arguments are not yet supported.
    LazyCall对象必须只使用关键字参数调用。
    位置参数尚未支持。

    Examples:
    ::
        from detectron2.config import instantiate, LazyCall

        layer_cfg = LazyCall(nn.Conv2d)(in_channels=32, out_channels=32)
        layer_cfg.out_channels = 64   # can edit it afterwards
        layer = instantiate(layer_cfg)
    # 示例:
    # ::
    #     from detectron2.config import instantiate, LazyCall
    #
    #     layer_cfg = LazyCall(nn.Conv2d)(in_channels=32, out_channels=32)
    #     layer_cfg.out_channels = 64   # 可以在之后编辑
    #     layer = instantiate(layer_cfg)
    """

    def __init__(self, target):
        if not (callable(target) or isinstance(target, (str, abc.Mapping))):
            raise TypeError(
                f"target of LazyCall must be a callable or defines a callable! Got {target}"
            )  # LazyCall的目标必须是可调用对象或定义可调用对象！
        self._target = target  # 存储目标可调用对象

    def __call__(self, **kwargs):
        if is_dataclass(self._target):
            # omegaconf object cannot hold dataclass type
            # https://github.com/omry/omegaconf/issues/784
            # omegaconf对象不能持有dataclass类型
            target = _convert_target_to_string(self._target)  # 转换数据类为字符串表示
        else:
            target = self._target  # 使用原始目标
        kwargs["_target_"] = target  # 将目标添加到关键字参数中

        return DictConfig(content=kwargs, flags={"allow_objects": True})  # 返回DictConfig对象


def _visit_dict_config(cfg, func):
    """
    Apply func recursively to all DictConfig in cfg.
    递归地将func应用于cfg中的所有DictConfig。
    """
    if isinstance(cfg, DictConfig):  # 如果是字典配置
        func(cfg)  # 对当前配置应用函数
        for v in cfg.values():  # 遍历所有值
            _visit_dict_config(v, func)  # 递归应用
    elif isinstance(cfg, ListConfig):  # 如果是列表配置
        for v in cfg:  # 遍历所有元素
            _visit_dict_config(v, func)  # 递归应用


def _validate_py_syntax(filename):
    # see also https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py
    # 另见 https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py
    with PathManager.open(filename, "r") as f:  # 打开文件
        content = f.read()  # 读取内容
    try:
        ast.parse(content)  # 尝试解析Python语法
    except SyntaxError as e:
        raise SyntaxError(f"Config file {filename} has syntax error!") from e  # 抛出语法错误


def _cast_to_config(obj):
    # if given a dict, return DictConfig instead
    # 如果给定一个dict，则返回DictConfig
    if isinstance(obj, dict):  # 如果是字典
        return DictConfig(obj, flags={"allow_objects": True})  # 转换为DictConfig
    return obj  # 否则返回原对象


_CFG_PACKAGE_NAME = "detectron2._cfg_loader"
"""
A namespace to put all imported config into.
一个命名空间，用于存放所有导入的配置。
"""


def _random_package_name(filename):
    # generate a random package name when loading config files
    # 加载配置文件时生成随机包名
    return _CFG_PACKAGE_NAME + str(uuid.uuid4())[:4] + "." + os.path.basename(filename)


@contextmanager
def _patch_import():
    """
    Enhance relative import statements in config files, so that they:
    增强配置文件中的相对导入语句，使它们：
    1. locate files purely based on relative location, regardless of packages.
       e.g. you can import file without having __init__
    # 1. 纯粹基于相对位置定位文件，而不考虑包。
    #    例如，你可以导入文件而不需要__init__
    2. do not cache modules globally; modifications of module states has no side effect
    # 2. 不在全局缓存模块；修改模块状态没有副作用
    3. support other storage system through PathManager
    # 3. 通过PathManager支持其他存储系统
    4. imported dict are turned into omegaconf.DictConfig automatically
    # 4. 导入的字典自动转换为omegaconf.DictConfig
    """
    old_import = builtins.__import__  # 保存原始的导入函数

    def find_relative_file(original_file, relative_import_path, level):
        cur_file = os.path.dirname(original_file)  # 获取当前文件目录
        for _ in range(level - 1):
            cur_file = os.path.dirname(cur_file)  # 根据level向上移动目录
        cur_name = relative_import_path.lstrip(".")  # 移除前导点
        for part in cur_name.split("."):
            cur_file = os.path.join(cur_file, part)  # 构建文件路径
        # NOTE: directory import is not handled. Because then it's unclear
        # if such import should produce python module or DictConfig. This can
        # be discussed further if needed.
        # 注意：不处理目录导入。因为不清楚这样的导入应该
        # 产生Python模块还是DictConfig。如果需要，可以进一步讨论。
        if not cur_file.endswith(".py"):
            cur_file += ".py"  # 添加.py后缀
        if not PathManager.isfile(cur_file):
            raise ImportError(
                f"Cannot import name {relative_import_path} from "
                f"{original_file}: {cur_file} has to exist."
            )  # 如果文件不存在则抛出导入错误
        return cur_file  # 返回找到的文件路径

    def new_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            # Only deal with relative imports inside config files
            # 只处理配置文件内的相对导入
            level != 0
            and globals is not None
            and (globals.get("__package__", "") or "").startswith(_CFG_PACKAGE_NAME)
        ):
            cur_file = find_relative_file(globals["__file__"], name, level)  # 查找相对文件
            _validate_py_syntax(cur_file)  # 验证Python语法
            spec = importlib.machinery.ModuleSpec(
                _random_package_name(cur_file), None, origin=cur_file
            )  # 创建模块规格
            module = importlib.util.module_from_spec(spec)  # 从规格创建模块
            module.__file__ = cur_file  # 设置文件路径
            with PathManager.open(cur_file) as f:
                content = f.read()  # 读取文件内容
            exec(compile(content, cur_file, "exec"), module.__dict__)  # 执行代码
            for name in fromlist:  # turn imported dict into DictConfig automatically
                                   # 自动将导入的字典转换为DictConfig
                val = _cast_to_config(module.__dict__[name])  # 转换为配置
                module.__dict__[name] = val  # 更新模块字典
            return module  # 返回模块
        return old_import(name, globals, locals, fromlist=fromlist, level=level)  # 使用原始导入

    builtins.__import__ = new_import  # 替换导入函数
    yield new_import  # 暂时交出控制权
    builtins.__import__ = old_import  # 恢复原始导入函数


class LazyConfig:
    """
    Provide methods to save, load, and overrides configurations.
    提供保存、加载和覆盖配置的方法。
    """

    @staticmethod
    def load_rel(filename: str, keys: Union[None, str, Tuple[str, ...]] = None):
        """
        Similar to :meth:`load()`, but load path relative to the caller's
        source file.

        This has the same functionality as a relative import, except that this method
        accepts filename as a string, so more characters are allowed in the filename.
        """
        caller_frame = inspect.stack()[1]
        caller_fname = caller_frame[0].f_code.co_filename
        assert caller_fname != "<string>", "load_rel Unable to find caller"
        caller_dir = os.path.dirname(caller_fname)
        filename = os.path.join(caller_dir, filename)
        return LazyConfig.load(filename, keys)

    @staticmethod
    def load(filename: str, keys: Union[None, str, Tuple[str, ...]] = None):
        """
        Load a config file.

        Args:
            filename: absolute path or relative path w.r.t. the current working directory
            keys: keys to load and return. If not given, return all keys
                (whose values are config objects) in a dict.
        """
        has_keys = keys is not None
        filename = filename.replace("/./", "/")  # redundant
        if os.path.splitext(filename)[1] not in [".py", ".yaml", ".yml"]:
            raise ValueError(f"Config file {filename} has to be a python or yaml file.")
        if filename.endswith(".py"):
            _validate_py_syntax(filename)

            with _patch_import():
                # Record the filename
                module_namespace = {
                    "__file__": filename,
                    "__package__": _random_package_name(filename),
                }
                with PathManager.open(filename) as f:
                    content = f.read()
                # Compile first with filename to:
                # 1. make filename appears in stacktrace
                # 1. 使用filename编译首先：
                # 1. 使filename出现在堆栈跟踪中
                # 2. make load_rel able to find its parent's (possibly remote) location
                # 2. 使load_rel能够找到其父级的（可能是远程的）位置
                exec(compile(content, filename, "exec"), module_namespace)

            ret = module_namespace
        else:
            with PathManager.open(filename) as f:
                obj = yaml.unsafe_load(f)
            ret = OmegaConf.create(obj, flags={"allow_objects": True})

        if has_keys:
            if isinstance(keys, str):
                return _cast_to_config(ret[keys])
            else:
                return tuple(_cast_to_config(ret[a]) for a in keys)
        else:
            if filename.endswith(".py"):
                # when not specified, only load those that are config objects
                ret = DictConfig(
                    {
                        name: _cast_to_config(value)
                        for name, value in ret.items()
                        if isinstance(value, (DictConfig, ListConfig, dict))
                        and not name.startswith("_")
                    },
                    flags={"allow_objects": True},
                )
            return ret

    @staticmethod
    def save(cfg, filename: str):
        """
        Save a config object to a yaml file.
        Note that when the config dictionary contains complex objects (e.g. lambda),
        it can't be saved to yaml. In that case we will print an error and
        attempt to save to a pkl file instead.

        Args:
            cfg: an omegaconf config object
            filename: yaml file name to save the config file
        """
        logger = logging.getLogger(__name__)
        try:
            cfg = deepcopy(cfg)
        except Exception:
            pass
        else:
            # if it's deep-copyable, then...
            def _replace_type_by_name(x):
                if "_target_" in x and callable(x._target_):
                    try:
                        x._target_ = _convert_target_to_string(x._target_)
                    except AttributeError:
                        pass

            # not necessary, but makes yaml looks nicer
            _visit_dict_config(cfg, _replace_type_by_name)

        save_pkl = False
        try:
            dict = OmegaConf.to_container(cfg, resolve=False)
            dumped = yaml.dump(dict, default_flow_style=None, allow_unicode=True, width=9999)
            with PathManager.open(filename, "w") as f:
                f.write(dumped)

            try:
                _ = yaml.unsafe_load(dumped)  # test that it is loadable
            except Exception:
                logger.warning(
                    "The config contains objects that cannot serialize to a valid yaml. "
                    f"{filename} is human-readable but cannot be loaded."
                )
                save_pkl = True
        except Exception:
            logger.exception("Unable to serialize the config to yaml. Error:")
            save_pkl = True

        if save_pkl:
            new_filename = filename + ".pkl"
            try:
                # retry by pickle
                with PathManager.open(new_filename, "wb") as f:
                    cloudpickle.dump(cfg, f)
                logger.warning(f"Config is saved using cloudpickle at {new_filename}.")
            except Exception:
                pass

    @staticmethod
    def apply_overrides(cfg, overrides: List[str]):
        """
        In-place override contents of cfg.
        就地覆盖cfg的内容。

        Args:
            cfg: an omegaconf config object
            # cfg: 一个omegaconf配置对象
            overrides: list of strings in the format of "a=b" to override configs.
                See https://hydra.cc/docs/next/advanced/override_grammar/basic/
                for syntax.
            # overrides: 格式为"a=b"的字符串列表，用于覆盖配置。
            #     语法见 https://hydra.cc/docs/next/advanced/override_grammar/basic/

        Returns:
            the cfg object
        # 返回：
        #     cfg对象
        """

        def safe_update(cfg, key, value):
            parts = key.split(".")  # 分割键路径
            for idx in range(1, len(parts)):
                prefix = ".".join(parts[:idx])  # 构建前缀
                v = OmegaConf.select(cfg, prefix, default=None)  # 尝试选择配置
                if v is None:
                    break  # 如果不存在则中断
                if not OmegaConf.is_config(v):
                    raise KeyError(
                        f"Trying to update key {key}, but {prefix} "
                        f"is not a config, but has type {type(v)}."
                    )  # 如果不是配置对象则抛出错误
            OmegaConf.update(cfg, key, value, merge=True)  # 更新配置

        from hydra.core.override_parser.overrides_parser import OverridesParser

        parser = OverridesParser.create()  # 创建解析器
        overrides = parser.parse_overrides(overrides)  # 解析覆盖
        for o in overrides:
            key = o.key_or_group  # 获取键或组
            value = o.value()  # 获取值
            if o.is_delete():
                # TODO support this
                # TODO 支持这个功能
                raise NotImplementedError("deletion is not yet a supported override")  # 删除尚不支持
            safe_update(cfg, key, value)  # 安全地更新配置
        return cfg  # 返回配置

    @staticmethod
    def to_py(cfg, prefix: str = "cfg."):
        """
        Try to convert a config object into Python-like psuedo code.
        尝试将配置对象转换为类似Python的伪代码。

        Note that perfect conversion is not always possible. So the returned
        results are mainly meant to be human-readable, and not meant to be executed.
        注意，完美的转换并不总是可能的。因此，返回的
        结果主要是为了人类可读，而不是为了执行。

        Args:
            cfg: an omegaconf config object
            # cfg: 一个omegaconf配置对象
            prefix: root name for the resulting code (default: "cfg.")
            # prefix: 结果代码的根名称（默认："cfg."）


        Returns:
            str of formatted Python code
        # 返回：
        #     格式化的Python代码字符串
        """
        import black  # 用于代码格式化

        cfg = OmegaConf.to_container(cfg, resolve=True)  # 转换为容器

        def _to_str(obj, prefix=None, inside_call=False):
            if prefix is None:
                prefix = []  # 初始化前缀
            if isinstance(obj, abc.Mapping) and "_target_" in obj:
                # Dict representing a function call
                # 表示函数调用的字典
                target = _convert_target_to_string(obj.pop("_target_"))  # 提取目标
                args = []
                for k, v in sorted(obj.items()):
                    args.append(f"{k}={_to_str(v, inside_call=True)}")  # 构建参数
                args = ", ".join(args)  # 连接参数
                call = f"{target}({args})"  # 构建调用
                return "".join(prefix) + call  # 返回带前缀的调用
            elif isinstance(obj, abc.Mapping) and not inside_call:
                # Dict that is not inside a call is a list of top-level config objects that we
                # render as one object per line with dot separated prefixes
                # 不在调用内部的字典是顶级配置对象的列表，
                # 我们将其渲染为每行一个对象，前缀用点分隔
                key_list = []
                for k, v in sorted(obj.items()):
                    if isinstance(v, abc.Mapping) and "_target_" not in v:
                        key_list.append(_to_str(v, prefix=prefix + [k + "."]))  # 递归处理嵌套映射
                    else:
                        key = "".join(prefix) + k  # 构建键
                        key_list.append(f"{key}={_to_str(v)}")  # 添加键值对
                return "\n".join(key_list)  # 返回换行分隔的键列表
            elif isinstance(obj, abc.Mapping):
                # Dict that is inside a call is rendered as a regular dict
                # 在调用内部的字典渲染为常规字典
                return (
                    "{"
                    + ",".join(
                        f"{repr(k)}: {_to_str(v, inside_call=inside_call)}"
                        for k, v in sorted(obj.items())
                    )  # 将每个键值对渲染为字符串
                    + "}"
                )
            elif isinstance(obj, list):
                return "[" + ",".join(_to_str(x, inside_call=inside_call) for x in obj) + "]"  # 渲染列表
            else:
                return repr(obj)  # 渲染基本类型

        py_str = _to_str(cfg, prefix=[prefix])  # 转换为Python字符串
        try:
            return black.format_str(py_str, mode=black.Mode())  # 使用black格式化
        except black.InvalidInput:
            return py_str  # 如果无法格式化，则返回原始字符串
