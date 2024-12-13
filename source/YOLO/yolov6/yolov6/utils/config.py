#!/usr/bin/env python
# -*- coding: utf-8 -*-
# The code is based on
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py
# Copyright (c) OpenMMLab.

import os.path as osp  # 导入os.path模块，并重命名为osp
import shutil  # 导入shutil模块，用于文件操作
import sys  # 导入sys模块，用于系统特定参数和函数
import tempfile  # 导入tempfile模块，用于创建临时文件和目录
from importlib import import_module  # 从importlib导入import_module函数，用于动态导入模块
from addict import Dict  # 从addict导入Dict类，用于创建字典的子类

class ConfigDict(Dict):
    # ConfigDict类继承自addict.Dict，允许通过属性访问字典的键

    def __missing__(self, name):
        raise KeyError(name)  # 如果键缺失，抛出KeyError

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)  # 尝试获取属性
        except KeyError:
            ex = AttributeError("'{}' object has no attribute '{}'".format(
                self.__class__.__name__, name))  # 如果键不存在，抛出AttributeError
        except Exception as e:
            ex = e  # 捕获其他异常
        else:
            return value  # 返回找到的值
        raise ex  # 抛出异常

class Config(object):
    # Config类用于加载和管理配置文件

    @staticmethod
    def _file2dict(filename):
        # 将文件转换为字典
        filename = str(filename)  # 将文件名转换为字符串
        if filename.endswith('.py'):  # 如果文件是Python文件
            with tempfile.TemporaryDirectory() as temp_config_dir:  # 创建临时目录
                shutil.copyfile(filename,
                                osp.join(temp_config_dir, '_tempconfig.py'))  # 复制文件到临时目录
                sys.path.insert(0, temp_config_dir)  # 将临时目录添加到系统路径
                mod = import_module('_tempconfig')  # 动态导入临时配置模块
                sys.path.pop(0)  # 从系统路径中移除临时目录
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith('__')  # 过滤掉以双下划线开头的属性
                }
                # delete imported module
                del sys.modules['_tempconfig']  # 删除导入的模块
        else:
            raise IOError('Only .py type are supported now!')  # 如果不是Python文件，抛出IOError
        cfg_text = filename + '\n'  # 初始化配置文本
        with open(filename, 'r') as f:
            cfg_text += f.read()  # 读取文件内容并添加到配置文本中

        return cfg_dict, cfg_text  # 返回配置字典和配置文本

    @staticmethod
    def fromfile(filename):
        # 从文件中加载配置
        cfg_dict, cfg_text = Config._file2dict(filename)  # 调用_file2dict方法
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)  # 返回Config实例

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        # 初始化Config实例
        if cfg_dict is None:
            cfg_dict = dict()  # 如果cfg_dict为None，则初始化为空字典
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but got {}'.format(
                type(cfg_dict)))  # 如果cfg_dict不是字典，抛出TypeError

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))  # 设置_cfg_dict属性为ConfigDict实例
        super(Config, self).__setattr__('_filename', filename)  # 设置_filename属性
        if cfg_text:
            text = cfg_text  # 如果cfg_text存在，使用它
        elif filename:
            with open(filename, 'r') as f:
                text = f.read()  # 从文件中读取文本
        else:
            text = ''  # 如果没有文本，设置为空字符串
        super(Config, self).__setattr__('_text', text)  # 设置_text属性

    @property
    def filename(self):
        return self._filename  # 返回文件名属性

    @property
    def text(self):
        return self._text  # 返回文本属性

    def __repr__(self):
        return 'Config (path: {}): {}'.format(self.filename,
                                              self._cfg_dict.__repr__())  # 返回Config的字符串表示

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)  # 通过属性访问_cfg_dict中的值

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)  # 如果值是字典，转换为ConfigDict实例
        self._cfg_dict.__setattr__(name, value)  # 设置_cfg_dict中的值