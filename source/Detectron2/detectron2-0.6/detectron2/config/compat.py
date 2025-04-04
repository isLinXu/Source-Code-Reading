# Copyright (c) Facebook, Inc. and its affiliates.
"""
Backward compatibility of configs.
配置的向后兼容性。

Instructions to bump version:
版本升级说明：
+ It's not needed to bump version if new keys are added.
  It's only needed when backward-incompatible changes happen
  (i.e., some existing keys disappear, or the meaning of a key changes)
+ 如果只是添加了新的键，不需要升级版本。
  只有当发生向后不兼容的更改时才需要升级版本
  （即，某些现有键消失，或键的含义发生变化）
+ To bump version, do the following:
+ 要升级版本，请执行以下操作：
    1. Increment _C.VERSION in defaults.py
    1. 增加defaults.py中的_C.VERSION
    2. Add a converter in this file.
    2. 在此文件中添加转换器。

      Each ConverterVX has a function "upgrade" which in-place upgrades config from X-1 to X,
      and a function "downgrade" which in-place downgrades config from X to X-1
      每个ConverterVX都有一个"upgrade"函数，用于将配置从X-1就地升级到X，
      以及一个"downgrade"函数，用于将配置从X就地降级到X-1

      In each function, VERSION is left unchanged.
      在每个函数中，VERSION保持不变。

      Each converter assumes that its input has the relevant keys
      (i.e., the input is not a partial config).
      每个转换器假设其输入具有相关键
      （即，输入不是部分配置）。
    3. Run the tests (test_config.py) to make sure the upgrade & downgrade
       functions are consistent.
    3. 运行测试（test_config.py）以确保升级和降级
       功能一致。
"""

import logging  # 导入日志记录模块
from typing import List, Optional, Tuple  # 导入类型提示

from .config import CfgNode as CN  # 导入配置节点类
from .defaults import _C  # 导入默认配置

__all__ = ["upgrade_config", "downgrade_config"]  # 定义公开的API


def upgrade_config(cfg: CN, to_version: Optional[int] = None) -> CN:
    """
    Upgrade a config from its current version to a newer version.
    将配置从当前版本升级到较新的版本。

    Args:
        cfg (CfgNode):
        # cfg (CfgNode): 配置节点
        to_version (int): defaults to the latest version.
        # to_version (int): 默认为最新版本。
    """
    cfg = cfg.clone()  # 克隆配置以避免修改原始配置
    if to_version is None:
        to_version = _C.VERSION  # 如果未指定目标版本，则使用最新版本

    assert cfg.VERSION <= to_version, "Cannot upgrade from v{} to v{}!".format(
        cfg.VERSION, to_version
    )  # 确保当前版本小于或等于目标版本
    for k in range(cfg.VERSION, to_version):
        converter = globals()["ConverterV" + str(k + 1)]  # 获取对应版本的转换器
        converter.upgrade(cfg)  # 应用升级
        cfg.VERSION = k + 1  # 更新版本号
    return cfg  # 返回升级后的配置


def downgrade_config(cfg: CN, to_version: int) -> CN:
    """
    Downgrade a config from its current version to an older version.
    将配置从当前版本降级到较旧的版本。

    Args:
        cfg (CfgNode):
        # cfg (CfgNode): 配置节点
        to_version (int):
        # to_version (int): 目标版本

    Note:
        A general downgrade of arbitrary configs is not always possible due to the
        different functionalities in different versions.
        The purpose of downgrade is only to recover the defaults in old versions,
        allowing it to load an old partial yaml config.
        Therefore, the implementation only needs to fill in the default values
        in the old version when a general downgrade is not possible.
    # 注意：
    #     由于不同版本的功能不同，对任意配置进行通用降级并不总是可能的。
    #     降级的目的仅仅是恢复旧版本中的默认值，
    #     允许它加载旧的部分yaml配置。
    #     因此，当无法进行通用降级时，实现只需要填写旧版本中的默认值。
    """
    cfg = cfg.clone()  # 克隆配置以避免修改原始配置
    assert cfg.VERSION >= to_version, "Cannot downgrade from v{} to v{}!".format(
        cfg.VERSION, to_version
    )  # 确保当前版本大于或等于目标版本
    for k in range(cfg.VERSION, to_version, -1):
        converter = globals()["ConverterV" + str(k)]  # 获取对应版本的转换器
        converter.downgrade(cfg)  # 应用降级
        cfg.VERSION = k - 1  # 更新版本号
    return cfg  # 返回降级后的配置


def guess_version(cfg: CN, filename: str) -> int:
    """
    Guess the version of a partial config where the VERSION field is not specified.
    Returns the version, or the latest if cannot make a guess.
    猜测未指定VERSION字段的部分配置的版本。
    返回版本，如果无法猜测则返回最新版本。

    This makes it easier for users to migrate.
    这使用户更容易迁移。
    """
    logger = logging.getLogger(__name__)  # 获取日志记录器

    def _has(name: str) -> bool:
        cur = cfg  # 从配置根开始
        for n in name.split("."):
            if n not in cur:
                return False  # 如果任何部分不存在，则返回False
            cur = cur[n]  # 否则继续深入配置
        return True  # 如果完整路径存在，则返回True

    # Most users' partial configs have "MODEL.WEIGHT", so guess on it
    # 大多数用户的部分配置有"MODEL.WEIGHT"，所以基于它猜测
    ret = None
    if _has("MODEL.WEIGHT") or _has("TEST.AUG_ON"):
        ret = 1  # 如果存在这些旧版本字段，猜测为版本1

    if ret is not None:
        logger.warning("Config '{}' has no VERSION. Assuming it to be v{}.".format(filename, ret))
        # 警告用户配置没有版本，假设为猜测的版本
    else:
        ret = _C.VERSION  # 如果无法猜测，使用最新版本
        logger.warning(
            "Config '{}' has no VERSION. Assuming it to be compatible with latest v{}.".format(
                filename, ret
            )
        )  # 警告用户配置没有版本，假设与最新版本兼容
    return ret  # 返回猜测的版本


def _rename(cfg: CN, old: str, new: str) -> None:
    old_keys = old.split(".")  # 将旧键路径分割为部分
    new_keys = new.split(".")  # 将新键路径分割为部分

    def _set(key_seq: List[str], val: str) -> None:
        cur = cfg  # 从配置根开始
        for k in key_seq[:-1]:
            if k not in cur:
                cur[k] = CN()  # 如果中间节点不存在，则创建它
            cur = cur[k]  # 移动到下一级
        cur[key_seq[-1]] = val  # 设置最终值

    def _get(key_seq: List[str]) -> CN:
        cur = cfg  # 从配置根开始
        for k in key_seq:
            cur = cur[k]  # 移动到下一级
        return cur  # 返回找到的值

    def _del(key_seq: List[str]) -> None:
        cur = cfg  # 从配置根开始
        for k in key_seq[:-1]:
            cur = cur[k]  # 移动到下一级
        del cur[key_seq[-1]]  # 删除最终键
        if len(cur) == 0 and len(key_seq) > 1:
            _del(key_seq[:-1])  # 如果节点为空，递归删除父节点

    _set(new_keys, _get(old_keys))  # 将旧键的值设置到新键
    _del(old_keys)  # 删除旧键


class _RenameConverter:
    """
    A converter that handles simple rename.
    一个处理简单重命名的转换器。
    """

    RENAME: List[Tuple[str, str]] = []  # list of tuples of (old name, new name)
                                        # 元组列表，每个元组是(旧名称，新名称)

    @classmethod
    def upgrade(cls, cfg: CN) -> None:
        for old, new in cls.RENAME:
            _rename(cfg, old, new)  # 将每个旧键重命名为新键

    @classmethod
    def downgrade(cls, cfg: CN) -> None:
        for old, new in cls.RENAME[::-1]:
            _rename(cfg, new, old)  # 反向应用重命名，将新键重命名回旧键


class ConverterV1(_RenameConverter):
    RENAME = [("MODEL.RPN_HEAD.NAME", "MODEL.RPN.HEAD_NAME")]  # 版本1的重命名列表


class ConverterV2(_RenameConverter):
    """
    A large bulk of rename, before public release.
    公开发布前的大量重命名。
    """

    RENAME = [
        ("MODEL.WEIGHT", "MODEL.WEIGHTS"),
        ("MODEL.PANOPTIC_FPN.SEMANTIC_LOSS_SCALE", "MODEL.SEM_SEG_HEAD.LOSS_WEIGHT"),
        ("MODEL.PANOPTIC_FPN.RPN_LOSS_SCALE", "MODEL.RPN.LOSS_WEIGHT"),
        ("MODEL.PANOPTIC_FPN.INSTANCE_LOSS_SCALE", "MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT"),
        ("MODEL.PANOPTIC_FPN.COMBINE_ON", "MODEL.PANOPTIC_FPN.COMBINE.ENABLED"),
        (
            "MODEL.PANOPTIC_FPN.COMBINE_OVERLAP_THRESHOLD",
            "MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH",
        ),
        (
            "MODEL.PANOPTIC_FPN.COMBINE_STUFF_AREA_LIMIT",
            "MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT",
        ),
        (
            "MODEL.PANOPTIC_FPN.COMBINE_INSTANCES_CONFIDENCE_THRESHOLD",
            "MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH",
        ),
        ("MODEL.ROI_HEADS.SCORE_THRESH", "MODEL.ROI_HEADS.SCORE_THRESH_TEST"),
        ("MODEL.ROI_HEADS.NMS", "MODEL.ROI_HEADS.NMS_THRESH_TEST"),
        ("MODEL.RETINANET.INFERENCE_SCORE_THRESHOLD", "MODEL.RETINANET.SCORE_THRESH_TEST"),
        ("MODEL.RETINANET.INFERENCE_TOPK_CANDIDATES", "MODEL.RETINANET.TOPK_CANDIDATES_TEST"),
        ("MODEL.RETINANET.INFERENCE_NMS_THRESHOLD", "MODEL.RETINANET.NMS_THRESH_TEST"),
        ("TEST.DETECTIONS_PER_IMG", "TEST.DETECTIONS_PER_IMAGE"),
        ("TEST.AUG_ON", "TEST.AUG.ENABLED"),
        ("TEST.AUG_MIN_SIZES", "TEST.AUG.MIN_SIZES"),
        ("TEST.AUG_MAX_SIZE", "TEST.AUG.MAX_SIZE"),
        ("TEST.AUG_FLIP", "TEST.AUG.FLIP"),
    ]  # 版本2的重命名列表

    @classmethod
    def upgrade(cls, cfg: CN) -> None:
        super().upgrade(cfg)  # 先应用基本重命名

        if cfg.MODEL.META_ARCHITECTURE == "RetinaNet":
            _rename(
                cfg, "MODEL.RETINANET.ANCHOR_ASPECT_RATIOS", "MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS"
            )  # 将RetinaNet特定的锚框比例移动到通用位置
            _rename(cfg, "MODEL.RETINANET.ANCHOR_SIZES", "MODEL.ANCHOR_GENERATOR.SIZES")  # 将RetinaNet特定的锚框大小移动到通用位置
            del cfg["MODEL"]["RPN"]["ANCHOR_SIZES"]  # 删除冗余配置
            del cfg["MODEL"]["RPN"]["ANCHOR_ASPECT_RATIOS"]  # 删除冗余配置
        else:
            _rename(cfg, "MODEL.RPN.ANCHOR_ASPECT_RATIOS", "MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS")  # 将RPN特定的锚框比例移动到通用位置
            _rename(cfg, "MODEL.RPN.ANCHOR_SIZES", "MODEL.ANCHOR_GENERATOR.SIZES")  # 将RPN特定的锚框大小移动到通用位置
            del cfg["MODEL"]["RETINANET"]["ANCHOR_SIZES"]  # 删除冗余配置
            del cfg["MODEL"]["RETINANET"]["ANCHOR_ASPECT_RATIOS"]  # 删除冗余配置
        del cfg["MODEL"]["RETINANET"]["ANCHOR_STRIDES"]  # 在所有情况下删除不再使用的配置

    @classmethod
    def downgrade(cls, cfg: CN) -> None:
        super().downgrade(cfg)  # 先应用基本重命名的反向操作

        _rename(cfg, "MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS", "MODEL.RPN.ANCHOR_ASPECT_RATIOS")  # 将通用锚框比例移回RPN特定位置
        _rename(cfg, "MODEL.ANCHOR_GENERATOR.SIZES", "MODEL.RPN.ANCHOR_SIZES")  # 将通用锚框大小移回RPN特定位置
        cfg.MODEL.RETINANET.ANCHOR_ASPECT_RATIOS = cfg.MODEL.RPN.ANCHOR_ASPECT_RATIOS  # 复制到RetinaNet
        cfg.MODEL.RETINANET.ANCHOR_SIZES = cfg.MODEL.RPN.ANCHOR_SIZES  # 复制到RetinaNet
        cfg.MODEL.RETINANET.ANCHOR_STRIDES = []  # this is not used anywhere in any version
                                                 # 这在任何版本中都未使用
