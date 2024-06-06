# damo_yolo

Directory tree
- tools
  - converter.py
  - calibrator.py
  - train.py
  - eval.py
  - demo.py
  - partial_quantization
    - utils.py
    - partial_quant.py
  - trt_eval.py
- damo
  - apis
    - __init__.py
    - detector_trainer.py
    - detector_inference_trt.py
    - detector_inference.py
  - structures
    - boxlist_ops.py
    - __init__.py
    - bounding_box.py
    - image_list.py
  - config
    - __init__.py
    - paths_catalog.py
    - base.py
    - augmentations.py
  - dataset
    - build.py
    - datasets
      - coco.py
      - __init__.py
      - mosaic_wrapper.py
      - evaluation
        - __init__.py
        - coco
          - __init__.py
          - coco_eval.py
    - __init__.py
    - transforms
      - build.py
      - transforms.py
      - __init__.py
      - tta_aug.py
      - transforms_keepratio.py
    - collate_batch.py
    - samplers
      - __init__.py
      - grouped_batch_sampler.py
      - distributed.py
      - iteration_based_batch_sampler.py
  - __init__.py
  - utils
    - model_utils.py
    - checkpoint.py
    - timer.py
    - __init__.py
    - boxes.py
    - visualize.py
    - logger.py
    - imports.py
    - metric.py
    - demo_utils.py
    - dist.py
    - debug_utils.py
  - augmentations
    - __init__.py
    - box_level_augs
      - box_level_augs.py
      - __init__.py
      - gaussian_maps.py
      - color_augs.py
      - geometric_augs.py
    - scale_aware_aug.py
  - detectors
    - detector.py
  - base_models
    - losses
      - distill_loss.py
      - gfocal_loss.py
    - core
      - end2end.py
      - ota_assigner.py
      - atss_assigner.py
      - ops.py
      - bbox_calculator.py
      - utils.py
      - weight_init.py
    - necks
      - __init__.py
      - giraffe_fpn_btn.py
    - __init__.py
    - heads
      - __init__.py
      - zero_head.py
    - backbones
      - tinynas_csp.py
      - tinynas_res.py
      - __init__.py
      - tinynas_mob.py
      - nas_backbones
        - tinynas_L45_kxkx.txt
        - tinynas_L35_kxkx.txt
        - tinynas_nano_middle.txt
        - tinynas_L20_k1kx.txt
        - tinynas_nano_small.txt
        - tinynas_L25_k1kx.txt
        - tinynas_nano_large.txt
        - tinynas_L20_k1kx_nano.txt
- datasets
- NOTICE
- README_cn.md
- configs
  - damoyolo_tinynasL18_Nm.py
  - damoyolo_tinynasL25_S.py
  - damoyolo_tinynasL20_Nl.py
  - damoyolo_tinynasL20_T.py
  - damoyolo_tinynasL20_N.py
  - damoyolo_tinynasL35_M.py
  - damoyolo_tinynasL18_Ns.py
  - damoyolo_tinynasL45_L.py
- scripts
  - coco_train.sh
  - coco_distill.sh
  - coco_eval.sh
- assets
  - dog.jpg
  - nano_curve.png
  - DAMO-YOLO.pdf
  - logo.png
  - applications
    - helmet_detection.png
    - facemask_detection.png
    - smartphone_detection.png
    - nflhelmet_detection.jpg
    - cigarette_detection.png
    - human_detection.png
    - trafficsign_detection.png
    - head_detection.png
  - CustomDatasetTutorial.md
  - 701class_cmp_horizontal.png
  - overview.gif
  - curve.png

---
<!-- TOC -->
# NOTICE

```
Copyright (c) 2021-2022 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
===========================================================


MIT License
-----------------------------

damo/utils/timer.py
damo/utils/imports.py
damo/dataset/transforms/tta_aug.py
damo/dataset/build.py
damo/dataset/collate_batch.py
damo/dataset/datasets/concat_dataset.py
damo/dataset/datasets/evaluation/coco/coco_eval.py
damo/dataset/samplers/distributed.py
damo/dataset/samplers/grouped_batch_sampler.py
damo/structures/bounding_box.py
damo/structures/boxlist_ops.py
damo/structures/image_list.py
Copyright (c) Facebook, Inc. and its affiliates.
All Rights Reserved.


Apache License 2.0 License
-----------------------------

damo/core/anchor.py
damo/core/sampler.py
damo/core/utils.py
damo/core/weight_init.py
damo/core/backbone/darknet.py
damo/core/ota_assigner.py
damo/core/atss_assigner.py
damo/base_models/losses/losses.py
damo/base_models/backbones/darknet.py
damo/base_models/necks/pafpn.py
opyright (c) OpenMMLab.
All rights reserved.


Creative Commons Attribution-NonCommercial 4.0 International Public License
-----------------------------

tools/onnx_inference.py

============================================================
Packages installed by codes or PIP.
============================
PyTorch's BSD-style license
============================
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

============================
Pillow's HPND License
============================
The Python Imaging Library (PIL) is

    Copyright © 1997-2011 by Secret Labs AB
    Copyright © 1995-2011 by Fredrik Lundh

Pillow is the friendly PIL fork. It is

    Copyright © 2010-2022 by Alex Clark and contributors

Like PIL, Pillow is licensed under the open source HPND License:

By obtaining, using, and/or copying this software and/or its associated
documentation, you agree that you have read, understood, and will comply
with the following terms and conditions:

Permission to use, copy, modify, and distribute this software and its
associated documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appears in all copies, and that
both that copyright notice and this permission notice appear in supporting
documentation, and that the name of Secret Labs AB or the author not be
used in advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.

============================
BSD License
============================
numpy==1.21.5
omegaconf
scikit-image


============================
Apache Software License
============================
Cython==0.29.24

============================
Apache License 2.0 License
============================
onnx==1.8.1
onnx-simplifier==0.3.5
ninja
timm

============================
MIT License
============================
tabulate==0.8.9
thop==0.0.31
opencv-python==4.5.4.60
thop
tensorboard
onnxruntime==1.8.0
loguru
scikit-image
tqdm

============================
LGPL-3.0 License
============================
easydict
```

## tools/converter.py

```python
#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import argparse
import sys

import onnx
import torch
from loguru import logger
from torch import nn

from damo.base_models.core.end2end import End2End
from damo.base_models.core.ops import RepConv, SiLU
from damo.config.base import parse_config
from damo.detectors.detector import build_local_model
from damo.utils.model_utils import get_model_info, replace_module


def make_parser():
    parser = argparse.ArgumentParser('damo converter deployment toolbox')
    # mode part
    parser.add_argument('--mode',
                        default='onnx',
                        type=str,
                        help='onnx, trt_16 or trt_32')
    # model part
    parser.add_argument(
        '-f',
        '--config_file',
        default=None,
        type=str,
        help='expriment description file',
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='if true, export without postprocess'
    )
    parser.add_argument('-c',
                        '--ckpt',
                        default=None,
                        type=str,
                        help='ckpt path')
    parser.add_argument('--trt',
                        action='store_true',
                        help='whether convert onnx into tensorrt')
    parser.add_argument(
        '--trt_type', type=str, default='fp32',
        help='one type of int8, fp16, fp32')
    parser.add_argument('--batch_size',
                        type=int,
                        default=None,
                        help='inference image batch nums')
    parser.add_argument('--img_size',
                        type=int,
                        default='640',
                        help='inference image shape')
    # onnx part
    parser.add_argument('--input',
                        default='images',
                        type=str,
                        help='input node name of onnx model')
    parser.add_argument('--output',
                        default='output',
                        type=str,
                        help='output node name of onnx model')
    parser.add_argument('-o',
                        '--opset',
                        default=11,
                        type=int,
                        help='onnx opset version')
    parser.add_argument('--end2end',
                        action='store_true',
                        help='export end2end onnx')
    parser.add_argument('--ort',
                        action='store_true',
                        help='export onnx for onnxruntime')
    parser.add_argument('--trt_eval',
                        action='store_true',
                        help='trt evaluation')
    parser.add_argument('--with-preprocess',
                        action='store_true',
                        help='export bgr2rgb and normalize')
    parser.add_argument('--topk-all',
                        type=int,
                        default=100,
                        help='topk objects for every images')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.65,
                        help='iou threshold for NMS')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.05,
                        help='conf threshold for NMS')
    parser.add_argument('--device',
                        default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


@logger.catch
def trt_export(onnx_path, batch_size, inference_h, inference_w, trt_mode, calib_loader=None, calib_cache='./damoyolo_calibration.cache'):
    import tensorrt as trt
    trt_version = int(trt.__version__[0])

    if trt_mode == 'int8':
        from calibrator import DataLoader, Calibrator
        calib_loader = DataLoader(1, 999, 'datasets/coco/val2017', 640, 640)

    TRT_LOGGER = trt.Logger()
    engine_path = onnx_path.replace('.onnx', f'_{trt_mode}_bs{batch_size}.trt')

    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    logger.info(f'trt_{trt_mode} converting ...')
    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(EXPLICIT_BATCH) as network, \
        trt.OnnxParser(network, TRT_LOGGER) as parser:

        logger.info('Loading ONNX file from path {}...'.format(onnx_path))
        with open(onnx_path, 'rb') as model:
            logger.info('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                logger.info('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    logger.info(parser.get_error(error))

        # builder.max_workspace_size = 1 << 30
        builder.max_batch_size = batch_size
        logger.info('Building an engine.  This would take a while...')
        config = builder.create_builder_config()
        config.max_workspace_size = 2 << 30

        if trt_mode == 'fp16':
            assert (builder.platform_has_fast_fp16 == True), 'not support fp16'
            # builder.fp16_mode = True
            config.flags |= 1 << int(trt.BuilderFlag.FP16)

        if trt_mode == 'int8':
            config.flags |= 1 << int(trt.BuilderFlag.INT8)
            config.flags |= 1 << int(trt.BuilderFlag.FP16)

        if calib_loader is not None:
            config.int8_calibrator = Calibrator(calib_loader, calib_cache)
            logger.info('Int8 calibation is enabled.')

        if trt_version >= 8:
            config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))
        engine = builder.build_engine(network, config)

        try:
            assert engine
        except AssertionError:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)  # Fixed format
            tb_info = traceback.extract_tb(tb)
            _, line, _, text = tb_info[-1]
            raise AssertionError(
                "Parsing failed on line {} in statement {}".format(line, text)
            )

        logger.info('generated trt engine named {}'.format(engine_path))
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        return engine_path


@logger.catch
def main():
    args = make_parser().parse_args()

    logger.info('args value: {}'.format(args))
    onnx_name = args.config_file.split('/')[-1].replace('.py', '.onnx')

    if args.end2end:
        onnx_name = onnx_name.replace('.onnx', '_end2end.onnx')

    # Check device
    cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.device}' if cuda else 'cpu')
    assert not (
        device.type == 'cpu' and args.trt_type != 'fp32'
    ), '{args.trt_type} only compatible with GPU export, i.e. use --device 0'
    # init and load model
    config = parse_config(args.config_file)
    config.merge(args.opts)
    if args.benchmark:
        config.model.head.export_with_post = False

    if args.batch_size is not None:
        config.test.batch_size = args.batch_size

    # build model
    model = build_local_model(config, device)
    # load model paramerters
    ckpt = torch.load(args.ckpt, map_location=device)

    model.eval()
    if 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt, strict=True)
    logger.info(f'loading checkpoint from {args.ckpt}.')

    model = replace_module(model, nn.SiLU, SiLU)

    for layer in model.modules():
        if isinstance(layer, RepConv):
            layer.switch_to_deploy()

    info = get_model_info(model, (args.img_size, args.img_size))
    logger.info(info)
    # decouple postprocess
    model.head.nms = False

    if args.end2end:
        import tensorrt as trt
        trt_version = int(trt.__version__[0])
        model = End2End(model,
                        max_obj=args.topk_all,
                        iou_thres=args.iou_thres,
                        score_thres=args.conf_thres,
                        device=device,
                        ort=args.ort,
                        trt_version=trt_version,
                        with_preprocess=args.with_preprocess)

    dummy_input = torch.randn(args.batch_size, 3, args.img_size,
                              args.img_size).to(device)
    _ = model(dummy_input)
    torch.onnx._export(
        model,
        dummy_input,
        onnx_name,
        input_names=[args.input],
        output_names=['num_dets', 'det_boxes', 'det_scores', 'det_classes']
        if args.end2end else [args.output],
        opset_version=args.opset,
    )
    onnx_model = onnx.load(onnx_name)
    # Fix output shape
    if args.end2end and not args.ort:
        shapes = [
            args.batch_size, 1, args.batch_size, args.topk_all, 4,
            args.batch_size, args.topk_all, args.batch_size, args.topk_all
        ]
        for i in onnx_model.graph.output:
            for j in i.type.tensor_type.shape.dim:
                j.dim_param = str(shapes.pop(0))

    try:
        import onnxsim
        logger.info('Starting to simplify ONNX...')
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check, 'check failed'
    except Exception as e:
        logger.info(f'simplify failed: {e}')
    onnx.save(onnx_model, onnx_name)
    logger.info('generated onnx model named {}'.format(onnx_name))
    if args.trt:
        trt_name = trt_export(onnx_name, args.batch_size, args.img_size,
                              args.img_size, args.trt_type)
        if args.trt_eval:
            from trt_eval import trt_inference
            logger.info('start trt inference on coco validataion dataset')
            trt_inference(config, trt_name, args.img_size, args.batch_size,
                          args.conf_thres, args.iou_thres, args.end2end)


if __name__ == '__main__':
    main()
```

## tools/calibrator.py

```python
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import glob

import ctypes
import logging
logger = logging.getLogger(__name__)
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=False, stride=32, return_int=False):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    if not return_int:
        return im, r, (dw, dh)
    else:
        return im, r, (left, top)




"""
There are 4 types calibrator in TensorRT.
trt.IInt8LegacyCalibrator
trt.IInt8EntropyCalibrator
trt.IInt8EntropyCalibrator2
trt.IInt8MinMaxCalibrator
"""

class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, stream, cache_file=""):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.stream = stream
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        self.cache_file = cache_file
        stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names):
        print("######################")
        print(names)
        print("######################")
        batch = self.stream.next_batch()
        if not batch.size:
            return None

        cuda.memcpy_htod(self.d_input, batch)
        return [int(self.d_input)]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info("Using calibration cache to save time: {:}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            logger.info("Caching calibration data for future use: {:}".format(self.cache_file))
            f.write(cache)


def precess_image(img_src, img_size, stride):
    '''Process image before image inference.'''
    image = letterbox(img_src, img_size, auto=False, return_int=True)[0]
    # Convert
    image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = np.ascontiguousarray(image).astype(np.float32)
    return image

class DataLoader:
    def __init__(self, batch_size, batch_num, calib_img_dir, input_w, input_h):
        self.index = 0
        self.length = batch_num
        self.batch_size = batch_size
        self.input_h = input_h
        self.input_w = input_w
        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        self.img_list = glob.glob(os.path.join(calib_img_dir, "*.jpg"))
        assert len(self.img_list) > self.batch_size * self.length, \
            '{} must contains more than '.format(calib_img_dir) + str(self.batch_size * self.length) + ' images to calib'
        print('found all {} images to calib.'.format(len(self.img_list)))
        self.calibration_data = np.zeros((self.batch_size, 3, input_h, input_w), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = precess_image(img, self.input_h, 32)
                self.calibration_data[i] = img

            self.index += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length
```

## tools/train.py

```python
#!/usr/bin/env python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import argparse
import copy

import torch
from loguru import logger

from damo.apis import Trainer
from damo.config.base import parse_config
from damo.utils import synchronize


def make_parser():
    """
    Create a parser with some common arguments used by users.

    Returns:
        argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser('Damo-Yolo train parser')

    parser.add_argument(
        '-f',
        '--config_file',
        default=None,
        type=str,
        help='plz input your config file',
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--tea_config', type=str, default=None)
    parser.add_argument('--tea_ckpt', type=str, default=None)
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    synchronize()
    if args.tea_config is not None:
        tea_config = parse_config(args.tea_config)
    else:
        tea_config = None

    config = parse_config(args.config_file)
    config.merge(args.opts)


    trainer = Trainer(config, args, tea_config)
    trainer.train(args.local_rank)


if __name__ == '__main__':
    main()
```

## tools/eval.py

```python
#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import argparse
import os

import torch
from loguru import logger

from damo.base_models.core.ops import RepConv
from damo.apis.detector_inference import inference
from damo.config.base import parse_config
from damo.dataset import build_dataloader, build_dataset
from damo.detectors.detector import build_ddp_model, build_local_model
from damo.utils import fuse_model, get_model_info, setup_logger, synchronize


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_parser():
    parser = argparse.ArgumentParser('damo eval')

    # distributed
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '-f',
        '--config_file',
        default=None,
        type=str,
        help='pls input your config file',
    )
    parser.add_argument('-c',
                        '--ckpt',
                        default=None,
                        type=str,
                        help='ckpt for eval')
    parser.add_argument('--conf', default=None, type=float, help='test conf')
    parser.add_argument('--nms',
                        default=None,
                        type=float,
                        help='test nms threshold')
    parser.add_argument('--tsize',
                        default=None,
                        type=int,
                        help='test img size')
    parser.add_argument('--seed', default=None, type=int, help='eval seed')
    parser.add_argument(
        '--fuse',
        dest='fuse',
        default=False,
        action='store_true',
        help='Fuse conv and bn for testing.',
    )
    parser.add_argument(
        '--test',
        dest='test',
        default=False,
        action='store_true',
        help='Evaluating on test-dev set.',
    )  # TODO
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    synchronize()

    device = 'cuda'
    config = parse_config(args.config_file)
    config.merge(args.opts)

    save_dir = os.path.join(config.miscs.output_dir, config.miscs.exp_name)

    if args.local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    setup_logger(save_dir,
                 distributed_rank=args.local_rank,
                 mode='w')
    logger.info('Args: {}'.format(args))

    model = build_local_model(config, device)
    model.head.nms = True

    model.cuda(args.local_rank)
    model.eval()

    ckpt_file = args.ckpt
    logger.info('loading checkpoint from {}'.format(ckpt_file))
    loc = 'cuda:{}'.format(args.local_rank)
    ckpt = torch.load(ckpt_file, map_location=loc)
    new_state_dict = {}
    for k, v in ckpt['model'].items():
        k = k.replace('module', '')
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    logger.info('loaded checkpoint done.')

    for layer in model.modules():
        if isinstance(layer, RepConv):
            layer.switch_to_deploy()

    infer_shape = sum(config.test.augment.transform.image_max_range) // 2
    logger.info('Model Summary: {}'.format(get_model_info(model,
        (infer_shape, infer_shape))))

    model = build_ddp_model(model, local_rank=args.local_rank)
    if args.fuse:
        logger.info('\tFusing model...')
        model = fuse_model(model)
    # start evaluate
    output_folders = [None] * len(config.dataset.val_ann)

    if args.local_rank == 0 and config.miscs.output_dir:
        for idx, dataset_name in enumerate(config.dataset.val_ann):
            output_folder = os.path.join(config.miscs.output_dir, 'inference',
                                         dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    val_dataset = build_dataset(config, config.dataset.val_ann, is_train=False)
    val_loader = build_dataloader(val_dataset,
                                  config.test.augment,
                                  batch_size=config.test.batch_size,
                                  num_workers=config.miscs.num_workers,
                                  is_train=False,
                                  size_div=32)

    for output_folder, dataset_name, data_loader_val in zip(
            output_folders, config.dataset.val_ann, val_loader):
        inference(
            model,
            data_loader_val,
            dataset_name,
            iou_types=('bbox', ),
            box_only=False,
            device=device,
            output_folder=output_folder,
        )


if __name__ == '__main__':
    main()
```

## tools/demo.py

```python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import argparse
import os

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

from damo.base_models.core.ops import RepConv
from damo.config.base import parse_config
from damo.detectors.detector import build_local_model
from damo.utils import get_model_info, vis, postprocess
from damo.utils.demo_utils import transform_img
from damo.structures.image_list import ImageList
from damo.structures.bounding_box import BoxList

IMAGES=['png', 'jpg']
VIDEOS=['mp4', 'avi']


class Infer():
    def __init__(self, config, infer_size=[640,640], device='cuda', output_dir='./', ckpt=None, end2end=False):

        self.ckpt_path = ckpt
        suffix = ckpt.split('.')[-1]
        if suffix == 'onnx':
            self.engine_type = 'onnx'
        elif suffix == 'trt':
            self.engine_type = 'tensorRT'
        elif suffix in ['pt', 'pth']:
            self.engine_type = 'torch'
        self.end2end = end2end # only work with tensorRT engine
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        if torch.cuda.is_available() and device=='cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        if "class_names" in config.dataset:
            self.class_names = config.dataset.class_names
        else:
            self.class_names = []
            for i in range(config.model.head.num_classes):
                self.class_names.append(str(i))
            self.class_names = tuple(self.class_names)

        self.infer_size = infer_size
        config.dataset.size_divisibility = 0
        self.config = config
        self.model = self._build_engine(self.config, self.engine_type)

    def _pad_image(self, img, target_size):
        n, c, h, w = img.shape
        assert n == 1
        assert h<=target_size[0] and w<=target_size[1]
        target_size = [n, c, target_size[0], target_size[1]]
        pad_imgs = torch.zeros(*target_size)
        pad_imgs[:, :c, :h, :w].copy_(img)

        img_sizes = [img.shape[-2:]]
        pad_sizes = [pad_imgs.shape[-2:]]

        return ImageList(pad_imgs, img_sizes, pad_sizes)


    def _build_engine(self, config, engine_type):

        print(f'Inference with {engine_type} engine!')
        if engine_type == 'torch':
            model = build_local_model(config, self.device)
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
            model.load_state_dict(ckpt['model'], strict=True)
            for layer in model.modules():
                if isinstance(layer, RepConv):
                    layer.switch_to_deploy()
            model.eval()
        elif engine_type == 'tensorRT':
            model = self.build_tensorRT_engine(self.ckpt_path)
        elif engine_type == 'onnx':
            model, self.input_name, self.infer_size, _, _ = self.build_onnx_engine(self.ckpt_path)
        else:
            NotImplementedError(f'{engine_type} is not supported yet! Please use one of [onnx, torch, tensorRT]')

        return model

    def build_tensorRT_engine(self, trt_path):

        import tensorrt as trt
        from cuda import cuda
        loggert = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(loggert, '')
        runtime = trt.Runtime(loggert)
        with open(trt_path, 'rb') as t:
            model = runtime.deserialize_cuda_engine(t.read())
            context = model.create_execution_context()

        allocations = []
        inputs = []
        outputs = []
        for i in range(context.engine.num_bindings):
            is_input = False
            if context.engine.binding_is_input(i):
                is_input = True
            name = context.engine.get_binding_name(i)
            dtype = context.engine.get_binding_dtype(i)
            shape = context.engine.get_binding_shape(i)
            if is_input:
                batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.cuMemAlloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
                'size': size
            }
            allocations.append(allocation[1])
            if context.engine.binding_is_input(i):
                inputs.append(binding)
            else:
                outputs.append(binding)
        trt_out = []
        for output in outputs:
            trt_out.append(np.zeros(output['shape'], output['dtype']))

        def predict(batch):  # result gets copied into output
            # transfer input data to device
            cuda.cuMemcpyHtoD(inputs[0]['allocation'][1],
                          np.ascontiguousarray(batch), int(inputs[0]['size']))
            # execute model
            context.execute_v2(allocations)
            # transfer predictions back
            for o in range(len(trt_out)):
                cuda.cuMemcpyDtoH(trt_out[o], outputs[o]['allocation'][1],
                              outputs[o]['size'])
            return trt_out

        return predict




    def build_onnx_engine(self, onnx_path):

        import onnxruntime

        session = onnxruntime.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape

        out_names = []
        out_shapes = []
        for idx in range(len(session.get_outputs())):
            out_names.append(session.get_outputs()[idx].name)
            out_shapes.append(session.get_outputs()[idx].shape)
        return session, input_name, input_shape[2:], out_names, out_shapes



    def preprocess(self, origin_img):

        img = transform_img(origin_img, 0,
                            **self.config.test.augment.transform,
                            infer_size=self.infer_size)
        # img is a image_list
        oh, ow, _  = origin_img.shape
        img = self._pad_image(img.tensors, self.infer_size)

        img = img.to(self.device)
        return img, (ow, oh)

    def postprocess(self, preds, image, origin_shape=None):

        if self.engine_type == 'torch':
            output = preds

        elif self.engine_type == 'onnx':
            scores = torch.Tensor(preds[0])
            bboxes = torch.Tensor(preds[1])
            output = postprocess(scores, bboxes,
                self.config.model.head.num_classes,
                self.config.model.head.nms_conf_thre,
                self.config.model.head.nms_iou_thre,
                image)
        elif self.engine_type == 'tensorRT':
            if self.end2end:
                nums = preds[0]
                boxes = preds[1]
                scores = preds[2]
                pred_classes = preds[3]
                batch_size = boxes.shape[0]
                output = [None for _ in range(batch_size)]
                for i in range(batch_size):
                    img_h, img_w = image.image_sizes[i]
                    boxlist = BoxList(torch.Tensor(boxes[i][:nums[i][0]]),
                              (img_w, img_h),
                              mode='xyxy')
                    boxlist.add_field(
                        'objectness',
                        torch.Tensor(np.ones_like(scores[i][:nums[i][0]])))
                    boxlist.add_field('scores', torch.Tensor(scores[i][:nums[i][0]]))
                    boxlist.add_field('labels',
                              torch.Tensor(pred_classes[i][:nums[i][0]] + 1))
                    output[i] = boxlist
            else:
                cls_scores = torch.Tensor(preds[0])
                bbox_preds = torch.Tensor(preds[1])
                output = postprocess(cls_scores, bbox_preds,
                             self.config.model.head.num_classes,
                             self.config.model.head.nms_conf_thre,
                             self.config.model.head.nms_iou_thre, image)

        output = output[0].resize(origin_shape)
        bboxes = output.bbox
        scores = output.get_field('scores')
        cls_inds = output.get_field('labels')

        return bboxes,  scores, cls_inds


    def forward(self, origin_image):

        image, origin_shape = self.preprocess(origin_image)

        if self.engine_type == 'torch':
            output = self.model(image)

        elif self.engine_type == 'onnx':
            image_np = np.asarray(image.tensors.cpu())
            output = self.model.run(None, {self.input_name: image_np})

        elif self.engine_type == 'tensorRT':
            image_np = np.asarray(image.tensors.cpu()).astype(np.float32)
            output = self.model(image_np)

        bboxes, scores, cls_inds = self.postprocess(output, image, origin_shape=origin_shape)

        return bboxes, scores, cls_inds

    def visualize(self, image, bboxes, scores, cls_inds, conf, save_name='vis.jpg', save_result=True):
        vis_img = vis(image, bboxes, scores, cls_inds, conf, self.class_names)
        if save_result:
            save_path = os.path.join(self.output_dir, save_name)
            print(f"save visualization results at {save_path}")
            cv2.imwrite(save_path, vis_img[:, :, ::-1])
        return vis_img


def make_parser():
    parser = argparse.ArgumentParser('DAMO-YOLO Demo')

    parser.add_argument('input_type',
                        default='image',
                        help="input type, support [image, video, camera]")
    parser.add_argument('-f',
                        '--config_file',
                        default=None,
                        type=str,
                        help='pls input your config file',)
    parser.add_argument('-p',
                        '--path',
                        default='./assets/dog.jpg',
                        type=str,
                        help='path to image or video')
    parser.add_argument('--camid',
                        type=int,
                        default=0,
                        help='camera id, necessary when input_type is camera')
    parser.add_argument('--engine',
                        default=None,
                        type=str,
                        help='engine for inference')
    parser.add_argument('--device',
                        default='cuda',
                        type=str,
                        help='device used to inference')
    parser.add_argument('--output_dir',
                        default='./demo',
                        type=str,
                        help='where to save inference results')
    parser.add_argument('--conf',
                        default=0.6,
                        type=float,
                        help='conf of visualization')
    parser.add_argument('--infer_size',
                        nargs='+',
                        type=int,
                        help='test img size')
    parser.add_argument('--end2end',
                        action='store_true',
                        help='trt engine with nms')
    parser.add_argument('--save_result',
                        default=True,
                        type=bool,
                        help='whether save visualization results')


    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    config = parse_config(args.config_file)
    input_type = args.input_type

    infer_engine = Infer(config, infer_size=args.infer_size, device=args.device,
        output_dir=args.output_dir, ckpt=args.engine, end2end=args.end2end)

    if input_type == 'image':
        origin_img = np.asarray(Image.open(args.path).convert('RGB'))
        bboxes, scores, cls_inds = infer_engine.forward(origin_img)
        vis_res = infer_engine.visualize(origin_img, bboxes, scores, cls_inds, conf=args.conf, save_name=os.path.basename(args.path), save_result=args.save_result)
        if not args.save_result:
            cv2.namedWindow("DAMO-YOLO", cv2.WINDOW_NORMAL)
            cv2.imshow("DAMO-YOLO", vis_res)

    elif input_type == 'video' or input_type == 'camera':
        cap = cv2.VideoCapture(args.path if input_type == 'video' else args.camid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        if args.save_result:
            save_path = os.path.join(args.output_dir, os.path.basename(args.path))
            print(f'inference result will be saved at {save_path}')
            vid_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (int(width), int(height)))
        while True:
            ret_val, frame = cap.read()
            if ret_val:
                bboxes, scores, cls_inds = infer_engine.forward(frame)
                result_frame = infer_engine.visualize(frame, bboxes, scores, cls_inds, conf=args.conf, save_result=False)
                if args.save_result:
                    vid_writer.write(result_frame)
                else:
                    cv2.namedWindow("DAMO-YOLO", cv2.WINDOW_NORMAL)
                    cv2.imshow("DAMO-YOLO", result_frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break



if __name__ == '__main__':
    main()
```

## tools/trt_eval.py

```python
#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import argparse
import os

import torch
from loguru import logger

import tensorrt as trt
from damo.apis.detector_inference_trt import inference
from damo.config.base import parse_config
from damo.dataset import build_dataloader, build_dataset
from damo.utils import setup_logger, synchronize


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_parser():
    parser = argparse.ArgumentParser('damo trt engine eval')

    parser.add_argument(
        '-f',
        '--config_file',
        default=None,
        type=str,
        help='pls input your config file',
    )
    parser.add_argument('-t',
                        '--trt',
                        default=None,
                        type=str,
                        help='trt for eval')
    parser.add_argument('--conf', default=None, type=float, help='test conf')
    parser.add_argument('--nms',
                        default=None,
                        type=float,
                        help='test nms threshold')
    parser.add_argument('--batch_size',
                        type=int,
                        default=None,
                        help='inference image batch nums')
    parser.add_argument('--img_size',
                        type=int,
                        default='640',
                        help='inference image shape')
    parser.add_argument(
        '--end2end',
        action='store_true',
        help='trt inference with nms',
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def trt_inference(config,
                  trt_name,
                  img_size,
                  batch_size=None,
                  conf=None,
                  nms=None,
                  end2end=False):

    # dist init
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    os.environ['WORLD_SIZE'] = '1'
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://',
                                         rank=0)
    synchronize()

    file_name = os.path.join(config.miscs.output_dir, config.miscs.exp_name)
    os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name,
                 distributed_rank=0,
                 mode='a')

    if conf is not None:
        config.model.head.nms_conf_thre = conf
    if nms is not None:
        config.model.head.nms_iou_thre = nms
    if batch_size is not None:
        config.test.batch_size = batch_size

    # set logs
    loggert = trt.Logger(trt.Logger.INFO)

    trt.init_libnvinfer_plugins(loggert, '')

    # initialize
    t = open(trt_name, 'rb')
    runtime = trt.Runtime(loggert)
    model = runtime.deserialize_cuda_engine(t.read())
    context = model.create_execution_context()

    # start evaluate
    output_folders = [None] * len(config.dataset.val_ann)

    if config.miscs.output_dir:
        for idx, dataset_name in enumerate(config.dataset.val_ann):
            output_folder = os.path.join(config.miscs.output_dir, 'inference',
                                         dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    val_dataset = build_dataset(config, config.dataset.val_ann, is_train=False)
    val_loader = build_dataloader(val_dataset,
                                  config.test.augment,
                                  batch_size=config.test.batch_size,
                                  num_workers=config.miscs.num_workers,
                                  is_train=False,
                                  size_div=img_size)

    for output_folder, dataset_name, data_loader_val in zip(
            output_folders, config.dataset.val_ann, val_loader):
        inference(
            config,
            context,
            data_loader_val,
            dataset_name,
            iou_types=('bbox', ),
            box_only=False,
            output_folder=output_folder,
            end2end=end2end,
        )


@logger.catch
def main():
    args = make_parser().parse_args()
    config = parse_config(args.config_file)
    config.merge(args.opts)

    trt_inference(config,
                  args.trt,
                  args.img_size,
                  batch_size=args.batch_size,
                  conf=args.conf,
                  nms=args.nms,
                  end2end=args.end2end)


if __name__ == '__main__':
    main()
```

### tools/partial_quantization/utils.py

```python
import os
import torch
import torch.nn as nn
import copy

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import tensor_quant
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from damo.dataset import build_dataloader, build_dataset


def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def get_module(model, submodule_key):
    sub_tokens = submodule_key.split('.')
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod


def module_quant_disable(ptq_model, k):
    verified_module = get_module(ptq_model, k)
    if hasattr(verified_module, '_input_quantizer'):
        verified_module._input_quantizer.disable()
    if hasattr(verified_module, '_weight_quantizer'):
        verified_module._weight_quantizer.disable()


def collect_stats(model, data_loader, batch_number, device='cuda'):
    """
      code mainly from https://github.com/NVIDIA/TensorRT/blob/99a11a5fcdd1f184739bb20a8c4a473262c8ecc8/tools/pytorch-quantization/examples/torchvision/classification_flow.py
      Feed data to the network and collect statistic
    """

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, data_tuple in enumerate(data_loader):
        images, targets, image_ids = data_tuple
        images = images.to(device)
        output = model(images)
        if i + 1 >= batch_number:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    """
      code mainly from https://github.com/NVIDIA/TensorRT/blob/99a11a5fcdd1f184739bb20a8c4a473262c8ecc8/tools/pytorch-quantization/examples/torchvision/classification_flow.py
      Load calib result
    """
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            print(F"{name:40}: {module}")
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)


def quantable_op_check(k, ops_to_quant):
    if ops_to_quant is None:
        return True

    if k in ops_to_quant:
        return True
    else:
        return False


def quant_model_init(ori_model, device):

    ptq_model = copy.deepcopy(ori_model)
    ptq_model.eval()
    ptq_model.to(device)
    quant_conv_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL
    quant_conv_desc_input = QuantDescriptor(num_bits=8, calib_method='histogram')

    quant_convtrans_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL
    quant_convtrans_desc_input = QuantDescriptor(num_bits=8, calib_method='histogram')

    for k, m in ptq_model.named_modules():
        if 'proj_conv' in k:
            print("Layer {} won't be quantized".format(k))
            continue

        if isinstance(m, nn.Conv2d):
            quant_conv = quant_nn.QuantConv2d(m.in_channels,
                                              m.out_channels,
                                              m.kernel_size,
                                              m.stride,
                                              m.padding,
                                              quant_desc_input = quant_conv_desc_input,
                                              quant_desc_weight = quant_conv_desc_weight)
            quant_conv.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_conv.bias.data.copy_(m.bias.detach())
            else:
                quant_conv.bias = None
            set_module(ptq_model, k, quant_conv)
        elif isinstance(m, nn.ConvTranspose2d):
            quant_convtrans = quant_nn.QuantConvTranspose2d(m.in_channels,
                                                       m.out_channels,
                                                       m.kernel_size,
                                                       m.stride,
                                                       m.padding,
                                                       quant_desc_input = quant_convtrans_desc_input,
                                                       quant_desc_weight = quant_convtrans_desc_weight)
            quant_convtrans.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_convtrans.bias.data.copy_(m.bias.detach())
            else:
                quant_convtrans.bias = None
            set_module(ptq_model, k, quant_convtrans)
        elif isinstance(m, nn.MaxPool2d):
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            dilation = m.dilation
            ceil_mode = m.ceil_mode
            quant_maxpool2d = quant_nn.QuantMaxPool2d(m.kernel_size,
                                                      m.stride,
                                                      m.padding,
                                                      m.dilation,
                                                      m.ceil_mode,
                                                      quant_desc_input = quant_conv_desc_input)
            set_module(ptq_model, k, quant_maxpool2d)
        else:
            continue

    return ptq_model.to(device)


def post_train_quant(ori_model, calib_data_loader, calib_img_number, device):
    ptq_model = quant_model_init(ori_model, device)
    with torch.no_grad():
        collect_stats(ptq_model, calib_data_loader, calib_img_number, device)
        compute_amax(ptq_model, method='entropy')
    return ptq_model


def load_quanted_model(model, calib_weights_path, device):
    ptq_model = quant_model_init(model, device)
    ptq_model.load_state_dict(torch.load(calib_weights_path)['model'].state_dict())
    return ptq_model


def execute_partial_quant(ptq_model, ops_to_quant=None):
    for k, m in ptq_model.named_modules():
        if quantable_op_check(k, ops_to_quant):
            continue
        # enable full-precision
        if isinstance(m, quant_nn.QuantConv2d) or \
            isinstance(m, quant_nn.QuantConvTranspose2d) or \
            isinstance(m, quant_nn.QuantMaxPool2d):
            module_quant_disable(ptq_model, k)


def init_calib_data_loader(config):
    # init dataloader
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    os.environ['WORLD_SIZE'] = '1'
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://',
                                         rank=0)

    val_dataset = build_dataset(config, config.dataset.val_ann, is_train=False)
    val_loader = build_dataloader(val_dataset,
                                  config.test.augment,
                                  batch_size=config.test.batch_size,
                                  num_workers=config.miscs.num_workers,
                                  is_train=False,
                                  size_div=32)

    return val_loader[0]



```

### tools/partial_quantization/partial_quant.py

```python
#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import os
import argparse
import sys

import onnx
import torch
from loguru import logger
from torch import nn

from damo.base_models.core.end2end import End2End
from damo.base_models.core.ops import RepConv, SiLU
from damo.config.base import parse_config
from damo.detectors.detector import build_local_model
from damo.utils.model_utils import get_model_info, replace_module
from tools.trt_eval import trt_inference

from tools.partial_quantization.utils import post_train_quant, load_quanted_model, execute_partial_quant, init_calib_data_loader

from pytorch_quantization import nn as quant_nn


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_parser():
    parser = argparse.ArgumentParser('damo converter deployment toolbox')
    # mode part
    parser.add_argument('--mode',
                        default='onnx',
                        type=str,
                        help='onnx, trt_16 or trt_32')
    # model part
    parser.add_argument(
        '-f',
        '--config_file',
        default=None,
        type=str,
        help='expriment description file',
    )
    parser.add_argument('-c',
                        '--ckpt',
                        default=None,
                        type=str,
                        help='ckpt path')
    parser.add_argument('--trt',
                        action='store_true',
                        help='whether convert onnx into tensorrt')
    parser.add_argument(
        '--trt_type', type=str, default='fp32',
        help='one type of int8, fp16, fp32')
    parser.add_argument('--batch_size',
                        type=int,
                        default=None,
                        help='inference image batch nums')
    parser.add_argument('--img_size',
                        type=int,
                        default='640',
                        help='inference image shape')
    # onnx part
    parser.add_argument('--input',
                        default='images',
                        type=str,
                        help='input node name of onnx model')
    parser.add_argument('--output',
                        default='output',
                        type=str,
                        help='output node name of onnx model')
    parser.add_argument('-o',
                        '--opset',
                        default=11,
                        type=int,
                        help='onnx opset version')
    parser.add_argument('--calib_weights',
                        type=str,
                        default=None,
                        help='calib weights')
    parser.add_argument('--model_type',
                        type=str,
                        default=None,
                        help='quant model type(tiny, small, medium)')
    parser.add_argument('--sensitivity_file',
                        type=str,
                        default=None,
                        help='sensitivity file')
    parser.add_argument('--end2end',
                        action='store_true',
                        help='export end2end onnx')
    parser.add_argument('--ort',
                        action='store_true',
                        help='export onnx for onnxruntime')
    parser.add_argument('--trt_eval',
                        action='store_true',
                        help='trt evaluation')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.65,
                        help='iou threshold for NMS')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.05,
                        help='conf threshold for NMS')
    parser.add_argument('--device',
                        default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser



@logger.catch
def trt_export(onnx_path, batch_size, inference_h, inference_w):
    import tensorrt as trt

    TRT_LOGGER = trt.Logger()
    engine_path = onnx_path.replace('.onnx', f'_bs{batch_size}.trt')

    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    
    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(EXPLICIT_BATCH) as network, \
        trt.OnnxParser(network, TRT_LOGGER) as parser:

        logger.info('Loading ONNX file from path {}...'.format(onnx_path))
        with open(onnx_path, 'rb') as model:
            logger.info('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                logger.info('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    logger.info(parser.get_error(error))

        builder.max_batch_size = batch_size
        logger.info('Building an engine.  This would take a while...')
        config = builder.create_builder_config()
        config.max_workspace_size = 2 << 30
        
        config.flags |= 1 << int(trt.BuilderFlag.INT8)
        config.flags |= 1 << int(trt.BuilderFlag.FP16)

        engine = builder.build_engine(network, config)
        try:
            assert engine
        except AssertionError:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)  # Fixed format
            tb_info = traceback.extract_tb(tb)
            _, line, _, text = tb_info[-1]
            raise AssertionError(
                "Parsing failed on line {} in statement {}".format(line, text)
            )

        logger.info('generated trt engine named {}'.format(engine_path))
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        return engine_path


@logger.catch
def main():
    args = make_parser().parse_args()

    logger.info('args value: {}'.format(args))

    onnx_name = args.config_file.split('/')[-1].replace('.py', '_partial_quant.onnx')
    # Check device
    cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.device}' if cuda else 'cpu')

    # init config
    config = parse_config(args.config_file)
    config.merge(args.opts)
    if args.batch_size is not None:
        config.test.batch_size = args.batch_size

    # build model
    model = build_local_model(config, 'cuda')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.eval()
    if 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt, strict=False)
    logger.info('loading checkpoint done.')
    model = replace_module(model, nn.SiLU, SiLU)
    for layer in model.modules():
        if isinstance(layer, RepConv):
            layer.switch_to_deploy()
    info = get_model_info(model, (args.img_size, args.img_size))
    logger.info(info)

    # decouple postprocess
    model.head.nms = False

    # 1. do post training quantization
    if args.calib_weights is None:
        calib_data_loader = init_calib_data_loader(config)
        ptq_model = post_train_quant(model, calib_data_loader, 1000, device)
        torch.save({'model': ptq_model}, args.ckpt.replace('.pth', '_calib.pth'))
    else:
        ptq_model = load_quanted_model(model, args.calib_weights, device)

    # 2. load sensitivity data
    all_ops = list()
    for k, m in ptq_model.named_modules():
        if isinstance(m, quant_nn.QuantConv2d) or \
           isinstance(m, quant_nn.QuantConvTranspose2d) or \
           isinstance(m, quant_nn.MaxPool2d):
            all_ops.append((k))

    quant_model = args.model_type
    if quant_model == 'tiny':
        backbone_inds = list(range(24))
        neck_inds = []
        head_inds = list(range(74, 80))
    elif quant_model == 'small':
        backbone_inds = list(range(30))
        neck_inds = list(range(30,31)) + list(range(32,40)) + list(range(40,41)) + list(range(42, 49)) + list(range(50,51)) + list(range(52, 59)) + list(range(60, 61)) + list(range(62, 69)) + list(range(70, 71)) + list(range(72, 79))
        head_inds = list(range(80, 86))
    elif quant_model == 'medium':
        backbone_inds = list(range(5)) + list(range(6, 15)) + list(range(16, 33)) + list(range(34, 46)) + list(range(47, 48))
        neck_inds = []
        head_inds = list(range(108, 114))
    else:
        raise ValueError("unsupported model type in requested schema(tiny, small, medium)")

    all_inds = backbone_inds + neck_inds + head_inds

    quantable_sensitivity = [all_ops[x] for x in all_inds]
    ops_to_quant = [qops for qops in quantable_sensitivity]        

    # 3. only quantize ops in quantable_ops list
    execute_partial_quant(ptq_model, ops_to_quant=ops_to_quant)


    # 4. ONNX export
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size).to(device)
    _ = ptq_model(dummy_input)
    torch.onnx._export(
        ptq_model,
        dummy_input,
        onnx_name,
        verbose=False,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output'],
        opset_version=13,
    )
    onnx_model = onnx.load(onnx_name)        # Fix output shape
    try:
        import onnxsim
        logger.info('Starting to simplify ONNX...')
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check, 'check failed'
    except Exception as e:
        logger.info(f'simplify failed: {e}')
    onnx.save(onnx_model, onnx_name)
    logger.info('generated onnx model named {}'.format(onnx_name))

    # 5. export trt
    if args.trt:
        trt_name = trt_export(onnx_name, args.batch_size, args.img_size, args.img_size)
        # 6. trt eval
        if args.trt_eval:
            logger.info('start trt inference on coco validataion dataset')
            trt_inference(config, trt_name, args.img_size, args.batch_size,
                          args.conf_thres, args.iou_thres, args.end2end)


if __name__ == '__main__':
    main()
```

## damo/__init__.py

```python
#!/usr/bin/env python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
from .apis import *
from .base_models import *
from .config import *
from .detectors import *
from .utils import *

__version__ = '0.1.0'
```

### damo/apis/__init__.py

```python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
from .detector_trainer import Trainer
```

### damo/apis/detector_trainer.py

```python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import datetime
import math
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP

from damo.apis.detector_inference import inference
from damo.base_models.losses.distill_loss import FeatureLoss
from damo.dataset import build_dataloader, build_dataset
from damo.detectors.detector import build_ddp_model, build_local_model
from damo.utils import (MeterBuffer, get_model_info, get_rank, gpu_mem_usage,
                        save_checkpoint, setup_logger, synchronize)

from torch.nn import GroupNorm, LayerNorm
from torch.nn.modules.batchnorm import _BatchNorm
NORMS = (GroupNorm, LayerNorm, _BatchNorm)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class cosine_scheduler:
    def __init__(self,
                 base_lr_per_img,
                 batch_size,
                 min_lr_ratio,
                 total_iters,
                 no_aug_iters,
                 warmup_iters,
                 warmup_start_lr=0):

        self.base_lr = base_lr_per_img * batch_size
        self.final_lr = self.base_lr * min_lr_ratio
        self.warmup_iters = warmup_iters
        self.warmup_start_lr = warmup_start_lr
        self.total_iters = total_iters
        self.no_aug_iters = no_aug_iters

    def get_lr(self, iters):

        if iters < self.warmup_iters:
            lr = (self.base_lr - self.warmup_start_lr) * pow(
                iters / float(self.warmup_iters), 2) + self.warmup_start_lr
        elif iters >= self.total_iters - self.no_aug_iters:
            lr = self.final_lr
        else:
            lr = self.final_lr + 0.5 * (self.base_lr - self.final_lr) * (
                1.0 + math.cos(math.pi * (iters - self.warmup_iters) /
                               (self.total_iters - self.warmup_iters -
                                self.no_aug_iters)))
        return lr


class ema_model:
    def __init__(self, student, ema_momentum):

        self.model = deepcopy(student).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.ema_scheduler = lambda x: ema_momentum * (1 - math.exp(-x / 2000))

    def update(self, iters, student):

        student = student.module.state_dict()
        with torch.no_grad():
            momentum = self.ema_scheduler(iters)
            for name, param in self.model.state_dict().items():
                if param.dtype.is_floating_point:
                    param *= momentum
                    param += (1.0 - momentum) * student[name].detach()


class Trainer:
    def __init__(self, cfg, args, tea_cfg=None, is_train=True):
        self.cfg = cfg
        self.tea_cfg = tea_cfg
        self.args = args
        self.output_dir = cfg.miscs.output_dir
        self.exp_name = cfg.miscs.exp_name
        self.device = 'cuda'

        # set_seed(cfg.miscs.seed)
        # metric record
        self.meter = MeterBuffer(window_size=cfg.miscs.print_interval_iters)
        self.file_name = os.path.join(cfg.miscs.output_dir, cfg.miscs.exp_name)

        # setup logger
        if get_rank() == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=get_rank(),
            mode='w',
            )

        # logger
        logger.info('args info: {}'.format(self.args))
        logger.info('cfg value:\n{}'.format(self.cfg))

        # build model
        self.model = build_local_model(self.cfg, self.device)
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        logger.info('model:', self.model)

        if tea_cfg is not None:
            self.distill = True
            self.grad_clip = 30
            self.tea_model = build_local_model(self.tea_cfg, self.device)
            self.tea_model.eval()
            tea_ckpt = torch.load(args.tea_ckpt, map_location=self.device)
            #self.tea_model.load_state_dict(tea_ckpt['model'], strict=True)
            self.tea_model.load_pretrain_detector(args.tea_ckpt)
            self.feature_loss = FeatureLoss(self.model.neck.out_channels,
                                            self.tea_model.neck.out_channels,
                                            distiller='cwd').to(self.device)
            self.optimizer = self.build_optimizer((self.model, self.feature_loss),
                cfg.train.optimizer)
        else:
            self.distill = False
            self.grad_clip = None

            self.optimizer = self.build_optimizer(self.model,
                cfg.train.optimizer)
        # resume model
        if self.cfg.train.finetune_path is not None:
            self.model.load_pretrain_detector(self.cfg.train.finetune_path)
            self.epoch = 0
            self.start_epoch = 0
        elif self.cfg.train.resume_path is not None:
            resume_epoch = self.resume_model(self.cfg.train.resume_path,
                                             load_optimizer=True)
            self.epoch = resume_epoch
            self.start_epoch = resume_epoch
            logger.info('Resume Training from Epoch: {}'.format(self.epoch))
        else:
            self.epoch = 0
            self.start_epoch = 0
            logger.info('Start Training...')

        if self.cfg.train.ema:
            logger.info(
                'Enable ema model! Ema model will be evaluated and saved.')
            self.ema_model = ema_model(self.model, cfg.train.ema_momentum)
        else:
            self.ema_model = None

        # dataloader
        self.train_loader, self.val_loader, iters = self.get_data_loader(cfg)

        # setup iters according epochs and iters_per_epoch
        self.setup_iters(iters, self.start_epoch, cfg.train.total_epochs,
                         cfg.train.warmup_epochs, cfg.train.no_aug_epochs,
                         cfg.miscs.eval_interval_epochs,
                         cfg.miscs.ckpt_interval_epochs,
                         cfg.miscs.print_interval_iters)

        self.lr_scheduler = cosine_scheduler(
            cfg.train.base_lr_per_img, cfg.train.batch_size,
            cfg.train.min_lr_ratio, self.total_iters, self.no_aug_iters,
            self.warmup_iters, cfg.train.warmup_start_lr)

        self.mosaic_mixup = 'mosaic_mixup' in cfg.train.augment

    def get_data_loader(self, cfg):

        train_dataset = build_dataset(
            cfg,
            cfg.dataset.train_ann,
            is_train=True,
            mosaic_mixup=cfg.train.augment.mosaic_mixup)
        val_dataset = build_dataset(cfg, cfg.dataset.val_ann, is_train=False)

        iters_per_epoch = math.ceil(
            len(train_dataset[0]) /
            cfg.train.batch_size)  # train_dataset is a list, however,

        train_loader = build_dataloader(train_dataset,
                                        cfg.train.augment,
                                        batch_size=cfg.train.batch_size,
                                        start_epoch=self.start_epoch,
                                        total_epochs=cfg.train.total_epochs,
                                        num_workers=cfg.miscs.num_workers,
                                        is_train=True,
                                        size_div=32)

        val_loader = build_dataloader(val_dataset,
                                      cfg.test.augment,
                                      batch_size=cfg.test.batch_size,
                                      num_workers=cfg.miscs.num_workers,
                                      is_train=False,
                                      size_div=32)

        return train_loader, val_loader, iters_per_epoch

    def setup_iters(self, iters_per_epoch, start_epoch, total_epochs,
                    warmup_epochs, no_aug_epochs, eval_interval_epochs,
                    ckpt_interval_epochs, print_interval_iters):
        self.iters_per_epoch = iters_per_epoch
        self.total_epochs = total_epochs
        self.iters_per_epoch = iters_per_epoch
        self.start_iter = start_epoch * iters_per_epoch
        self.total_iters = total_epochs * iters_per_epoch
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.no_aug_iters = no_aug_epochs * iters_per_epoch
        self.no_aug = self.start_iter >= self.total_iters - self.no_aug_iters
        self.eval_interval_iters = eval_interval_epochs * iters_per_epoch
        self.ckpt_interval_iters = ckpt_interval_epochs * iters_per_epoch
        self.print_interval_iters = print_interval_iters

    def build_optimizer(self, models, cfg, exp_module=None):
        if not isinstance(models, (tuple, list)):
            models = (models, )

        param_dict = {}
        base_wd = cfg.get('weight_decay', None)
        optimizer_name = cfg.pop('name')
        optim_cls = getattr(torch.optim, optimizer_name)
        for model in models:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                assert param not in param_dict
                param_dict[param] = {"name": name}

            # weight decay of bn is always 0.
            for name, m in model.named_modules():
                if isinstance(m, NORMS):
                    if hasattr(m, "bias") and m.bias is not None:
                        param_dict[m.bias].update({"weight_decay": 0})
                    param_dict[m.weight].update({"weight_decay": 0})

            # weight decay of bias is always 0.
            for name, m in model.named_modules():
                if hasattr(m, "bias") and m.bias is not None:
                    param_dict[m.bias].update({"weight_decay": 0})
            param_groups = []
        for p, pconfig in param_dict.items():
            name = pconfig.pop("name", None)
            param_groups += [{"params": p, **pconfig}]


        optimizer = optim_cls(param_groups, **cfg)

        return optimizer

    def train(self, local_rank):

        infer_shape = sum(self.cfg.test.augment.transform.image_max_range) // 2
        logger.info('Model Summary: {}'.format(
            get_model_info(self.model, (infer_shape, infer_shape))))

        # distributed model init
        self.model = build_ddp_model(self.model, local_rank)
        logger.info('Model: {}'.format(self.model))

        logger.info('Training start...')

        # ----------- start training ------------------------- #
        self.model.train()
        iter_start_time = time.time()
        iter_end_time = time.time()
        for data_iter, (inps, targets, ids) in enumerate(self.train_loader):
            cur_iter = self.start_iter + data_iter

            lr = self.lr_scheduler.get_lr(cur_iter)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            inps = inps.to(self.device)  # ImageList: tensors, img_size
            targets = [target.to(self.device)
                       for target in targets]  # BoxList: bbox, num_boxes ...

            model_start_time = time.time()

            if self.distill:
                outputs, fpn_outs = self.model(inps, targets, stu=True)
                loss = outputs['total_loss']
                with torch.no_grad():
                    fpn_outs_tea = self.tea_model(inps, targets, tea=True)
                distill_weight = (
                    (1 - math.cos(cur_iter * math.pi / len(self.train_loader)))
                    / 2) * (0.1 - 1) + 1

                distill_loss = distill_weight * self.feature_loss(
                    fpn_outs, fpn_outs_tea)
                loss = loss + distill_loss
                outputs['distill_loss'] = distill_loss

            else:

                outputs = self.model(inps, targets)
                loss = outputs['total_loss']

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         max_norm=self.grad_clip,
                                         norm_type=2)  # for stable training

            self.optimizer.step()

            if self.ema_model is not None:
                self.ema_model.update(cur_iter, self.model)

            iter_start_time = iter_end_time
            iter_end_time = time.time()

            outputs_array = {_name: _v.item() for _name, _v in outputs.items()}
            self.meter.update(
                iter_time=iter_end_time - iter_start_time,
                model_time=iter_end_time - model_start_time,
                lr=lr,
                **outputs_array,
            )

            if cur_iter + 1 > self.total_iters - self.no_aug_iters:
                if self.mosaic_mixup:
                    logger.info('--->turn OFF mosaic aug now!')
                    self.train_loader.batch_sampler.set_mosaic(False)
                    self.eval_interval_iters = self.iters_per_epoch
                    self.ckpt_interval_iters = self.iters_per_epoch
                    self.mosaic_mixup = False

            # log needed information
            if (cur_iter + 1) % self.print_interval_iters == 0:
                left_iters = self.total_iters - (cur_iter + 1)
                eta_seconds = self.meter['iter_time'].global_avg * left_iters
                eta_str = 'ETA: {}'.format(
                    datetime.timedelta(seconds=int(eta_seconds)))

                progress_str = 'epoch: {}/{}, iter: {}/{}'.format(
                    self.epoch + 1, self.total_epochs,
                    (cur_iter + 1) % self.iters_per_epoch,
                    self.iters_per_epoch)
                loss_meter = self.meter.get_filtered_meter('loss')
                loss_str = ', '.join([
                    '{}: {:.1f}'.format(k, v.avg)
                    for k, v in loss_meter.items()
                ])

                time_meter = self.meter.get_filtered_meter('time')
                time_str = ', '.join([
                    '{}: {:.3f}s'.format(k, v.avg)
                    for k, v in time_meter.items()
                ])

                logger.info('{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}'.format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter['lr'].latest,
                ) + (', size: ({:d}, {:d}), {}'.format(
                    inps.tensors.shape[2], inps.tensors.shape[3], eta_str)))
                self.meter.clear_meters()

            if (cur_iter + 1) % self.ckpt_interval_iters == 0:
                self.save_ckpt('epoch_%d' % (self.epoch + 1),
                               local_rank=local_rank)

            if (cur_iter + 1) % self.eval_interval_iters == 0:
                time.sleep(0.003)
                self.evaluate(local_rank, self.cfg.dataset.val_ann)
                self.model.train()
            synchronize()

            if (cur_iter + 1) % self.iters_per_epoch == 0:
                self.epoch = self.epoch + 1

        self.save_ckpt(ckpt_name='latest', local_rank=local_rank)

    def save_ckpt(self, ckpt_name, local_rank, update_best_ckpt=False):
        if local_rank == 0:
            if self.ema_model is not None:
                save_model = self.ema_model.model
            else:
                save_model = self.model.module
            logger.info('Save weights to {}'.format(self.file_name))
            ckpt_state = {
                'epoch':
                self.epoch + 1,
                'model':
                save_model.state_dict(),
                'optimizer':
                self.optimizer.state_dict(),
                'feature_loss':
                self.feature_loss.state_dict() if self.distill else None,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

    def resume_model(self, resume_path, load_optimizer=False):
        ckpt_file_path = resume_path
        ckpt = torch.load(ckpt_file_path, map_location=self.device)

        self.model.load_state_dict(ckpt['model'])
        if load_optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            if self.distill:
                self.feature_loss.load_state_dict(ckpt['feature_loss'])
            resume_epoch = ckpt['epoch']
            return resume_epoch

    def evaluate(self, local_rank, val_ann):
        assert len(self.val_loader) == len(val_ann)
        if self.ema_model is not None:
            evalmodel = self.ema_model.model
        else:
            evalmodel = self.model
            if isinstance(evalmodel, DDP):
                evalmodel = evalmodel.module

        output_folders = [None] * len(val_ann)
        for idx, dataset_name in enumerate(val_ann):
            output_folder = os.path.join(self.output_dir, self.exp_name,
                                         'inference', dataset_name)
            if local_rank == 0:
                mkdir(output_folder)
            output_folders[idx] = output_folder

        for output_folder, dataset_name, data_loader_val in zip(
                output_folders, val_ann, self.val_loader):
            inference(
                evalmodel,
                data_loader_val,
                dataset_name,
                device=self.device,
                output_folder=output_folder,
            )
```

### damo/apis/detector_inference_trt.py

```python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import os

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

import tensorrt as trt
from cuda import cuda
from damo.dataset.datasets.evaluation import evaluate
from damo.structures.bounding_box import BoxList
from damo.utils import postprocess
from damo.utils.timer import Timer

COCO_CLASSES = []
for i in range(80):
    COCO_CLASSES.append(str(i))
COCO_CLASSES = tuple(COCO_CLASSES)


def compute_on_dataset(config,
                       context,
                       data_loader,
                       device,
                       timer=None,
                       end2end=False):

    results_dict = {}
    cpu_device = torch.device('cpu')
    allocations = []
    inputs = []
    outputs = []
    for i in range(context.engine.num_bindings):
        is_input = False
        if context.engine.binding_is_input(i):
            is_input = True
        name = context.engine.get_binding_name(i)
        dtype = context.engine.get_binding_dtype(i)
        shape = context.engine.get_binding_shape(i)
        if is_input:
            batch_size = shape[0]
        size = np.dtype(trt.nptype(dtype)).itemsize
        for s in shape:
            size *= s
        allocation = cuda.cuMemAlloc(size)
        binding = {
            'index': i,
            'name': name,
            'dtype': np.dtype(trt.nptype(dtype)),
            'shape': list(shape),
            'allocation': allocation,
            'size': size
        }
        allocations.append(allocation[1])
        if context.engine.binding_is_input(i):
            inputs.append(binding)
        else:
            outputs.append(binding)

    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            images_np = images.tensors.numpy()
            input_batch = images_np.astype(np.float32)

            trt_out = []
            for output in outputs:
                trt_out.append(np.zeros(output['shape'], output['dtype']))

            def predict(batch):  # result gets copied into output
                # transfer input data to device
                cuda.cuMemcpyHtoD(inputs[0]['allocation'][1],
                                  np.ascontiguousarray(batch),
                                  int(inputs[0]['size']))
                # execute model
                context.execute_v2(allocations)
                # transfer predictions back
                for o in range(len(trt_out)):
                    cuda.cuMemcpyDtoH(trt_out[o], outputs[o]['allocation'][1],
                                      outputs[o]['size'])
                return trt_out

            pred_out = predict(input_batch)
            # trt with nms
            if end2end:
                nums = pred_out[0]
                boxes = pred_out[1]
                scores = pred_out[2]
                pred_classes = pred_out[3]
                batch_size = boxes.shape[0]
                output = [None for _ in range(batch_size)]
                for i in range(batch_size):
                    img_h, img_w = images.image_sizes[i]
                    boxlist = BoxList(torch.Tensor(boxes[i][:nums[i][0]]),
                                      (img_w, img_h),
                                      mode='xyxy')
                    boxlist.add_field(
                        'objectness',
                        torch.Tensor(np.ones_like(scores[i][:nums[i][0]])))
                    boxlist.add_field('scores',
                                      torch.Tensor(scores[i][:nums[i][0]]))
                    boxlist.add_field(
                        'labels',
                        torch.Tensor(pred_classes[i][:nums[i][0]] + 1))
                    output[i] = boxlist
            else:
                cls_scores = torch.Tensor(pred_out[0])
                bbox_preds = torch.Tensor(pred_out[1])
                output = postprocess(cls_scores, bbox_preds,
                                     config.model.head.num_classes,
                                     config.model.head.nms_conf_thre,
                                     config.model.head.nms_iou_thre,
                                     images)

            if timer:
                torch.cuda.synchronize()
                timer.toc()

            output = [o.to(cpu_device) if o is not None else o for o in output]
        results_dict.update(
            {img_id: result
             for img_id, result in zip(image_ids, output)})
    return results_dict


def inference(
    config,
    context,
    data_loader,
    dataset_name,
    iou_types=('bbox', ),
    box_only=False,
    device='cuda',
    expected_results=(),
    expected_results_sigma_tol=4,
    output_folder=None,
    end2end=False,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    dataset = data_loader.dataset
    logger.info('Start evaluation on {} dataset({} images).'.format(
        dataset_name, len(dataset)))

    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(config, context, data_loader, device,
                                     inference_timer, end2end)
    # convert to a list
    image_ids = list(sorted(predictions.keys()))
    predictions = [predictions[i] for i in image_ids]

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, 'predictions.pth'))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
```

### damo/apis/detector_inference.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import os

import torch
from loguru import logger
from tqdm import tqdm

from damo.dataset.datasets.evaluation import evaluate
from damo.utils import all_gather, get_world_size, is_main_process, synchronize
from damo.utils.timer import Timer, get_time_str


def compute_on_dataset(model, data_loader, device, timer=None, tta=False):
    model.eval()
    results_dict = {}
    cpu_device = torch.device('cpu')
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
                output = model(images.to(device))
            if timer:
                # torch.cuda.synchronize() # consume much time
                timer.toc()
            output = [o.to(cpu_device) if o is not None else o for o in output]
        results_dict.update(
            {img_id: result
             for img_id, result in zip(image_ids, output)})
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu,
                                               multi_gpu_infer):
    if multi_gpu_infer:
        all_predictions = all_gather(predictions_per_gpu)
    else:
        all_predictions = [predictions_per_gpu]
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger.warning(
            'Number of images that were gathered from multiple processes is'
            'not a contiguous set. Some images might be missing from the'
            'evaluation')

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
    model,
    data_loader,
    dataset_name,
    iou_types=('bbox', ),
    box_only=False,
    device='cuda',
    expected_results=(),
    expected_results_sigma_tol=4,
    output_folder=None,
    multi_gpu_infer=True,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    dataset = data_loader.dataset
    logger.info('Start evaluation on {} dataset({} images).'.format(
        dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device,
                                     inference_timer)
    # wait for all processes to complete before measuring the time
    if multi_gpu_infer:
        synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        'Total run time: {} ({} s / img per device, on {} devices)'.format(
            total_time_str, total_time * num_devices / len(dataset),
            num_devices))
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        'Model inference time: {} ({} s / img per device, on {} devices)'.
        format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        ))

    predictions = _accumulate_predictions_from_multiple_gpus(
        predictions, multi_gpu_infer)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, 'predictions.pth'))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
```

### damo/structures/boxlist_ops.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .bounding_box import BoxList


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size
    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    xywh_boxes = boxlist.convert('xywh').bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = ((ws >= min_size) & (hs >= min_size)).nonzero().squeeze(1)
    return boxlist[keep]


def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].
    Returns:
      (tensor) iou, sized [N,M].
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
            'boxlists should have same image size, got {}, {}'.format(
                boxlist1, boxlist2))

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only
    a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList
    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size,
                        mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes
```

### damo/structures/__init__.py

```python
```

### damo/structures/bounding_box.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box,
    such as labels.
    """
    def __init__(self, bbox, image_size, mode='xyxy'):
        device = bbox.device if isinstance(
            bbox, torch.Tensor) else torch.device('cpu')
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError('bbox should have 2 dimensions, got {}'.format(
                bbox.ndimension()))
        if bbox.size(-1) != 4:
            raise ValueError('last dimension of bbox should have a '
                             'size of 4, got {}'.format(bbox.size(-1)))
        if mode not in ('xyxy', 'xywh'):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ('xyxy', 'xywh'):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == 'xyxy':
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 0
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE),
                dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == 'xyxy':
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == 'xywh':
            TO_REMOVE = 0
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError('Should not be here')

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box
        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """
        ratios = tuple(
            float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox

        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1)
        bbox = BoxList(scaled_box, size, mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)

        return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                'Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented')

        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 0
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin

        transposed_boxes = torch.cat((transposed_xmin, transposed_ymin,
                                      transposed_xmax, transposed_ymax),
                                     dim=-1)
        bbox = BoxList(transposed_boxes, self.size, mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        cropped_box = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1)
        bbox = BoxList(cropped_box, (w, h), mode='xyxy')
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    # Tensor-like methods

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, 'to'):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 0
        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def area(self):
        box = self.bbox
        if self.mode == 'xyxy':
            TO_REMOVE = 0
            area = (box[:, 2] - box[:, 0] +
                    TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == 'xywh':
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError('Should not be here')

        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(
                    field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_boxes={}, '.format(len(self))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={}, '.format(self.size[1])
        s += 'mode={})'.format(self.mode)
        return s
```

### damo/structures/image_list.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import division

import torch


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """
    def __init__(self, tensors, image_sizes, pad_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes
        self.pad_sizes = pad_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes, self.pad_sizes)


def to_image_list(tensors, size_divisible=0, max_size=None):
    """
    tensors can be an ImageList, a torch.Tensor or
    an iterable of Tensors. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the Tensors with zeros so that they have the same
    shape
    """
    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        tensors = [tensors]

    if isinstance(tensors, ImageList):
        return tensors
    elif isinstance(tensors, torch.Tensor):
        # single tensor shape can be inferred
        if tensors.dim() == 3:
            tensors = tensors[None]
        assert tensors.dim() == 4
        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        return ImageList(tensors, image_sizes, image_sizes)
    elif isinstance(tensors, (tuple, list)):
        if max_size is None:
            max_size = tuple(
                max(s) for s in zip(*[img.shape for img in tensors]))
        # TODO Ideally, just remove this and let me model handle arbitrary
        # input sizs
        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(tensors), ) + max_size
        batched_imgs = tensors[0].new(*batch_shape).zero_()  # + 114
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)

        image_sizes = [im.shape[-2:] for im in tensors]
        pad_sizes = [batched_imgs.shape[-2:] for im in batched_imgs]

        return ImageList(batched_imgs, image_sizes, pad_sizes)
    else:
        raise TypeError('Unsupported type for to_image_list: {}'.format(
            type(tensors)))
```

### damo/config/__init__.py

```python
#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

from .base import Config
```

### damo/config/paths_catalog.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
"""Centralized catalog of paths."""
import os


class DatasetCatalog(object):
    DATA_DIR = 'datasets'
    DATASETS = {
        'coco_2017_train': {
            'img_dir': 'coco/train2017',
            'ann_file': 'coco/annotations/instances_train2017.json'
        },
        'coco_2017_val': {
            'img_dir': 'coco/val2017',
            'ann_file': 'coco/annotations/instances_val2017.json'
        },
        'coco_2017_test_dev': {
            'img_dir': 'coco/test2017',
            'ann_file': 'coco/annotations/image_info_test-dev2017.json'
        },
        }

    @staticmethod
    def get(name):
        if 'coco' in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs['img_dir']),
                ann_file=os.path.join(data_dir, attrs['ann_file']),
            )
            return dict(
                factory='COCODataset',
                args=args,
            )
        else:
            raise RuntimeError('Only support coco format now!')
        return None
```

### damo/config/base.py

```python
#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import ast
import importlib
import os
import pprint
import sys
from abc import ABCMeta
from os.path import dirname, join

from easydict import EasyDict as easydict
from tabulate import tabulate

from .augmentations import test_aug, train_aug
from .paths_catalog import DatasetCatalog

miscs = easydict({
    'print_interval_iters': 50,    # print interval
    'output_dir': './workdirs',    # save dir
    'exp_name': os.path.split(os.path.realpath(__file__))[1].split('.')[0],
    'seed': 1234,                  # rand seed for initialize
    'eval_interval_epochs': 10,    # evaluation interval
    'ckpt_interval_epochs': 10,    # ckpt saving interval
    'num_workers': 4,
})

train = easydict({
    # ema
    'ema': True,                   # enable ema
    'ema_momentum': 0.9998,        # ema momentum
    'warmup_start_lr': 0,          # warmup start learning rate
    # scheduler
    'min_lr_ratio': 0.05,          # min lr ratio after closing augmentation
    'batch_size': 64,              # training batch size
    'total_epochs': 300,           # training total epochs
    'warmup_epochs': 5,            # warmup epochs
    'no_aug_epochs': 16,           # training epochs after closing augmentation
    'resume_path': None,           # ckpt path for resuming training
    'finetune_path': None,         # ckpt path for finetuning
    'augment': train_aug,          # augmentation config for training
    # optimizer
    'optimizer': {
        'momentum': 0.9,
        'name': "SGD",
        'weight_decay': 5e-4,
        'nesterov': True,
        'lr': 0.04,
        },
})

test = easydict({
    'augment': test_aug,           # augmentation config for testing
    'batch_size': 128,             # testing batch size
})

dataset = easydict({
    'paths_catalog': join(dirname(__file__), 'paths_catalog.py'),
    'train_ann': ('coco_2017_train', ),
    'val_ann': ('coco_2017_val', ),
    'data_dir': None,
    'aspect_ratio_grouping': False,
    'class_names': None,
})


class Config(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.model = easydict({'backbone': None, 'neck': None, 'head': None})
        self.train = train
        self.test = test
        self.dataset = dataset
        self.miscs = miscs

    def get_data(self, name):
        if 'coco' in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=join(data_dir, attrs['img_dir']),
                ann_file=join(data_dir, attrs['ann_file']),
            )
            return dict(
                factory='COCODataset',
                args=args,
            )
        else:
            raise RuntimeError('Only support coco format dataset now!')

    def __repr__(self):
        table_header = ['keys', 'values']
        exp_table = [(str(k), pprint.pformat(v, compact=True))
                     for k, v in vars(self).items() if not k.startswith('_')]
        return tabulate(exp_table, headers=table_header, tablefmt='fancy_grid')

    def merge(self, cfg_list):
        assert len(cfg_list) % 2 == 0
        for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
            # only update value with same key
            if hasattr(self, k):
                src_value = getattr(self, k)
                src_type = type(src_value)
                if src_value is not None and src_type != type(v):
                    try:
                        v = src_type(v)
                    except Exception:
                        v = ast.literal_eval(v)
                setattr(self, k, v)

    def read_structure(self, path):

        with open(path, 'r') as f:
            structure = f.read()

        return structure


def get_config_by_file(config_file):
    try:
        sys.path.append(os.path.dirname(config_file))
        current_config = importlib.import_module(
            os.path.basename(config_file).split('.')[0])
        exp = current_config.Config()
    except Exception:
        raise ImportError(
            "{} doesn't contains class named 'Config'".format(config_file))
    return exp


def parse_config(config_file):
    """
    get config object by file.
    Args:
        config_file (str): file path of config.
    """
    assert (config_file is not None), 'plz provide config file'
    if config_file is not None:
        return get_config_by_file(config_file)
```

### damo/config/augmentations.py

```python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
SADA = {
    'box_prob':
    0.3,
    'num_subpolicies':
    5,
    'scale_splits': [2048, 10240, 51200],
    'autoaug_params':
    (6, 9, 5, 3, 3, 4, 2, 4, 4, 4, 5, 2, 4, 1, 4, 2, 6, 4, 2, 2, 2, 6, 2, 2, 2,
     0, 5, 1, 3, 0, 8, 5, 2, 8, 7, 5, 1, 3, 3, 3),
}
Mosaic_Mixup = {
    'mosaic_prob': 1.0,
    'mosaic_scale': (0.1, 2.0),
    'mosaic_size': (640, 640),
    'mixup_prob': 1.0,
    'mixup_scale': (0.5, 1.5),
    'degrees': 10.0,
    'translate': 0.2,
    'shear': 2.0,
    'keep_ratio': False,
}

train_transform = {
    'image_mean': [0.0, 0.0, 0.0],
    'image_std': [1.0, 1.0, 1.0],
    'image_max_range': (640, 640),
    'flip_prob': 0.5,
    'keep_ratio': False,
    'autoaug_dict': SADA,
}
test_transform = {
    'image_mean': [0.0, 0.0, 0.0],
    'image_std': [1.0, 1.0, 1.0],
    'image_max_range': (640, 640),
    'flip_prob': 0.0,
    'keep_ratio': False,
}

train_aug = {
    'mosaic_mixup': Mosaic_Mixup,
    'transform': train_transform,
}

test_aug = {
    'transform': test_transform,
}
```

### damo/dataset/build.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import math

import torch.utils.data

from damo.utils import get_world_size

from . import datasets as D
from .collate_batch import BatchCollator
from .datasets import MosaicWrapper
from .samplers import DistributedSampler, IterationBasedBatchSampler
from .transforms import build_transforms


def build_dataset(cfg, ann_files, is_train=True, mosaic_mixup=None):

    if not isinstance(ann_files, (list, tuple)):
        raise RuntimeError(
            'datasets should be a list of strings, got {}'.format(ann_files))
    datasets = []
    for dataset_name in ann_files:
        # read data from config first
        data = cfg.get_data(dataset_name)
        factory = getattr(D, data['factory'])
        args = data['args']
        args['transforms'] = None
        args['class_names'] = cfg.dataset.class_names
        # make dataset from factory
        dataset = factory(**args)

        # mosaic wrapped
        if is_train and mosaic_mixup is not None:
            dataset = MosaicWrapper(dataset=dataset,
                                    img_size=mosaic_mixup.mosaic_size,
                                    mosaic_prob=mosaic_mixup.mosaic_prob,
                                    mixup_prob=mosaic_mixup.mixup_prob,
                                    transforms=None,
                                    degrees=mosaic_mixup.degrees,
                                    translate=mosaic_mixup.translate,
                                    shear=mosaic_mixup.shear,
                                    mosaic_scale=mosaic_mixup.mosaic_scale,
                                    mixup_scale=mosaic_mixup.mixup_scale,
                                    keep_ratio=mosaic_mixup.keep_ratio)

        datasets.append(dataset)

    return datasets


def make_data_sampler(dataset, shuffle):

    return DistributedSampler(dataset, shuffle=shuffle)


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info['height']) / float(img_info['width'])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_sampler(dataset,
                       sampler,
                       images_per_batch,
                       num_iters=None,
                       start_iter=0,
                       mosaic_warpper=False):
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler,
                                                          images_per_batch,
                                                          drop_last=False)
    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter, enable_mosaic=mosaic_warpper)
    return batch_sampler


def build_dataloader(datasets,
                     augment,
                     batch_size=128,
                     start_epoch=None,
                     total_epochs=None,
                     no_aug_epochs=0,
                     is_train=True,
                     num_workers=8,
                     size_div=32):

    num_gpus = get_world_size()
    assert (
            batch_size % num_gpus == 0
        ), 'training_imgs_per_batch ({}) must be divisible by the number ' \
        'of GPUs ({}) used.'.format(batch_size, num_gpus)
    images_per_gpu = batch_size // num_gpus

    if is_train:
        iters_per_epoch = math.ceil(len(datasets[0]) / batch_size)
        shuffle = True
        num_iters = total_epochs * iters_per_epoch
        start_iter = start_epoch * iters_per_epoch
    else:
        iters_per_epoch = math.ceil(len(datasets[0]) / batch_size)
        shuffle = False
        num_iters = None
        start_iter = 0

    transforms = augment.transform
    enable_mosaic_mixup = 'mosaic_mixup' in augment

    transforms = build_transforms(start_epoch, total_epochs, no_aug_epochs,
                                  iters_per_epoch, num_workers, batch_size,
                                  num_gpus, **transforms)

    for dataset in datasets:
        dataset._transforms = transforms
        if hasattr(dataset, '_dataset'):
            dataset._dataset._transforms = transforms

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle)
        batch_sampler = make_batch_sampler(dataset, sampler, images_per_gpu,
                                           num_iters, start_iter,
                                           enable_mosaic_mixup)
        collator = BatchCollator(size_div)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
    if is_train:
        assert len(
            data_loaders) == 1, 'multi-training set is not supported yet!'
        return data_loaders[0]
    return data_loaders
```

### damo/dataset/__init__.py

```python
#!/usr/bin/env python3
from .build import build_dataloader, build_dataset
```

### damo/dataset/collate_batch.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from damo.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """
    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return images, targets, img_ids


class TTACollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """
    def __call__(self, batch):
        return list(zip(*batch))
```

#### damo/dataset/datasets/coco.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import cv2
import numpy as np
import torch
from torchvision.datasets.coco import CocoDetection

from damo.structures.bounding_box import BoxList

cv2.setNumThreads(0)


class COCODataset(CocoDetection):
    def __init__(self, ann_file, root, transforms=None, class_names=None):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        assert (class_names is not None), 'plz provide class_names'

        self.contiguous_class2id = {
            class_name: i
            for i, class_name in enumerate(class_names)
        }
        self.contiguous_id2class = {
            i: class_name
            for i, class_name in enumerate(class_names)
        }

        categories = self.coco.dataset['categories']
        cat_names = [cat['name'] for cat in categories]
        cat_ids = [cat['id'] for cat in categories]
        self.ori_class2id = {
            class_name: i
            for class_name, i in zip(cat_names, cat_ids)
        }
        self.ori_id2class = {
            i: class_name
            for class_name, i in zip(cat_names, cat_ids)
        }

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, inp):
        if type(inp) is tuple:
            idx = inp[1]
        else:
            idx = inp
        img, anno = super(COCODataset, self).__getitem__(idx)
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')

        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        classes = torch.tensor(classes)
        target.add_field('labels', classes)


        target = target.clip_to_image(remove_empty=True)

        # PIL to numpy array
        img = np.asarray(img)  # rgb

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target, idx

    def pull_item(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')
        target = target.clip_to_image(remove_empty=True)

        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        obj_masks = []
        for obj in anno:
            obj_mask = []
            if 'segmentation' in obj:
                for mask in obj['segmentation']:
                    obj_mask += mask
                if len(obj_mask) > 0:
                    obj_masks.append(obj_mask)
        seg_masks = [
            np.array(obj_mask, dtype=np.float32).reshape(-1, 2)
            for obj_mask in obj_masks
        ]

        res = np.zeros((len(target.bbox), 5))
        for idx in range(len(target.bbox)):
            res[idx, 0:4] = target.bbox[idx]
            res[idx, 4] = classes[idx]

        img = np.asarray(img)  # rgb

        return img, res, seg_masks, idx

    def load_anno(self, idx):
        _, anno = super(COCODataset, self).__getitem__(idx)
        anno = [obj for obj in anno if obj['iscrowd'] == 0]
        classes = [obj['category_id'] for obj in anno]
        classes = [self.contiguous_class2id[self.ori_id2class[c]] 
                   for c in classes]

        return classes

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
```

#### damo/dataset/datasets/__init__.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .coco import COCODataset
from .mosaic_wrapper import MosaicWrapper

__all__ = [
    'COCODataset',
    'MosaicWrapper',
]
```

#### damo/dataset/datasets/mosaic_wrapper.py

```python
# Copyright (c) Megvii Inc. All rights reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import math
import random

import cv2
import numpy as np
import torch

from damo.structures.bounding_box import BoxList
from damo.utils import adjust_box_anns, get_rank


def xyn2xy(x, scale_w, scale_h, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = scale_w * x[:, 0] + padw  # top left x
    y[:, 1] = scale_h * x[:, 1] + padh  # top left y
    return y


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([
            np.interp(x, xp, s[:, i]) for i in range(2)
        ]).reshape(2, -1).T  # segment xy
    return segments


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint,
    # i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(),
                     y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            'Affine params should be either a sequence containing two values\
                          or single float values. Got {}'.format(value))


def box_candidates(box1,
                   box2,
                   wh_thr=2,
                   ar_thr=20,
                   area_thr=0.1,
                   eps=1e-16):  # box1(4,n), box2(4,n)
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 /
                                            (w1 * h1 + eps) > area_thr) & (
                                                ar < ar_thr)  # candidates


def get_transform_matrix(img_shape, new_shape, degrees, scale, shear,
                         translate):
    new_height, new_width = new_shape
    # Center
    C = np.eye(3)
    C[0, 2] = -img_shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img_shape[0] / 2  # y translation (pixels)
    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = get_aug_params(scale, center=1.0)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi /
                       180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi /
                       180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(
        0.5 - translate, 0.5 + translate) * new_width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 +
                             translate) * new_height  # y transla ion (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT
    return M, s


def random_affine(
        img,
        targets=(),
        segments=None,
        target_size=(640, 640),
        degrees=10,
        translate=0.1,
        scales=0.1,
        shear=10,
):
    M, scale = get_transform_matrix(img.shape[:2], target_size, degrees,
                                    scales, shear, translate)

    if (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img,
                             M[:2],
                             dsize=target_size,
                             borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if (n and len(segments)==0) or (len(segments) != len(targets)):
        new = np.zeros((n, 4))

        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate(
            (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, target_size[0])
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, target_size[1])

    else:
        segments = resample_segments(segments)  # upsample
        new = np.zeros((len(targets), 4))
        assert len(segments) <= len(targets)
        for i, segment in enumerate(segments):
            xy = np.ones((len(segment), 3))
            xy[:, :2] = segment
            xy = xy @ M.T  # transform
            xy = xy[:, :2]  # perspective rescale or affine
            # clip
            new[i] = segment2box(xy, target_size[0], target_size[1])

    # filter candidates
    i = box_candidates(box1=targets[:, 0:4].T * scale,
                       box2=new.T,
                       area_thr=0.1)
    targets = targets[i]
    targets[:, 0:4] = new[i]

    return img, targets


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h,
                          input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w,
                                     input_w * 2), min(input_h * 2,
                                                       yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class MosaicWrapper(torch.utils.data.dataset.Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""
    def __init__(self,
                 dataset,
                 img_size,
                 mosaic_prob=1.0,
                 mixup_prob=1.0,
                 transforms=None,
                 degrees=10.0,
                 translate=0.1,
                 mosaic_scale=(0.1, 2.0),
                 mixup_scale=(0.5, 1.5),
                 shear=2.0,
                 keep_ratio=True,
                 *args):
        super().__init__()
        self._dataset = dataset
        self.input_dim = img_size
        self._transforms = transforms
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.mixup_scale = mixup_scale
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.keep_ratio = keep_ratio
        self.local_rank = get_rank()

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, inp):
        if type(inp) is tuple:
            enable_mosaic_mixup = inp[0]
            idx = inp[1]
        else:
            enable_mosaic_mixup = False
            idx = inp
        img, labels, segments, img_id = self._dataset.pull_item(idx)

        if enable_mosaic_mixup:
            if random.random() < self.mosaic_prob:
                mosaic_labels = []
                mosaic_segments = []
                input_h, input_w = self.input_dim[0], self.input_dim[1]

                yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
                xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

                # 3 additional image indices
                indices = [idx] + [
                    random.randint(0,
                                   len(self._dataset) - 1) for _ in range(3)
                ]

                for i_mosaic, index in enumerate(indices):
                    img, _labels, _segments, img_id = self._dataset.pull_item(
                        index)
                    h0, w0 = img.shape[:2]  # orig hw
                    if not self.keep_ratio:
                        scale_h, scale_w = 1. * input_h / h0, 1. * input_w / w0
                    else:
                        scale_h = min(1. * input_h / h0, 1. * input_w / w0)
                        scale_w = scale_h

                    img = cv2.resize(img, (int(w0 * scale_w), int(h0 * scale_h)),
                                     interpolation=cv2.INTER_LINEAR)
                    # generate output mosaic image
                    (h, w, c) = img.shape[:3]
                    if i_mosaic == 0:
                        mosaic_img = np.full((input_h * 2, input_w * 2, c),
                                             114,
                                             dtype=np.uint8)  # pad 114

                    (l_x1, l_y1, l_x2,
                     l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                         mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w)

                    mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2,
                                                           s_x1:s_x2]
                    padw, padh = l_x1 - s_x1, l_y1 - s_y1

                    labels = _labels.copy()
                    # Normalized xywh to pixel xyxy format
                    if _labels.size > 0:
                        labels[:, 0] = scale_w * _labels[:, 0] + padw
                        labels[:, 1] = scale_h * _labels[:, 1] + padh
                        labels[:, 2] = scale_w * _labels[:, 2] + padw
                        labels[:, 3] = scale_h * _labels[:, 3] + padh
                    segments = [
                        xyn2xy(x, scale_w, scale_h, padw, padh) for x in _segments
                    ]
                    mosaic_segments.extend(segments)
                    mosaic_labels.append(labels)

                if len(mosaic_labels):
                    mosaic_labels = np.concatenate(mosaic_labels, 0)
                    np.clip(mosaic_labels[:, 0],
                            0,
                            2 * input_w,
                            out=mosaic_labels[:, 0])
                    np.clip(mosaic_labels[:, 1],
                            0,
                            2 * input_h,
                            out=mosaic_labels[:, 1])
                    np.clip(mosaic_labels[:, 2],
                            0,
                            2 * input_w,
                            out=mosaic_labels[:, 2])
                    np.clip(mosaic_labels[:, 3],
                            0,
                            2 * input_h,
                            out=mosaic_labels[:, 3])

                if len(mosaic_segments):
                    assert input_w == input_h
                    for x in mosaic_segments:
                        np.clip(x, 0, 2 * input_w,
                                out=x)  # clip when using random_perspective()

                img, labels = random_affine(
                    mosaic_img,
                    mosaic_labels,
                    mosaic_segments,
                    target_size=(input_w, input_h),
                    degrees=self.degrees,
                    translate=self.translate,
                    scales=self.scale,
                    shear=self.shear,
                )

            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            if (not len(labels) == 0 and random.random() < self.mixup_prob):
                img, labels = self.mixup(img, labels, self.input_dim)

            # transfer labels to BoxList
            h_tmp, w_tmp = img.shape[:2]
            boxes = np.array([label[:4] for label in labels])
            boxes = torch.as_tensor(boxes).reshape(-1, 4)
            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            valid_idx = areas > 4

            target = BoxList(boxes[valid_idx], (w_tmp, h_tmp), mode='xyxy')

            classes = np.array([label[4] for label in labels])
            classes = torch.tensor(classes)[valid_idx]
            target.add_field('labels', classes.long())

            if self._transforms is not None:
                img, target = self._transforms(img, target)

            # -----------------------------------------------------------------
            # img_info and img_id are not used for training.
            # They are also hard to be specified on a mosaic image.
            # -----------------------------------------------------------------
            return img, target, img_id

        else:
            return self._dataset.__getitem__(idx)

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        img, cp_labels, _, _ = self._dataset.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3),
                             dtype=np.uint8) * 114  # pad 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114  # pad 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0],
                             input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio),
             int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        cp_img[:int(img.shape[0] *
                    cp_scale_ratio), :int(img.shape[1] *
                                          cp_scale_ratio)] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor),
             int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3),
            dtype=np.uint8)
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[y_offset:y_offset + target_h,
                                        x_offset:x_offset + target_w]

        cp_bboxes_origin_np = adjust_box_anns(cp_labels[:, :4].copy(),
                                              cp_scale_ratio, 0, 0, origin_w,
                                              origin_h)
        if FLIP:
            cp_bboxes_origin_np[:,
                                0::2] = (origin_w -
                                         cp_bboxes_origin_np[:, 0::2][:, ::-1])
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w)
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h)

        cls_labels = cp_labels[:, 4:5].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, cls_labels))
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(
            np.float32)

        return origin_img.astype(np.uint8), origin_labels

    def get_img_info(self, index):
        return self._dataset.get_img_info(index)
```

##### damo/dataset/datasets/evaluation/__init__.py

```python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

from damo.dataset import datasets

from .coco import coco_evaluation


def evaluate(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(dataset=dataset,
                predictions=predictions,
                output_folder=output_folder,
                **kwargs)
    if isinstance(dataset, datasets.COCODataset):
        return coco_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError(
            'Unsupported dataset type {}.'.format(dataset_name))
```

###### damo/dataset/datasets/evaluation/coco/__init__.py

```python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

from .coco_eval import do_coco_evaluation


def coco_evaluation(
    dataset,
    predictions,
    output_folder,
    box_only,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    return do_coco_evaluation(
        dataset=dataset,
        predictions=predictions,
        box_only=box_only,
        output_folder=output_folder,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )
```

###### damo/dataset/datasets/evaluation/coco/coco_eval.py

```python
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import tempfile
from collections import OrderedDict

import torch
from loguru import logger

from damo.structures.bounding_box import BoxList
from damo.structures.boxlist_ops import boxlist_iou


def do_coco_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):

    if box_only:
        logger.info('Evaluating bbox proposals')
        areas = {'all': '', 'small': 's', 'medium': 'm', 'large': 'l'}
        res = COCOResults('box_proposal')
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = evaluate_box_proposals(predictions,
                                               dataset,
                                               area=area,
                                               limit=limit)
                key = 'AR{}@{:d}'.format(suffix, limit)
                res.results['box_proposal'][key] = stats['ar'].item()
        logger.info(res)
        check_expected_results(res, expected_results,
                               expected_results_sigma_tol)
        if output_folder:
            torch.save(res, os.path.join(output_folder, 'box_proposals.pth'))
        return
    logger.info('Preparing results for COCO format')
    coco_results = {}
    if 'bbox' in iou_types:
        logger.info('Preparing bbox results')
        coco_results['bbox'] = prepare_for_coco_detection(predictions, dataset)

    results = COCOResults(*iou_types)
    logger.info('Evaluating predictions')
    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            if output_folder:
                file_path = os.path.join(output_folder, iou_type + '.json')
            res = evaluate_predictions_on_coco(dataset.coco,
                                               coco_results[iou_type],
                                               file_path, iou_type)
            results.update(res)
    logger.info(results)
    check_expected_results(results, expected_results,
                           expected_results_sigma_tol)
    if output_folder:
        torch.save(results, os.path.join(output_folder, 'coco_results.pth'))
    return results, coco_results


def prepare_for_coco_detection(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
        image_width = img_info['width']
        image_height = img_info['height']
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert('xywh')

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field('scores').tolist()
        labels = prediction.get_field('labels').tolist()

        mapped_labels = [
            dataset.ori_class2id[dataset.contiguous_id2class[i]] for i in labels
        ]
        coco_results.extend([{
            'image_id': original_id,
            'category_id': mapped_labels[k],
            'bbox': box,
            'score': scores[k],
        } for k, box in enumerate(boxes)])
    return coco_results


# inspired from Detectron
def evaluate_box_proposals(predictions,
                           dataset,
                           thresholds=None,
                           area='all',
                           limit=None):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code.
    However, it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        'all': 0,
        'small': 1,
        'medium': 2,
        'large': 3,
        '96-128': 4,
        '128-256': 5,
        '256-512': 6,
        '512-inf': 7,
    }
    area_ranges = [
        [0**2, 1e5**2],  # all
        [0**2, 32**2],  # small
        [32**2, 96**2],  # medium
        [96**2, 1e5**2],  # large
        [96**2, 128**2],  # 96-128
        [128**2, 256**2],  # 128-256
        [256**2, 512**2],  # 256-512
        [512**2, 1e5**2],
    ]  # 512-inf
    assert area in areas, 'Unknown area range: {}'.format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]

        img_info = dataset.get_img_info(image_id)
        image_width = img_info['width']
        image_height = img_info['height']
        prediction = prediction.resize((image_width, image_height))
        # prediction = prediction.resize((image_height, image_width))

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = prediction.get_field('objectness').sort(descending=True)[1]
        prediction = prediction[inds]

        ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
        anno = dataset.coco.loadAnns(ann_ids)
        gt_boxes = [obj['bbox'] for obj in anno if obj['iscrowd'] == 0]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(
            -1, 4)  # guard against no boxes
        gt_boxes = BoxList(gt_boxes, (image_width, image_height),
                           mode='xywh').convert('xyxy')
        gt_areas = torch.as_tensor(
            [obj['area'] for obj in anno if obj['iscrowd'] == 0])

        if len(gt_boxes) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <=
                                                       area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if len(prediction) == 0:
            continue

        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]

        overlaps = boxlist_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(prediction), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        'ar': ar,
        'recalls': recalls,
        'thresholds': thresholds,
        'gt_overlaps': gt_overlaps,
        'num_pos': num_pos,
    }


def evaluate_predictions_on_coco(coco_gt,
                                 coco_results,
                                 json_result_file,
                                 iou_type='bbox'):
    import json

    with open(json_result_file, 'w') as f:
        json.dump(coco_results, f)

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_dt = coco_gt.loadRes(
        str(json_result_file)) if coco_results else COCO()

    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # compute_thresholds_for_classes(coco_eval)

    return coco_eval


def compute_thresholds_for_classes(coco_eval):
    '''
    The function is used to compute the thresholds corresponding to best
    f-measure. The resulting thresholds are used in fcos_demo.py.
    '''
    import numpy as np
    # dimension of precision: [TxRxKxAxM]
    precision = coco_eval.eval['precision']
    # we compute thresholds with IOU being 0.5
    precision = precision[0, :, :, 0, -1]
    scores = coco_eval.eval['scores']
    scores = scores[0, :, :, 0, -1]

    recall = np.linspace(0, 1, num=precision.shape[0])
    recall = recall[:, None]

    f_measure = (2 * precision * recall) / (np.maximum(precision + recall,
                                                       1e-6))
    max_f_measure = f_measure.max(axis=0)
    max_f_measure_inds = f_measure.argmax(axis=0)
    scores = scores[max_f_measure_inds, range(len(max_f_measure_inds))]

    print('Maximum f-measures for classes:')
    print(list(max_f_measure))
    print('Score thresholds for classes (used in demos for visualization):')
    print(list(scores))


class COCOResults(object):
    METRICS = {
        'bbox': ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl'],
        'segm': ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl'],
        'box_proposal': [
            'AR@100',
            'ARs@100',
            'ARm@100',
            'ARl@100',
            'AR@1000',
            'ARs@1000',
            'ARm@1000',
            'ARl@1000',
        ],
        'keypoints': ['AP', 'AP50', 'AP75', 'APm', 'APl'],
    }

    def __init__(self, *iou_types):
        allowed_types = ('box_proposal', 'bbox', 'segm', 'keypoints')
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict([
                (metric, -1) for metric in COCOResults.METRICS[iou_type]
            ])
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = ('{} > {} sanity check (actual vs. expected): '
               '{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})'
               ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = 'FAIL: ' + msg
            logger.error(msg)
        else:
            msg = 'PASS: ' + msg
            logger.info(msg)
```

#### damo/dataset/transforms/build.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
from damo.augmentations.scale_aware_aug import SA_Aug

from . import transforms as T


def build_transforms(start_epoch,
                     total_epochs,
                     no_aug_epochs,
                     iters_per_epoch,
                     num_workers,
                     batch_size,
                     num_gpus,
                     image_max_range=(640, 640),
                     flip_prob=0.5,
                     image_mean=[0, 0, 0],
                     image_std=[1., 1., 1.],
                     autoaug_dict=None,
                     keep_ratio=True):

    transform = [
        T.Resize(image_max_range, keep_ratio=keep_ratio),
        T.RandomHorizontalFlip(flip_prob),
        T.ToTensor(),
        T.Normalize(mean=image_mean, std=image_std),
    ]

    if autoaug_dict is not None:
        transform += [
            SA_Aug(iters_per_epoch, start_epoch, total_epochs, no_aug_epochs,
                   batch_size, num_gpus, num_workers, autoaug_dict)
        ]

    transform = T.Compose(transform)

    return transform
```

#### damo/dataset/transforms/transforms.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import random

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Resize(object):
    def __init__(self, max_range, target_size=None, keep_ratio=True):
        if not isinstance(max_range, (list, tuple)):
            max_range = (max_range, )
        self.max_range = max_range
        self.target_size = target_size
        self.keep_ratio = keep_ratio

    def get_size_ratio(self, image_size):
        if self.target_size is None:
            target_size = random.choice(self.max_range)
            t_w, t_h = target_size, target_size
        else:
            t_w, t_h = self.target_size[1], self.target_size[0]
        if not self.keep_ratio:
            return t_w, t_h
        w, h = image_size
        r = min(t_w / w, t_h / h)
        o_w, o_h = int(w * r), int(h * r)
        return (o_w, o_h)

    def __call__(self, image, target=None):
        h, w = image.shape[:2]
        size = self.get_size_ratio((w, h))

        image = cv2.resize(image, size,
                           interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        image = image.transpose((2, 0, 1))
        image = np.ascontiguousarray(image, dtype=np.float32)
        if isinstance(target, list):
            target = [t.resize(size) for t in target]
        elif target is None:
            return image, target
        else:
            target = target.resize(size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image[:, :, ::-1]
            image = np.ascontiguousarray(image, dtype=np.float32)
            if target is not None:
                target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return torch.from_numpy(image), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
```

#### damo/dataset/transforms/__init__.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
from .build import build_transforms
from .transforms import (Compose, Normalize, RandomHorizontalFlip, Resize,
                         ToTensor)
```

#### damo/dataset/transforms/tta_aug.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import torch

from damo.dataset.transforms import transforms as T
from damo.structures.bounding_box import BoxList
from damo.structures.image_list import to_image_list
from damo.utils.boxes import filter_results


def im_detect_bbox_aug(model, images, device, config):
    # Collect detections computed under different transformations
    boxlists_ts = []
    for _ in range(len(images)):
        boxlists_ts.append([])

    def add_preds_t(boxlists_t):
        for i, boxlist_t in enumerate(boxlists_t):
            if len(boxlists_ts[i]) == 0:
                # The first one is identity transform,
                # no need to resize the boxlist
                boxlists_ts[i].append(boxlist_t)
            else:
                # Resize the boxlist as the first one
                boxlists_ts[i].append(boxlist_t.resize(boxlists_ts[i][0].size))

    # Compute detections for the original image (identity transform)
    boxlists_i = im_detect_bbox(model, images, config.testing.input_min_size,
                                config.testing.input_max_size, device, config)
    add_preds_t(boxlists_i)

    # Perform detection on the horizontally flipped image
    if config.testing.augmentation.hflip:
        boxlists_hf = im_detect_bbox_hflip(model, images,
                                           config.testing.input_min_size,
                                           config.testing.input_max_size,
                                           device, config)
        add_preds_t(boxlists_hf)

    # Compute detections at different scales
    for scale in config.testing.augmentation.scales:
        max_size = config.testing.augmentation.scales_max_size
        boxlists_scl = im_detect_bbox_scale(model, images, scale, max_size,
                                            device, config)
        add_preds_t(boxlists_scl)

        if config.testing.augmentation.scales_hflip:
            boxlists_scl_hf = im_detect_bbox_scale(model,
                                                   images,
                                                   scale,
                                                   max_size,
                                                   device,
                                                   config,
                                                   hflip=True)
            add_preds_t(boxlists_scl_hf)

    # Merge boxlists detected by different bbox aug params
    boxlists = []
    for i, boxlist_ts in enumerate(boxlists_ts):
        bbox = torch.cat([boxlist_t.bbox for boxlist_t in boxlist_ts])
        scores = torch.cat(
            [boxlist_t.get_field('scores') for boxlist_t in boxlist_ts])
        labels = torch.cat(
            [boxlist_t.get_field('labels') for boxlist_t in boxlist_ts])
        boxlist = BoxList(bbox, boxlist_ts[0].size, boxlist_ts[0].mode)
        boxlist.add_field('scores', scores)
        boxlist.add_field('labels', labels)
        boxlists.append(boxlist)

    # Apply NMS and limit the final detections
    results = []
    for boxlist in boxlists:
        results.append(
            filter_results(boxlist, config.model.head.num_classes,
                           config.testing.augmentation.nms_thres))

    return results


def im_detect_bbox(model, images, target_scale, target_max_size, device,
                   config):
    """
    Performs bbox detection on the original image.
    """
    transform = T.Compose([
        T.Resize(target_scale, target_max_size),
        T.ToTensor(),
        T.Normalize(mean=config.dataset.input_pixel_mean,
                    std=config.dataset.input_pixel_std,
                    to_bgr255=config.dataset.input_to_bgr255)
    ])
    images = [transform(image)[0] for image in images]
    images = to_image_list(images, config.dataset.size_divisibility)
    return model(images.to(device))


def im_detect_bbox_hflip(model, images, target_scale, target_max_size, device,
                         config):
    """
    Performs bbox detection on the horizontally flipped image.
    Function signature is the same as for im_detect_bbox.
    """
    transform = T.Compose([
        T.Resize(target_scale, target_max_size),
        T.RandomHorizontalFlip(1.0),
        T.ToTensor(),
        T.Normalize(mean=config.dataset.input_pixel_mean,
                    std=config.dataset.input_pixel_std,
                    to_bgr255=config.dataset.input_to_bgr255)
    ])
    images = [transform(image)[0] for image in images]
    images = to_image_list(images, config.dataset.size_divisibility)
    boxlists = model(images.to(device))

    # Invert the detections computed on the flipped image
    boxlists_inv = [boxlist.transpose(0) for boxlist in boxlists]
    return boxlists_inv


def im_detect_bbox_scale(model,
                         images,
                         target_scale,
                         target_max_size,
                         device,
                         config,
                         hflip=False):
    """
    Computes bbox detections at the given scale.
    Returns predictions in the scaled image space.
    """
    if hflip:
        boxlists_scl = im_detect_bbox_hflip(model, images, target_scale,
                                            target_max_size, device, config)
    else:
        boxlists_scl = im_detect_bbox(model, images, target_scale,
                                      target_max_size, device, config)
    return boxlists_scl
```

#### damo/dataset/transforms/transforms_keepratio.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import random

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Resize(object):
    def __init__(self, max_range, target_size=None, keep_ratio=True):
        if not isinstance(max_range, (list, tuple)):
            max_range = (max_range, )
        self.max_range = max_range
        self.target_size = target_size
        self.keep_ratio = keep_ratio

    def get_size_ratio(self, image_size):
        if self.target_size is None:
            target_size = random.choice(self.max_range)
            t_w, t_h = target_size, target_size
        else:
            t_w, t_h = self.target_size[1], self.target_size[0]
        if not self.keep_ratio:
            return t_w, t_h
        w, h = image_size
        r = min(t_w / w, t_h / h)
        o_w, o_h = int(w * r), int(h * r)
        return (o_w, o_h)

    def __call__(self, image, target=None):
        h, w = image.shape[:2]
        size = self.get_size_ratio((w, h))

        image = cv2.resize(image, size,
                           interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        image = image.transpose((2, 0, 1))
        image = np.ascontiguousarray(image, dtype=np.float32)
        if isinstance(target, list):
            target = [t.resize(size) for t in target]
        elif target is None:
            return image, target
        else:
            target = target.resize(size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image[:, :, ::-1]
            image = np.ascontiguousarray(image, dtype=np.float32)
            if target is not None:
                target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return torch.from_numpy(image), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
```

#### damo/dataset/samplers/__init__.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .distributed import DistributedSampler
from .grouped_batch_sampler import GroupedBatchSampler
from .iteration_based_batch_sampler import IterationBasedBatchSampler

__all__ = [
    'DistributedSampler', 'GroupedBatchSampler', 'IterationBasedBatchSampler'
]
```

#### damo/dataset/samplers/grouped_batch_sampler.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import itertools

import torch
from torch.utils.data.sampler import BatchSampler, Sampler


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces elements from the same group appear in groups of batch_size.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_uneven (bool): If ``True``, the sampler will drop the batches
            whose size is less than ``batch_size``
    """
    def __init__(self, sampler, group_ids, batch_size, drop_uneven=False):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                'sampler should be an instance of '
                'torch.utils.data.Sampler, but got sampler={}'.format(sampler))
        self.sampler = sampler
        self.group_ids = torch.as_tensor(group_ids)
        assert self.group_ids.dim() == 1
        self.batch_size = batch_size
        self.drop_uneven = drop_uneven

        self.groups = torch.unique(self.group_ids).sort(0)[0]

        self._can_reuse_batches = False

    def _prepare_batches(self):
        dataset_size = len(self.group_ids)
        # get the sampled indices from the sampler
        sampled_ids = torch.as_tensor(list(self.sampler))
        # potentially not all elements of the dataset were sampled
        # by the sampler (e.g., DistributedSampler).
        # construct a tensor which contains -1 if the element was
        # not sampled, and a non-negative number indicating the
        # order where the element was sampled.
        # for example. if sampled_ids = [3, 1] and dataset_size = 5,
        # the order is [-1, 1, -1, 0, -1]
        order = torch.full((dataset_size, ), -1, dtype=torch.int64)
        order[sampled_ids] = torch.arange(len(sampled_ids))

        # get a mask with the elements that were sampled
        mask = order >= 0

        # find the elements that belong to each individual cluster
        clusters = [(self.group_ids == i) & mask for i in self.groups]
        # get relative order of the elements inside each cluster
        # that follows the order from the sampler
        relative_order = [order[cluster] for cluster in clusters]
        # with the relative order, find the absolute order in the
        # sampled space
        permutation_ids = [s[s.sort()[1]] for s in relative_order]
        # permute each cluster so that they follow the order from
        # the sampler
        permuted_clusters = [sampled_ids[idx] for idx in permutation_ids]

        # splits each cluster in batch_size, and merge as a list of tensors
        splits = [c.split(self.batch_size) for c in permuted_clusters]
        merged = tuple(itertools.chain.from_iterable(splits))

        # now each batch internally has the right order, but
        # they are grouped by clusters. Find the permutation between
        # different batches that brings them as close as possible to
        # the order that we have in the sampler. For that, we will consider the
        # ordering as coming from the first element of each batch, and sort
        # correspondingly
        first_element_of_batch = [t[0].item() for t in merged]
        # get and inverse mapping from sampled indices and the position where
        # they occur (as returned by the sampler)
        inv_sampled_ids_map = {
            v: k
            for k, v in enumerate(sampled_ids.tolist())
        }
        # from the first element in each batch, get a relative ordering
        first_index_of_batch = torch.as_tensor(
            [inv_sampled_ids_map[s] for s in first_element_of_batch])

        # permute the batches so that they approximately follow the order
        # from the sampler
        permutation_order = first_index_of_batch.sort(0)[1].tolist()
        # finally, permute the batches
        batches = [merged[i].tolist() for i in permutation_order]

        if self.drop_uneven:
            kept = []
            for batch in batches:
                if len(batch) == self.batch_size:
                    kept.append(batch)
            batches = kept
        return batches

    def __iter__(self):
        if self._can_reuse_batches:
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()
        self._batches = batches
        return iter(batches)

    def __len__(self):
        if not hasattr(self, '_batches'):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True
        return len(self._batches)
```

#### damo/dataset/samplers/distributed.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Code is copy-pasted exactly as in torch.utils.data.distributed.
# FIXME remove this once c10d fixes the bug it has
import math

import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
```

#### damo/dataset/samplers/iteration_based_batch_sampler.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
from torch.utils.data.sampler import BatchSampler


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """
    def __init__(self,
                 batch_sampler,
                 num_iterations,
                 start_iter=0,
                 enable_mosaic=False):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter
        self.enable_mosaic = enable_mosaic

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, 'set_epoch'):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield [(self.enable_mosaic, idx) for idx in batch]

    def __len__(self):
        return self.num_iterations

    def set_mosaic(self, enable_mosaic):
        self.enable_mosaic = enable_mosaic
```

### damo/utils/model_utils.py

```python
#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import time
from copy import deepcopy

import torch
import torch.nn as nn
from thop import profile

__all__ = [
    'fuse_conv_and_bn',
    'fuse_model',
    'get_model_info',
    'replace_module',
    'make_divisible'
]

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def get_latency(model, inp, iters=500, warmup=2):

    start = time.time()
    for i in range(iters):
        out = model(inp)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if i <= warmup:
            start = time.time()
    latency = (time.time() - start) / (iters - warmup)

    return out, latency


def get_model_info(model, tsize):
    stride = 640
    model = model.eval()
    backbone = model.backbone
    neck = model.neck
    head = model.head
    h, w = tsize
    img = torch.randn((1, 3, stride, stride),
                      device=next(model.parameters()).device)

    bf, bp = profile(deepcopy(backbone), inputs=(img, ), verbose=False)
    bo, bl = get_latency(backbone, img, iters=10)

    nf, np = profile(deepcopy(neck), inputs=(bo, ), verbose=False)
    no, nl = get_latency(neck, bo, iters=10)

    hf, hp = profile(deepcopy(head), inputs=(no, ), verbose=False)
    ho, hl = get_latency(head, no, iters=10)

    _, total_latency = get_latency(model, img)
    total_flops = 0
    total_params = 0
    info = ''
    for name, flops, params, latency in zip(('backbone', 'neck', 'head'),
                                            (bf, nf, hf), (bp, np, hp),
                                            (bl, nl, hl)):
        params /= 1e6
        flops /= 1e9
        flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
        total_flops += flops
        total_params += params
        info += f"{name}'s params(M): {params:.2f}, " + \
                f'flops(G): {flops:.2f}, latency(ms): {latency*1000:.3f}\n'
    info += f'total latency(ms): {total_latency*1000:.3f}, ' + \
            f'total flops(G): {total_flops:.2f}, ' + f'total params(M): {total_params:.2f}\n'
    return info


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = (nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True,
    ).requires_grad_(False).to(conv.weight.device))

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (torch.zeros(conv.weight.size(0), device=conv.weight.device)
              if conv.bias is None else conv.bias)
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(
        torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def fuse_model(model):
    from damo.base_models.core.ops import ConvBNAct
    from damo.base_models.backbones.tinynas_res import ConvKXBN

    for m in model.modules():
        if type(m) is ConvBNAct and hasattr(m, 'bn'):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, 'bn')  # remove batchnorm
            m.forward = m.fuseforward  # update forward
        elif type(m) is ConvKXBN and hasattr(m, 'bn1'):
            m.conv1 = fuse_conv_and_bn(m.conv1, m.bn1)  # update conv
            delattr(m, 'bn1')  # remove batchnorm
            m.forward = m.fuseforward  # update forward

    return model


def replace_module(module,
                   replaced_module_type,
                   new_module_type,
                   replace_func=None):
    """
    Replace given type in module to a new type. mostly used in deploy.

    Args:
        module (nn.Module): model to apply replace operation.
        replaced_module_type (Type): module type to be replaced.
        new_module_type (Type)
        replace_func (function): python function to describe replace logic.
                                 Defalut value None.

    Returns:
        model (nn.Module): module that already been replaced.
    """
    def default_replace_func(replaced_module_type, new_module_type):
        return new_module_type()

    if replace_func is None:
        replace_func = default_replace_func

    model = module
    if isinstance(module, replaced_module_type):
        model = replace_func(replaced_module_type, new_module_type)
    else:  # recurrsively replace
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type,
                                       new_module_type)
            if new_child is not child:  # child is already replaced
                model.add_module(name, new_child)

    return model
```

### damo/utils/checkpoint.py

```python
#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.
import os
import shutil

import torch
from loguru import logger


def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            logger.warning('{} is not in the ckpt. \
                 Please double check and see if this is desired.'.format(
                key_model))
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            logger.warning('Shape of {} in checkpoint is {}, \
                 while shape of {} in model is {}.'.format(
                key_model, v_ckpt.shape, key_model, v.shape))
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model


def save_checkpoint(state, is_best, save_dir, model_name=''):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + '_ckpt.pth')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, 'best_ckpt.pth')
        shutil.copyfile(filename, best_filename)
```

### damo/utils/timer.py

```python
# Copyright (c) Facebook, Inc. and its affiliates.

import datetime
import time


class Timer(object):
    def __init__(self):
        self.reset()

    @property
    def average_time(self):
        return self.total_time / self.calls if self.calls > 0 else 0.0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.add(time.time() - self.start_time)
        if average:
            return self.average_time
        else:
            return self.diff

    def add(self, time_diff):
        self.diff = time_diff
        self.total_time += self.diff
        self.calls += 1

    def reset(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0

    def avg_time_str(self):
        time_str = str(datetime.timedelta(seconds=self.average_time))
        return time_str


def get_time_str(time_diff):
    time_str = str(datetime.timedelta(seconds=time_diff))
    return time_str
```

### damo/utils/__init__.py

```python
#!/usr/bin/env python3

from .boxes import *
from .checkpoint import load_ckpt, save_checkpoint
from .dist import *
from .imports import *
from .logger import setup_logger
from .metric import *
from .model_utils import *
from .visualize import *
```

### damo/utils/boxes.py

```python
#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Megvii Inc. All rights reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import numpy as np
import torch
import torchvision

from damo.structures.bounding_box import BoxList

__all__ = [
    'filter_box',
    'postprocess',
    'bboxes_iou',
    'matrix_iou',
    'adjust_box_anns',
    'xyxy2xywh',
    'xyxy2cxcywh',
]


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   iou_thr,
                   max_num=100,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1)
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0),
                                              num_classes, 4)
    scores = multi_scores
    # filter out boxes with low scores
    valid_mask = scores > score_thr  # 1000 * 80 bool

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # (TODO): as ONNX does not support repeat now,
    # we have to use this ugly code
    # bboxes -> 1000, 4
    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                    -1)).view(-1, 4)  # mask->  1000*80*4, 80000*4
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        scores = multi_bboxes.new_zeros((0, ))

        return bboxes, scores, labels

    keep = torchvision.ops.batched_nms(bboxes, scores, labels, iou_thr)

    if max_num > 0:
        keep = keep[:max_num]

    return bboxes[keep], scores[keep], labels[keep]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def filter_results(boxlist, num_classes, nms_thre):
    boxes = boxlist.bbox
    scores = boxlist.get_field('scores')
    cls = boxlist.get_field('labels')
    nms_out_index = torchvision.ops.batched_nms(
        boxes,
        scores,
        cls,
        nms_thre,
    )
    boxlist = boxlist[nms_out_index]

    return boxlist


def postprocess(cls_scores,
                bbox_preds,
                num_classes,
                conf_thre=0.7,
                nms_thre=0.45,
                imgs=None):
    batch_size = bbox_preds.size(0)
    output = [None for _ in range(batch_size)]
    for i in range(batch_size):
        # If none are remaining => process next image
        if not bbox_preds[i].size(0):
            continue
        detections, scores, labels = multiclass_nms(bbox_preds[i],
                                                    cls_scores[i], conf_thre,
                                                    nms_thre, 500)
        detections = torch.cat((detections, torch.ones_like(
            scores[:, None]), scores[:, None], labels[:, None]),
                               dim=1)

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    # transfer to BoxList
    for i in range(len(output)):
        res = output[i]
        if res is None or imgs is None:
            boxlist = BoxList(torch.zeros(0, 4), (0, 0), mode='xyxy')
            boxlist.add_field('objectness', 0)
            boxlist.add_field('scores', 0)
            boxlist.add_field('labels', -1)

        else:
            img_h, img_w = imgs.image_sizes[i]
            boxlist = BoxList(res[:, :4], (img_w, img_h), mode='xyxy')
            boxlist.add_field('objectness', res[:, 4])
            boxlist.add_field('scores', res[:, 5])
            boxlist.add_field('labels', res[:, 6])
        output[i] = boxlist

    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes
```

### damo/utils/visualize.py

```python
#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ['vis']


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255,
                                                                      255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(img, (x0, y0 + 1),
                      (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                      txt_bk_color, -1)
        cv2.putText(img,
                    text, (x0, y0 + txt_size[1]),
                    font,
                    0.4,
                    txt_color,
                    thickness=1)

    return img


_COLORS = np.array([
    0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494,
    0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078,
    0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
    1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000,
    0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667,
    0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
    0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000,
    1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
    0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
    0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667,
    0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333,
    0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
    0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333,
    0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000,
    1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000,
    1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.333,
    0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000,
    0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000,
    0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000,
    1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000,
    0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000,
    0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286, 0.286, 0.286, 0.429,
    0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714, 0.714, 0.857, 0.857,
    0.857, 0.000, 0.447, 0.741, 0.314, 0.717, 0.741, 0.50, 0.5, 0
]).astype(np.float32).reshape(-1, 3)
```

### damo/utils/logger.py

```python
#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.
import inspect
import os
import sys
import datetime

from loguru import logger


def get_caller_name(depth=0):
    """
    Args:
        depth (int): Depth of caller conext, use 0 for caller depth.
        Default value: 0.

    Returns:
        str: module name of the caller
    """
    # the following logic is a little bit faster than inspect.stack() logic
    frame = inspect.currentframe().f_back
    for _ in range(depth):
        frame = frame.f_back

    return frame.f_globals['__name__']


class StreamToLoguru:
    """
    stream object that redirects writes to a logger instance.
    """
    def __init__(self, level='INFO', caller_names=('apex', 'pycocotools')):
        """
        Args:
            level(str): log level string of loguru. Default value: "INFO".
            caller_names(tuple): caller names of redirected module.
                Default value: (apex, pycocotools).
        """
        self.level = level
        self.linebuf = ''
        self.caller_names = caller_names

    def write(self, buf):
        full_name = get_caller_name(depth=1)
        module_name = full_name.rsplit('.', maxsplit=-1)[0]
        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines():
                # use caller level log
                logger.opt(depth=2).log(self.level, line.rstrip())
        else:
            sys.__stdout__.write(buf)

    def flush(self):
        pass


def redirect_sys_output(log_level='INFO'):
    redirect_logger = StreamToLoguru(log_level)
    sys.stderr = redirect_logger
    sys.stdout = redirect_logger


def setup_logger(save_dir, distributed_rank=0, mode='a'):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        mode(str): log file write mode, `append` or `override`. default is `a`.

    Return:
        logger instance.
    """
    loguru_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<level>{level: <8}</level> | '
        '<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>')

    logger.remove()

    # only keep logger in rank0 process
    if distributed_rank == 0:
        filename = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
        save_file = os.path.join(save_dir, filename)
        logger.add(
            sys.stderr,
            format=loguru_format,
            level='INFO',
            enqueue=True,
        )
        logger.add(save_file)

    # redirect stdout/stderr to loguru
    redirect_sys_output('INFO')
```

### damo/utils/imports.py

```python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import sys

if sys.version_info[0] == 3 and sys.version_info[1] >= 7:
    import importlib
    import importlib.util
    import sys

    def import_file(module_name, file_path, make_importable=False):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if make_importable:
            sys.modules[module_name] = module
        return module
else:
    import imp

    def import_file(module_name, file_path, make_importable=None):
        module = imp.load_source(module_name, file_path)
        return module
```

### damo/utils/metric.py

```python
#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import functools
import os
from collections import defaultdict, deque

import numpy as np
import torch

__all__ = [
    'AverageMeter',
    'MeterBuffer',
    'get_total_and_free_memory_in_Mb',
    'gpu_mem_usage',
]


def get_total_and_free_memory_in_Mb(cuda_device):
    devices_info_str = os.popen(
        'nvidia-smi --query-gpu=memory.total,memory.used \
         --format=csv,nounits,noheader')
    devices_info = devices_info_str.read().strip().split('\n')
    total, used = devices_info[int(cuda_device)].split(',')
    return int(total), int(used)


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)


class AverageMeter:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=50):
        self._deque = deque(maxlen=window_size)
        self._total = 0.0
        self._count = 0

    def update(self, value):
        self._deque.append(value)
        self._count += 1
        self._total += value

    @property
    def median(self):
        d = np.array(list(self._deque))
        return np.median(d)

    @property
    def avg(self):
        # if deque is empty, nan will be returned.
        d = np.array(list(self._deque))
        return d.mean()

    @property
    def global_avg(self):
        return self._total / max(self._count, 1e-5)

    @property
    def latest(self):
        return self._deque[-1] if len(self._deque) > 0 else None

    @property
    def total(self):
        return self._total

    def reset(self):
        self._deque.clear()
        self._total = 0.0
        self._count = 0

    def clear(self):
        self._deque.clear()


class MeterBuffer(defaultdict):
    """Computes and stores the average and current value"""
    def __init__(self, window_size=20):
        factory = functools.partial(AverageMeter, window_size=window_size)
        super().__init__(factory)

    def reset(self):
        for v in self.values():
            v.reset()

    def get_filtered_meter(self, filter_key='time'):
        return {k: v for k, v in self.items() if filter_key in k}

    def update(self, values=None, **kwargs):
        if values is None:
            values = {}
        values.update(kwargs)
        for k, v in values.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            self[k].update(v)

    def clear_meters(self):
        for v in self.values():
            v.clear()
```

### damo/utils/demo_utils.py

```python
#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import os

import numpy as np

from damo.dataset.transforms import transforms as T
from damo.structures.image_list import to_image_list

__all__ = [
    'mkdir', 'nms', 'multiclass_nms', 'demo_postprocess', 'transform_img'
]


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def demo_postprocess(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


def transform_img(origin_img, size_divisibility, image_max_range, flip_prob,
                  image_mean, image_std, keep_ratio, infer_size=None):
    transform = [
        T.Resize(image_max_range, target_size=infer_size, keep_ratio=keep_ratio),
        T.RandomHorizontalFlip(flip_prob),
        T.ToTensor(),
        T.Normalize(mean=image_mean, std=image_std),
    ]
    transform = T.Compose(transform)

    img, _ = transform(origin_img)
    img = to_image_list(img, size_divisibility)
    return img
```

### damo/utils/dist.py

```python
#!/usr/bin/env python3
# This file mainly comes from
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/comm.py
# Copyright (c) Facebook, Inc. and its affiliates.
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import functools
import os
import pickle
import time
from contextlib import contextmanager

import numpy as np
import torch
from loguru import logger
from torch import distributed as dist

__all__ = [
    'get_num_devices',
    'wait_for_the_master',
    'is_main_process',
    'synchronize',
    'get_world_size',
    'get_rank',
    'get_local_rank',
    'get_local_size',
    'time_synchronized',
    'gather',
    'all_gather',
]

_LOCAL_PROCESS_GROUP = None


def get_num_devices():
    gpu_list = os.getenv('CUDA_VISIBLE_DEVICES', None)
    if gpu_list is not None:
        return len(gpu_list.split(','))
    else:
        devices_list_info = os.popen('nvidia-smi -L')
        devices_list_info = devices_list_info.read().strip().split('\n')
        return len(devices_list_info)


@contextmanager
def wait_for_the_master(local_rank: int):
    """
    Make all processes waiting for the master to do some task.
    """
    if local_rank > 0:
        dist.barrier()
    yield
    if local_rank == 0:
        if not dist.is_available():
            return
        if not dist.is_initialized():
            return
        else:
            dist.barrier()


def synchronize():
    """
    Helper function to synchronize (barrier)
    among all processes when using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the
        local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    return get_rank() == 0


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == 'nccl':
        return dist.new_group(backend='gloo')
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ['gloo', 'nccl']
    device = torch.device('cpu' if backend == 'gloo' else 'cuda')

    buffer = pickle.dumps(data)
    if len(buffer) > 1024**3:
        logger.warning(
            'Rank {} trying to all-gather {:.2f} GB of data on device {}'.
            format(get_rank(),
                   len(buffer) / (1024**3), device))
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), 'comm.gather/all_gather must be called from ranks within the group!'
    local_size = torch.tensor([tensor.numel()],
                              dtype=torch.int64,
                              device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size, ),
                              dtype=torch.uint8,
                              device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size, ), dtype=torch.uint8, device=tensor.device)
        for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group=group) == 1:
        return [data]
    rank = dist.get_rank(group=group)

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving Tensor from all ranks
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [
            torch.empty((max_size, ), dtype=torch.uint8, device=tensor.device)
            for _ in size_list
        ]
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        dist.gather(tensor, [], dst=dst, group=group)
        return []


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
            If workers need a shared RNG, they can use this shared seed to
            create one.
    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2**31)
    all_ints = all_gather(ints)
    return all_ints[0]


def time_synchronized():
    """pytorch-accurate time"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
```

### damo/utils/debug_utils.py

```python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import cv2
import numpy as np


def debug_input_vis(imgs, targets, ids, train_loader):

    std = np.array([1.0, 1.0, 1.0]).reshape(3, 1, 1)
    mean = np.array([0.0, 0.0, 0.0]).reshape(3, 1, 1)

    n, c, h, w = imgs.shape
    for i in range(n):
        img = imgs[i, :, :, :].cpu()
        bboxs = targets[i].bbox.cpu().numpy()
        cls = targets[i].get_field('labels').cpu().numpy()
        if True:
            # if self.config.training_mosaic:
            img_id = train_loader.dataset._dataset.id_to_img_map[ids[i]]
        else:
            img_id = train_loader.dataset.id_to_img_map[ids[i]]

        img = np.clip(
            (img.numpy() * std + mean).transpose(1, 2,
                                                 0).copy().astype(np.uint8), 0,
            255)
        for bbox, obj_cls in zip(bboxs, cls):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img,
                          pt1=(x1, y1),
                          pt2=(x2, y2),
                          color=(0, 0, 255),
                          thickness=2)
            cv2.putText(img, f'{obj_cls}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255))

        cv2.imwrite(f'visimgs/vis_{img_id}.jpg', img)
```

### damo/augmentations/__init__.py

```python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
```

### damo/augmentations/scale_aware_aug.py

```python
# This file mainly comes from
# https://github.com/dvlab-research/SA-AutoAug/blob/master/FCOS/fcos_core/augmentations/scale_aware_aug.py
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import copy

from damo.augmentations.box_level_augs.box_level_augs import Box_augs
from damo.augmentations.box_level_augs.color_augs import color_aug_func
from damo.augmentations.box_level_augs.geometric_augs import geometric_aug_func


class SA_Aug(object):
    def __init__(self, iters_per_epoch, start_epoch, total_epochs,
                 no_aug_epochs, batch_size, num_gpus, num_workers, sada_cfg):

        autoaug_list = sada_cfg.autoaug_params
        num_policies = sada_cfg.num_subpolicies
        scale_splits = sada_cfg.scale_splits
        box_prob = sada_cfg.box_prob

        self.batch_size = batch_size / num_gpus
        self.num_workers = num_workers
        self.max_iters = (total_epochs - no_aug_epochs) * iters_per_epoch
        self.count = start_epoch * iters_per_epoch
        if self.num_workers == 0:
            self.num_workers += 1

        box_aug_list = autoaug_list[4:]
        color_aug_types = list(color_aug_func.keys())
        geometric_aug_types = list(geometric_aug_func.keys())
        policies = []
        for i in range(num_policies):
            _start_pos = i * 6
            sub_policy = [
                (
                    color_aug_types[box_aug_list[_start_pos + 0] %
                                    len(color_aug_types)],
                    box_aug_list[_start_pos + 1] * 0.1,
                    box_aug_list[_start_pos + 2],
                ),  # box_color policy
                (geometric_aug_types[box_aug_list[_start_pos + 3] %
                                     len(geometric_aug_types)],
                 box_aug_list[_start_pos + 4] * 0.1,
                 box_aug_list[_start_pos + 5])
            ]  # box_geometric policy
            policies.append(sub_policy)

        _start_pos = num_policies * 6
        scale_ratios = {
            'area': [
                box_aug_list[_start_pos + 0], box_aug_list[_start_pos + 1],
                box_aug_list[_start_pos + 2]
            ],
            'prob': [
                box_aug_list[_start_pos + 3], box_aug_list[_start_pos + 4],
                box_aug_list[_start_pos + 5]
            ]
        }

        box_augs_dict = {'policies': policies, 'scale_ratios': scale_ratios}

        self.box_augs = Box_augs(box_augs_dict=box_augs_dict,
                                 max_iters=self.max_iters,
                                 scale_splits=scale_splits,
                                 box_prob=box_prob)

    def __call__(self, tensor, target):
        iteration = self.count // self.batch_size * self.num_workers
        tensor = copy.deepcopy(tensor)
        target = copy.deepcopy(target)
        tensor, target = self.box_augs(tensor, target, iteration=iteration)

        self.count += 1

        return tensor, target
```

#### damo/augmentations/box_level_augs/box_level_augs.py

```python
# This file mainly comes from
# https://github.com/dvlab-research/SA-AutoAug/blob/master/FCOS/fcos_core/augmentations/box_level_augs/box_level_augs.py
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import random

import numpy as np

from .color_augs import color_aug_func
from .geometric_augs import geometric_aug_func


def _box_sample_prob(bbox, scale_ratios_splits, box_prob=0.3):
    scale_ratios, scale_splits = scale_ratios_splits

    ratios = np.array(scale_ratios)
    ratios = ratios / ratios.sum()
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if area == 0:
        return 0
    if area < scale_splits[0]:
        scale_ratio = ratios[0]
    elif area < scale_splits[1]:
        scale_ratio = ratios[1]
    else:
        scale_ratio = ratios[2]
    return box_prob * scale_ratio


def _box_aug_per_img(img,
                     target,
                     aug_type=None,
                     scale_ratios=None,
                     scale_splits=None,
                     img_prob=0.1,
                     box_prob=0.3,
                     level=1):
    if random.random() > img_prob:
        return img, target
    img /= 255.0

    tag = 'prob' if aug_type in geometric_aug_func else 'area'
    scale_ratios_splits = [scale_ratios[tag], scale_splits]
    if scale_ratios is None:
        box_sample_prob = [box_prob] * len(target.bbox)
    else:
        box_sample_prob = [
            _box_sample_prob(bbox, scale_ratios_splits, box_prob=box_prob)
            for bbox in target.bbox
        ]

    if aug_type in color_aug_func:
        img_aug = color_aug_func[aug_type](
            img, level, target, [scale_ratios['area'], scale_splits],
            box_sample_prob)
    elif aug_type in geometric_aug_func:
        img_aug, target = geometric_aug_func[aug_type](img, level, target,
                                                       box_sample_prob)
    else:
        raise ValueError('Unknown box-level augmentation function %s.' %
                         (aug_type))
    out = img_aug * 255.0

    return out, target


class Box_augs(object):
    def __init__(self, box_augs_dict, max_iters, scale_splits, box_prob=0.3):
        self.max_iters = max_iters
        self.box_prob = box_prob
        self.scale_splits = scale_splits
        self.policies = box_augs_dict['policies']
        self.scale_ratios = box_augs_dict['scale_ratios']

    def __call__(self, tensor, target, iteration):
        iter_ratio = float(iteration) / self.max_iters
        sub_policy = random.choice(self.policies)

        h, w = tensor.shape[-2:]
        ratio = min(h, w) / 800

        scale_splits = [area * ratio for area in self.scale_splits]
        if iter_ratio <= 1:
            tensor, _ = _box_aug_per_img(tensor,
                                         target,
                                         aug_type=sub_policy[0][0],
                                         scale_ratios=self.scale_ratios,
                                         scale_splits=scale_splits,
                                         img_prob=sub_policy[0][1] *
                                         iter_ratio,
                                         box_prob=self.box_prob,
                                         level=sub_policy[0][2])
            tensor, target = _box_aug_per_img(tensor,
                                              target,
                                              aug_type=sub_policy[1][0],
                                              scale_ratios=self.scale_ratios,
                                              scale_splits=scale_splits,
                                              img_prob=sub_policy[1][1] *
                                              iter_ratio,
                                              box_prob=self.box_prob,
                                              level=sub_policy[1][2])

        return tensor, target
```

#### damo/augmentations/box_level_augs/__init__.py

```python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
```

#### damo/augmentations/box_level_augs/gaussian_maps.py

```python
# This file mainly comes from
# https://github.com/dvlab-research/SA-AutoAug/blob/master/FCOS/fcos_core/augmentations/box_level_augs/gaussian_maps.py
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import math

import torch


def _gaussian_map(img, boxes, scale_splits=None, scale_ratios=None):
    g_maps = torch.zeros(*img.shape[1:]).to(img.device)
    height, width = img.shape[1], img.shape[2]

    x_range = torch.arange(0, height, 1).to(img.device)
    y_range = torch.arange(0, width, 1).to(img.device)
    xx, yy = torch.meshgrid(x_range, y_range)
    pos = torch.empty(xx.shape + (2, )).to(img.device)
    pos[:, :, 0] = xx
    pos[:, :, 1] = yy

    for j, box in enumerate(boxes):
        y1, x1, y2, x2 = box
        x, y, h, w = x1, y1, x2 - x1, y2 - y1
        mean_torch = torch.tensor([x + h // 2, y + w // 2]).to(img.device)
        if scale_ratios is None:
            scale_ratio = 1.0
        else:
            ratio_list = [0.2, 0.4, 0.6, 0.8, 1.0, 2, 4, 6, 8, 10]
            if h * w < scale_splits[0]:
                scale_ratio = ratio_list[scale_ratios[0]] * scale_splits[0] / (
                    h * w)
            elif h * w < scale_splits[1]:
                scale_ratio = ratio_list[scale_ratios[1]] * (
                    scale_splits[0] + scale_splits[1]) / 2.0 / (h * w)
            elif h * w < scale_splits[2]:
                scale_ratio = ratio_list[scale_ratios[2]] * scale_splits[2] / (
                    h * w)
            else:
                scale_ratio = ratio_list[scale_ratios[2]]

        r_var = (scale_ratio * height * width / (2 * math.pi))**0.5
        var_x = torch.tensor([(h / height) * r_var],
                             dtype=torch.float32).to(img.device)
        var_y = torch.tensor([(w / width) * r_var],
                             dtype=torch.float32).to(img.device)
        g_map = torch.exp(-(((xx.float() - mean_torch[0])**2 /
                             (2.0 * var_x**2) +
                             (yy.float() - mean_torch[1])**2 /
                             (2.0 * var_y**2)))).to(img.device)
        g_maps += g_map
    return g_maps


def _merge_gaussian(img, img_aug, boxes, scale_ratios, scale_splits):
    g_maps = _gaussian_map(img, boxes, scale_splits, scale_ratios)
    g_maps = g_maps.clamp(min=0, max=1.0)
    out = img * (1 - g_maps) + img_aug * g_maps
    return out
```

#### damo/augmentations/box_level_augs/color_augs.py

```python
# This file mainly comes from
# https://github.com/dvlab-research/SA-AutoAug/blob/master/FCOS/fcos_core/augmentations/box_level_augs/color_augs.py
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import random

import torch
import torch.nn.functional as F

from damo.augmentations.box_level_augs.gaussian_maps import _merge_gaussian

_MAX_LEVEL = 10.0


def blend(image1, image2, factor):
    """Blend image1 and image2 using 'factor'.
    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 1.0.
    """

    if factor == 0.0:
        return image1
    if factor == 1.0:
        return image2

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = image1 + scaled

    # Interpolate
    if factor > 0.0 and factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return temp

    # Extrapolate:
    #
    # We need to clip and then cast.
    return torch.clamp(temp, 0.0, 1.0)


def solarize(image, threshold=0.5):
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract 255 from the pixel.
    return torch.where(image <= threshold, image, 1.0 - image)


def solarize_add(image, addition=0, threshold=0.5):
    # For each pixel in the image less than threshold
    # we add 'addition' amount to it and then clip the
    # pixel value to be between 0 and 255. The value
    # of 'addition' is between -128 and 128.
    added_image = image + addition
    added_image = torch.clamp(added_image, 0.0, 1.0)
    return torch.where(image <= threshold, added_image, image)


def rgb2gray(rgb):
    gray = rgb[0] * 0.2989 + rgb[1] * 0.5870 + rgb[2] * 0.1140
    gray = gray.unsqueeze(0).repeat((3, 1, 1))
    return gray


def color(img, factor):
    """Equivalent of PIL Color."""
    if img.shape[0] == 0 or img.shape[1] == 0:
        return img

    degenerate = rgb2gray(img)
    return blend(degenerate, img, factor)


def contrast(img, factor):
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    mean = torch.mean(rgb2gray(img).to(dtype), dim=(-3, -2, -1), keepdim=True)
    return blend(mean, img, max(factor, 1e-6))


def brightness(image, factor):
    """Equivalent of PIL Brightness."""
    degenerate = torch.zeros(image.shape)
    return blend(degenerate, image, factor)


def sharpness(image, factor):
    """Implements Sharpness function from PIL using TF ops."""
    if image.shape[0] == 0 or image.shape[1] == 0:
        return image
    channels = image.shape[0]
    kernel = torch.Tensor([[1, 1, 1], [1, 5, 1], [1, 1, 1]]).reshape(
        1, 1, 3, 3) / 13.0
    kernel = kernel.repeat((3, 1, 1, 1))
    image_newaxis = image.unsqueeze(0)
    image_pad = F.pad(image_newaxis, (1, 1, 1, 1), mode='reflect')
    degenerate = F.conv2d(image_pad, weight=kernel, groups=channels).squeeze(0)
    return blend(degenerate, image, factor)


def equalize(image):
    """Implements Equalize function from PIL using PyTorch ops based on:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/
    autoaugment.py#L352"""
    image = image * 255

    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = im[c, :, :]
        # Compute the histogram of the image channel.
        histo = torch.histc(im, bins=256, min=0, max=255)  # .type(torch.int32)
        # For the purposes of computing the step, filter out the nonzeros.
        nonzero_histo = torch.reshape(histo[histo != 0], [-1])
        step = (torch.sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (torch.cumsum(histo, 0) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = torch.cat([torch.zeros(1), lut[:-1]])
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return torch.clamp(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        if step == 0:
            result = im
        else:
            # can't index using 2d index. Have to flatten and then reshape
            result = torch.gather(build_lut(histo, step), 0,
                                  im.flatten().long())
            result = result.reshape_as(im)

        return result  # .type(torch.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = torch.stack([s1, s2, s3], 0) / 255.0
    return image


def autocontrast(image):
    def scale_channel(image):
        """Scale the 2D image using the autocontrast rule."""
        lo = torch.min(image)
        hi = torch.max(image)

        # Scale the image, making the lowest value 0 and the highest value 1.
        def scale_values(im):
            scale = 1.0 / (hi - lo)
            offset = -lo * scale
            im = im * scale + offset
            im = torch.clamp(im, 0.0, 1.0)
            return im

        if hi > lo:
            result = scale_values(image)
        else:
            result = image

        return result

    # Assumes RGB for now. Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image[0, :, :])
    s2 = scale_channel(image[1, :, :])
    s3 = scale_channel(image[2, :, :])
    image = torch.stack([s1, s2, s3], 0)
    return image


def posterize(image, bits):
    """Equivalent of PIL Posterize."""
    image *= 255
    image = image.long()
    shift = bits  # 8 - bits
    image_rightshift = image >> shift
    image_leftshift = image_rightshift << shift
    image_leftshift = image_leftshift.float() / 255.0
    return image_leftshift


def _color_aug_func(img, img_aug, target, scale_ratios_splits,
                    box_sample_probs):
    scale_ratios, scale_splits = scale_ratios_splits
    boxes = [
        bbox for i, bbox in enumerate(target.bbox)
        if random.random() < box_sample_probs[i]
    ]
    img_aug = _merge_gaussian(img, img_aug, boxes, scale_ratios, scale_splits)
    return img_aug


color_aug_func = {
    'AutoContrast':
    lambda x, level, target,
    scale_ratios_splits, box_sample_probs: _color_aug_func(
        x, autocontrast(x), target, scale_ratios_splits, box_sample_probs),
    'Equalize':
    lambda x, leve, target,
    scale_ratios_splits, box_sample_probs: _color_aug_func(
        x, equalize(x), target, scale_ratios_splits, box_sample_probs),
    'SolarizeAdd':
    lambda x, level, target, scale_ratios_splits, box_sample_probs:
    _color_aug_func(x, solarize_add(x, level / _MAX_LEVEL * 0.4296875), target,
                    scale_ratios_splits, box_sample_probs),
    'Color':
    lambda x, level, target, scale_ratios_splits, box_sample_probs:
    _color_aug_func(x, color(x, level / _MAX_LEVEL * 1.8 + 0.1), target,
                    scale_ratios_splits, box_sample_probs),
    'Contrast':
    lambda x, level, target, scale_ratios_splits, box_sample_probs:
    _color_aug_func(x, contrast(x, level / _MAX_LEVEL * 1.8 + 0.1), target,
                    scale_ratios_splits, box_sample_probs),
    'Brightness':
    lambda x, level, target, scale_ratios_splits, box_sample_probs:
    _color_aug_func(x, brightness(x, level / _MAX_LEVEL * 1.8 + 0.1), target,
                    scale_ratios_splits, box_sample_probs),
    'Sharpness':
    lambda x, level, target, scale_ratios_splits, box_sample_probs:
    _color_aug_func(x, sharpness(x, level / _MAX_LEVEL * 1.8 + 0.1), target,
                    scale_ratios_splits, box_sample_probs),
}
```

#### damo/augmentations/box_level_augs/geometric_augs.py

```python
# This file mainly comes from
# https://github.com/dvlab-research/SA-AutoAug/blob/master/FCOS/fcos_core/augmentations/box_level_augs/geometric_augs.py
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import copy
import random

import torch
import torchvision.transforms as transforms

from damo.augmentations.box_level_augs.gaussian_maps import _gaussian_map

_MAX_LEVEL = 10.0
pixel_mean = [102.9801, 115.9465, 122.7717]


def scale_area(box, height, width, scale_ratio=1.0):
    y1, x1, y2, x2 = box
    h, w = x2 - x1, y2 - y1
    h_new, w_new = h * scale_ratio, w * scale_ratio
    x1, y1 = max(x1 + h / 2 - h_new / 2, 0), max(y1 + w / 2 - w_new / 2, 0)
    x2, y2 = min(x1 + h_new, height), min(y1 + w_new, width)
    box_new = torch.Tensor([y1, x1, y2, x2])
    return box_new


def _geometric_aug_func(x,
                        target,
                        angle=0,
                        translate=(0, 0),
                        scale=1,
                        shear=(0, 0),
                        hflip=False,
                        boxes_sample_prob=[],
                        scale_ratio=1.0):
    boxes_and_labels = [(
        target.bbox[i],
        target.extra_fields['labels'][i],
    ) for i in range(len(target.bbox))
                        if random.random() < boxes_sample_prob[i]]
    boxes = [b_and_l[0] for b_and_l in boxes_and_labels]
    labels = [b_and_l[1] for b_and_l in boxes_and_labels]

    if random.random() < 0.5:
        angle *= -1
        translate = (-translate[0], -translate[1])
        shear = (-shear[0], -shear[1])

    height, width = x.shape[1], x.shape[2]

    x_crops = []
    boxes_crops = []
    boxes_new = []
    labels_new = []
    for i, box in enumerate(boxes):
        box_crop = scale_area(box, height, width, scale_ratio)
        y1, x1, y2, x2 = box_crop.long()

        x_crop = x[:, x1:x2, y1:y2]
        boxes_crops.append(box_crop)

        if x1 >= x2 or y1 >= y2:
            x_crops.append(x_crop)
            continue

        if hflip:
            x_crop = x_crop.flip(-1)
        elif translate[0] + translate[1] != 0:
            offset_y = (y2 + translate[0]).clamp(0, width).long().tolist() - y2
            offset_x = (x2 + translate[1]).clamp(0,
                                                 height).long().tolist() - x2
            if offset_x != 0 or offset_y != 0:
                offset = [offset_y, offset_x]
                boxes_new.append(box + torch.Tensor(offset * 2))
                labels_new.append(labels[i])
        else:
            x_crop = transforms.functional.to_pil_image(x_crop.cpu())
            try:
                x_crop = transforms.functional.affine(
                    x_crop,
                    angle,
                    translate,
                    scale,
                    shear,
                    resample=2,
                    fillcolor=tuple([int(i) for i in pixel_mean]))
            except:
                x_crop = transforms.functional.affine(
                    x_crop,
                    angle,
                    translate,
                    scale,
                    shear,
                    interpolation=2,
                    fill=tuple([int(i) for i in pixel_mean]))
            x_crop = transforms.functional.to_tensor(x_crop).to(x.device)
        x_crops.append(x_crop)
    y = _transform(x, x_crops, boxes_crops, translate)

    if translate[0] + translate[1] != 0 and len(boxes_new) > 0:
        target.bbox = torch.cat((target.bbox, torch.stack(boxes_new)))
        target.extra_fields['labels'] = torch.cat(
            (target.extra_fields['labels'], torch.Tensor(labels_new).long()))

    return y, target


def _transform(x, x_crops, boxes_crops, translate=(0, 0)):
    y = copy.deepcopy(x)
    height, width = x.shape[1], x.shape[2]

    for i, box in enumerate(boxes_crops):
        y1_c, x1_c, y2_c, x2_c = boxes_crops[i].long()

        y1_c = (y1_c + translate[0]).clamp(0, width).long().tolist()
        x1_c = (x1_c + translate[1]).clamp(0, height).long().tolist()
        y2_c = (y2_c + translate[0]).clamp(0, width).long().tolist()
        x2_c = (x2_c + translate[1]).clamp(0, height).long().tolist()

        y_crop = copy.deepcopy(y[:, x1_c:x2_c, y1_c:y2_c])
        x_crop = x_crops[i][:, :y_crop.shape[1], :y_crop.shape[2]]

        if y_crop.shape[1] * y_crop.shape[2] == 0:
            continue

        g_maps = _gaussian_map(x_crop,
                               [[0, 0, y_crop.shape[2], y_crop.shape[1]]])
        _, _h, _w = y[:, x1_c:x2_c, y1_c:y2_c].shape
        y[:, x1_c:x1_c + x_crop.shape[1],
          y1_c:y1_c + x_crop.shape[2]] = g_maps * x_crop + (
              1 - g_maps) * y_crop[:, :x_crop.shape[1], :x_crop.shape[2]]
    return y


geometric_aug_func = {
    'hflip':
    lambda x, level, target, boxes_sample_probs: _geometric_aug_func(
        x, target, hflip=True, boxes_sample_prob=boxes_sample_probs),
    'rotate':
    lambda x, level, target, boxes_sample_probs: _geometric_aug_func(
        x,
        target,
        level / _MAX_LEVEL * 30,
        boxes_sample_prob=boxes_sample_probs),
    'shearX':
    lambda x, level, target, boxes_sample_probs: _geometric_aug_func(
        x,
        target,
        shear=(level / _MAX_LEVEL * 15, 0),
        boxes_sample_prob=boxes_sample_probs),
    'shearY':
    lambda x, level, target, boxes_sample_probs: _geometric_aug_func(
        x,
        target,
        shear=(0, level / _MAX_LEVEL * 15),
        boxes_sample_prob=boxes_sample_probs),
    'translateX':
    lambda x, level, target, boxes_sample_probs: _geometric_aug_func(
        x,
        target,
        translate=(level / _MAX_LEVEL * 120.0, 0),
        boxes_sample_prob=boxes_sample_probs),
    'translateY':
    lambda x, level, target, boxes_sample_probs: _geometric_aug_func(
        x,
        target,
        translate=(0, level / _MAX_LEVEL * 120.0),
        boxes_sample_prob=boxes_sample_probs)
}
```

### damo/detectors/detector.py

```python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import torch
import torch.nn as nn
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP

from damo.base_models.backbones import build_backbone
from damo.base_models.heads import build_head
from damo.base_models.necks import build_neck
from damo.structures.image_list import to_image_list


class Detector(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.backbone = build_backbone(config.model.backbone)
        self.neck = build_neck(config.model.neck)
        self.head = build_head(config.model.head)

        self.config = config

    def init_bn(self, M):

        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def init_model(self):

        self.apply(self.init_bn)

        self.backbone.init_weights()
        self.neck.init_weights()
        self.head.init_weights()

    def load_pretrain_detector(self, pretrain_model):

        state_dict = torch.load(pretrain_model, map_location='cpu')['model']
        logger.info(f'Finetune from {pretrain_model}................')
        new_state_dict = {}
        for k, v in self.state_dict().items():
            k = k.replace('module.', '')
            if 'head' in k:
                new_state_dict[k] = self.state_dict()[k]
                continue
            new_state_dict[k] = state_dict[k]

        self.load_state_dict(new_state_dict, strict=True)

    def forward(self, x, targets=None, tea=False, stu=False):
        images = to_image_list(x)
        feature_outs = self.backbone(images.tensors)  # list of tensor
        fpn_outs = self.neck(feature_outs)

        if tea:
            return fpn_outs
        else:
            outputs = self.head(
                fpn_outs,
                targets,
                imgs=images,
            )
            if stu:
                return outputs, fpn_outs
            else:
                return outputs


def build_local_model(config, device):
    model = Detector(config)
    model.init_model()
    model.to(device)

    return model


def build_ddp_model(model, local_rank):
    model = DDP(model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True)

    return model
```

### damo/base_models/__init__.py

```python
#!/usr/bin/env python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
from .backbones import build_backbone
from .heads import build_head
from .necks import build_neck
```

#### damo/base_models/losses/distill_loss.py

```python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureLoss(nn.Module):
    def __init__(self,
                 channels_s,
                 channels_t,
                 distiller='cwd',
                 loss_weight=1.0):
        super(FeatureLoss, self).__init__()
        self.loss_weight = loss_weight

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.align_module = nn.ModuleList([
            nn.Conv2d(channel, tea_channel, kernel_size=1, stride=1,
                      padding=0).to(device)
            for channel, tea_channel in zip(channels_s, channels_t)
        ])
        self.norm = [
            nn.BatchNorm2d(tea_channel, affine=False).to(device)
            for tea_channel in channels_t
        ]

        if (distiller == 'mimic'):
            self.feature_loss = MimicLoss(channels_s, channels_t)
        elif (distiller == 'mgd'):
            self.feature_loss = MGDLoss(channels_s, channels_t)
        elif (distiller == 'cwd'):
            self.feature_loss = CWDLoss(channels_s, channels_t)
        else:
            raise NotImplementedError

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        tea_feats = []
        stu_feats = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            s = self.align_module[idx](s)
            s = self.norm[idx](s)
            t = self.norm[idx](t)
            tea_feats.append(t)
            stu_feats.append(s)

        loss = self.feature_loss(stu_feats, tea_feats)
        return self.loss_weight * loss


class MimicLoss(nn.Module):
    def __init__(self, channels_s, channels_t):
        super(MimicLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mse = nn.MSELoss()

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            losses.append(self.mse(s, t))
        loss = sum(losses)
        return loss


class MGDLoss(nn.Module):
    def __init__(self,
                 channels_s,
                 channels_t,
                 alpha_mgd=0.00002,
                 lambda_mgd=0.65):
        super(MGDLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        self.generation = [
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3,
                          padding=1)).to(device) for channel in channels_t
        ]

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        loss = sum(losses)
        return loss

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N

        return dis_loss


class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    """
    def __init__(self, channels_s, channels_t, tau=1.0):
        super(CWDLoss, self).__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            N, C, H, W = s.shape
            # normalize in channel diemension
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau,
                                       dim=1)  # [N*C, H*W]

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (
                    self.tau**2)

            losses.append(cost / (C * N))
        loss = sum(losses)

        return loss

```

#### damo/base_models/losses/gfocal_loss.py

```python
# This file mainly comes from
# https://github.com/implus/GFocalV2/blob/master/mmdet/models/losses/gfocal_loss.py

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.bbox_calculator import bbox_overlaps


def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """
    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


@weighted_loss
def giou_loss(pred, target, eps=1e-7):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.
    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss


class GIoULoss(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * giou_loss(pred,
                                            target,
                                            weight,
                                            eps=self.eps,
                                            reduction=reduction,
                                            avg_factor=avg_factor,
                                            **kwargs)
        return loss


@weighted_loss
def distribution_focal_loss(pred, label):
    r"""Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.
    Args:
        pred (torch.Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        label (torch.Tensor): Target distance label for bounding boxes with
            shape (N,).
    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    dis_left = label.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - label
    weight_right = label - dis_left.float()
    loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left \
        + F.cross_entropy(pred, dis_right, reduction='none') * weight_right
    return loss


class DistributionFocalLoss(nn.Module):
    r"""Distribution Focal Loss (DFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.
    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
    """
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): Predicted general distribution of bounding
                boxes (before softmax) with shape (N, n+1), n is the max value
                of the integral set `{0, ..., n}` in paper.
            target (torch.Tensor): Target distance label for bounding boxes
                with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        loss_cls = self.loss_weight * distribution_focal_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_cls


@weighted_loss
def quality_focal_loss(pred, target, beta=2.0, use_sigmoid=True):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.
    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target
    if use_sigmoid:
        func = F.binary_cross_entropy_with_logits
    else:
        func = F.binary_cross_entropy
    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid() if use_sigmoid else pred
    scale_factor = pred_sigmoid  # 8400, 81
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = func(pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    bg_class_ind = pred.size(1)
    pos = ((label >= 0) &
           (label < bg_class_ind)).nonzero(as_tuple=False).squeeze(1)
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos,
         pos_label] = func(pred[pos, pos_label], score[pos],
                           reduction='none') * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)
    return loss


class QualityFocalLoss(nn.Module):
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.
    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """
    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        # assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        loss_cls = self.loss_weight * quality_focal_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            use_sigmoid=self.use_sigmoid,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_cls
```

#### damo/base_models/core/end2end.py

```python
import random

import torch
import torch.nn as nn


class ORT_NMS(torch.autograd.Function):
    '''ONNX-Runtime NMS operation'''
    @staticmethod
    def forward(ctx,
                boxes,
                scores,
                max_output_boxes_per_class=torch.tensor([100]),
                iou_threshold=torch.tensor([0.45]),
                score_threshold=torch.tensor([0.25])):
        device = boxes.device
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det, )).sort()[0].to(device)
        idxs = torch.arange(100, 100 + num_det).to(device)
        zeros = torch.zeros((num_det, ), dtype=torch.int64).to(device)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]],
                                     0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return selected_indices

    @staticmethod
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold,
                 score_threshold):
        return g.op('NonMaxSuppression', boxes, scores,
                    max_output_boxes_per_class, iou_threshold, score_threshold)


class TRT8_NMS(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0,
                                max_output_boxes, (batch_size, 1),
                                dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0,
                                    num_classes,
                                    (batch_size, max_output_boxes),
                                    dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 background_class=-1,
                 box_coding=1,
                 iou_threshold=0.45,
                 max_output_boxes=100,
                 plugin_version='1',
                 score_activation=0,
                 score_threshold=0.25):
        out = g.op('TRT::EfficientNMS_TRT',
                   boxes,
                   scores,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   iou_threshold_f=iou_threshold,
                   max_output_boxes_i=max_output_boxes,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   score_threshold_f=score_threshold,
                   outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class TRT7_NMS(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        plugin_version='1',
        shareLocation=1,
        backgroundLabelId=-1,
        numClasses=80,
        topK=1000,
        keepTopK=100,
        scoreThreshold=0.25,
        iouThreshold=0.45,
        isNormalized=0,
        clipBoxes=0,
        scoreBits=16,
        caffeSemantics=1,
    ):
        batch_size, num_boxes, numClasses = scores.shape
        num_det = torch.randint(0,
                                keepTopK, (batch_size, 1),
                                dtype=torch.int32)
        det_boxes = torch.randn(batch_size, keepTopK, 4)
        det_scores = torch.randn(batch_size, keepTopK)
        det_classes = torch.randint(0, numClasses,
                                    (batch_size, keepTopK)).float()
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        plugin_version='1',
        shareLocation=1,
        backgroundLabelId=-1,
        numClasses=80,
        topK=1000,
        keepTopK=100,
        scoreThreshold=0.25,
        iouThreshold=0.45,
        isNormalized=0,
        clipBoxes=0,
        scoreBits=16,
        caffeSemantics=1,
    ):
        out = g.op('TRT::BatchedNMSDynamic_TRT',
                   boxes,
                   scores,
                   shareLocation_i=shareLocation,
                   plugin_version_s=plugin_version,
                   backgroundLabelId_i=backgroundLabelId,
                   numClasses_i=numClasses,
                   topK_i=topK,
                   keepTopK_i=keepTopK,
                   scoreThreshold_f=scoreThreshold,
                   iouThreshold_f=iouThreshold,
                   isNormalized_i=isNormalized,
                   clipBoxes_i=clipBoxes,
                   scoreBits_i=scoreBits,
                   caffeSemantics_i=caffeSemantics,
                   outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class ONNX_ORT(nn.Module):
    '''onnx module with ONNX-Runtime NMS operation.'''
    def __init__(self,
                 max_obj=100,
                 iou_thres=0.45,
                 score_thres=0.25,
                 device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')
        self.max_obj = torch.tensor([max_obj]).to(device)
        self.iou_threshold = torch.tensor([iou_thres]).to(device)
        self.score_threshold = torch.tensor([score_thres]).to(device)
        self.convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
            dtype=torch.float32,
            device=self.device)

    def forward(self, score, box):
        batch, anchors, _ = score.shape

        nms_box = box @ self.convert_matrix
        nms_score = score.transpose(1, 2).contiguous()

        selected_indices = ORT_NMS.apply(nms_box, nms_score, self.max_obj,
                                         self.iou_threshold,
                                         self.score_threshold)
        batch_inds, cls_inds, box_inds = selected_indices.unbind(1)
        selected_score = nms_score[batch_inds, cls_inds, box_inds].unsqueeze(1)
        selected_box = nms_box[batch_inds, box_inds, ...]

        dets = torch.cat([selected_box, selected_score], dim=1)

        batched_dets = dets.unsqueeze(0).repeat(batch, 1, 1)
        batch_template = torch.arange(0,
                                      batch,
                                      dtype=batch_inds.dtype,
                                      device=batch_inds.device)
        batched_dets = batched_dets.where(
            (batch_inds == batch_template.unsqueeze(1)).unsqueeze(-1),
            batched_dets.new_zeros(1))

        batched_labels = cls_inds.unsqueeze(0).repeat(batch, 1)
        batched_labels = batched_labels.where(
            (batch_inds == batch_template.unsqueeze(1)),
            batched_labels.new_ones(1) * -1)

        N = batched_dets.shape[0]

        batched_dets = torch.cat(
            (batched_dets, batched_dets.new_zeros((N, 1, 5))), 1)
        batched_labels = torch.cat((batched_labels, -batched_labels.new_ones(
            (N, 1))), 1)

        _, topk_inds = batched_dets[:, :, -1].sort(dim=1, descending=True)

        topk_batch_inds = torch.arange(batch,
                                       dtype=topk_inds.dtype,
                                       device=topk_inds.device).view(-1, 1)
        batched_dets = batched_dets[topk_batch_inds, topk_inds, ...]
        det_classes = batched_labels[topk_batch_inds, topk_inds, ...]
        det_boxes, det_scores = batched_dets.split((4, 1), -1)
        det_scores = det_scores.squeeze(-1)
        num_det = (det_scores > 0).sum(1, keepdim=True)
        return num_det, det_boxes, det_scores, det_classes


class ONNX_TRT7(nn.Module):
    '''onnx module with TensorRT NMS operation.'''
    def __init__(self,
                 max_obj=100,
                 iou_thres=0.45,
                 score_thres=0.25,
                 device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')
        self.shareLocation = 1
        self.backgroundLabelId = -1
        self.numClasses = 80
        self.topK = 1000
        self.keepTopK = max_obj
        self.scoreThreshold = score_thres
        self.iouThreshold = iou_thres
        self.isNormalized = 0
        self.clipBoxes = 0
        self.scoreBits = 16
        self.caffeSemantics = 1
        self.plugin_version = '1'
        self.convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
            dtype=torch.float32,
            device=self.device)

    def forward(self, score, box):
        # box @= self.convert_matrix
        box = box.unsqueeze(2)
        self.numClasses = int(score.shape[2])
        num_det, det_boxes, det_scores, det_classes = TRT7_NMS.apply(
            box,
            score,
            self.plugin_version,
            self.shareLocation,
            self.backgroundLabelId,
            self.numClasses,
            self.topK,
            self.keepTopK,
            self.scoreThreshold,
            self.iouThreshold,
            self.isNormalized,
            self.clipBoxes,
            self.scoreBits,
            self.caffeSemantics,
        )
        return num_det, det_boxes, det_scores, det_classes.int()


class ONNX_TRT8(nn.Module):
    '''onnx module with TensorRT NMS operation.'''
    def __init__(self,
                 max_obj=100,
                 iou_thres=0.45,
                 score_thres=0.25,
                 device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')
        self.background_class = -1,
        self.box_coding = 1,
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres

    def forward(self, score, box):
        num_det, det_boxes, det_scores, det_classes = TRT8_NMS.apply(
            box, score, self.background_class, self.box_coding,
            self.iou_threshold, self.max_obj, self.plugin_version,
            self.score_activation, self.score_threshold)
        return num_det, det_boxes, det_scores, det_classes


class End2End(nn.Module):
    '''export onnx or tensorrt model with NMS operation.'''
    def __init__(self,
                 model,
                 max_obj=100,
                 iou_thres=0.45,
                 score_thres=0.25,
                 device=None,
                 ort=False,
                 trt_version=7,
                 with_preprocess=False):
        super().__init__()
        device = device if device else torch.device('cpu')
        self.with_preprocess = with_preprocess
        self.model = model.to(device)
        TRT = ONNX_TRT8 if trt_version >= 8 else ONNX_TRT7
        self.patch_model = ONNX_ORT if ort else TRT
        self.end2end = self.patch_model(max_obj, iou_thres, score_thres,
                                        device)
        self.end2end.eval()

    def forward(self, x):
        if self.with_preprocess:
            x = x[:, [2, 1, 0], ...]
            x = x * (1 / 255)
        x = self.model(x)

        x = self.end2end(x[0], x[1])
        return x
```

#### damo/base_models/core/ota_assigner.py

```python
# Copyright (c) OpenMMLab. All rights reserved.

import warnings

import torch
import torch.nn.functional as F

from .bbox_calculator import bbox_overlaps


class BaseAssigner(object):
    """Base assigner that assigns boxes to ground truth boxes."""
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign boxes to either a ground truth boxes or a negative boxes."""


class AssignResult(object):
    """Stores assignments between predicted and truth boxes.
    Attributes:
        num_gts (int): the number of truth boxes considered when computing this
            assignment
        gt_inds (LongTensor): for each predicted box indicates the 1-based
            index of the assigned truth box. 0 means unassigned and -1 means
            ignore.
        max_overlaps (FloatTensor): the iou between the predicted box and its
            assigned truth box.
        labels (None | LongTensor): If specified, for each predicted box
            indicates the category label of the assigned truth box.
    Example:
        >>> # An assign result between 4 predicted boxes and 9 true boxes
        >>> # where only two boxes were assigned.
        >>> num_gts = 9
        >>> max_overlaps = torch.LongTensor([0, .5, .9, 0])
        >>> gt_inds = torch.LongTensor([-1, 1, 2, 0])
        >>> labels = torch.LongTensor([0, 3, 4, 0])
        >>> self = AssignResult(num_gts, gt_inds, max_overlaps, labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(4,), max_overlaps.shape=(4,),
                      labels.shape=(4,))>
        >>> # Force addition of gt labels (when adding gt as proposals)
        >>> new_labels = torch.LongTensor([3, 4, 5])
        >>> self.add_gt_(new_labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(7,), max_overlaps.shape=(7,),
                      labels.shape=(7,))>
    """
    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        # Interface for possible user-defined properties
        self._extra_properties = {}

    @property
    def num_preds(self):
        """int: the number of predictions in this assignment"""
        return len(self.gt_inds)

    def set_extra_property(self, key, value):
        """Set user-defined new property."""
        assert key not in self.info
        self._extra_properties[key] = value

    def get_extra_property(self, key):
        """Get user-defined property."""
        return self._extra_properties.get(key, None)

    @property
    def info(self):
        """dict: a dictionary of info about the object"""
        basic_info = {
            'num_gts': self.num_gts,
            'num_preds': self.num_preds,
            'gt_inds': self.gt_inds,
            'max_overlaps': self.max_overlaps,
            'labels': self.labels,
        }
        basic_info.update(self._extra_properties)
        return basic_info

    @classmethod
    def random(cls, **kwargs):
        """Create random AssignResult for tests or debugging.
        Args:
            num_preds: number of predicted boxes
            num_gts: number of true boxes
            p_ignore (float): probability of a predicted box assinged to an
                ignored truth
            p_assigned (float): probability of a predicted box not being
                assigned
            p_use_label (float | bool): with labels or not
            rng (None | int | numpy.random.RandomState): seed or state
        Returns:
            :obj:`AssignResult`: Randomly generated assign results.
        Example:
            >>> from mmdet.core.bbox.assigners.assign_result import *  # NOQA
            >>> self = AssignResult.random()
            >>> print(self.info)
        """
        from mmdet.core.bbox import demodata
        rng = demodata.ensure_rng(kwargs.get('rng', None))

        num_gts = kwargs.get('num_gts', None)
        num_preds = kwargs.get('num_preds', None)
        p_ignore = kwargs.get('p_ignore', 0.3)
        p_assigned = kwargs.get('p_assigned', 0.7)
        p_use_label = kwargs.get('p_use_label', 0.5)
        num_classes = kwargs.get('p_use_label', 3)

        if num_gts is None:
            num_gts = rng.randint(0, 8)
        if num_preds is None:
            num_preds = rng.randint(0, 16)

        if num_gts == 0:
            max_overlaps = torch.zeros(num_preds, dtype=torch.float32)
            gt_inds = torch.zeros(num_preds, dtype=torch.int64)
            if p_use_label is True or p_use_label < rng.rand():
                labels = torch.zeros(num_preds, dtype=torch.int64)
            else:
                labels = None
        else:
            import numpy as np
            # Create an overlap for each predicted box
            max_overlaps = torch.from_numpy(rng.rand(num_preds))

            # Construct gt_inds for each predicted box
            is_assigned = torch.from_numpy(rng.rand(num_preds) < p_assigned)
            # maximum number of assignments constraints
            n_assigned = min(num_preds, min(num_gts, is_assigned.sum()))

            assigned_idxs = np.where(is_assigned)[0]
            rng.shuffle(assigned_idxs)
            assigned_idxs = assigned_idxs[0:n_assigned]
            assigned_idxs.sort()

            is_assigned[:] = 0
            is_assigned[assigned_idxs] = True

            is_ignore = torch.from_numpy(
                rng.rand(num_preds) < p_ignore) & is_assigned

            gt_inds = torch.zeros(num_preds, dtype=torch.int64)

            true_idxs = np.arange(num_gts)
            rng.shuffle(true_idxs)
            true_idxs = torch.from_numpy(true_idxs)
            gt_inds[is_assigned] = true_idxs[:n_assigned]

            gt_inds = torch.from_numpy(
                rng.randint(1, num_gts + 1, size=num_preds))
            gt_inds[is_ignore] = -1
            gt_inds[~is_assigned] = 0
            max_overlaps[~is_assigned] = 0

            if p_use_label is True or p_use_label < rng.rand():
                if num_classes == 0:
                    labels = torch.zeros(num_preds, dtype=torch.int64)
                else:
                    labels = torch.from_numpy(
                        # remind that we set FG labels to [0, num_class-1]
                        # since mmdet v2.0
                        # BG cat_id: num_class
                        rng.randint(0, num_classes, size=num_preds))
                    labels[~is_assigned] = 0
            else:
                labels = None

        self = cls(num_gts, gt_inds, max_overlaps, labels)
        return self

    def add_gt_(self, gt_labels):
        """Add ground truth as assigned results.
        Args:
            gt_labels (torch.Tensor): Labels of gt boxes
        """
        self_inds = torch.arange(1,
                                 len(gt_labels) + 1,
                                 dtype=torch.long,
                                 device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])

        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(len(gt_labels)), self.max_overlaps])

        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])


class AlignOTAAssigner(BaseAssigner):
    """Computes matching between predictions and ground truth.
    Args:
        center_radius (int | float, optional): Ground truth center size
            to judge whether a prior is in center. Default 2.5.
        candidate_topk (int, optional): The candidate top-k which used to
            get top-k ious to calculate dynamic-k. Default 10.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 3.0.
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
    """
    def __init__(self,
                 center_radius=2.5,
                 candidate_topk=10,
                 iou_weight=3.0,
                 cls_weight=1.0):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight

    def assign(self,
               pred_scores,
               priors,
               decoded_bboxes,
               gt_bboxes,
               gt_labels,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Assign gt to priors using SimOTA. It will switch to CPU mode when
        GPU is out of memory.
        Args:
            pred_scores (Tensor): Classification scores of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Predicted bboxes, a 2D-Tensor with shape
                [num_priors, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            eps (float): A value added to the denominator for numerical
                stability. Default 1e-7.
        Returns:
            assign_result (obj:`AssignResult`): The assigned result.
        """
        try:
            assign_result = self._assign(pred_scores, priors, decoded_bboxes,
                                         gt_bboxes, gt_labels,
                                         gt_bboxes_ignore, eps)
            return assign_result
        except RuntimeError:
            origin_device = pred_scores.device
            warnings.warn('OOM RuntimeError is raised due to the huge memory '
                          'cost during label assignment. CPU mode is applied '
                          'in this batch. If you want to avoid this issue, '
                          'try to reduce the batch size or image size.')
            torch.cuda.empty_cache()

            pred_scores = pred_scores.cpu()
            priors = priors.cpu()
            decoded_bboxes = decoded_bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu().float()
            gt_labels = gt_labels.cpu()

            assign_result = self._assign(pred_scores, priors, decoded_bboxes,
                                         gt_bboxes, gt_labels,
                                         gt_bboxes_ignore, eps)
            assign_result.gt_inds = assign_result.gt_inds.to(origin_device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(
                origin_device)
            assign_result.labels = assign_result.labels.to(origin_device)

            return assign_result

    def _assign(self,
                pred_scores,
                priors,
                decoded_bboxes,
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore=None,
                eps=1e-7):
        """Assign gt to priors using SimOTA.
        Args:
            pred_scores (Tensor): Classification scores of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Predicted bboxes, a 2D-Tensor with shape
                [num_priors, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            eps (float): A value added to the denominator for numerical
                stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        INF = 100000000
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes, ),
                                                   0,
                                                   dtype=torch.long)
        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(
            priors, gt_bboxes)
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                          -1,
                                                          dtype=torch.long)
            return AssignResult(num_gt,
                                assigned_gt_inds,
                                max_overlaps,
                                labels=assigned_labels)

        pairwise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + eps)

        gt_onehot_label = (F.one_hot(gt_labels.to(
            torch.int64), pred_scores.shape[-1]).float().unsqueeze(0).repeat(
                num_valid, 1, 1))

        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores

        cls_cost = F.binary_cross_entropy(
            valid_pred_scores, soft_label,
            reduction='none') * scale_factor.abs().pow(2.0)

        cls_cost = cls_cost.sum(dim=-1)
        cost_matrix = (cls_cost * self.cls_weight +
                       iou_cost * self.iou_weight +
                       (~is_in_boxes_and_center) * INF)
        matched_pred_ious, matched_gt_inds = \
            self.dynamic_k_matching(
                cost_matrix, pairwise_ious, num_gt, valid_mask)

        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes, ),
                                                 -INF,
                                                 dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(num_gt,
                            assigned_gt_inds,
                            max_overlaps,
                            labels=assigned_labels)

    def get_in_gt_and_in_center_info(self, priors, gt_bboxes):
        num_gt = gt_bboxes.size(0)

        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)

        # is prior centers in gt bboxes, shape: [n_prior, n_gt]
        l_ = repeated_x - gt_bboxes[:, 0]
        t_ = repeated_y - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - repeated_x
        b_ = gt_bboxes[:, 3] - repeated_y

        deltas = torch.stack([l_, t_, r_, b_], dim=1)
        is_in_gts = deltas.min(dim=1).values > 0
        is_in_gts_all = is_in_gts.sum(dim=1) > 0

        # is prior centers in gt centers
        gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        ct_box_l = gt_cxs - self.center_radius * repeated_stride_x
        ct_box_t = gt_cys - self.center_radius * repeated_stride_y
        ct_box_r = gt_cxs + self.center_radius * repeated_stride_x
        ct_box_b = gt_cys + self.center_radius * repeated_stride_y

        cl_ = repeated_x - ct_box_l
        ct_ = repeated_y - ct_box_t
        cr_ = ct_box_r - repeated_x
        cb_ = ct_box_b - repeated_y

        ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
        is_in_cts = ct_deltas.min(dim=1).values > 0
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        # in boxes or in centers, shape: [num_priors]
        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all

        # both in boxes and centers, shape: [num_fg, num_gt]
        is_in_boxes_and_centers = (is_in_gts[is_in_gts_or_centers, :]
                                   & is_in_cts[is_in_gts_or_centers, :])
        return is_in_gts_or_centers, is_in_boxes_and_centers

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        matching_matrix = torch.zeros_like(cost)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx],
                                    k=dynamic_ks[gt_idx].item(),
                                    largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(cost[prior_match_gt_mask, :],
                                              dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0.0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix *
                             pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds
```

#### damo/base_models/core/atss_assigner.py

```python
# Copyright (c) OpenMMLab. All rights reserved.

import torch

from .bbox_calculator import BboxOverlaps2D


class AssignResult(object):
    """Stores assignments between predicted and truth boxes.
    Attributes:
        num_gts (int): the number of truth boxes considered when computing this
            assignment
        gt_inds (LongTensor): for each predicted box indicates the 1-based
            index of the assigned truth box. 0 means unassigned and -1 means
            ignore.
        max_overlaps (FloatTensor): the iou between the predicted box and its
            assigned truth box.
        labels (None | LongTensor): If specified, for each predicted box
            indicates the category label of the assigned truth box.
    Example:
        >>> # An assign result between 4 predicted boxes and 9 true boxes
        >>> # where only two boxes were assigned.
        >>> num_gts = 9
        >>> max_overlaps = torch.LongTensor([0, .5, .9, 0])
        >>> gt_inds = torch.LongTensor([-1, 1, 2, 0])
        >>> labels = torch.LongTensor([0, 3, 4, 0])
        >>> self = AssignResult(num_gts, gt_inds, max_overlaps, labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(4,), max_overlaps.shape=(4,),
                      labels.shape=(4,))>
        >>> # Force addition of gt labels (when adding gt as proposals)
        >>> new_labels = torch.LongTensor([3, 4, 5])
        >>> self.add_gt_(new_labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(7,), max_overlaps.shape=(7,),
                      labels.shape=(7,))>
    """
    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        # Interface for possible user-defined properties
        self._extra_properties = {}

    @property
    def num_preds(self):
        """int: the number of predictions in this assignment"""
        return len(self.gt_inds)

    def set_extra_property(self, key, value):
        """Set user-defined new property."""
        assert key not in self.info
        self._extra_properties[key] = value

    def get_extra_property(self, key):
        """Get user-defined property."""
        return self._extra_properties.get(key, None)

    @property
    def info(self):
        """dict: a dictionary of info about the object"""
        basic_info = {
            'num_gts': self.num_gts,
            'num_preds': self.num_preds,
            'gt_inds': self.gt_inds,
            'max_overlaps': self.max_overlaps,
            'labels': self.labels,
        }
        basic_info.update(self._extra_properties)
        return basic_info

    @classmethod
    def random(cls, **kwargs):
        """Create random AssignResult for tests or debugging.
        Args:
            num_preds: number of predicted boxes
            num_gts: number of true boxes
            p_ignore (float): probability of a predicted box assinged to an
                ignored truth
            p_assigned (float): probability of a predicted box not being
                assigned
            p_use_label (float | bool): with labels or not
            rng (None | int | numpy.random.RandomState): seed or state
        Returns:
            :obj:`AssignResult`: Randomly generated assign results.
        Example:
            >>> from mmdet.core.bbox.assigners.assign_result import *  # NOQA
            >>> self = AssignResult.random()
            >>> print(self.info)
        """
        from mmdet.core.bbox import demodata
        rng = demodata.ensure_rng(kwargs.get('rng', None))

        num_gts = kwargs.get('num_gts', None)
        num_preds = kwargs.get('num_preds', None)
        p_ignore = kwargs.get('p_ignore', 0.3)
        p_assigned = kwargs.get('p_assigned', 0.7)
        p_use_label = kwargs.get('p_use_label', 0.5)
        num_classes = kwargs.get('p_use_label', 3)

        if num_gts is None:
            num_gts = rng.randint(0, 8)
        if num_preds is None:
            num_preds = rng.randint(0, 16)

        if num_gts == 0:
            max_overlaps = torch.zeros(num_preds, dtype=torch.float32)
            gt_inds = torch.zeros(num_preds, dtype=torch.int64)
            if p_use_label is True or p_use_label < rng.rand():
                labels = torch.zeros(num_preds, dtype=torch.int64)
            else:
                labels = None
        else:
            import numpy as np
            # Create an overlap for each predicted box
            max_overlaps = torch.from_numpy(rng.rand(num_preds))

            # Construct gt_inds for each predicted box
            is_assigned = torch.from_numpy(rng.rand(num_preds) < p_assigned)
            # maximum number of assignments constraints
            n_assigned = min(num_preds, min(num_gts, is_assigned.sum()))

            assigned_idxs = np.where(is_assigned)[0]
            rng.shuffle(assigned_idxs)
            assigned_idxs = assigned_idxs[0:n_assigned]
            assigned_idxs.sort()

            is_assigned[:] = 0
            is_assigned[assigned_idxs] = True

            is_ignore = torch.from_numpy(
                rng.rand(num_preds) < p_ignore) & is_assigned

            gt_inds = torch.zeros(num_preds, dtype=torch.int64)

            true_idxs = np.arange(num_gts)
            rng.shuffle(true_idxs)
            true_idxs = torch.from_numpy(true_idxs)
            gt_inds[is_assigned] = true_idxs[:n_assigned]

            gt_inds = torch.from_numpy(
                rng.randint(1, num_gts + 1, size=num_preds))
            gt_inds[is_ignore] = -1
            gt_inds[~is_assigned] = 0
            max_overlaps[~is_assigned] = 0

            if p_use_label is True or p_use_label < rng.rand():
                if num_classes == 0:
                    labels = torch.zeros(num_preds, dtype=torch.int64)
                else:
                    labels = torch.from_numpy(
                        # remind that we set FG labels to [0, num_class-1]
                        # since mmdet v2.0
                        # BG cat_id: num_class
                        rng.randint(0, num_classes, size=num_preds))
                    labels[~is_assigned] = 0
            else:
                labels = None

        self = cls(num_gts, gt_inds, max_overlaps, labels)
        return self

    def add_gt_(self, gt_labels):
        """Add ground truth as assigned results.
        Args:
            gt_labels (torch.Tensor): Labels of gt boxes
        """
        self_inds = torch.arange(1,
                                 len(gt_labels) + 1,
                                 dtype=torch.long,
                                 device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])

        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(len(gt_labels)), self.max_overlaps])

        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])


class ATSSAssigner:
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
    """
    def __init__(self,
                 topk,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1):
        self.topk = topk
        self.iou_calculator = BboxOverlaps2D()
        self.ignore_iof_thr = ignore_iof_thr

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as postive
        6. limit the positive sample's center in gt


        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes)

        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(num_gt,
                                assigned_gt_inds,
                                max_overlaps,
                                labels=assigned_labels)

        # compute center distance between all bbox and gt
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)

        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(bboxes,
                                                  gt_bboxes_ignore,
                                                  mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_k = min(self.topk, bboxes_per_level)
            _, topk_idxs_per_level = distances_per_level.topk(selectable_k,
                                                              dim=0,
                                                              largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]
        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(assigned_gt_inds > 0,
                                     as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1].type(torch.int64)
        else:
            assigned_labels = None
        return AssignResult(num_gt,
                            assigned_gt_inds,
                            max_overlaps,
                            labels=assigned_labels)
```

#### damo/base_models/core/ops.py

```python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .weight_init import kaiming_init, constant_init
from damo.utils import make_divisible


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


def get_activation(name='silu', inplace=True):
    if name is None:
        return nn.Identity()

    if isinstance(name, str):
        if name == 'silu':
            module = nn.SiLU(inplace=inplace)
        elif name == 'relu':
            module = nn.ReLU(inplace=inplace)
        elif name == 'lrelu':
            module = nn.LeakyReLU(0.1, inplace=inplace)
        elif name == 'swish':
            module = Swish(inplace=inplace)
        elif name == 'hardsigmoid':
            module = nn.Hardsigmoid(inplace=inplace)
        elif name == 'identity':
            module = nn.Identity()
        else:
            raise AttributeError('Unsupported act type: {}'.format(name))
        return module

    elif isinstance(name, nn.Module):
        return name

    else:
        raise AttributeError('Unsupported act type: {}'.format(name))


def get_norm(name, out_channels):
    if name == 'bn':
        module = nn.BatchNorm2d(out_channels)
    elif name == 'gn':
        module = nn.GroupNorm(out_channels)
    else:
        raise NotImplementedError
    return module


class ConvBNAct(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""
    def __init__(
        self,
        in_channels,
        out_channels,
        ksize,
        stride=1,
        groups=1,
        bias=False,
        act='silu',
        norm='bn',
        reparam=False,
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        if norm is not None:
            self.bn = get_norm(norm, out_channels)
        if act is not None:
            self.act = get_activation(act, inplace=True)
        self.with_norm = norm is not None
        self.with_act = act is not None

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        if self.with_act:
            x = self.act(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 activation='silu'):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBNAct(in_channels,
                               hidden_channels,
                               1,
                               stride=1,
                               act=activation)
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvBNAct(conv2_channels,
                               out_channels,
                               1,
                               stride=1,
                               act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x

def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
    return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)



class Focus(nn.Module):
    """Focus width and height information into channel space."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=1,
                 stride=1,
                 act='silu'):
        super().__init__()
        self.conv = ConvBNAct(in_channels * 4,
                              out_channels,
                              ksize,
                              stride,
                              act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class MobileV3Block(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 kernel_size,
                 stride,
                 act='silu',
                 reparam=False,
                 block_type='k1kx',
                 depthwise=False,):
        super(MobileV3Block, self).__init__()
        self.stride = stride
        self.exp_ratio = 3.0
        branch_features = math.ceil(out_c * self.exp_ratio)
        branch_features = make_divisible(branch_features)

        #assert (self.stride != 1) or (in_c == branch_features << 1)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            get_activation(act),
            depthwise_conv(
                branch_features,
                branch_features,
                kernel_size=5,
                stride=self.stride,
                padding=2,
            ),
            nn.BatchNorm2d(branch_features),
            get_activation(act),
            nn.Conv2d(
                branch_features,
                out_c,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_c),
        )
        self.use_shotcut = self.stride == 1 and in_c == out_c

    def forward(self, x):
        if self.use_shotcut:
            return x + self.conv(x)
        else:
            return self.conv(x)





class BasicBlock_3x3_Reverse(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_hidden_ratio,
                 ch_out,
                 act='relu',
                 shortcut=True,
                 depthwise=False):
        super(BasicBlock_3x3_Reverse, self).__init__()
        assert ch_in == ch_out
        ch_hidden = int(ch_in * ch_hidden_ratio)
        if not depthwise:
            self.conv1 = ConvBNAct(ch_hidden, ch_out, 3, stride=1, act=act)
            self.conv2 = RepConv(ch_in, ch_hidden, 3, stride=1, act=act)
        else:
            self.conv = MobileV3Block(in_c=ch_in, out_c=ch_out, btn_c=None,
                kernel_size=5, stride=1, act=act)


        self.shortcut = shortcut
        self.depthwise = depthwise

    def forward(self, x):
        if not self.depthwise:
            y = self.conv2(x)
            y = self.conv1(y)
            if self.shortcut:
                return x + y
            else:
                return y
        else:
            return self.conv(x)



class DepthwiseConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias="auto",
        norm_cfg="bn",
        act="ReLU",
        inplace=True,
        order=("depthwise", "dwnorm", "act", "pointwise", "pwnorm", "act"),
    ):
        super(DepthwiseConv, self).__init__()
        assert act is None or isinstance(act, str)
        self.act = act
        self.inplace = inplace
        self.order = order
        padding = (kernel_size - 1) //2

        self.with_norm = norm_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = False if self.with_norm else True
        self.with_bias = bias

        # build convolution layer
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.depthwise.in_channels
        self.out_channels = self.pointwise.out_channels
        self.kernel_size = self.depthwise.kernel_size
        self.stride = self.depthwise.stride
        self.padding = self.depthwise.padding
        self.dilation = self.depthwise.dilation
        self.transposed = self.depthwise.transposed
        self.output_padding = self.depthwise.output_padding

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if 'dwnorm' in self.order:
                self.dwnorm = get_norm(norm_cfg, in_channels)
            if 'pwnorm' in self.order:
                self.pwnorm = get_norm(norm_cfg, out_channels)

        # build activation layer
        if self.act:
            self.act = get_activation(self.act)

        # Use msra init by default
        self.init_weights()

    def init_weights(self):
        if self.act == "lrelu":
            nonlinearity = "leaky_relu"
        else:
            nonlinearity = "relu"
        kaiming_init(self.depthwise, nonlinearity=nonlinearity)
        kaiming_init(self.pointwise, nonlinearity=nonlinearity)

        if self.with_norm:
            if 'dwnorm' in self.order:
                constant_init(self.dwnorm, 1, bias=0)
            if 'pwnorm' in self.order:
                constant_init(self.pwnorm, 1, bias=0)

    def forward(self, x):
        for layer_name in self.order:
            if layer_name != "act":
                layer = self.__getattr__(layer_name)
                x = layer(x)
            elif layer_name == "act" and self.act:
                x = self.act(x)
        return x




class SPP(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        k,
        pool_size,
        act='swish',
    ):
        super(SPP, self).__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(kernel_size=size,
                                stride=1,
                                padding=size // 2,
                                ceil_mode=False)
            self.add_module('pool{}'.format(i), pool)
            self.pool.append(pool)
        self.conv = ConvBNAct(ch_in, ch_out, k, act=act)

    def forward(self, x):
        outs = [x]

        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)

        y = self.conv(y)
        return y


class CSPStage(nn.Module):
    def __init__(self,
                 block_fn,
                 ch_in,
                 ch_hidden_ratio,
                 ch_out,
                 n,
                 act='swish',
                 spp=False,
                 depthwise=False):
        super(CSPStage, self).__init__()

        split_ratio = 2
        ch_first = int(ch_out // split_ratio)
        ch_mid = int(ch_out - ch_first)
        self.conv1 = ConvBNAct(ch_in, ch_first, 1, act=act)
        self.conv2 = ConvBNAct(ch_in, ch_mid, 1, act=act)
        self.convs = nn.Sequential()

        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == 'BasicBlock_3x3_Reverse':
                self.convs.add_module(
                    str(i),
                    BasicBlock_3x3_Reverse(next_ch_in,
                                           ch_hidden_ratio,
                                           ch_mid,
                                           act=act,
                                           shortcut=True,
                                           depthwise=depthwise))
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module(
                    'spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
            next_ch_in = ch_mid
        self.conv3 = ConvBNAct(ch_mid * n + ch_first, ch_out, 1, act=act)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)

        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)
        y = self.conv3(y)
        return y


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module(
        'conv',
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  groups=groups,
                  bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepConv(nn.Module):
    '''RepConv is a basic rep-style block, including training and deploy status
    Code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 deploy=False,
                 act='relu',
                 norm=None):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        if isinstance(act, str):
            self.nonlinearity = get_activation(act)
        else:
            self.nonlinearity = act

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = None
            self.rbr_dense = conv_bn(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=stride,
                                   padding=padding_11,
                                   groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True
```

#### damo/base_models/core/bbox_calculator.py

```python
# Copyright (c) OpenMMLab. All rights reserved.

import torch


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.
    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.
    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
            Possible keys includes the following.
            - iou_thr (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
                number of boxes is large (e.g., 200k). To avoid OOM during
                training, the users could set `split_thr` to a small value.
                If the number of boxes is greater than the threshold, it will
                perform NMS on each group of boxes separately and sequentially.
                Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class.
    Returns:
        tuple: kept dets and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]

    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr or torch.onnx.is_in_onnx_export():
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        # -1 indexing works abnormal in TensorRT
        # This assumes `dets` has 5 dimensions where
        # the last dimension is score.
        # TODO: more elegant way to handle the dimension issue.
        # Some type of nms would reweight the score, such as SoftNMS
        scores = dets[:, 4]
    else:
        max_num = nms_cfg_.pop('max_num', -1)
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = scores.new_zeros(scores.size())
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero(as_tuple=False).view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    return torch.cat([boxes, scores[:, None]], -1), keep


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0),
                                              num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # (TODO): as ONNX does not support repeat now,
    # we have to use this ugly code
    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                    -1)).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        return bboxes, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]


def fast_nms(multi_bboxes,
             multi_scores,
             multi_coeffs,
             score_thr,
             iou_thr,
             top_k,
             max_num=-1):
    """Fast NMS in `YOLACT <https://arxiv.org/abs/1904.02689>`_.

    Fast NMS allows already-removed detections to suppress other detections so
    that every instance can be decided to be kept or discarded in parallel,
    which is not possible in traditional NMS. This relaxation allows us to
    implement Fast NMS entirely in standard GPU-accelerated matrix operations.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
        multi_coeffs (Tensor): shape (n, #class*coeffs_dim).
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): IoU threshold to be considered as conflicted.
        top_k (int): if there are more than top_k bboxes before NMS,
            only top top_k will be kept.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept. If -1, keep all the bboxes.
            Default: -1.

    Returns:
        tuple: (bboxes, labels, coefficients), tensors of shape (k, 5), (k, 1),
            and (k, coeffs_dim). Labels are 0-based.
    """

    scores = multi_scores[:, :-1].t()  # [#class, n]
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]  # [#class, topk]
    num_classes, num_dets = idx.size()
    boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    coeffs = multi_coeffs[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = bbox_overlaps(boxes, boxes)  # [#class, topk, topk]
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = iou_max <= iou_thr

    # Second thresholding introduces 0.2 mAP gain at negligible time cost
    keep *= scores > score_thr

    # Assign each kept detection to its corresponding class
    classes = torch.arange(num_classes,
                           device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    coeffs = coeffs[keep]
    scores = scores[keep]

    # Only keep the top max_num highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    if max_num > 0:
        idx = idx[:max_num]
        scores = scores[:max_num]

    classes = classes[idx]
    boxes = boxes[idx]
    coeffs = coeffs[idx]

    cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
    return cls_dets, classes, coeffs


class BboxOverlaps2D(object):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""
    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.
        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.
        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]
        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.
    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.
    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or "iof" (intersection over
            foreground).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.
    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )
    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] -
                                                   bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] -
                                                   bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious
```

#### damo/base_models/core/utils.py

```python
# Copyright (c) OpenMMLab. All rights reserved.

from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn


class Scale(nn.Module):
    """A learnable scale parameter.
    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.
    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """
    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.
    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.
    Args:
        func (Function): A function that will be applied to a list of
            arguments
    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret


def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.
    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets
```

#### damo/base_models/core/weight_init.py

```python
# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch.nn as nn


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init
```

#### damo/base_models/necks/__init__.py

```python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import copy

from .giraffe_fpn_btn import GiraffeNeckV2


def build_neck(cfg):
    neck_cfg = copy.deepcopy(cfg)
    name = neck_cfg.pop('name')
    if name == 'GiraffeNeckV2':
        return GiraffeNeckV2(**neck_cfg)
    else:
        raise NotImplementedError
```

#### damo/base_models/necks/giraffe_fpn_btn.py

```python
import torch
import torch.nn as nn

from ..core.ops import ConvBNAct, CSPStage, DepthwiseConv


class GiraffeNeckV2(nn.Module):
    def __init__(
        self,
        depth=1.0,
        hidden_ratio=1.0,
        in_features=[2, 3, 4],
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        act='silu',
        spp=False,
        block_name='BasicBlock',
        depthwise=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.in_channels = in_channels
        self.out_channels = out_channels
        Conv = DepthwiseConv if depthwise else ConvBNAct

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # node x3: input x0, x1
        self.bu_conv13 = Conv(in_channels[1], in_channels[1], 3, 2, act=act)
        self.merge_3 = CSPStage(block_name,
                                in_channels[1] + in_channels[2],
                                hidden_ratio,
                                in_channels[2],
                                round(3 * depth),
                                act=act,
                                spp=spp,
                                depthwise=depthwise)

        # node x4: input x1, x2, x3
        self.bu_conv24 = Conv(in_channels[0], in_channels[0], 3, 2, act=act)
        self.merge_4 = CSPStage(block_name,
                                in_channels[0] + in_channels[1] +
                                in_channels[2],
                                hidden_ratio,
                                in_channels[1],
                                round(3 * depth),
                                act=act,
                                spp=spp,
                                depthwise=depthwise)

        # node x5: input x2, x4
        self.merge_5 = CSPStage(block_name,
                                in_channels[1] + in_channels[0],
                                hidden_ratio,
                                out_channels[0],
                                round(3 * depth),
                                act=act,
                                spp=spp,
                                depthwise=depthwise)

        # node x7: input x4, x5
        self.bu_conv57 = Conv(out_channels[0], out_channels[0], 3, 2, act=act)
        self.merge_7 = CSPStage(block_name,
                                out_channels[0] + in_channels[1],
                                hidden_ratio,
                                out_channels[1],
                                round(3 * depth),
                                act=act,
                                spp=spp,
                                depthwise=depthwise)

        # node x6: input x3, x4, x7
        self.bu_conv46 = Conv(in_channels[1], in_channels[1], 3, 2, act=act)
        self.bu_conv76 = Conv(out_channels[1], out_channels[1], 3, 2, act=act)
        self.merge_6 = CSPStage(block_name,
                                in_channels[1] + out_channels[1] +
                                in_channels[2],
                                hidden_ratio,
                                out_channels[2],
                                round(3 * depth),
                                act=act,
                                spp=spp,
                                depthwise=depthwise)

    def init_weights(self):
        pass

    def forward(self, out_features):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        [x2, x1, x0] = out_features

        # node x3
        x13 = self.bu_conv13(x1)
        x3 = torch.cat([x0, x13], 1)
        x3 = self.merge_3(x3)

        # node x4
        x34 = self.upsample(x3)
        x24 = self.bu_conv24(x2)
        x4 = torch.cat([x1, x24, x34], 1)
        x4 = self.merge_4(x4)

        # node x5
        x45 = self.upsample(x4)
        x5 = torch.cat([x2, x45], 1)
        x5 = self.merge_5(x5)

        # node x8
        # x8 = x5

        # node x7
        x57 = self.bu_conv57(x5)
        x7 = torch.cat([x4, x57], 1)
        x7 = self.merge_7(x7)

        # node x6
        x46 = self.bu_conv46(x4)
        x76 = self.bu_conv76(x7)
        x6 = torch.cat([x3, x46, x76], 1)
        x6 = self.merge_6(x6)

        outputs = (x5, x7, x6)
        return outputs
```

#### damo/base_models/heads/__init__.py

```python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import copy

from .zero_head import ZeroHead


def build_head(cfg):

    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop('name')
    if name == 'ZeroHead':
        return ZeroHead(**head_cfg)
    else:
        raise NotImplementedError
```

#### damo/base_models/heads/zero_head.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from damo.utils import postprocess

from ..core.ops import ConvBNAct
from ..core.ota_assigner import AlignOTAAssigner
from ..core.utils import Scale, multi_apply, reduce_mean
from ..core.weight_init import bias_init_with_prob, normal_init
from ..losses.gfocal_loss import (DistributionFocalLoss, GIoULoss,
                                  QualityFocalLoss)

from loguru import logger

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    """
    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        """
        b, hw, _, _ = x.size()
        x = x.reshape(b * hw * 4, self.reg_max + 1)
        y = self.project.type_as(x).unsqueeze(1)
        x = torch.matmul(x, y).reshape(b, hw, 4)
        return x


class ZeroHead(nn.Module):
    """Ref to Generalized Focal Loss V2: Learning Reliable Localization Quality
    Estimation for Dense Object Detection.
    """
    def __init__(
            self,
            num_classes,
            in_channels,
            stacked_convs=4,  # 4
            feat_channels=256,
            reg_max=12,
            strides=[8, 16, 32],
            norm='gn',
            act='relu',
            nms_conf_thre=0.05,
            nms_iou_thre=0.7,
            nms=True,
            legacy=True,
            last_kernel_size=3,
            export_with_post=True,
            **kwargs):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.stacked_convs = stacked_convs
        self.last_kernel_size = last_kernel_size
        self.export_with_post = export_with_post
        self.act = act
        self.strides = strides
        if stacked_convs == 0:
            feat_channels = in_channels
        if isinstance(feat_channels, list):
            self.feat_channels = feat_channels
        else:
            self.feat_channels = [feat_channels] * len(self.strides)
        if legacy:
            # add 1 for keep consistance with former models
            self.cls_out_channels = num_classes + 1
        else:
            self.cls_out_channels = num_classes
        self.reg_max = reg_max

        self.nms = nms
        self.nms_conf_thre = nms_conf_thre
        self.nms_iou_thre = nms_iou_thre

        self.assigner = AlignOTAAssigner(center_radius=2.5,
                                         cls_weight=1.0,
                                         iou_weight=3.0)

        self.feat_size = [torch.zeros(4) for _ in strides]

        super(ZeroHead, self).__init__()
        self.integral = Integral(self.reg_max)
        self.loss_dfl = DistributionFocalLoss(loss_weight=0.25)
        self.loss_cls = QualityFocalLoss(use_sigmoid=False,
                                         beta=2.0,
                                         loss_weight=1.0)
        self.loss_bbox = GIoULoss(loss_weight=2.0)

        self._init_layers()

    def _build_not_shared_convs(self, in_channel, feat_channels):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = feat_channels if i > 0 else in_channel
            kernel_size = 3 if i > 0 else 1
            cls_convs.append(
                ConvBNAct(chn,
                          feat_channels,
                          kernel_size,
                          stride=1,
                          groups=1,
                          norm='bn',
                          act=self.act))
            reg_convs.append(
                ConvBNAct(chn,
                          feat_channels,
                          kernel_size,
                          stride=1,
                          groups=1,
                          norm='bn',
                          act=self.act))

        return cls_convs, reg_convs

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for i in range(len(self.strides)):
            cls_convs, reg_convs = self._build_not_shared_convs(
                self.in_channels[i], self.feat_channels[i])
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

        self.gfl_cls = nn.ModuleList([
            nn.Conv2d(self.feat_channels[i],
                      self.cls_out_channels,
                      self.last_kernel_size, # 3
                      padding=self.last_kernel_size//2) for i in range(len(self.strides))
        ])

        self.gfl_reg = nn.ModuleList([
            nn.Conv2d(self.feat_channels[i],
                      4 * (self.reg_max + 1),
                      self.last_kernel_size, # 3
                      padding=self.last_kernel_size//2) for i in range(len(self.strides))
        ])

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        for cls_conv in self.cls_convs:
            for m in cls_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        for reg_conv in self.reg_convs:
            for m in reg_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
            normal_init(self.gfl_reg[i], std=0.01)

    def forward(self, xin, labels=None, imgs=None, aux_targets=None):
        if self.training:
            return self.forward_train(xin=xin, labels=labels, imgs=imgs)
        else:
            return self.forward_eval(xin=xin, labels=labels, imgs=imgs)

    def forward_train(self, xin, labels=None, imgs=None, aux_targets=None):

        # prepare labels during training
        b, c, h, w = xin[0].shape
        if labels is not None:
            gt_bbox_list = []
            gt_cls_list = []
            for label in labels:
                gt_bbox_list.append(label.bbox)
                gt_cls_list.append((label.get_field('labels')).long())

        # prepare priors for label assignment and bbox decode
        mlvl_priors_list = [
            self.get_single_level_center_priors(xin[i].shape[0],
                                                xin[i].shape[-2:],
                                                stride,
                                                dtype=torch.float32,
                                                device=xin[0].device)
            for i, stride in enumerate(self.strides)
        ]
        mlvl_priors = torch.cat(mlvl_priors_list, dim=1)

        # forward for bboxes and classification prediction
        cls_scores, bbox_preds, bbox_before_softmax = multi_apply(
            self.forward_single,
            xin,
            self.cls_convs,
            self.reg_convs,
            self.gfl_cls,
            self.gfl_reg,
            self.scales,
        )
        cls_scores = torch.cat(cls_scores, dim=1)
        bbox_preds = torch.cat(bbox_preds, dim=1)
        bbox_before_softmax = torch.cat(bbox_before_softmax, dim=1)

        # calculating losses
        loss = self.loss(
            cls_scores,
            bbox_preds,
            bbox_before_softmax,
            gt_bbox_list,
            gt_cls_list,
            mlvl_priors,
        )
        return loss

    def forward_eval(self, xin, labels=None, imgs=None):

        # prepare priors for label assignment and bbox decode
        if self.feat_size[0] != xin[0].shape:
            mlvl_priors_list = [
                self.get_single_level_center_priors(xin[i].shape[0],
                                                    xin[i].shape[-2:],
                                                    stride,
                                                    dtype=torch.float32,
                                                    device=xin[0].device)
                for i, stride in enumerate(self.strides)
            ]
            self.mlvl_priors = torch.cat(mlvl_priors_list, dim=1)
            self.feat_size[0] = xin[0].shape

        # forward for bboxes and classification prediction
        cls_scores, bbox_preds = multi_apply(
            self.forward_single,
            xin,
            self.cls_convs,
            self.reg_convs,
            self.gfl_cls,
            self.gfl_reg,
            self.scales,
        )

        if self.export_with_post:
            cls_scores_new, bbox_preds_new = [], []
            for cls_score, bbox_pred in zip(cls_scores, bbox_preds):
                N, C, H, W = bbox_pred.size()
                bbox_pred = F.softmax(bbox_pred.reshape(N, 4, self.reg_max + 1, H, W),
                    dim=2)
                bbox_pred = bbox_pred.reshape(N, 4, self.reg_max + 1, H, W)
                cls_score = cls_score.flatten(start_dim=2).permute(
                    0, 2, 1)
                bbox_pred = bbox_pred.flatten(start_dim=3).permute(
                    0, 3, 1, 2)
                cls_scores_new.append(cls_score)
                bbox_preds_new.append(bbox_pred)

            cls_scores = torch.cat(cls_scores_new, dim=1)[:, :, :self.num_classes]
            bbox_preds = torch.cat(bbox_preds_new, dim=1)
            bbox_preds = self.integral(bbox_preds) * self.mlvl_priors[..., 2, None]
            bbox_preds = distance2bbox(self.mlvl_priors[..., :2], bbox_preds)

            if self.nms:
                output = postprocess(cls_scores, bbox_preds, self.num_classes,
                                 self.nms_conf_thre, self.nms_iou_thre, imgs)
                return output
        return cls_scores, bbox_preds

    def forward_single(self, x, cls_convs, reg_convs, gfl_cls, gfl_reg, scale):
        """Forward feature of a single scale level.

        """
        cls_feat = x
        reg_feat = x

        for cls_conv, reg_conv in zip(cls_convs, reg_convs):
            cls_feat = cls_conv(cls_feat)
            reg_feat = reg_conv(reg_feat)

        bbox_pred = scale(gfl_reg(reg_feat)).float()
        cls_score = gfl_cls(cls_feat).sigmoid()
        N, C, H, W = bbox_pred.size()
        if self.training:
            bbox_before_softmax = bbox_pred.reshape(N, 4, self.reg_max + 1, H,
                                                    W)
            bbox_before_softmax = bbox_before_softmax.flatten(
                start_dim=3).permute(0, 3, 1, 2)

            bbox_pred = F.softmax(bbox_pred.reshape(N, 4, self.reg_max + 1, H, W),
                                 dim=2)

            bbox_pred = bbox_pred.reshape(N, 4, self.reg_max + 1, H, W)

            cls_score = cls_score.flatten(start_dim=2).permute(
                0, 2, 1)  # N, h*w, self.num_classes+1
            bbox_pred = bbox_pred.flatten(start_dim=3).permute(
                0, 3, 1, 2)  # N, h*w, 4, self.reg_max+1

        if self.training:
            return cls_score, bbox_pred, bbox_before_softmax
        else:
            return cls_score, bbox_pred

    def get_single_level_center_priors(self, batch_size, featmap_size, stride,
                                       dtype, device):

        h, w = featmap_size
        x_range = (torch.arange(0, int(w), dtype=dtype,
                                device=device)) * stride
        y_range = (torch.arange(0, int(h), dtype=dtype,
                                device=device)) * stride

        x = x_range.repeat(h, 1)
        y = y_range.unsqueeze(-1).repeat(1, w)

        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0], ), stride)
        priors = torch.stack([x, y, strides, strides], dim=-1)

        return priors.unsqueeze(0).repeat(batch_size, 1, 1)

    def loss(
        self,
        cls_scores,
        bbox_preds,
        bbox_before_softmax,
        gt_bboxes,
        gt_labels,
        mlvl_center_priors,
        gt_bboxes_ignore=None,
    ):
        """Compute losses of the head.

        """
        device = cls_scores[0].device

        # get decoded bboxes for label assignment
        dis_preds = self.integral(bbox_preds) * mlvl_center_priors[..., 2,
                                                                   None]
        decoded_bboxes = distance2bbox(mlvl_center_priors[..., :2], dis_preds)
        cls_reg_targets = self.get_targets(cls_scores,
                                           decoded_bboxes,
                                           gt_bboxes,
                                           mlvl_center_priors,
                                           gt_labels_list=gt_labels)

        if cls_reg_targets is None:
            return None

        (labels_list, label_scores_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, dfl_targets_list, num_pos) = cls_reg_targets

        num_total_pos = max(
            reduce_mean(torch.tensor(num_pos).type(
                torch.float).to(device)).item(), 1.0)

        labels = torch.cat(labels_list, dim=0)
        label_scores = torch.cat(label_scores_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        dfl_targets = torch.cat(dfl_targets_list, dim=0)

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # bbox_preds = bbox_preds.reshape(-1, 4 * (self.reg_max + 1))
        bbox_before_softmax = bbox_before_softmax.reshape(
            -1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)

        loss_qfl = self.loss_cls(cls_scores, (labels, label_scores),
                                 avg_factor=num_total_pos)

        pos_inds = torch.nonzero((labels >= 0) & (labels < self.num_classes),
                                 as_tuple=False).squeeze(1)

        weight_targets = cls_scores.detach()
        weight_targets = weight_targets.max(dim=1)[0][pos_inds]
        norm_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)

        if len(pos_inds) > 0:
            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=1.0 * norm_factor,
            )
            loss_dfl = self.loss_dfl(
                bbox_before_softmax[pos_inds].reshape(-1, self.reg_max + 1),
                dfl_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * norm_factor,
            )
        else:
            loss_bbox = bbox_preds.sum() / norm_factor * 0.0
            loss_dfl = bbox_preds.sum() / norm_factor * 0.0
            logger.info(f'No Positive Samples on {bbox_preds.device}! May cause performance decrease. loss_bbox:{loss_bbox:.3f}, loss_dfl:{loss_dfl:.3f}, loss_qfl:{loss_qfl:.3f} ')

        total_loss = loss_qfl + loss_bbox + loss_dfl

        return dict(
            total_loss=total_loss,
            loss_cls=loss_qfl,
            loss_bbox=loss_bbox,
            loss_dfl=loss_dfl,
        )

    def get_targets(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    mlvl_center_priors,
                    gt_labels_list=None,
                    unmap_outputs=True):
        """Get targets for GFL head.

        """
        num_imgs = mlvl_center_priors.shape[0]

        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        (all_labels, all_label_scores, all_label_weights, all_bbox_targets,
         all_bbox_weights, all_dfl_targets, all_pos_num) = multi_apply(
             self.get_target_single,
             mlvl_center_priors,
             cls_scores,
             bbox_preds,
             gt_bboxes_list,
             gt_labels_list,
         )
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        all_pos_num = sum(all_pos_num)

        return (all_labels, all_label_scores, all_label_weights,
                all_bbox_targets, all_bbox_weights, all_dfl_targets,
                all_pos_num)

    def get_target_single(self,
                          center_priors,
                          cls_scores,
                          bbox_preds,
                          gt_bboxes,
                          gt_labels,
                          unmap_outputs=True,
                          gt_bboxes_ignore=None):
        """Compute regression, classification targets for anchors in a single
        image.

        """
        # assign gt and sample anchors

        num_valid_center = center_priors.shape[0]

        labels = center_priors.new_full((num_valid_center, ),
                                        self.num_classes,
                                        dtype=torch.long)
        label_weights = center_priors.new_zeros(num_valid_center,
                                                dtype=torch.float)
        label_scores = center_priors.new_zeros(num_valid_center,
                                               dtype=torch.float)

        bbox_targets = torch.zeros_like(center_priors)
        bbox_weights = torch.zeros_like(center_priors)
        dfl_targets = torch.zeros_like(center_priors)

        if gt_labels.size(0) == 0:

            return (labels, label_scores, label_weights, bbox_targets,
                    bbox_weights, dfl_targets, 0)

        assign_result = self.assigner.assign(cls_scores.detach(),
                                             center_priors,
                                             bbox_preds.detach(), gt_bboxes,
                                             gt_labels)

        pos_inds, neg_inds, pos_bbox_targets, pos_assign_gt_inds = self.sample(
            assign_result, gt_bboxes)
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
            labels[pos_inds] = gt_labels[pos_assign_gt_inds]
            label_scores[pos_inds] = pos_ious
            label_weights[pos_inds] = 1.0

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            dfl_targets[pos_inds, :] = (bbox2distance(
                center_priors[pos_inds, :2] / center_priors[pos_inds, None, 2],
                pos_bbox_targets / center_priors[pos_inds, None, 2],
                self.reg_max))
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        # map up to original set of anchors

        return (labels, label_scores, label_weights, bbox_targets,
                bbox_weights, dfl_targets, pos_inds.size(0))

    def sample(self, assign_result, gt_bboxes):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0,
                                 as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0,
                                 as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]

        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds
```

#### damo/base_models/backbones/tinynas_csp.py

```python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import torch
import torch.nn as nn

from ..core.ops import Focus, RepConv, SPPBottleneck, get_activation


class ConvKXBN(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super(ConvKXBN, self).__init__()
        self.conv1 = nn.Conv2d(in_c,
                               out_c,
                               kernel_size,
                               stride, (kernel_size - 1) // 2,
                               groups=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn1(self.conv1(x))

    def fuseforward(self, x):
        return self.conv1(x)


class ConvKXBNRELU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, act='silu'):
        super(ConvKXBNRELU, self).__init__()
        self.conv = ConvKXBN(in_c, out_c, kernel_size, stride)
        if act is None:
            self.activation_function = torch.relu
        else:
            self.activation_function = get_activation(act)

    def forward(self, x):
        output = self.conv(x)
        return self.activation_function(output)


class ResConvBlock(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 kernel_size,
                 stride,
                 act='silu',
                 reparam=False,
                 block_type='k1kx'):
        super(ResConvBlock, self).__init__()
        self.stride = stride
        if block_type == 'k1kx':
            self.conv1 = ConvKXBN(in_c, btn_c, kernel_size=1, stride=1)
        else:
            self.conv1 = ConvKXBN(in_c,
                                  btn_c,
                                  kernel_size=kernel_size,
                                  stride=1)
        if not reparam:
            self.conv2 = ConvKXBN(btn_c, out_c, kernel_size, stride)
        else:
            self.conv2 = RepConv(btn_c,
                                 out_c,
                                 kernel_size,
                                 stride,
                                 act='identity')

        self.activation_function = get_activation(act)

        if in_c != out_c and stride != 2:
            self.residual_proj = ConvKXBN(in_c, out_c, kernel_size=1, stride=1)
        else:
            self.residual_proj = None

    def forward(self, x):
        if self.residual_proj is not None:
            reslink = self.residual_proj(x)
        else:
            reslink = x
        x = self.conv1(x)
        x = self.activation_function(x)
        x = self.conv2(x)
        if self.stride != 2:
            x = x + reslink
        x = self.activation_function(x)
        return x


class CSPStem(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 stride,
                 kernel_size,
                 num_blocks,
                 act='silu',
                 reparam=False,
                 block_type='k1kx'):
        super(CSPStem, self).__init__()
        self.in_channels = in_c
        self.out_channels = out_c
        self.stride = stride
        if self.stride == 2:
            self.num_blocks = num_blocks - 1
        else:
            self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.act = act
        self.block_type = block_type
        out_c = out_c // 2

        if act is None:
            self.act = torch.relu
        else:
            self.act = get_activation(act)
        self.block_list = nn.ModuleList()
        for block_id in range(self.num_blocks):
            if self.stride == 1 and block_id == 0:
                in_c = in_c // 2
            else:
                in_c = out_c
            the_block = ResConvBlock(in_c,
                                     out_c,
                                     btn_c,
                                     kernel_size,
                                     stride=1,
                                     act=act,
                                     reparam=reparam,
                                     block_type=block_type)
            self.block_list.append(the_block)

    def forward(self, x):
        output = x
        for block in self.block_list:
            output = block(output)
        return output


class TinyNAS(nn.Module):
    def __init__(self,
                 structure_info=None,
                 out_indices=[2, 3, 4],
                 with_spp=False,
                 use_focus=False,
                 act='silu',
                 reparam=False):
        super(TinyNAS, self).__init__()
        self.out_indices = out_indices
        self.block_list = nn.ModuleList()
        self.stride_list = []

        for idx, block_info in enumerate(structure_info):
            the_block_class = block_info['class']
            if the_block_class == 'ConvKXBNRELU':
                if use_focus and idx == 0:
                    the_block = Focus(block_info['in'],
                                      block_info['out'],
                                      block_info['k'],
                                      act=act)
                else:
                    the_block = ConvKXBNRELU(block_info['in'],
                                             block_info['out'],
                                             block_info['k'],
                                             block_info['s'],
                                             act=act)
            elif the_block_class == 'SuperResConvK1KX':
                the_block = CSPStem(block_info['in'],
                                    block_info['out'],
                                    block_info['btn'],
                                    block_info['s'],
                                    block_info['k'],
                                    block_info['L'],
                                    act=act,
                                    reparam=reparam,
                                    block_type='k1kx')
            elif the_block_class == 'SuperResConvKXKX':
                the_block = CSPStem(block_info['in'],
                                    block_info['out'],
                                    block_info['btn'],
                                    block_info['s'],
                                    block_info['k'],
                                    block_info['L'],
                                    act=act,
                                    reparam=reparam,
                                    block_type='kxkx')
            else:
                raise NotImplementedError

            self.block_list.append(the_block)

        self.csp_stage = nn.ModuleList()
        self.csp_stage.append(self.block_list[0])
        self.csp_stage.append(CSPWrapper(self.block_list[1]))
        self.csp_stage.append(CSPWrapper(self.block_list[2]))
        self.csp_stage.append(
            CSPWrapper((self.block_list[3], self.block_list[4])))
        self.csp_stage.append(CSPWrapper(self.block_list[5],
                                         with_spp=with_spp))
        del self.block_list

    def init_weights(self, pretrain=None):
        pass

    def forward(self, x):
        output = x
        stage_feature_list = []
        for idx, block in enumerate(self.csp_stage):
            output = block(output)
            if idx in self.out_indices:
                stage_feature_list.append(output)
        return stage_feature_list


class CSPWrapper(nn.Module):
    def __init__(self, convstem, act='relu', reparam=False, with_spp=False):

        super(CSPWrapper, self).__init__()
        self.with_spp = with_spp
        if isinstance(convstem, tuple):
            in_c = convstem[0].in_channels
            out_c = convstem[-1].out_channels
            hidden_dim = convstem[0].out_channels // 2
            _convstem = nn.ModuleList()
            for modulelist in convstem:
                for layer in modulelist.block_list:
                    _convstem.append(layer)
        else:
            in_c = convstem.in_channels
            out_c = convstem.out_channels
            hidden_dim = out_c // 2
            _convstem = convstem.block_list

        self.convstem = nn.ModuleList()
        for layer in _convstem:
            self.convstem.append(layer)

        self.act = get_activation(act)
        self.downsampler = ConvKXBNRELU(in_c,
                                        hidden_dim * 2,
                                        3,
                                        2,
                                        act=self.act)
        if self.with_spp:
            self.spp = SPPBottleneck(hidden_dim * 2, hidden_dim * 2)
        if len(self.convstem) > 0:
            self.conv_start = ConvKXBNRELU(hidden_dim * 2,
                                           hidden_dim,
                                           1,
                                           1,
                                           act=self.act)
            self.conv_shortcut = ConvKXBNRELU(hidden_dim * 2,
                                              out_c // 2,
                                              1,
                                              1,
                                              act=self.act)
            self.conv_fuse = ConvKXBNRELU(out_c, out_c, 1, 1, act=self.act)

    def forward(self, x):
        x = self.downsampler(x)
        if self.with_spp:
            x = self.spp(x)
        if len(self.convstem) > 0:
            shortcut = self.conv_shortcut(x)
            x = self.conv_start(x)
            for block in self.convstem:
                x = block(x)
            x = torch.cat((x, shortcut), dim=1)
            x = self.conv_fuse(x)
        return x


def load_tinynas_net(backbone_cfg):
    # load masternet model to path
    import ast

    struct_str = ''.join([x.strip() for x in backbone_cfg.net_structure_str])
    struct_info = ast.literal_eval(struct_str)
    for layer in struct_info:
        if 'nbitsA' in layer:
            del layer['nbitsA']
        if 'nbitsW' in layer:
            del layer['nbitsW']

    model = TinyNAS(structure_info=struct_info,
                    out_indices=backbone_cfg.out_indices,
                    with_spp=backbone_cfg.with_spp,
                    use_focus=backbone_cfg.use_focus,
                    act=backbone_cfg.act,
                    reparam=backbone_cfg.reparam)

    return model
```

#### damo/base_models/backbones/tinynas_res.py

```python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import torch
import torch.nn as nn

from ..core.ops import Focus, RepConv, SPPBottleneck, get_activation


class ConvKXBN(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super(ConvKXBN, self).__init__()
        self.conv1 = nn.Conv2d(in_c,
                               out_c,
                               kernel_size,
                               stride, (kernel_size - 1) // 2,
                               groups=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn1(self.conv1(x))

    def fuseforward(self, x):
        return self.conv1(x)


class ConvKXBNRELU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, act='silu'):
        super(ConvKXBNRELU, self).__init__()
        self.conv = ConvKXBN(in_c, out_c, kernel_size, stride)
        if act is None:
            self.activation_function = torch.relu
        else:
            self.activation_function = get_activation(act)

    def forward(self, x):
        output = self.conv(x)
        return self.activation_function(output)


class ResConvBlock(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 kernel_size,
                 stride,
                 act='silu',
                 reparam=False,
                 block_type='k1kx'):
        super(ResConvBlock, self).__init__()
        self.stride = stride
        if block_type == 'k1kx':
            self.conv1 = ConvKXBN(in_c, btn_c, kernel_size=1, stride=1)
        else:
            self.conv1 = ConvKXBN(in_c,
                                  btn_c,
                                  kernel_size=kernel_size,
                                  stride=1)

        if not reparam:
            self.conv2 = ConvKXBN(btn_c, out_c, kernel_size, stride)
        else:
            self.conv2 = RepConv(btn_c,
                                 out_c,
                                 kernel_size,
                                 stride,
                                 act='identity')

        self.activation_function = get_activation(act)

        if in_c != out_c and stride != 2:
            self.residual_proj = ConvKXBN(in_c, out_c, 1, 1)
        else:
            self.residual_proj = None

    def forward(self, x):
        if self.residual_proj is not None:
            reslink = self.residual_proj(x)
        else:
            reslink = x
        x = self.conv1(x)
        x = self.activation_function(x)
        x = self.conv2(x)
        if self.stride != 2:
            x = x + reslink
        x = self.activation_function(x)
        return x


class SuperResStem(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 kernel_size,
                 stride,
                 num_blocks,
                 with_spp=False,
                 act='silu',
                 reparam=False,
                 block_type='k1kx'):
        super(SuperResStem, self).__init__()
        if act is None:
            self.act = torch.relu
        else:
            self.act = get_activation(act)
        self.block_list = nn.ModuleList()
        for block_id in range(num_blocks):
            if block_id == 0:
                in_channels = in_c
                out_channels = out_c
                this_stride = stride
                this_kernel_size = kernel_size
            else:
                in_channels = out_c
                out_channels = out_c
                this_stride = 1
                this_kernel_size = kernel_size
            the_block = ResConvBlock(in_channels,
                                     out_channels,
                                     btn_c,
                                     this_kernel_size,
                                     this_stride,
                                     act=act,
                                     reparam=reparam,
                                     block_type=block_type)
            self.block_list.append(the_block)
            if block_id == 0 and with_spp:
                self.block_list.append(
                    SPPBottleneck(out_channels, out_channels))

    def forward(self, x):
        output = x
        for block in self.block_list:
            output = block(output)
        return output


class TinyNAS(nn.Module):
    def __init__(self,
                 structure_info=None,
                 out_indices=[2, 4, 5],
                 with_spp=False,
                 use_focus=False,
                 act='silu',
                 reparam=False):
        super(TinyNAS, self).__init__()
        self.out_indices = out_indices
        self.block_list = nn.ModuleList()

        for idx, block_info in enumerate(structure_info):
            the_block_class = block_info['class']
            if the_block_class == 'ConvKXBNRELU':
                if use_focus:
                    the_block = Focus(block_info['in'],
                                      block_info['out'],
                                      block_info['k'],
                                      act=act)
                else:
                    the_block = ConvKXBNRELU(block_info['in'],
                                             block_info['out'],
                                             block_info['k'],
                                             block_info['s'],
                                             act=act)
                self.block_list.append(the_block)
            elif the_block_class == 'SuperResConvK1KX':
                spp = with_spp if idx == len(structure_info) - 1 else False
                the_block = SuperResStem(block_info['in'],
                                         block_info['out'],
                                         block_info['btn'],
                                         block_info['k'],
                                         block_info['s'],
                                         block_info['L'],
                                         spp,
                                         act=act,
                                         reparam=reparam,
                                         block_type='k1kx')
                self.block_list.append(the_block)
            elif the_block_class == 'SuperResConvKXKX':
                spp = with_spp if idx == len(structure_info) - 1 else False
                the_block = SuperResStem(block_info['in'],
                                         block_info['out'],
                                         block_info['btn'],
                                         block_info['k'],
                                         block_info['s'],
                                         block_info['L'],
                                         spp,
                                         act=act,
                                         reparam=reparam,
                                         block_type='kxkx')
                self.block_list.append(the_block)
            else:
                raise NotImplementedError

    def init_weights(self, pretrain=None):
        pass

    def forward(self, x):
        output = x
        stage_feature_list = []
        for idx, block in enumerate(self.block_list):
            output = block(output)
            if idx in self.out_indices:
                stage_feature_list.append(output)
        return stage_feature_list


def load_tinynas_net(backbone_cfg):
    # load masternet model to path
    import ast

    struct_str = ''.join([x.strip() for x in backbone_cfg.net_structure_str])
    struct_info = ast.literal_eval(struct_str)
    for layer in struct_info:
        if 'nbitsA' in layer:
            del layer['nbitsA']
        if 'nbitsW' in layer:
            del layer['nbitsW']

    model = TinyNAS(structure_info=struct_info,
                    out_indices=backbone_cfg.out_indices,
                    with_spp=backbone_cfg.with_spp,
                    use_focus=backbone_cfg.use_focus,
                    act=backbone_cfg.act,
                    reparam=backbone_cfg.reparam)

    return model
```

#### damo/base_models/backbones/__init__.py

```python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import copy

from .tinynas_csp import load_tinynas_net as load_tinynas_net_csp
from .tinynas_res import load_tinynas_net as load_tinynas_net_res
from .tinynas_mob import load_tinynas_net as load_tinynas_net_mob


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop('name')
    if name == 'TinyNAS_res':
        return load_tinynas_net_res(backbone_cfg)
    elif name == 'TinyNAS_csp':
        return load_tinynas_net_csp(backbone_cfg)
    elif name == 'TinyNAS_mob':
        return load_tinynas_net_mob(backbone_cfg)
    else:
        print(f'{name} is not supported yet!')
```

#### damo/base_models/backbones/tinynas_mob.py

```python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..core.ops import Focus, RepConv, SPPBottleneck, get_activation, DepthwiseConv
from damo.utils import make_divisible


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvKXBN(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, depthwise=False):
        super(ConvKXBN, self).__init__()
        if depthwise:
            self.conv1 = DepthwiseConv(in_channels=in_c,
                                       out_channels=out_c,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=(kernel_size-1) // 2,
                                       norm_cfg="bn",
                                       act="relu",
                                       order=("depthwise","pointwise")
                                       )
        else:
            self.conv1 = nn.Conv2d(in_c,
                               out_c,
                               kernel_size,
                               stride, (kernel_size - 1) // 2,
                               groups=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn1(self.conv1(x))

    def fuseforward(self, x):
        return self.conv1(x)


class ConvKXBNRELU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, act='silu'):
        super(ConvKXBNRELU, self).__init__()
        self.conv = ConvKXBN(in_c, out_c, kernel_size, stride)
        if act is None:
            self.activation_function = torch.relu
        else:
            self.activation_function = get_activation(act)

    def forward(self, x):
        output = self.conv(x)
        return self.activation_function(output)

def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
    return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class MobileV3Block(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 kernel_size,
                 stride,
                 act='silu',
                 reparam=False,
                 block_type='k1kx',
                 depthwise=False,
                 use_se=False,
                 block_pos=None):
        super(MobileV3Block, self).__init__()
        self.stride = stride
        self.exp_ratio = 2.5
        if block_pos is not None:
            self.exp_ratio = 3.5 + (block_pos-1) * 0.5

        branch_features = math.ceil(out_c * self.exp_ratio)
        branch_features = make_divisible(branch_features)

        # assert (self.stride != 1) or (in_c == branch_features << 1)

        if use_se:
            SELayer = SEModule
        else:
            SELayer = nn.Identity

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            get_activation(act),
            depthwise_conv(
                branch_features,
                branch_features,
                kernel_size=5,
                stride=self.stride,
                padding=2,
            ),
            nn.BatchNorm2d(branch_features),
            SELayer(branch_features),
            get_activation(act),
            nn.Conv2d(
                branch_features,
                out_c,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_c),
        )
        self.use_shotcut = self.stride == 1 and in_c == out_c

    def forward(self, x):
        if self.use_shotcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SuperResStem(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 kernel_size,
                 stride,
                 num_blocks,
                 with_spp=False,
                 act='silu',
                 reparam=False,
                 block_type='k1kx',
                 depthwise=False,
                 use_se=False,
                 block_pos=None,):
        super(SuperResStem, self).__init__()
        if act is None:
            self.act = torch.relu
        else:
            self.act = get_activation(act)
        self.block_list = nn.ModuleList()
        basic_block = MobileV3Block
        for block_id in range(num_blocks):
            if block_id == 0:
                in_channels = in_c
                out_channels = out_c
                this_stride = stride
                this_kernel_size = kernel_size
            else:
                in_channels = out_c
                out_channels = out_c
                this_stride = 1
                this_kernel_size = kernel_size
            the_block = basic_block(in_channels,
                                     out_channels,
                                     btn_c,
                                     this_kernel_size,
                                     this_stride,
                                     act=act,
                                     reparam=reparam,
                                     block_type=block_type,
                                     depthwise=depthwise,
                                     use_se=use_se,
                                     block_pos=block_pos,)
            self.block_list.append(the_block)
            if block_id == 0 and with_spp:
                self.block_list.append(
                    SPPBottleneck(out_channels, out_channels))

    def forward(self, x):
        output = x
        for block in self.block_list:
            output = block(output)
        return output


class TinyNAS(nn.Module):
    def __init__(self,
                 structure_info=None,
                 out_indices=[2, 4, 5],
                 with_spp=False,
                 use_focus=False,
                 act='silu',
                 reparam=False,
                 depthwise=False,
                 use_se=False,):
        super(TinyNAS, self).__init__()
        self.out_indices = out_indices
        self.block_list = nn.ModuleList()

        for idx, block_info in enumerate(structure_info):
            the_block_class = block_info['class']
            if the_block_class == 'ConvKXBNRELU':
                if use_focus:
                    the_block = Focus(block_info['in'],
                                      block_info['out'],
                                      block_info['k'],
                                      act=act)
                else:
                    the_block = ConvKXBNRELU(3,
                                             block_info['out'],
                                             block_info['k'],
                                             2,
                                             act=act)
                self.block_list.append(the_block)
            elif the_block_class == 'SuperResConvK1KX':
                spp = with_spp if idx == len(structure_info) - 1 else False
                the_block = SuperResStem(block_info['in'],
                                         block_info['out'],
                                         block_info['btn'],
                                         block_info['k'],
                                         block_info['s'],
                                         block_info['L'],
                                         spp,
                                         act=act,
                                         reparam=reparam,
                                         block_type='k1kx',
                                         depthwise=depthwise,
                                         use_se=use_se,
                                         block_pos=idx)
                self.block_list.append(the_block)
            elif the_block_class == 'SuperResConvKXKX':
                spp = with_spp if idx == len(structure_info) - 1 else False
                the_block = SuperResStem(block_info['in'],
                                         block_info['out'],
                                         block_info['btn'],
                                         block_info['k'],
                                         block_info['s'],
                                         block_info['L'],
                                         spp,
                                         act=act,
                                         reparam=reparam,
                                         block_type='kxkx',
                                         depthwise=depthwise,
                                         use_se=use_se)
                self.block_list.append(the_block)
            else:
                raise NotImplementedError

    def init_weights(self, pretrain=None):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if "first" in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)

    def forward(self, x):
        output = x
        stage_feature_list = []
        for idx, block in enumerate(self.block_list):
            output = block(output)
            if idx in self.out_indices:
                stage_feature_list.append(output)
        return stage_feature_list


def load_tinynas_net(backbone_cfg):
    # load masternet model to path
    import ast

    struct_str = ''.join([x.strip() for x in backbone_cfg.net_structure_str])
    struct_info = ast.literal_eval(struct_str)
    for layer in struct_info:
        if 'nbitsA' in layer:
            del layer['nbitsA']
        if 'nbitsW' in layer:
            del layer['nbitsW']

    model = TinyNAS(structure_info=struct_info,
                    out_indices=backbone_cfg.out_indices,
                    with_spp=backbone_cfg.with_spp,
                    use_focus=backbone_cfg.use_focus,
                    act=backbone_cfg.act,
                    reparam=backbone_cfg.reparam,
                    depthwise=backbone_cfg.depthwise,
                    use_se=backbone_cfg.use_se,)

    return model
```

##### damo/base_models/backbones/nas_backbones/tinynas_L45_kxkx.txt

```
  [ {'class': 'ConvKXBNRELU', 'in': 3, 'k': 3, 'nbitsA': 8, 'nbitsW': 8, 'out': 32, 's': 1},
                             { 'L': 3,
                               'btn': 96,
                               'class': 'SuperResConvKXKX',
                               'in': 32,
                               'inner_class': 'ResConvKXKX',
                               'k': 3,
                               'nbitsA': [8, 8, 8, 8],
                               'nbitsW': [8, 8, 8, 8],
                               'out': 128,
                               's': 2},
                             { 'L': 5,
                               'btn': 96,
                               'class': 'SuperResConvKXKX',
                               'in': 128,
                               'inner_class': 'ResConvKXKX',
                               'k': 3,
                               'nbitsA': [8, 8, 8, 8, 8, 8, 8, 8],
                               'nbitsW': [8, 8, 8, 8, 8, 8, 8, 8],
                               'out': 128,
                               's': 2},
                             { 'L': 5,
                               'btn': 384,
                               'class': 'SuperResConvKXKX',
                               'in': 128,
                               'inner_class': 'ResConvKXKX',
                               'k': 3,
                               'nbitsA': [8, 8, 8, 8, 8, 8, 8, 8],
                               'nbitsW': [8, 8, 8, 8, 8, 8, 8, 8],
                               'out': 256,
                               's': 2},
                             { 'L': 5,
                               'btn': 384,
                               'class': 'SuperResConvKXKX',
                               'in': 256,
                               'inner_class': 'ResConvKXKX',
                               'k': 3,
                               'nbitsA': [8, 8, 8, 8, 8, 8, 8, 8],
                               'nbitsW': [8, 8, 8, 8, 8, 8, 8, 8],
                               'out': 256,
                               's': 1},
                             { 'L': 4,
                               'btn': 384,
                               'class': 'SuperResConvKXKX',
                               'in': 256,
                               'inner_class': 'ResConvKXKX',
                               'k': 3,
                               'nbitsA': [8, 8, 8, 8, 8, 8],
                               'nbitsW': [8, 8, 8, 8, 8, 8],
                               'out': 512,
                               's': 2}
]
```

##### damo/base_models/backbones/nas_backbones/tinynas_L35_kxkx.txt

```
  [ {'class': 'ConvKXBNRELU', 'in': 3, 'k': 3, 'nbitsA': 8, 'nbitsW': 8, 'out': 32, 's': 1},
                             { 'L': 2,
                               'btn': 64,
                               'class': 'SuperResConvKXKX',
                               'in': 32,
                               'inner_class': 'ResConvKXKX',
                               'k': 3,
                               'nbitsA': [8, 8, 8, 8],
                               'nbitsW': [8, 8, 8, 8],
                               'out': 128,
                               's': 2},
                             { 'L': 4,
                               'btn': 64,
                               'class': 'SuperResConvKXKX',
                               'in': 128,
                               'inner_class': 'ResConvKXKX',
                               'k': 3,
                               'nbitsA': [8, 8, 8, 8, 8, 8, 8, 8],
                               'nbitsW': [8, 8, 8, 8, 8, 8, 8, 8],
                               'out': 128,
                               's': 2},
                             { 'L': 4,
                               'btn': 256,
                               'class': 'SuperResConvKXKX',
                               'in': 128,
                               'inner_class': 'ResConvKXKX',
                               'k': 3,
                               'nbitsA': [8, 8, 8, 8, 8, 8, 8, 8],
                               'nbitsW': [8, 8, 8, 8, 8, 8, 8, 8],
                               'out': 256,
                               's': 2},
                             { 'L': 4,
                               'btn': 256,
                               'class': 'SuperResConvKXKX',
                               'in': 256,
                               'inner_class': 'ResConvKXKX',
                               'k': 3,
                               'nbitsA': [8, 8, 8, 8, 8, 8, 8, 8],
                               'nbitsW': [8, 8, 8, 8, 8, 8, 8, 8],
                               'out': 256,
                               's': 1},
                             { 'L': 3,
                               'btn': 256,
                               'class': 'SuperResConvKXKX',
                               'in': 256,
                               'inner_class': 'ResConvKXKX',
                               'k': 3,
                               'nbitsA': [8, 8, 8, 8, 8, 8],
                               'nbitsW': [8, 8, 8, 8, 8, 8],
                               'out': 512,
                               's': 2}
]
```

##### damo/base_models/backbones/nas_backbones/tinynas_nano_middle.txt

```
[ {'class': 'ConvKXBNRELU', 'in': 3, 'k': 3, 'nbitsA': 8, 'nbitsW': 8, 'out': 16, 's': 1},
  { 'L': 2,
    'btn': 24,
    'class': 'SuperResConvK1KX',
    'in': 16,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 40,
    's': 2},
  { 'L': 2,
    'btn': 64,
    'class': 'SuperResConvK1KX',
    'in': 40,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 64,
    's': 2},
  { 'L': 2,
    'btn': 40,
    'class': 'SuperResConvK1KX',
    'in': 64,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 112,
    's': 2},
  { 'L': 2,
    'btn': 152,
    'class': 'SuperResConvK1KX',
    'in': 112,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 128,
    's': 1},
  { 'L': 1,
    'btn': 192,
    'class': 'SuperResConvK1KX',
    'in': 128,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8],
    'nbitsW': [8, 8],
    'out': 256,
    's': 2}]
```

##### damo/base_models/backbones/nas_backbones/tinynas_L20_k1kx.txt

```
[ {'class': 'ConvKXBNRELU', 'in': 3, 'k': 3, 'nbitsA': 8, 'nbitsW': 8, 'out': 24, 's': 1},
  { 'L': 2,
    'btn': 24,
    'class': 'SuperResConvK1KX',
    'in': 24,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 64,
    's': 2},
  { 'L': 2,
    'btn': 64,
    'class': 'SuperResConvK1KX',
    'in': 64,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 96,
    's': 2},
  { 'L': 2,
    'btn': 96,
    'class': 'SuperResConvK1KX',
    'in': 96,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 192,
    's': 2},
  { 'L': 2,
    'btn': 152,
    'class': 'SuperResConvK1KX',
    'in': 192,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 192,
    's': 1},
  { 'L': 1,
    'btn': 192,
    'class': 'SuperResConvK1KX',
    'in': 192,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8],
    'nbitsW': [8, 8],
    'out': 384,
    's': 2}]
```

##### damo/base_models/backbones/nas_backbones/tinynas_nano_small.txt

```
[ {'class': 'ConvKXBNRELU', 'in': 3, 'k': 3, 'nbitsA': 8, 'nbitsW': 8, 'out': 16, 's': 1},
  { 'L': 1,
    'btn': 24,
    'class': 'SuperResConvK1KX',
    'in': 16,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 24,
    's': 2},
  { 'L': 2,
    'btn': 64,
    'class': 'SuperResConvK1KX',
    'in': 24,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 40,
    's': 2},
  { 'L': 2,
    'btn': 40,
    'class': 'SuperResConvK1KX',
    'in': 40,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 64,
    's': 2},
  { 'L': 2,
    'btn': 152,
    'class': 'SuperResConvK1KX',
    'in': 64,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 80,
    's': 1},
  { 'L': 2,
    'btn': 192,
    'class': 'SuperResConvK1KX',
    'in': 80,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8],
    'nbitsW': [8, 8],
    'out': 160,
    's': 2}]
```

##### damo/base_models/backbones/nas_backbones/tinynas_L25_k1kx.txt

```
[ {'class': 'ConvKXBNRELU', 'in': 3, 'k': 3, 'nbitsA': 8, 'nbitsW': 8, 'out': 32, 's': 1},
  { 'L': 1,
    'btn': 24,
    'class': 'SuperResConvK1KX',
    'in': 32,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8],
    'nbitsW': [8, 8],
    'out': 128,
    's': 2},
  { 'L': 5,
    'btn': 88,
    'class': 'SuperResConvK1KX',
    'in': 128,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    'out': 128,
    's': 2},
  { 'L': 3,
    'btn': 128,
    'class': 'SuperResConvK1KX',
    'in': 128,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8, 8, 8],
    'out': 256,
    's': 2},
  { 'L': 2,
    'btn': 120,
    'class': 'SuperResConvK1KX',
    'in': 256,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 256,
    's': 1},
  { 'L': 1,
    'btn': 144,
    'class': 'SuperResConvK1KX',
    'in': 256,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8],
    'nbitsW': [8, 8],
    'out': 512,
    's': 2}]
```

##### damo/base_models/backbones/nas_backbones/tinynas_nano_large.txt

```
[ {'class': 'ConvKXBNRELU', 'in': 3, 'k': 3, 'nbitsA': 8, 'nbitsW': 8, 'out': 24, 's': 1},
  { 'L': 1,
    'btn': 24,
    'class': 'SuperResConvK1KX',
    'in': 24,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 48,
    's': 2},
  { 'L': 2,
    'btn': 64,
    'class': 'SuperResConvK1KX',
    'in': 48,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 80,
    's': 2},
  { 'L': 2,
    'btn': 40,
    'class': 'SuperResConvK1KX',
    'in': 80,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 160,
    's': 2},
  { 'L': 3,
    'btn': 152,
    'class': 'SuperResConvK1KX',
    'in': 160,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 160,
    's': 1},
  { 'L': 2,
    'btn': 192,
    'class': 'SuperResConvK1KX',
    'in': 160,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8],
    'nbitsW': [8, 8],
    'out': 320,
    's': 2}]
```

##### damo/base_models/backbones/nas_backbones/tinynas_L20_k1kx_nano.txt

```
[ {'class': 'ConvKXBNRELU', 'in': 3, 'k': 3, 'nbitsA': 8, 'nbitsW': 8, 'out': 16, 's': 1},
  { 'L': 2,
    'btn': 24,
    'class': 'SuperResConvK1KX',
    'in': 16,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 40,
    's': 2},
  { 'L': 2,
    'btn': 64,
    'class': 'SuperResConvK1KX',
    'in': 40,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 80,
    's': 2},
  { 'L': 2,
    'btn': 40,
    'class': 'SuperResConvK1KX',
    'in': 80,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 112,
    's': 2},
  { 'L': 2,
    'btn': 152,
    'class': 'SuperResConvK1KX',
    'in': 112,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8, 8, 8],
    'nbitsW': [8, 8, 8, 8],
    'out': 112,
    's': 1},
  { 'L': 1,
    'btn': 192,
    'class': 'SuperResConvK1KX',
    'in': 112,
    'inner_class': 'ResConvK1KX',
    'k': 3,
    'nbitsA': [8, 8],
    'nbitsW': [8, 8],
    'out': 160,
    's': 2}]
```

## datasets/coco

## configs/damoyolo_tinynasL18_Nm.py

```python
#!/usr/bin/env python3

import os

from damo.config import Config as MyConfig


class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.miscs.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split('.')[0]
        self.miscs.eval_interval_epochs = 10
        self.miscs.ckpt_interval_epochs = 10
        # optimizer
        self.train.batch_size = 256
        self.train.base_lr_per_img = 0.001 / 64
        self.train.min_lr_ratio = 0.05
        self.train.no_aug_epochs = 16
        self.train.warmup_epochs = 5

        self.train.optimizer = {
            'name': "AdamW",
            'weight_decay': 1e-2,
            'lr': 4e-3,
            }

        # augment
        self.train.augment.transform.image_max_range = (416, 416)
        self.train.augment.transform.keep_ratio = False
        self.test.augment.transform.keep_ratio = False
        self.test.augment.transform.image_max_range = (416, 416)
        self.train.augment.mosaic_mixup.mixup_prob = 0.15
        self.train.augment.mosaic_mixup.degrees = 10.0
        self.train.augment.mosaic_mixup.translate = 0.2
        self.train.augment.mosaic_mixup.shear = 0.2
        self.train.augment.mosaic_mixup.mosaic_scale = (0.1, 2.0)
        self.train.augment.mosaic_mixup.keep_ratio = False

        self.dataset.train_ann = ('coco_2017_train', )
        self.dataset.val_ann = ('coco_2017_val', )

        # backbone
        structure = self.read_structure(
            './damo/base_models/backbones/nas_backbones/tinynas_nano_middle.txt')
        TinyNAS = {
            'name': 'TinyNAS_mob',
            'net_structure_str': structure,
            'out_indices': (2, 4, 5),
            'with_spp': True,
            'use_focus': False,
            'act': 'silu',
            'reparam': False,
            'depthwise': True,
            'use_se': False,
        }

        self.model.backbone = TinyNAS

        GiraffeNeckV2 = {
            'name': 'GiraffeNeckV2',
            'depth': 0.5,
            'hidden_ratio': 0.5,
            'in_channels': [64, 128, 256],
            'out_channels': [64, 128, 256],
            'act': 'silu',
            'spp': False,
            'block_name': 'BasicBlock_3x3_Reverse',
            'depthwise': True,
        }

        self.model.neck = GiraffeNeckV2

        ZeroHead = {
            'name': 'ZeroHead',
            'num_classes': 80,
            'in_channels': [64, 128, 256],
            'stacked_convs': 0,
            'reg_max': 7,
            'act': 'silu',
            'nms_conf_thre': 0.03,
            'nms_iou_thre': 0.65,
            'legacy': False,
            'last_kernel_size': 1,
        }
        self.model.head = ZeroHead

        self.dataset.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
```

## configs/damoyolo_tinynasL25_S.py

```python
#!/usr/bin/env python3

import os

from damo.config import Config as MyConfig


class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.miscs.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split('.')[0]
        self.miscs.eval_interval_epochs = 10
        self.miscs.ckpt_interval_epochs = 10
        # optimizer
        self.train.batch_size = 256
        self.train.base_lr_per_img = 0.01 / 64
        self.train.min_lr_ratio = 0.05
        self.train.weight_decay = 5e-4
        self.train.momentum = 0.9
        self.train.no_aug_epochs = 16
        self.train.warmup_epochs = 5

        # augment
        self.train.augment.transform.image_max_range = (640, 640)
        self.train.augment.mosaic_mixup.mixup_prob = 0.15
        self.train.augment.mosaic_mixup.degrees = 10.0
        self.train.augment.mosaic_mixup.translate = 0.2
        self.train.augment.mosaic_mixup.shear = 2.0
        self.train.augment.mosaic_mixup.mosaic_scale = (0.1, 2.0)

        self.dataset.train_ann = ('coco_2017_train', )
        self.dataset.val_ann = ('coco_2017_val', )

        # backbone
        structure = self.read_structure(
            './damo/base_models/backbones/nas_backbones/tinynas_L25_k1kx.txt')
        TinyNAS = {
            'name': 'TinyNAS_res',
            'net_structure_str': structure,
            'out_indices': (2, 4, 5),
            'with_spp': True,
            'use_focus': True,
            'act': 'relu',
            'reparam': True,
        }

        self.model.backbone = TinyNAS

        GiraffeNeckV2 = {
            'name': 'GiraffeNeckV2',
            'depth': 1.0,
            'hidden_ratio': 0.75,
            'in_channels': [128, 256, 512],
            'out_channels': [128, 256, 512],
            'act': 'relu',
            'spp': False,
            'block_name': 'BasicBlock_3x3_Reverse',
        }

        self.model.neck = GiraffeNeckV2

        ZeroHead = {
            'name': 'ZeroHead',
            'num_classes': 80,
            'in_channels': [128, 256, 512],
            'stacked_convs': 0,
            'reg_max': 16,
            'act': 'silu',
            'nms_conf_thre': 0.05,
            'nms_iou_thre': 0.7,
            'legacy': False,
        }
        self.model.head = ZeroHead

        self.dataset.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
```

## configs/damoyolo_tinynasL20_Nl.py

```python
#!/usr/bin/env python3

import os

from damo.config import Config as MyConfig


class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.miscs.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split('.')[0]
        self.miscs.eval_interval_epochs = 10
        self.miscs.ckpt_interval_epochs = 10
        # optimizer
        self.train.batch_size = 256
        self.train.base_lr_per_img = 0.001 / 64
        self.train.min_lr_ratio = 0.05
        self.train.no_aug_epochs = 16
        self.train.warmup_epochs = 5

        self.train.optimizer = {
            'name': "AdamW",
            'weight_decay': 1e-2,
            'lr': 4e-3,
            }

        # augment
        self.train.augment.transform.image_max_range = (416, 416)
        self.train.augment.transform.keep_ratio = False
        self.test.augment.transform.keep_ratio = False
        self.test.augment.transform.image_max_range = (416, 416)
        self.train.augment.mosaic_mixup.mixup_prob = 0.15
        self.train.augment.mosaic_mixup.degrees = 10.0
        self.train.augment.mosaic_mixup.translate = 0.2
        self.train.augment.mosaic_mixup.shear = 0.2
        self.train.augment.mosaic_mixup.mosaic_scale = (0.1, 2.0)
        self.train.augment.mosaic_mixup.keep_ratio = False

        self.dataset.train_ann = ('coco_2017_train', )
        self.dataset.val_ann = ('coco_2017_val', )

        # backbone
        structure = self.read_structure(
            './damo/base_models/backbones/nas_backbones/tinynas_nano_large.txt')
        TinyNAS = {
            'name': 'TinyNAS_mob',
            'net_structure_str': structure,
            'out_indices': (2, 4, 5),
            'with_spp': True,
            'use_focus': False,
            'act': 'silu',
            'reparam': False,
            'depthwise': True,
            'use_se': False,
        }

        self.model.backbone = TinyNAS

        GiraffeNeckV2 = {
            'name': 'GiraffeNeckV2',
            'depth': 0.5,
            'hidden_ratio': 0.5,
            'in_channels': [80, 160, 320],
            'out_channels': [80, 160, 320],
            'act': 'silu',
            'spp': False,
            'block_name': 'BasicBlock_3x3_Reverse',
            'depthwise': True,
        }

        self.model.neck = GiraffeNeckV2

        ZeroHead = {
            'name': 'ZeroHead',
            'num_classes': 80,
            'in_channels': [80, 160, 320],
            'stacked_convs': 0,
            'reg_max': 7,
            'act': 'silu',
            'nms_conf_thre': 0.03,
            'nms_iou_thre': 0.65,
            'legacy': False,
            'last_kernel_size': 1,
        }
        self.model.head = ZeroHead

        self.dataset.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
```

## configs/damoyolo_tinynasL20_T.py

```python
#!/usr/bin/env python3

import os

from damo.config import Config as MyConfig


class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.miscs.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split('.')[0]
        self.miscs.eval_interval_epochs = 10
        self.miscs.ckpt_interval_epochs = 10
        # optimizer
        self.train.batch_size = 256
        self.train.base_lr_per_img = 0.01 / 64
        self.train.min_lr_ratio = 0.05
        self.train.weight_decay = 5e-4
        self.train.momentum = 0.9
        self.train.no_aug_epochs = 16
        self.train.warmup_epochs = 5

        # augment
        self.train.augment.transform.image_max_range = (640, 640)
        self.train.augment.mosaic_mixup.mixup_prob = 0.15
        self.train.augment.mosaic_mixup.degrees = 10.0
        self.train.augment.mosaic_mixup.translate = 0.2
        self.train.augment.mosaic_mixup.shear = 0.2
        self.train.augment.mosaic_mixup.mosaic_scale = (0.1, 2.0)

        self.dataset.train_ann = ('coco_2017_train', )
        self.dataset.val_ann = ('coco_2017_val', )

        # backbone
        structure = self.read_structure(
            './damo/base_models/backbones/nas_backbones/tinynas_L20_k1kx.txt')
        TinyNAS = {
            'name': 'TinyNAS_res',
            'net_structure_str': structure,
            'out_indices': (2, 4, 5),
            'with_spp': True,
            'use_focus': True,
            'act': 'relu',
            'reparam': True,
        }

        self.model.backbone = TinyNAS

        GiraffeNeckV2 = {
            'name': 'GiraffeNeckV2',
            'depth': 1.0,
            'hidden_ratio': 1.0,
            'in_channels': [96, 192, 384],
            'out_channels': [64, 128, 256],
            'act': 'relu',
            'spp': False,
            'block_name': 'BasicBlock_3x3_Reverse',
        }

        self.model.neck = GiraffeNeckV2

        ZeroHead = {
            'name': 'ZeroHead',
            'num_classes': 80,
            'in_channels': [64, 128, 256],
            'stacked_convs': 0,
            'reg_max': 16,
            'act': 'silu',
            'nms_conf_thre': 0.05,
            'nms_iou_thre': 0.7,
            'legacy': False,
        }
        self.model.head = ZeroHead

        self.dataset.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
```

## configs/damoyolo_tinynasL20_N.py

```python
#!/usr/bin/env python3

import os

from damo.config import Config as MyConfig


class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.miscs.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split('.')[0]
        self.miscs.eval_interval_epochs = 10
        self.miscs.ckpt_interval_epochs = 10
        # optimizer
        self.train.batch_size = 256
        self.train.base_lr_per_img = 0.001 / 64
        self.train.min_lr_ratio = 0.05
        self.train.no_aug_epochs = 16
        self.train.warmup_epochs = 5

        self.train.optimizer = {
            'name': "AdamW",
            'weight_decay': 1e-2,
            'lr': 4e-3,
            }

        # augment
        self.train.augment.transform.image_max_range = (416, 416)
        self.train.augment.transform.keep_ratio = False
        self.test.augment.transform.keep_ratio = False
        self.test.augment.transform.image_max_range = (416, 416)
        self.train.augment.mosaic_mixup.mixup_prob = 0.15
        self.train.augment.mosaic_mixup.degrees = 10.0
        self.train.augment.mosaic_mixup.translate = 0.2
        self.train.augment.mosaic_mixup.shear = 0.2
        self.train.augment.mosaic_mixup.mosaic_scale = (0.1, 2.0)
        self.train.augment.mosaic_mixup.keep_ratio = False

        self.dataset.train_ann = ('coco_2017_train', )
        self.dataset.val_ann = ('coco_2017_val', )

        # backbone
        structure = self.read_structure(
            './damo/base_models/backbones/nas_backbones/tinynas_L20_k1kx_nano.txt')
        TinyNAS = {
            'name': 'TinyNAS_mob',
            'net_structure_str': structure,
            'out_indices': (2, 4, 5),
            'with_spp': True,
            'use_focus': True,
            'act': 'silu',
            'reparam': False,
            'depthwise': True,
            'use_se': False,
        }

        self.model.backbone = TinyNAS

        GiraffeNeckV2 = {
            'name': 'GiraffeNeckV2',
            'depth': 0.5,
            'hidden_ratio': 0.5,
            'in_channels': [80, 112, 160],
            'out_channels': [64, 128, 256],
            'act': 'silu',
            'spp': False,
            'block_name': 'BasicBlock_3x3_Reverse',
            'depthwise': True,
        }

        self.model.neck = GiraffeNeckV2

        ZeroHead = {
            'name': 'ZeroHead',
            'num_classes': 80,
            'in_channels': [64, 128, 256],
            'stacked_convs': 0,
            'reg_max': 7,
            'act': 'silu',
            'nms_conf_thre': 0.03,
            'nms_iou_thre': 0.65,
            'legacy': False,
        }
        self.model.head = ZeroHead

        self.dataset.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
```

## configs/damoyolo_tinynasL35_M.py

```python
#!/usr/bin/env python3

import os

from damo.config import Config as MyConfig


class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.miscs.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split('.')[0]
        self.miscs.eval_interval_epochs = 10
        self.miscs.ckpt_interval_epochs = 10
        # optimizer
        self.train.batch_size = 256
        self.train.base_lr_per_img = 0.01 / 64
        self.train.min_lr_ratio = 0.05
        self.train.weight_decay = 5e-4
        self.train.momentum = 0.9
        self.train.no_aug_epochs = 16
        self.train.warmup_epochs = 5

        # augment
        self.train.augment.transform.image_max_range = (640, 640)
        self.train.augment.mosaic_mixup.mixup_prob = 0.15
        self.train.augment.mosaic_mixup.degrees = 10.0
        self.train.augment.mosaic_mixup.translate = 0.2
        self.train.augment.mosaic_mixup.shear = 2.0
        self.train.augment.mosaic_mixup.mosaic_scale = (0.1, 2.0)

        self.dataset.train_ann = ('coco_2017_train', )
        self.dataset.val_ann = ('coco_2017_val', )

        # backbone
        structure = self.read_structure(
            './damo/base_models/backbones/nas_backbones/tinynas_L35_kxkx.txt')
        TinyNAS = {
            'name': 'TinyNAS_csp',
            'net_structure_str': structure,
            'out_indices': (2, 3, 4),
            'with_spp': True,
            'use_focus': True,
            'act': 'silu',
            'reparam': True,
        }

        self.model.backbone = TinyNAS

        GiraffeNeckV2 = {
            'name': 'GiraffeNeckV2',
            'depth': 1.5,
            'hidden_ratio': 1.0,
            'in_channels': [128, 256, 512],
            'out_channels': [128, 256, 512],
            'act': 'silu',
            'spp': False,
            'block_name': 'BasicBlock_3x3_Reverse',
        }

        self.model.neck = GiraffeNeckV2

        ZeroHead = {
            'name': 'ZeroHead',
            'num_classes': 80,
            'in_channels': [128, 256, 512],
            'stacked_convs': 0,
            'reg_max': 16,
            'act': 'silu',
            'nms_conf_thre': 0.05,
            'nms_iou_thre': 0.7,
            'legacy': False,
        }
        self.model.head = ZeroHead

        self.dataset.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
```

## configs/damoyolo_tinynasL18_Ns.py

```python
#!/usr/bin/env python3

import os

from damo.config import Config as MyConfig


class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.miscs.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split('.')[0]
        self.miscs.eval_interval_epochs = 10
        self.miscs.ckpt_interval_epochs = 10
        # optimizer
        self.train.batch_size = 256
        self.train.base_lr_per_img = 0.001 / 64
        self.train.min_lr_ratio = 0.05
        self.train.no_aug_epochs = 16
        self.train.warmup_epochs = 5

        self.train.optimizer = {
            'name': "AdamW",
            'weight_decay': 1e-2,
            'lr': 4e-3,
            }

        # augment
        self.train.augment.transform.image_max_range = (416, 416)
        self.train.augment.transform.keep_ratio = False
        self.test.augment.transform.keep_ratio = False
        self.test.augment.transform.image_max_range = (416, 416)
        self.train.augment.mosaic_mixup.mixup_prob = 0.15
        self.train.augment.mosaic_mixup.degrees = 10.0
        self.train.augment.mosaic_mixup.translate = 0.2
        self.train.augment.mosaic_mixup.shear = 0.2
        self.train.augment.mosaic_mixup.mosaic_scale = (0.75, 1.25)
        self.train.augment.mosaic_mixup.keep_ratio = False

        self.dataset.train_ann = ('coco_2017_train', )
        self.dataset.val_ann = ('coco_2017_val', )

        # backbone
        structure = self.read_structure(
            './damo/base_models/backbones/nas_backbones/tinynas_nano_small.txt')
        TinyNAS = {
            'name': 'TinyNAS_mob',
            'net_structure_str': structure,
            'out_indices': (2, 4, 5),
            'with_spp': True,
            'use_focus': False,
            'act': 'silu',
            'reparam': False,
            'depthwise': True,
            'use_se': False,
        }

        self.model.backbone = TinyNAS

        GiraffeNeckV2 = {
            'name': 'GiraffeNeckV2',
            'depth': 0.50,
            'hidden_ratio': 0.5, # 0.5
            'in_channels': [40, 80, 160],
            'out_channels': [40, 80, 160],
            'act': 'silu',
            'spp': False,
            'block_name': 'BasicBlock_3x3_Reverse',
            'depthwise': True,
        }

        self.model.neck = GiraffeNeckV2

        ZeroHead = {
            'name': 'ZeroHead',
            'num_classes': 80,
            'in_channels': [40, 80, 160],
            'stacked_convs': 0,
            'reg_max': 7,
            'act': 'silu',
            'nms_conf_thre': 0.03,
            'nms_iou_thre': 0.65,
            'legacy': False,
            'last_kernel_size': 1,
        }
        self.model.head = ZeroHead

        self.dataset.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
```

## configs/damoyolo_tinynasL45_L.py

```python
#!/usr/bin/env python3

import os

from damo.config import Config as MyConfig


class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.miscs.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split('.')[0]
        self.miscs.eval_interval_epochs = 10
        self.miscs.ckpt_interval_epochs = 10
        # optimizer
        self.train.batch_size = 256
        self.train.base_lr_per_img = 0.01 / 64
        self.train.min_lr_ratio = 0.05
        self.train.weight_decay = 5e-4
        self.train.momentum = 0.9
        self.train.no_aug_epochs = 16
        self.train.warmup_epochs = 5

        # augment
        self.train.augment.transform.image_max_range = (640, 640)
        self.train.augment.mosaic_mixup.mixup_prob = 0.15
        self.train.augment.mosaic_mixup.degrees = 10.0
        self.train.augment.mosaic_mixup.translate = 0.2
        self.train.augment.mosaic_mixup.shear = 2.0
        self.train.augment.mosaic_mixup.mosaic_scale = (0.1, 2.0)

        self.dataset.train_ann = ('coco_2017_train', )
        self.dataset.val_ann = ('coco_2017_val', )

        # backbone
        structure = self.read_structure(
            './damo/base_models/backbones/nas_backbones/tinynas_L45_kxkx.txt')
        TinyNAS = {
            'name': 'TinyNAS_csp',
            'net_structure_str': structure,
            'out_indices': (2, 3, 4),
            'with_spp': True,
            'use_focus': True,
            'act': 'silu',
            'reparam': True,
        }

        self.model.backbone = TinyNAS

        GiraffeNeckV2 = {
            'name': 'GiraffeNeckV2',
            'depth': 2.0,
            'hidden_ratio': 1.0,
            'in_channels': [128, 256, 512],
            'out_channels': [128, 256, 512],
            'act': 'silu',
            'spp': False,
            'block_name': 'BasicBlock_3x3_Reverse',
        }

        self.model.neck = GiraffeNeckV2

        ZeroHead = {
            'name': 'ZeroHead',
            'num_classes': 80,
            'in_channels': [128, 256, 512],
            'stacked_convs': 0,
            'reg_max': 16,
            'act': 'silu',
            'nms_conf_thre': 0.05,
            'nms_iou_thre': 0.7,
            'legacy': False,
        }
        self.model.head = ZeroHead

        self.dataset.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
```

## scripts/coco_train.sh

```bash
#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=8  tools/train.py -f configs/damoyolo_tinynasL25_S.py
```

## scripts/coco_distill.sh

```bash

# from scratch distillation
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py -f configs/damoyolo_tinynasL25_S.py     --tea_config configs/damoyolo_tinynasL35_M.py --tea_ckpt ../damoyolo_tinynasL35_M.pth
```

## scripts/coco_eval.sh

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29502 tools/eval.py -f configs/damoyolo_tinynasL25_S.py -c ../damoyolo_tinynasL25_S.pth
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29502 tools/eval.py -f configs/damoyolo_tinynasL20_T.py -c ../damoyolo_tinynasL20_T.pth
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29502 tools/eval.py -f configs/damoyolo_tinynasL35_M.py -c ../damoyolo_tinynasL35_M.pth
```

