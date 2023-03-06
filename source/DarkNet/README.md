# DarkNet

---

## Index

```shell
darknet/
├── cfg
│   ├── alexnet.cfg
│   ├── cifar.cfg
│   ├── cifar.test.cfg
│   ├── coco.data
│   ├── combine9k.data
│   ├── darknet19_448.cfg
│   ├── darknet19.cfg
│   ├── darknet53_448.cfg
│   ├── darknet53.cfg
│   ├── darknet9000.cfg
│   ├── darknet.cfg
│   ├── densenet201.cfg
│   ├── extraction22k.cfg
│   ├── extraction.cfg
│   ├── extraction.conv.cfg
│   ├── go.cfg
│   ├── go.test.cfg
│   ├── gru.cfg
│   ├── imagenet1k.data
│   ├── imagenet22k.dataset
│   ├── imagenet9k.hierarchy.dataset
│   ├── jnet-conv.cfg
│   ├── openimages.data
│   ├── resnet101.cfg
│   ├── resnet152.cfg
│   ├── resnet18.cfg
│   ├── resnet34.cfg
│   ├── resnet50.cfg
│   ├── resnext101-32x4d.cfg
│   ├── resnext152-32x4d.cfg
│   ├── resnext50.cfg
│   ├── rnn.cfg
│   ├── rnn.train.cfg
│   ├── strided.cfg
│   ├── t1.test.cfg
│   ├── tiny.cfg
│   ├── vgg-16.cfg
│   ├── vgg-conv.cfg
│   ├── voc.data
│   ├── writing.cfg
│   ├── yolo9000.cfg
│   ├── yolov1.cfg
│   ├── yolov1-tiny.cfg
│   ├── yolov2.cfg
│   ├── yolov2-tiny.cfg
│   ├── yolov2-tiny-voc.cfg
│   ├── yolov2-voc.cfg
│   ├── yolov3.cfg
│   ├── yolov3-openimages.cfg
│   ├── yolov3-spp.cfg
│   ├── yolov3-tiny.cfg
│   └── yolov3-voc.cfg
├── data
│   ├── 9k.labels
│   ├── 9k.names
│   ├── 9k.tree
│   ├── coco9k.map
│   ├── coco.names
│   ├── dog.jpg
│   ├── eagle.jpg
│   ├── giraffe.jpg
│   ├── goal.txt
│   ├── horses.jpg
│   ├── imagenet.labels.list
│   ├── imagenet.shortnames.list
│   ├── inet9k.map
│   ├── kite.jpg
│   ├── labels
│   │   ├── 100_0.png
│   │   └── make_labels.py
│   ├── openimages.names
│   ├── person.jpg
│   ├── scream.jpg
│   └── voc.names
├── examples
│   ├── art.c
│   ├── attention.c
│   ├── captcha.c
│   ├── cifar.c
│   ├── classifier.c
│   ├── coco.c
│   ├── darknet.c
│   ├── detector.c
│   ├── detector.py
│   ├── detector-scipy-opencv.py
│   ├── dice.c
│   ├── go.c
│   ├── instance-segmenter.c
│   ├── lsd.c
│   ├── nightmare.c
│   ├── regressor.c
│   ├── rnn.c
│   ├── rnn_vid.c
│   ├── segmenter.c
│   ├── super.c
│   ├── swag.c
│   ├── tag.c
│   ├── voxel.c
│   ├── writing.c
│   └── yolo.c
├── include
│   └── darknet.h
├── LICENSE
├── LICENSE.fuck
├── LICENSE.gen
├── LICENSE.gpl
├── LICENSE.meta
├── LICENSE.mit
├── LICENSE.v1
├── Makefile
├── python
│   ├── darknet.py
│   └── proverbot.py
├── README.md
├── scripts
│   ├── dice_label.sh
│   ├── gen_tactic.sh
│   ├── get_coco_dataset.sh
│   ├── imagenet_label.sh
│   └── voc_label.py
└── src
    ├── activation_kernels.cu
    ├── activation_layer.c
    ├── activation_layer.h
    ├── activations.c
    ├── activations.h
    ├── avgpool_layer.c
    ├── avgpool_layer.h
    ├── avgpool_layer_kernels.cu
    ├── batchnorm_layer.c
    ├── batchnorm_layer.h
    ├── blas.c
    ├── blas.h
    ├── blas_kernels.cu
    ├── box.c
    ├── box.h
    ├── classifier.h
    ├── col2im.c
    ├── col2im.h
    ├── col2im_kernels.cu
    ├── compare.c
    ├── connected_layer.c
    ├── connected_layer.h
    ├── convolutional_kernels.cu
    ├── convolutional_layer.c
    ├── convolutional_layer.h
    ├── cost_layer.c
    ├── cost_layer.h
    ├── crnn_layer.c
    ├── crnn_layer.h
    ├── crop_layer.c
    ├── crop_layer.h
    ├── crop_layer_kernels.cu
    ├── cuda.c
    ├── cuda.h
    ├── data.c
    ├── data.h
    ├── deconvolutional_kernels.cu
    ├── deconvolutional_layer.c
    ├── deconvolutional_layer.h
    ├── demo.c
    ├── demo.h
    ├── detection_layer.c
    ├── detection_layer.h
    ├── dropout_layer.c
    ├── dropout_layer.h
    ├── dropout_layer_kernels.cu
    ├── gemm.c
    ├── gemm.h
    ├── gru_layer.c
    ├── gru_layer.h
    ├── im2col.c
    ├── im2col.h
    ├── im2col_kernels.cu
    ├── image.c
    ├── image.h
    ├── image_opencv.cpp
    ├── iseg_layer.c
    ├── iseg_layer.h
    ├── l2norm_layer.c
    ├── l2norm_layer.h
    ├── layer.c
    ├── layer.h
    ├── list.c
    ├── list.h
    ├── local_layer.c
    ├── local_layer.h
    ├── logistic_layer.c
    ├── logistic_layer.h
    ├── lstm_layer.c
    ├── lstm_layer.h
    ├── matrix.c
    ├── matrix.h
    ├── maxpool_layer.c
    ├── maxpool_layer.h
    ├── maxpool_layer_kernels.cu
    ├── network.c
    ├── network.h
    ├── normalization_layer.c
    ├── normalization_layer.h
    ├── option_list.c
    ├── option_list.h
    ├── parser.c
    ├── parser.h
    ├── region_layer.c
    ├── region_layer.h
    ├── reorg_layer.c
    ├── reorg_layer.h
    ├── rnn_layer.c
    ├── rnn_layer.h
    ├── route_layer.c
    ├── route_layer.h
    ├── shortcut_layer.c
    ├── shortcut_layer.h
    ├── softmax_layer.c
    ├── softmax_layer.h
    ├── stb_image.h
    ├── stb_image_write.h
    ├── tree.c
    ├── tree.h
    ├── upsample_layer.c
    ├── upsample_layer.h
    ├── utils.c
    ├── utils.h
    ├── yolo_layer.c
    └── yolo_layer.h

8 directories, 978 files
```

