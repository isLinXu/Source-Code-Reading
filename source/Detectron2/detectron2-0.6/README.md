bao<img src=".github/Detectron2-Logo-Horz.svg" width="300" >

Detectron2 is Facebook AI Research's next generation library
that provides state-of-the-art detection and segmentation algorithms.
It is the successor of
[Detectron](https://github.com/facebookresearch/Detectron/)
and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).
It supports a number of computer vision research projects and production applications in Facebook.

<div align="center">
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>

### What's New
* Includes new capabilities such as panoptic segmentation, Densepose, Cascade R-CNN, rotated bounding boxes, PointRend,
  DeepLab, etc.
* Used as a library to support building [research projects](projects/) on top of it.
* Models can be exported to TorchScript format or Caffe2 format for deployment.
* It [trains much faster](https://detectron2.readthedocs.io/notes/benchmarks.html).

See our [blog post](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/)
to see more demos and learn about detectron2.

## Installation

See [installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).

## Getting Started

See [Getting Started with Detectron2](https://detectron2.readthedocs.io/tutorials/getting_started.html),
and the [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
to learn about basic usage.

Learn more at our [documentation](https://detectron2.readthedocs.org).
And see [projects/](projects/) for some projects that are built on top of detectron2.

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Detectron2 Model Zoo](MODEL_ZOO.md).

## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing Detectron2

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```

graph TD
    A[detectron2-0.6] --> B[.circleci]
    A --> C[.github]
    A --> D[configs]
    A --> E[detectron2]
    A --> F[docker]
    A --> G[projects]
    A --> H[tools]
    A --> I[datasets]
    A --> J[demo]
    A --> K[dev]
    A --> L[docs]
    A --> M[tests]
    
    B --> B1[config.yml]
    
    C --> C1[CODE_OF_CONDUCT.md]
    C --> C2[CONTRIBUTING.md]
    C --> C3[ISSUE_TEMPLATE]
    C --> C4[workflows]
    
    D --> D1[Base-RCNN-C4.yaml]
    D --> D2[Base-RCNN-FPN.yaml]
    D --> D3[COCO-Detection]
    D --> D4[COCO-InstanceSegmentation]
    D --> D5[COCO-Keypoints]
    D --> D6[COCO-PanopticSegmentation]
    D --> D7[Cityscapes]
    D --> D8[LVISv0.5-InstanceSegmentation]
    D --> D9[PascalVOC-Detection]
    D --> D10[new_baselines]
    
    E --> E1[checkpoint]
    E --> E2[config]
    E --> E3[data]
    E --> E4[engine]
    E --> E5[evaluation]
    E --> E6[export]
    E --> E7[layers]
    E --> E8[modeling]
    E --> E9[model_zoo]
    E --> E10[solver]
    E --> E11[structures]
    E --> E12[utils]
