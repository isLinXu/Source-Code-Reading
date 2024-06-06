# dynamic_det

Directory tree
- train_step1.py
- demo
  - demo.jpg
  - demo.mp4
- get_dynamic_thres.py
- figure
  - performance.png
- utils
  - metrics.py
  - checkpoint.py
  - datasets.py
  - __init__.py
  - loss.py
  - plots.py
  - wandb_logging
    - log_dataset.py
    - __init__.py
    - wandb_utils.py
  - autoanchor.py
  - torch_utils.py
  - general.py
- models
  - __init__.py
  - common.py
  - yolo.py
- test.py
- hyp
  - hyp.finetune.dynamic.adam.yaml
  - hyp.scratch.p5.yaml
  - hyp.scratch.p6.yaml
- cfg
  - dy-yolov7x-step1.yaml
  - dy-yolov7-w6-step2.yaml
  - dy-yolov7-step2.yaml
  - dy-yolov7-step1.yaml
  - dy-yolov7x-step2.yaml
  - dy-yolov7-w6-step1.yaml
- data
  - coco.yaml
- train_step2.py
- detect.py

---
<!-- TOC -->
# train_step1.py

```python
import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread
import yaml
from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    set_logging, one_cycle, colorstr
from utils.loss import ComputeLoss, ComputeLossOTA, ComputeLossOTADual
from utils.plots import plot_images, plot_results, plot_lr_scheduler
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from utils.checkpoint import get_state_dict


logger = logging.getLogger(__name__)

def train(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weight, rank, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weight, opt.global_rank, opt.freeze

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    if not opt.resume:
        with open(save_dir / 'cfg.yaml', 'w') as f_d:
            with open(opt.cfg) as f_s:
                cfg_yaml = yaml.load(f_s, Loader=yaml.SafeLoader)
            yaml.dump(cfg_yaml, f_d, sort_keys=False)

    # Configure
    plots = True
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    is_coco = opt.data.endswith('coco.yaml')

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weight, map_location='cpu')['wandb_id'] if weight.endswith('.pt') and os.path.isfile(weight) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weight, epochs, hyp = opt.weight, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weight.endswith('.pt')
    if pretrained:
        ckpt = torch.load(weight, map_location='cpu')  # load checkpoint
        state_dict = ckpt['model']
        model = Model(opt.cfg, ch=3, nc=nc)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        model.to(device)
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weight))  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc).to(device)  # create
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']

    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'):
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'):
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):
                pg0.append(v.rbr_dense.vector)

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    plot_lr_scheduler(optimizer, scheduler, epochs, save_dir)

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'])
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weight, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weight, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model_h2[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    if opt.single_backbone:
        compute_loss_ota = ComputeLossOTA(model)  # init loss class
    else:
        compute_loss_ota = ComputeLossOTADual(model)  # init loss class
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 10:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # add model graph
                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                results, maps, times = test.test(data_dict,
                                                 batch_size=batch_size * 2,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss,
                                                 is_coco=is_coco,
                                                 v5_metric=opt.v5_metric)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            if (not opt.nosave) or final_epoch:  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': get_state_dict(deepcopy(model.module if is_parallel(model) else model).half()),
                        'ema': get_state_dict(deepcopy(ema.ema).half()),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if (best_fitness == fi) and (epoch >= 200):
                    torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
                if epoch == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                elif ((epoch+1) % 25) == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                elif epoch >= (epochs-5):
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
        if wandb_logger.wandb:  # Log the stripped model
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--weight', type=str, default='', help='initial weights path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='hyp/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--single-backbone', action='store_true', help='train single backbone model')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)

    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weight, opt.resume = os.path.relpath(Path(ckpt).parent.parent / 'cfg.yaml'), ckpt, True
        opt.batch_size, opt.global_rank, opt.local_rank = opt.total_batch_size, *apriori  # reinstate
        opt.save_dir = os.path.relpath(Path(ckpt).parent.parent)
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg), 'cfg must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    tb_writer = None  # init loggers
    if opt.global_rank in [-1, 0]:
        prefix = colorstr('tensorboard: ')
        logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
        tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
    train(hyp, opt, device, tb_writer)
```

# get_dynamic_thres.py

```python
import argparse
import os
import logging
from pathlib import Path
from threading import Thread
import yaml
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import check_dataset, check_file, check_img_size, set_logging, colorstr
from utils.torch_utils import select_device


logger = logging.getLogger(__name__)

def get_thres(data,
              cfg=None,
              weight=None,
              batch_size=32,
              imgsz=640,
              augment=False,
              half_precision=True):
    set_logging()
    device = select_device(opt.device, batch_size=batch_size)
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Load model
    model = Model(cfg, ch=3, nc=nc)  # create
    state_dict = torch.load(weight, map_location='cpu')['model']
    model.load_state_dict(state_dict, strict=True)  # load
    model.to(device)
    logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weight))  # report
    for p in model.parameters():
        p.requires_grad = False
    model.float().fuse().eval()
    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size

    model.get_score = True

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Dataloader
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
    dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'{task}: '))[0]

    score_list = []
    for batch_i, (img, _, _, _) in enumerate(tqdm(dataloader)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        with torch.no_grad():
            # Run model
            cur_score = model(img, augment=augment)  # inference and training outputs
            score_list.append(cur_score.item())

    thres = ['0']
    for i in list(range(10, 100, 10)):
        thres.append(str(np.percentile(score_list, i)))
    thres.append('1')
    return thres


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--weight', type=str, default='', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--task', default='val', help='train, val, test')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.single_cls = False
    opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.task in ('train', 'val', 'test'):  # run normally
        thres = get_thres(opt.data, opt.cfg, opt.weight, opt.batch_size, opt.img_size, opt.augment)
        print()
        print('***************************************************')
        print(' '.join(thres))
        for idx, thr in enumerate(thres):
            print('First: {}%\tSecond: {}%\tThreshold: {}'.format(100 - idx * 10, idx  * 10, thr))
        print('***************************************************')
    else:
        raise NotImplementedError
```

# test.py

```python
import argparse
import json
import logging
from pathlib import Path
from threading import Thread
import yaml
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized, intersect_dicts


logger = logging.getLogger(__name__)

def test(data,
         cfg=None,
         weight=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         is_coco=False,
         v5_metric=False,
         dy_thres=0.5,
         save_results=False):
    # Initialize/load model and set device
    training = model is not None

    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    if not training:
        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = Model(cfg, ch=3, nc=nc)  # create
        state_dict = torch.load(weight, map_location='cpu')['model']
        state_dict = intersect_dicts(state_dict, model.state_dict())  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        model.to(device)
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weight))  # report
        for p in model.parameters():
            p.requires_grad = False
        model.float().fuse().eval()
        # Compatibility updates
        for m in model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility

        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size
        if hasattr(model, 'dy_thres'):
            model.dy_thres = dy_thres
            logger.info('Set dynamic threshold to %f' % dy_thres)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    if v5_metric:
        print("Testing with YOLOv5 AP metric...")

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weight).stem if weight is not None else ''  # weight
        anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
            if save_results:
                results_txt = str(save_dir / f"{w}_results.txt")
                with open(results_txt, 'a') as f:
                    f.write(f'map: {round(map, 3)}, map50: {round(map50, 3)}, dy_thres: {dy_thres}\n')
                    f.write('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g\n' % t)
                    f.write('***********************************************************************************\n')
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--weight', type=str, default='', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--dy-thres', type=float, default=0.5, help='dynamic threshold')
    parser.add_argument('--save-results', action='store_true', help='save results')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.cfg,
             opt.weight,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             v5_metric=opt.v5_metric,
             dy_thres=opt.dy_thres,
             save_results=opt.save_results
             )

    else:
        raise NotImplementedError```

# train_step2.py

```python
import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread
import yaml
from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    set_logging, colorstr
from utils.loss import ComputeLoss, ComputeLossOTADy
from utils.plots import plot_images, plot_results, plot_lr_scheduler
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from utils.checkpoint import get_state_dict


logger = logging.getLogger(__name__)

def train(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weight, rank, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weight, opt.global_rank, opt.freeze

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    if not opt.resume:
        with open(save_dir / 'cfg.yaml', 'w') as f_d:
            with open(opt.cfg) as f_s:
                cfg_yaml = yaml.load(f_s, Loader=yaml.SafeLoader)
            yaml.dump(cfg_yaml, f_d, sort_keys=False)

    # Configure
    plots = True
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    is_coco = opt.data.endswith('coco.yaml')

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weight, map_location='cpu')['wandb_id'] if weight.endswith('.pt') and os.path.isfile(weight) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weight, epochs, hyp = opt.weight, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weight.endswith('.pt')
    if pretrained:
        ckpt = torch.load(weight, map_location='cpu')  # load checkpoint
        state_dict = ckpt['model']
        model = Model(opt.cfg, ch=3, nc=nc)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        model.to(device)
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weight))  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc).to(device)  # create
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']

    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # freeze the parameters of the cascaded detector
    for k, v in model.named_parameters():
        if 'router' not in k:
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if 'router' in k:
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

    if opt.adam:
        optimizer = optim.AdamW(pg1, lr=hyp['lr0'], weight_decay=hyp['weight_decay'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg1, lr=hyp['lr0'], weight_decay=hyp['weight_decay'], momentum=hyp['momentum'], nesterov=True)
    if len(pg0):
        optimizer.add_param_group({'params': pg0, 'weight_decay': 0})  # add pg0 without weight_decay
    if len(pg2):
        optimizer.add_param_group({'params': pg2, 'weight_decay': 0})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    plot_lr_scheduler(optimizer, scheduler, epochs, save_dir)

    # EMA
    ema_decay = 0.
    ema = ModelEMA(model, decay=ema_decay) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'])
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weight, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weight, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model_h2[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        test_batch_size = 1
        testloader = create_dataloader(test_path, imgsz_test, test_batch_size, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss_ota_dy = ComputeLossOTADy(model)
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        model.close_all_bn()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'diff', 'score.0', 'score.1', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss_ota_dy(pred, targets.to(device), imgs)  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 10:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # add model graph
                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                test_batch_size = 1
                results, maps, times = test.test(data_dict,
                                                 batch_size=test_batch_size,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss,
                                                 is_coco=is_coco,
                                                 v5_metric=opt.v5_metric)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            if (not opt.nosave) or final_epoch:  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': get_state_dict(deepcopy(model.module if is_parallel(model) else model).half()),
                        'ema': get_state_dict(deepcopy(ema.ema).half()),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if (best_fitness == fi) and (epoch >= 200):
                    torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
                if epoch == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                elif ((epoch+1) % 25) == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                elif epoch >= (epochs-5):
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
        if wandb_logger.wandb:  # Log the stripped model
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--weight', type=str, default='', help='initial weights path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='hyp/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)

    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weight, opt.resume = os.path.relpath(Path(ckpt).parent.parent / 'cfg.yaml'), ckpt, True
        opt.batch_size, opt.global_rank, opt.local_rank = opt.total_batch_size, *apriori  # reinstate
        opt.save_dir = os.path.relpath(Path(ckpt).parent.parent)
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg), 'cfg must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    # DDP mode
    opt.total_batch_size = opt.batch_size
    assert opt.batch_size == 1, 'set batch_size = 1 for dynamic training'

    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    tb_writer = None  # init loggers
    if opt.global_rank in [-1, 0]:
        prefix = colorstr('tensorboard: ')
        logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
        tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
    train(hyp, opt, device, tb_writer)
```

# detect.py

```python
import argparse
import time
import logging
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from numpy import random

from models.yolo import Model
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, intersect_dicts


logger = logging.getLogger(__name__)

def detect(save_img=False):
    source, cfg, weight, view_img, save_txt, nc, imgsz = opt.source, opt.cfg, opt.weight, \
        opt.view_img, opt.save_txt, opt.num_classes, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Model(cfg, ch=3, nc=nc)  # create
    state_dict = torch.load(weight, map_location='cpu')['model']
    state_dict = intersect_dicts(state_dict, model.state_dict())  # intersect
    model.load_state_dict(state_dict, strict=False)  # load
    model.to(device)
    logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weight))  # report
    for p in model.parameters():
        p.requires_grad = False
    model.float().fuse().eval()
    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if hasattr(model, 'dy_thres'):
        model.dy_thres = opt.dy_thres
        logger.info('Set dynamic threshold to %f' % opt.dy_thres)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--weight', type=str, default='', help='model.pt path(s)')
    parser.add_argument('--num-classes', type=int, default=80, help='number of classes')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--dy-thres', type=float, default=0.5, help='dynamic thres')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
```

## utils/metrics.py

```python
# Model validation metrics

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import general


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def ap_per_class(tp, conf, pred_cls, target_cls, v5_metric=False, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j], v5_metric=v5_metric)
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision, v5_metric=False):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
        v5_metric: Assume maximum recall to be 1.0, as in YOLOv5, MMDetetion etc.
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    if v5_metric:  # New YOLOv5 metric, same as MMDetection and Detectron2 repositories
        mrec = np.concatenate(([0.], recall, [1.0]))
    else:  # Old YOLOv5 metric, i.e. default YOLOv7 metric
        mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = general.box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[gc, detection_classes[m1[j]]] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def plot(self, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            pass

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
```

## utils/checkpoint.py

```python
from collections import OrderedDict

def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.
    Args:
        state_dict (OrderedDict): Model weights on GPU.
    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    # Keep metadata in state_dict
    state_dict_cpu._metadata = getattr(  # type: ignore
        state_dict, '_metadata', OrderedDict())
    return state_dict_cpu


def get_state_dict(model):
    return weights_to_cpu(model.state_dict())
```

## utils/datasets.py

```python
# Dataset utils and dataloaders

import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

import pickle
from copy import deepcopy
#from pycocotools import mask as maskUtils
from torchvision.utils import save_image
from torchvision.ops import roi_pool, roi_align, ps_roi_pool, ps_roi_align

from utils.general import check_requirements, xyxy2xywh, xywh2xyxy, xywhn2xyxy, xyn2xy, segment2box, segments2boxes, \
    resample_segments, clean_str
from utils.torch_utils import torch_distributed_zero_first

# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix=''):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=opt.single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            #print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride

        if pipe.isnumeric():
            pipe = eval(pipe)  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        print(f'webcam {self.count}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            url = eval(s) if s.isnumeric() else s
            if 'youtube.com/' in str(url) or 'youtu.be/' in str(url):  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                url = pafy.new(url).getbest(preftype="mp4").url
            cap = cv2.VideoCapture(url)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) % 100

            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f' success ({w}x{h} at {self.fps:.2f} FPS).')
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(1 / self.fps)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        #self.albumentations = Albumentations() if augment else None

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {help_url}')

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')  # cached labels
        if cache_path.is_file():
            cache, exists = torch.load(cache_path), True  # load
            #if cache['hash'] != get_hash(self.label_files + self.img_files) or 'version' not in cache:  # changed
            #    cache, exists = self.cache_labels(cache_path, prefix), False  # re-cache
        else:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {help_url}'

        # Read cache
        cache.pop('hash')  # remove hash
        cache.pop('version')  # remove version
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            if cache_images == 'disk':
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x
                    gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im.format.lower() in img_formats, f'invalid image format {im.format}'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines()]
                        if any([len(x) > 8 for x in l]):  # is segment
                            classes = np.array([x[0] for x in l], dtype=np.float32)
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                            l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                        l = np.array(l, dtype=np.float32)
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape, segments]
            except Exception as e:
                nc += 1
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.1  # cache version
        torch.save(x, path)  # save for next time
        logging.info(f'{prefix}New cache created: {path}')
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            if random.random() < 0.8:
                img, labels = load_mosaic(self, index)
            else:
                img, labels = load_mosaic9(self, index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']:
                if random.random() < 0.8:
                    img2, labels2 = load_mosaic(self, random.randint(0, len(self.labels) - 1))
                else:
                    img2, labels2 = load_mosaic9(self, random.randint(0, len(self.labels) - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            # Augment imagespace
            if not mosaic:
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])


            #img, labels = self.albumentations(img, labels)

            # Augment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

            if random.random() < hyp['paste_in']:
                sample_labels, sample_images, sample_masks = [], [], []
                while len(sample_labels) < 30:
                    sample_labels_, sample_images_, sample_masks_ = load_samples(self, random.randint(0, len(self.labels) - 1))
                    sample_labels += sample_labels_
                    sample_images += sample_images_
                    sample_masks += sample_masks_
                    #print(len(sample_labels))
                    if len(sample_labels) == 0:
                        break
                labels = pastein(img, labels, sample_labels, sample_images, sample_masks)

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def hist_equalize(img, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'img' with img.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def load_mosaic(self, index):
    # loads images in a 4-mosaic

    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    #img4, labels4, segments4 = remove_background(img4, labels4, segments4)
    #sample_segments(img4, labels4, segments4, probability=self.hyp['copy_paste'])
    img4, labels4, segments4 = copy_paste(img4, labels4, segments4, probability=self.hyp['copy_paste'])
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4


def load_mosaic9(self, index):
    # loads images in a 9-mosaic

    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
    #img9, labels9, segments9 = remove_background(img9, labels9, segments9)
    img9, labels9, segments9 = copy_paste(img9, labels9, segments9, probability=self.hyp['copy_paste'])
    img9, labels9 = random_perspective(img9, labels9, segments9,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img9, labels9


def load_samples(self, index):
    # loads images in a 4-mosaic

    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    #img4, labels4, segments4 = remove_background(img4, labels4, segments4)
    sample_labels, sample_images, sample_masks = sample_segments(img4, labels4, segments4, probability=0.5)

    return sample_labels, sample_images, sample_masks


def copy_paste(img, labels, segments, probability=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if probability and n:
        h, w, c = img.shape  # height, width, channels
        im_new = np.zeros(img.shape, np.uint8)
        for j in random.sample(range(n), k=round(probability * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=img, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        img[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug

    return img, labels, segments


def remove_background(img, labels, segments):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    h, w, c = img.shape  # height, width, channels
    im_new = np.zeros(img.shape, np.uint8)
    img_new = np.ones(img.shape, np.uint8) * 114
    for j in range(n):
        cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=img, src2=im_new)

        i = result > 0  # pixels to replace
        img_new[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug

    return img_new, labels, segments


def sample_segments(img, labels, segments, probability=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    sample_labels = []
    sample_images = []
    sample_masks = []
    if probability and n:
        h, w, c = img.shape  # height, width, channels
        for j in random.sample(range(n), k=round(probability * n)):
            l, s = labels[j], segments[j]
            box = l[1].astype(int).clip(0,w-1), l[2].astype(int).clip(0,h-1), l[3].astype(int).clip(0,w-1), l[4].astype(int).clip(0,h-1)

            #print(box)
            if (box[2] <= box[0]) or (box[3] <= box[1]):
                continue

            sample_labels.append(l[0])

            mask = np.zeros(img.shape, np.uint8)

            cv2.drawContours(mask, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)
            sample_masks.append(mask[box[1]:box[3],box[0]:box[2],:])

            result = cv2.bitwise_and(src1=img, src2=mask)
            i = result > 0  # pixels to replace
            mask[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug
            #print(box)
            sample_images.append(mask[box[1]:box[3],box[0]:box[2],:])

    return sample_labels, sample_images, sample_masks


def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def random_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1.1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def bbox_ioa(box1, box2):
    # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

    # Intersection over box2 area
    return inter_area / box2_area


def cutout(image, labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def pastein(image, labels, sample_labels, sample_images, sample_masks):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    # create random masks
    scales = [0.75] * 2 + [0.5] * 4 + [0.25] * 4 + [0.125] * 4 + [0.0625] * 6  # image size fraction
    for s in scales:
        if random.random() < 0.2:
            continue
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        if len(labels):
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
        else:
            ioa = np.zeros(1)

        if (ioa < 0.30).all() and len(sample_labels) and (xmax > xmin+20) and (ymax > ymin+20):  # allow 30% obscuration of existing labels
            sel_ind = random.randint(0, len(sample_labels)-1)
            #print(len(sample_labels))
            #print(sel_ind)
            #print((xmax-xmin, ymax-ymin))
            #print(image[ymin:ymax, xmin:xmax].shape)
            #print([[sample_labels[sel_ind], *box]])
            #print(labels.shape)
            hs, ws, cs = sample_images[sel_ind].shape
            r_scale = min((ymax-ymin)/hs, (xmax-xmin)/ws)
            r_w = int(ws*r_scale)
            r_h = int(hs*r_scale)

            if (r_w > 10) and (r_h > 10):
                r_mask = cv2.resize(sample_masks[sel_ind], (r_w, r_h))
                r_image = cv2.resize(sample_images[sel_ind], (r_w, r_h))
                temp_crop = image[ymin:ymin+r_h, xmin:xmin+r_w]
                m_ind = r_mask > 0
                if m_ind.astype(np.int).sum() > 60:
                    temp_crop[m_ind] = r_image[m_ind]
                    #print(sample_labels[sel_ind])
                    #print(sample_images[sel_ind].shape)
                    #print(temp_crop.shape)
                    box = np.array([xmin, ymin, xmin+r_w, ymin+r_h], dtype=np.float32)
                    if len(labels):
                        labels = np.concatenate((labels, [[sample_labels[sel_ind], *box]]), 0)
                    else:
                        labels = np.array([[sample_labels[sel_ind], *box]])

                    image[ymin:ymin+r_h, xmin:xmin+r_w] = temp_crop

    return labels

class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        import albumentations as A

        self.transform = A.Compose([
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.01),
            A.RandomGamma(gamma_limit=[80, 120], p=0.01),
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.ImageCompression(quality_lower=75, p=0.01),],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

            #logging.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path='../coco'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='../coco/'):  # from utils.datasets import *; extract_boxes('../coco128')
    # Convert detection dataset into classification dataset, with one directory per class

    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in img_formats:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path='../coco', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit('../coco')
    Arguments
        path:           Path to images directory
        weights:        Train, val, test weights (list)
        annotated_only: Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sum([list(path.rglob(f"*.{img_ext}")) for img_ext in img_formats], [])  # image files only
    n = len(files)  # number of files
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path / x).unlink() for x in txt if (path / x).exists()]  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path / txt[i], 'a') as f:
                f.write(str(img) + '\n')  # add image to txt file


def load_segmentations(self, index):
    key = '/work/handsomejw66/coco17/' + self.img_files[index]
    #print(key)
    # /work/handsomejw66/coco17/
    return self.segs[key]
```

## utils/__init__.py

```python
# init```

## utils/loss.py

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import bbox_iou, box_iou, xywh2xyxy
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model_h2[-1] if is_parallel(model) else model.model_h2[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        #self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.1, .05])  # P3-P7
        #self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.5, 0.4, .1])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    #t[t==self.cp] = iou.detach().clamp(0).type(t.dtype)
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


class ComputeLossOTA:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossOTA, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model_h2[-1] if is_parallel(model) else model.model_h2[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, imgs):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p, targets, imgs)
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p]


        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                #pxy = ps[:, :2].sigmoid() * 3. - 1.
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets, imgs):

        #indices, anch = self.find_positive(p, targets)
        indices, anch = self.find_3_positive(p, targets)
        #indices, anch = self.find_4_positive(p, targets)
        #indices, anch = self.find_5_positive(p, targets)
        #indices, anch = self.find_9_positive(p, targets)

        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]

        nl = len(p)

        for batch_idx in range(p[0].shape[0]):

            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue

            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []

            for i, pi in enumerate(p):

                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)

                fg_pred = pi[b, a, gj, gi]
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])

                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)

            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)

            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_

            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]

            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch


class ComputeLossOTADual(ComputeLossOTA):
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossOTADual, self).__init__(model, autobalance)

    def __call__(self, p, targets, imgs):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        bs_2, as_2_, gjs_2, gis_2, targets_2, anchors_2 = self.build_targets(p[self.nl:], targets, imgs)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p[:self.nl], targets, imgs)
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]]
        pre_gen_gains_2 = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[self.nl:]]

        # Losses
        for i in range(self.nl):  # layer index, layer predictions
            pi = p[i]
            pi_2 = p[i+self.nl]
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            b_2, a_2, gj_2, gi_2 = bs_2[i], as_2_[i], gjs_2[i], gis_2[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            tobj_2 = torch.zeros_like(pi_2[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                #pxy = ps[:, :2].sigmoid() * 3. - 1.
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            n_2 = b_2.shape[0]  # number of targets
            if n_2:
                ps_2 = pi_2[b_2, a_2, gj_2, gi_2]  # prediction subset corresponding to targets

                # Regression
                grid_2 = torch.stack([gi_2, gj_2], dim=1)
                pxy_2 = ps_2[:, :2].sigmoid() * 2. - 0.5
                #pxy = ps[:, :2].sigmoid() * 3. - 1.
                pwh_2 = (ps_2[:, 2:4].sigmoid() * 2) ** 2 * anchors_2[i]
                pbox_2 = torch.cat((pxy_2, pwh_2), 1)  # predicted box
                selected_tbox_2 = targets_2[i][:, 2:6] * pre_gen_gains_2[i]
                selected_tbox_2[:, :2] -= grid_2
                iou_2 = bbox_iou(pbox_2.T, selected_tbox_2, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou_2).mean()  # iou loss

                # Objectness
                tobj_2[b_2, a_2, gj_2, gi_2] = (1.0 - self.gr) + self.gr * iou_2.detach().clamp(0).type(tobj_2.dtype)  # iou ratio

                # Classification
                selected_tcls_2 = targets_2[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t_2 = torch.full_like(ps_2[:, 5:], self.cn, device=device)  # targets
                    t_2[range(n_2), selected_tcls_2] = self.cp
                    lcls += self.BCEcls(ps_2[:, 5:], t_2)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]


            obji = self.BCEobj(pi[..., 4], tobj)
            obji_2 = self.BCEobj(pi_2[..., 4], tobj_2)
            lobj += obji * self.balance[i]  # obj loss
            lobj += obji_2 * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


class ComputeLossOTADy(ComputeLossOTA):
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossOTADy, self).__init__(model, autobalance)
        self.tracked_diff = 0
        self.iter_count = 0
        self.diff_list = []

    def __call__(self, ps, targets, imgs):  # predictions, targets, model
        p, score = ps
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        lcls_2, lbox_2, lobj_2 = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        bs_2, as_2_, gjs_2, gis_2, targets_2, anchors_2 = self.build_targets(p[self.nl:], targets, imgs)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p[:self.nl], targets, imgs)
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.nl]]
        pre_gen_gains_2 = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[self.nl:]]

        # Losses
        for i in range(self.nl):  # layer index, layer predictions
            pi = p[i]
            pi_2 = p[i+self.nl]
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            b_2, a_2, gj_2, gi_2 = bs_2[i], as_2_[i], gjs_2[i], gis_2[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            tobj_2 = torch.zeros_like(pi_2[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                #pxy = ps[:, :2].sigmoid() * 3. - 1.
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            n_2 = b_2.shape[0]  # number of targets
            if n_2:
                ps_2 = pi_2[b_2, a_2, gj_2, gi_2]  # prediction subset corresponding to targets

                # Regression
                grid_2 = torch.stack([gi_2, gj_2], dim=1)
                pxy_2 = ps_2[:, :2].sigmoid() * 2. - 0.5
                #pxy = ps[:, :2].sigmoid() * 3. - 1.
                pwh_2 = (ps_2[:, 2:4].sigmoid() * 2) ** 2 * anchors_2[i]
                pbox_2 = torch.cat((pxy_2, pwh_2), 1)  # predicted box
                selected_tbox_2 = targets_2[i][:, 2:6] * pre_gen_gains_2[i]
                selected_tbox_2[:, :2] -= grid_2
                iou_2 = bbox_iou(pbox_2.T, selected_tbox_2, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox_2 += (1.0 - iou_2).mean()  # iou loss

                # Objectness
                tobj_2[b_2, a_2, gj_2, gi_2] = (1.0 - self.gr) + self.gr * iou_2.detach().clamp(0).type(tobj_2.dtype)  # iou ratio

                # Classification
                selected_tcls_2 = targets_2[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t_2 = torch.full_like(ps_2[:, 5:], self.cn, device=device)  # targets
                    t_2[range(n_2), selected_tcls_2] = self.cp
                    lcls_2 += self.BCEcls(ps_2[:, 5:], t_2)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]


            obji = self.BCEobj(pi[..., 4], tobj)
            obji_2 = self.BCEobj(pi_2[..., 4], tobj_2)
            lobj += obji * self.balance[i]  # obj loss
            lobj_2 += obji_2 * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lbox_2 *= self.hyp['box']
        lobj_2 *= self.hyp['obj']
        lcls_2 *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = (lbox + lobj + lcls).detach()
        loss_2 = (lbox_2 + lobj_2 + lcls_2).detach()

        self.iter_count += 1
        current_diff = loss - loss_2
        self.diff_list.append(current_diff.item())

        # (online) calculation of the loss difference
        if self.iter_count < 10000:
            self.tracked_diff = np.median(self.diff_list)
        elif self.iter_count % 1000 == 0:
            self.tracked_diff = np.median(self.diff_list)

        loss = loss - self.tracked_diff / 2
        loss_2 = loss_2 + self.tracked_diff / 2
        adaptive_loss = score[:, 0] * loss + (1 - score[:, 0]) * loss_2
        return adaptive_loss * bs, torch.cat((current_diff, score[:, 0], 1 - score[:, 0], adaptive_loss)).detach()
```

## utils/plots.py

```python
# Plotting utils

import glob
import math
import os
import random
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, filtfilt

from utils.general import xywh2xyxy, xyxy2xywh
from utils.metrics import fitness

# Settings
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # for writing to files only


def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()]  # or BASE_ (8), CSS4_ (148), XKCD_ (949)


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def output_to_target(output):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):
        for *box, conf, cls in o.cpu().numpy():
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])
    return np.array(targets)


def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
    # Plot image grid with labels

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    colors = color_list()  # list of colors
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            labels = image_targets.shape[1] == 6  # labels if no conf column
            conf = None if labels else image_targets[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif scale_factor < 1:  # absolute coords need scale if image scales
                    boxes *= scale_factor
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = colors[cls % len(colors)]
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = '%s' % cls if labels else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)

        # Draw image filename labels
        if paths:
            label = Path(paths[i]).name[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname:
        r = min(1280. / max(h, w) / ns, 1.0)  # ratio to limit image size
        mosaic = cv2.resize(mosaic, (int(ns * w * r), int(ns * h * r)), interpolation=cv2.INTER_AREA)
        # cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))  # cv2 save
        Image.fromarray(mosaic).save(fname)  # PIL save
    return mosaic


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)
    plt.close()


def plot_results(start=0, stop=0, bucket='', id=(), labels=(), save_dir=''):
    # Plot training 'results*.txt'. from utils.plots import *; plot_results(save_dir='runs/train/exp')
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    s = ['Box', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val Box', 'val Objectness', 'val Classification', 'mAP@0.5', 'mAP@0.5:0.95']
    if bucket:
        # files = ['https://storage.googleapis.com/%s/results%g.txt' % (bucket, x) for x in id]
        files = ['results%g.txt' % x for x in id]
        c = ('gsutil cp ' + '%s ' * len(files) + '.') % tuple('gs://%s/results%g.txt' % (bucket, x) for x in id)
        os.system(c)
    else:
        files = list(Path(save_dir).glob('results*.txt'))
    assert len(files), 'No results.txt files found in %s, nothing to plot.' % os.path.abspath(save_dir)
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
            n = results.shape[1]  # number of rows
            x = range(start, min(stop, n) if stop else n)
            for i in range(10):
                y = results[i, x]
                if i in [0, 1, 2, 5, 6, 7]:
                    y[y == 0] = np.nan  # don't show zero loss values
                    # y /= y[0]  # normalize
                label = labels[fi] if len(labels) else f.stem
                ax[i].plot(x, y, marker='.', label=label, linewidth=2, markersize=8)
                ax[i].set_title(s[i])
                # if i in [5, 6, 7]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            print('Warning: Plotting error for %s; %s' % (f, e))

    ax[1].legend()
    fig.savefig(Path(save_dir) / 'results.png', dpi=200)
```

## utils/autoanchor.py

```python
# Auto-anchor utils

import numpy as np
import torch
import yaml
from scipy.cluster.vq import kmeans
from tqdm import tqdm

from utils.general import colorstr


def check_anchor_order(m):
    # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # Check anchor fit to data, recompute if necessary
    prefix = colorstr('autoanchor: ')
    print(f'\n{prefix}Analyzing anchors... ', end='')
    m = model.module.model_h2[-1] if hasattr(model, 'module') else model.model_h2[-1]  # Detect()
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1. / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1. / thr).float().mean()  # best possible recall
        return bpr, aat

    anchors = m.anchor_grid.clone().cpu().view(-1, 2)  # current anchors
    bpr, aat = metric(anchors)
    print(f'anchors/target = {aat:.2f}, Best Possible Recall (BPR) = {bpr:.4f}', end='')
    if bpr < 0.98:  # threshold to recompute
        print('. Attempting to improve anchors, please wait...')
        na = m.anchor_grid.numel() // 2  # number of anchors
        try:
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        except Exception as e:
            print(f'{prefix}ERROR: {e}')
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchor_grid[:] = anchors.clone().view_as(m.anchor_grid)  # for inference
            check_anchor_order(m)
            m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss
            print(f'{prefix}New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            print(f'{prefix}Original anchors better than new anchors. Proceeding with original anchors.')
    print('')  # newline


def kmean_anchors(path='./data/coco.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            path: path to dataset *.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    thr = 1. / thr
    prefix = colorstr('autoanchor: ')

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        print(f'{prefix}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr')
        print(f'{prefix}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, '
              f'past_thr={x[x > thr].mean():.3f}-mean: ', end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    if isinstance(path, str):  # *.yaml file
        with open(path) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        from utils.datasets import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)
    else:
        dataset = path  # dataset

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print(f'{prefix}WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels
    # wh = wh * (np.random.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans calculation
    print(f'{prefix}Running kmeans for {n} anchors on {len(wh)} points...')
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    assert len(k) == n, print(f'{prefix}ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}')
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unfiltered
    k = print_results(k)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    npr = np.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc=f'{prefix}Evolving anchors with Genetic Algorithm:')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{prefix}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k)

    return print_results(k)
```

## utils/torch_utils.py

```python
import datetime
import logging
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None
logger = logging.getLogger(__name__)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def git_describe(path=Path(__file__).parent):  # path must be a directory
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''  # not a git repository


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'DynamicDet 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def profile(x, ops, n=100, device=None):
    # profile a pytorch module or list of modules. Example usage:
    #     x = torch.randn(16, 3, 640, 640)  # input
    #     m1 = lambda x: x * torch.sigmoid(x)
    #     m2 = nn.SiLU()
    #     profile(x, [m1, m2], n=100)  # profile speed over 100 iterations

    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    x.requires_grad = True
    print(torch.__version__, device.type, torch.cuda.get_device_properties(0) if device.type == 'cuda' else '')
    print(f"\n{'Params':>12s}{'GFLOPS':>12s}{'forward (ms)':>16s}{'backward (ms)':>16s}{'input':>24s}{'output':>24s}")
    for m in ops if isinstance(ops, list) else [ops]:
        m = m.to(device) if hasattr(m, 'to') else m  # device
        m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m  # type
        dtf, dtb, t = 0., 0., [0., 0., 0.]  # dt forward, backward
        try:
            flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # GFLOPS
        except:
            flops = 0

        for _ in range(n):
            t[0] = time_synchronized()
            y = m(x)
            t[1] = time_synchronized()
            try:
                _ = y.sum().backward()
                t[2] = time_synchronized()
            except:  # no backward method
                t[2] = float('nan')
            dtf += (t[1] - t[0]) * 1000 / n  # ms per op forward
            dtb += (t[2] - t[1]) * 1000 / n  # ms per op backward

        s_in = tuple(x.shape) if isinstance(x, torch.Tensor) else 'list'
        s_out = tuple(y.shape) if isinstance(y, torch.Tensor) else 'list'
        p = sum(list(x.numel() for x in m.parameters())) if isinstance(m, nn.Module) else 0  # parameters
        print(f'{p:12}{flops:12.4g}{dtf:16.4g}{dtb:16.4g}{str(s_in):>24s}{str(s_out):>24s}')


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # Finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    # Return global model sparsity
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent
    print(' %.3g global sparsity' % sparsity(model))


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False, img_size=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPS
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = ', %.1f GFLOPS' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPS
    except (ImportError, Exception):
        fs = ''

    logger.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


class BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        # The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
        # is this method that is overwritten by the sub-class
        # This original goal of this method was for tensor sanity checks
        # If you're ok bypassing those sanity checks (eg. if you trust your inference
        # to provide the right dimensional inputs), then you can just use this method
        # for easy conversion from SyncBatchNorm
        # (unfortunately, SyncBatchNorm does not store the original class - if it did
        #  we could return the one that was originally created)
        return

def revert_sync_batchnorm(module):
    # this is very similar to the function that it is trying to revert:
    # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        new_cls = BatchNormXd
        module_output = BatchNormXd(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output
```

## utils/general.py

```python
import glob
import logging
import math
import os
import platform
import random
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml

from utils.metrics import fitness
from utils.torch_utils import init_torch_seeds

# Settings
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))  # NumExpr max threads


def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def get_latest_run(search_dir='.'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def isdocker():
    # Is environment a Docker container
    return Path('/workspace').exists()  # or Path('/.dockerenv').exists()


def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


def check_online():
    # Check internet connectivity
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accesability
        return True
    except OSError:
        return False


def check_git_status():
    # Recommend 'git pull' if code is out of date
    print(colorstr('github: '), end='')
    try:
        assert Path('.git').exists(), 'skipping check (not a git repository)'
        assert not isdocker(), 'skipping check (Docker image)'
        assert check_online(), 'skipping check (offline)'

        cmd = 'git fetch && git config --get remote.origin.url'
        url = subprocess.check_output(cmd, shell=True).decode().strip().rstrip('.git')  # github repo url
        branch = subprocess.check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()  # checked out
        n = int(subprocess.check_output(f'git rev-list {branch}..origin/master --count', shell=True))  # commits behind
        if n > 0:
            s = f"⚠️ WARNING: code is out of date by {n} commit{'s' * (n > 1)}. " \
                f"Use 'git pull' to update or 'git clone {url}' to download latest."
        else:
            s = f'up to date with {url} ✅'
        print(emojis(s))  # emoji-safe
    except Exception as e:
        print(e)


def check_requirements(requirements='requirements.txt', exclude=()):
    # Check installed dependencies meet requirements (pass *.txt file or list of packages)
    import pkg_resources as pkg
    prefix = colorstr('red', 'bold', 'requirements:')
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        file = Path(requirements)
        if not file.exists():
            print(f"{prefix} {file.resolve()} not found, check failed.")
            return
        requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(file.open()) if x.name not in exclude]
    else:  # list or tuple of packages
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # number of packages updates
    for r in requirements:
        try:
            pkg.require(r)
        except Exception as e:  # DistributionNotFound or VersionConflict if requirements not met
            n += 1
            print(f"{prefix} {e.req} not found, attempting auto-update...")
            print(subprocess.check_output(f"pip install '{e.req}'", shell=True).decode())

    if n:  # if packages updated
        source = file.resolve() if 'file' in locals() else requirements
        s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        print(emojis(s))  # emoji-safe


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def check_imshow():
    # Check if environment supports image displays
    try:
        assert not isdocker(), 'cv2.imshow() is disabled in Docker environments'
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False


def check_file(file):
    # Search for file if not found
    if Path(file).is_file() or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), f'File Not Found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def check_dataset(dict):
    # Download dataset if not found locally
    val, s = dict.get('val'), dict.get('download')
    if val and len(val):
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            if s and len(s):  # download script
                print('Downloading %s ...' % s)
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    f = Path(s).name  # filename
                    torch.hub.download_url_to_file(s, f)
                    r = os.system('unzip -q %s -d ../ && rm %s' % (f, f))  # unzip
                else:  # bash script
                    r = os.system(s)
                print('Dataset autodownload %s\n' % ('success' if r == 0 else 'failure'))  # analyze return value
            else:
                raise Exception('Dataset not found.')


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class_weights and image contents
    class_counts = np.array([np.bincount(x[:, 0].astype(np.int), minlength=nc) for x in labels])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def bbox_alpha_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False, alpha=2, eps=1e-9):
    # Returns tsqrt_he IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    # change iou into pow(iou+eps)
    # iou = inter / union
    iou = torch.pow(inter/union + eps, alpha)
    # beta = 2 * alpha
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = (cw ** 2 + ch ** 2) ** alpha + eps  # convex diagonal
            rho_x = torch.abs(b2_x1 + b2_x2 - b1_x1 - b1_x2)
            rho_y = torch.abs(b2_y1 + b2_y2 - b1_y1 - b1_y2)
            rho2 = ((rho_x ** 2 + rho_y ** 2) / 4) ** alpha  # center distance
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha_ciou = v / ((1 + eps) - inter / union + v)
                # return iou - (rho2 / c2 + v * alpha_ciou)  # CIoU
                return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            # c_area = cw * ch + eps  # convex area
            # return iou - (c_area - union) / c_area  # GIoU
            c_area = torch.max(cw * ch + eps, union) # convex area
            return iou - torch.pow((c_area - union) / c_area + eps, alpha)  # GIoU
    else:
        return iou # torch.log(iou+eps) or iou


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def box_giou(box1, box2):
    """
    Return generalized intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise generalized IoU values
        for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    union = (area1[:, None] + area2 - inter)

    iou = inter / union

    lti = torch.min(box1[:, None, :2], box2[:, :2])
    rbi = torch.max(box1[:, None, 2:], box2[:, 2:])

    whi = (rbi - lti).clamp(min=0)  # [N,M,2]
    areai = whi[:, :, 0] * whi[:, :, 1]

    return iou - (areai - union) / areai


def box_ciou(box1, box2, eps: float = 1e-7):
    """
    Return complete intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
        eps (float, optional): small number to prevent division by zero. Default: 1e-7
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise complete IoU values
        for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    union = (area1[:, None] + area2 - inter)

    iou = inter / union

    lti = torch.min(box1[:, None, :2], box2[:, :2])
    rbi = torch.max(box1[:, None, 2:], box2[:, 2:])

    whi = (rbi - lti).clamp(min=0)  # [N,M,2]
    diagonal_distance_squared = (whi[:, :, 0] ** 2) + (whi[:, :, 1] ** 2) + eps

    # centers of boxes
    x_p = (box1[:, None, 0] + box1[:, None, 2]) / 2
    y_p = (box1[:, None, 1] + box1[:, None, 3]) / 2
    x_g = (box2[:, 0] + box2[:, 2]) / 2
    y_g = (box2[:, 1] + box2[:, 3]) / 2
    # The distance between boxes' centers squared.
    centers_distance_squared = (x_p - x_g) ** 2 + (y_p - y_g) ** 2

    w_pred = box1[:, None, 2] - box1[:, None, 0]
    h_pred = box1[:, None, 3] - box1[:, None, 1]

    w_gt = box2[:, 2] - box2[:, 0]
    h_gt = box2[:, 3] - box2[:, 1]

    v = (4 / (torch.pi ** 2)) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (centers_distance_squared / diagonal_distance_squared) - alpha * v


def box_diou(box1, box2, eps: float = 1e-7):
    """
    Return distance intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
        eps (float, optional): small number to prevent division by zero. Default: 1e-7
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise distance IoU values
        for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    union = (area1[:, None] + area2 - inter)

    iou = inter / union

    lti = torch.min(box1[:, None, :2], box2[:, :2])
    rbi = torch.max(box1[:, None, 2:], box2[:, 2:])

    whi = (rbi - lti).clamp(min=0)  # [N,M,2]
    diagonal_distance_squared = (whi[:, :, 0] ** 2) + (whi[:, :, 1] ** 2) + eps

    # centers of boxes
    x_p = (box1[:, None, 0] + box1[:, None, 2]) / 2
    y_p = (box1[:, None, 1] + box1[:, None, 3]) / 2
    x_g = (box2[:, 0] + box2[:, 2]) / 2
    y_g = (box2[:, 1] + box2[:, 3]) / 2
    # The distance between boxes' centers squared.
    centers_distance_squared = (x_p - x_g) ** 2 + (y_p - y_g) ** 2

    # The distance IoU is the IoU penalized by a normalized
    # distance between boxes' centers squared.
    return iou - (centers_distance_squared / diagonal_distance_squared)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':  # keys
        x[k] = None
    x['epoch'] = -1
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path
```

### utils/wandb_logging/log_dataset.py

```python
import argparse

import yaml

from wandb_utils import WandbLogger

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'


def create_dataset_artifact(opt):
    with open(opt.data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    logger = WandbLogger(opt, '', None, data, job_type='Dataset Creation')
```

### utils/wandb_logging/__init__.py

```python
# init```

### utils/wandb_logging/wandb_utils.py

```python
import json
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))  # add utils/ to path
from utils.datasets import LoadImagesAndLabels
from utils.datasets import img2label_paths
from utils.general import colorstr, xywh2xyxy, check_dataset

try:
    import wandb
    from wandb import init, finish
except ImportError:
    wandb = None

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'


def remove_prefix(from_string, prefix=WANDB_ARTIFACT_PREFIX):
    return from_string[len(prefix):]


def check_wandb_config_file(data_config_file):
    wandb_config = '_wandb.'.join(data_config_file.rsplit('.', 1))  # updated data.yaml path
    if Path(wandb_config).is_file():
        return wandb_config
    return data_config_file


def get_run_info(run_path):
    run_path = Path(remove_prefix(run_path, WANDB_ARTIFACT_PREFIX))
    run_id = run_path.stem
    project = run_path.parent.stem
    model_artifact_name = 'run_' + run_id + '_model'
    return run_id, project, model_artifact_name


def check_wandb_resume(opt):
    process_wandb_config_ddp_mode(opt) if opt.global_rank not in [-1, 0] else None
    if isinstance(opt.resume, str):
        if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
            if opt.global_rank not in [-1, 0]:  # For resuming DDP runs
                run_id, project, model_artifact_name = get_run_info(opt.resume)
                api = wandb.Api()
                artifact = api.artifact(project + '/' + model_artifact_name + ':latest')
                modeldir = artifact.download()
                opt.weights = str(Path(modeldir) / "last.pt")
            return True
    return None


def process_wandb_config_ddp_mode(opt):
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    train_dir, val_dir = None, None
    if isinstance(data_dict['train'], str) and data_dict['train'].startswith(WANDB_ARTIFACT_PREFIX):
        api = wandb.Api()
        train_artifact = api.artifact(remove_prefix(data_dict['train']) + ':' + opt.artifact_alias)
        train_dir = train_artifact.download()
        train_path = Path(train_dir) / 'data/images/'
        data_dict['train'] = str(train_path)

    if isinstance(data_dict['val'], str) and data_dict['val'].startswith(WANDB_ARTIFACT_PREFIX):
        api = wandb.Api()
        val_artifact = api.artifact(remove_prefix(data_dict['val']) + ':' + opt.artifact_alias)
        val_dir = val_artifact.download()
        val_path = Path(val_dir) / 'data/images/'
        data_dict['val'] = str(val_path)
    if train_dir or val_dir:
        ddp_data_path = str(Path(val_dir) / 'wandb_local_data.yaml')
        with open(ddp_data_path, 'w') as f:
            yaml.dump(data_dict, f)
        opt.data = ddp_data_path


class WandbLogger():
    def __init__(self, opt, name, run_id, data_dict, job_type='Training'):
        # Pre-training routine --
        self.job_type = job_type
        self.wandb, self.wandb_run, self.data_dict = wandb, None if not wandb else wandb.run, data_dict
        # It's more elegant to stick to 1 wandb.init call, but useful config data is overwritten in the WandbLogger's wandb.init call
        if isinstance(opt.resume, str):  # checks resume from artifact
            if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
                run_id, project, model_artifact_name = get_run_info(opt.resume)
                model_artifact_name = WANDB_ARTIFACT_PREFIX + model_artifact_name
                assert wandb, 'install wandb to resume wandb runs'
                # Resume wandb-artifact:// runs here| workaround for not overwriting wandb.config
                self.wandb_run = wandb.init(id=run_id, project=project, resume='allow')
                opt.resume = model_artifact_name
        elif self.wandb:
            self.wandb_run = wandb.init(config=opt,
                                        resume="allow",
                                        project='DynamicDet' if opt.project == 'runs/train' else Path(opt.project).stem,
                                        name=name,
                                        job_type=job_type,
                                        id=run_id) if not wandb.run else wandb.run
        if self.wandb_run:
            if self.job_type == 'Training':
                if not opt.resume:
                    wandb_data_dict = self.check_and_upload_dataset(opt) if opt.upload_dataset else data_dict
                    # Info useful for resuming from artifacts
                    self.wandb_run.config.opt = vars(opt)
                    self.wandb_run.config.data_dict = wandb_data_dict
                self.data_dict = self.setup_training(opt, data_dict)
            if self.job_type == 'Dataset Creation':
                self.data_dict = self.check_and_upload_dataset(opt)
        else:
            prefix = colorstr('wandb: ')
            print(f"{prefix}Install Weights & Biases for DynamicDet logging with 'pip install wandb' (recommended)")

    def check_and_upload_dataset(self, opt):
        assert wandb, 'Install wandb to upload dataset'
        check_dataset(self.data_dict)
        config_path = self.log_dataset_artifact(opt.data,
                                                opt.single_cls,
                                                'DynamicDet' if opt.project == 'runs/train' else Path(opt.project).stem)
        print("Created dataset config file ", config_path)
        with open(config_path) as f:
            wandb_data_dict = yaml.load(f, Loader=yaml.SafeLoader)
        return wandb_data_dict

    def setup_training(self, opt, data_dict):
        self.log_dict, self.current_epoch, self.log_imgs = {}, 0, 16  # Logging Constants
        self.bbox_interval = opt.bbox_interval
        if isinstance(opt.resume, str):
            modeldir, _ = self.download_model_artifact(opt)
            if modeldir:
                self.weights = Path(modeldir) / "last.pt"
                config = self.wandb_run.config
                opt.weights, opt.save_period, opt.batch_size, opt.bbox_interval, opt.epochs, opt.hyp = str(
                    self.weights), config.save_period, config.total_batch_size, config.bbox_interval, config.epochs, \
                                                                                                       config.opt['hyp']
            data_dict = dict(self.wandb_run.config.data_dict)  # eliminates the need for config file to resume
        if 'val_artifact' not in self.__dict__:  # If --upload_dataset is set, use the existing artifact, don't download
            self.train_artifact_path, self.train_artifact = self.download_dataset_artifact(data_dict.get('train'),
                                                                                           opt.artifact_alias)
            self.val_artifact_path, self.val_artifact = self.download_dataset_artifact(data_dict.get('val'),
                                                                                       opt.artifact_alias)
            self.result_artifact, self.result_table, self.val_table, self.weights = None, None, None, None
            if self.train_artifact_path is not None:
                train_path = Path(self.train_artifact_path) / 'data/images/'
                data_dict['train'] = str(train_path)
            if self.val_artifact_path is not None:
                val_path = Path(self.val_artifact_path) / 'data/images/'
                data_dict['val'] = str(val_path)
                self.val_table = self.val_artifact.get("val")
                self.map_val_table_path()
        if self.val_artifact is not None:
            self.result_artifact = wandb.Artifact("run_" + wandb.run.id + "_progress", "evaluation")
            self.result_table = wandb.Table(["epoch", "id", "prediction", "avg_confidence"])
        if opt.bbox_interval == -1:
            self.bbox_interval = opt.bbox_interval = (opt.epochs // 10) if opt.epochs > 10 else 1
        return data_dict

    def download_dataset_artifact(self, path, alias):
        if isinstance(path, str) and path.startswith(WANDB_ARTIFACT_PREFIX):
            dataset_artifact = wandb.use_artifact(remove_prefix(path, WANDB_ARTIFACT_PREFIX) + ":" + alias)
            assert dataset_artifact is not None, "'Error: W&B dataset artifact doesn\'t exist'"
            datadir = dataset_artifact.download()
            return datadir, dataset_artifact
        return None, None

    def download_model_artifact(self, opt):
        if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
            model_artifact = wandb.use_artifact(remove_prefix(opt.resume, WANDB_ARTIFACT_PREFIX) + ":latest")
            assert model_artifact is not None, 'Error: W&B model artifact doesn\'t exist'
            modeldir = model_artifact.download()
            epochs_trained = model_artifact.metadata.get('epochs_trained')
            total_epochs = model_artifact.metadata.get('total_epochs')
            assert epochs_trained < total_epochs, 'training to %g epochs is finished, nothing to resume.' % (
                total_epochs)
            return modeldir, model_artifact
        return None, None

    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        model_artifact = wandb.Artifact('run_' + wandb.run.id + '_model', type='model', metadata={
            'original_url': str(path),
            'epochs_trained': epoch + 1,
            'save period': opt.save_period,
            'project': opt.project,
            'total_epochs': opt.epochs,
            'fitness_score': fitness_score
        })
        model_artifact.add_file(str(path / 'last.pt'), name='last.pt')
        wandb.log_artifact(model_artifact,
                           aliases=['latest', 'epoch ' + str(self.current_epoch), 'best' if best_model else ''])
        print("Saving model artifact on epoch ", epoch + 1)

    def log_dataset_artifact(self, data_file, single_cls, project, overwrite_config=False):
        with open(data_file) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
        nc, names = (1, ['item']) if single_cls else (int(data['nc']), data['names'])
        names = {k: v for k, v in enumerate(names)}  # to index dictionary
        self.train_artifact = self.create_dataset_table(LoadImagesAndLabels(
            data['train']), names, name='train') if data.get('train') else None
        self.val_artifact = self.create_dataset_table(LoadImagesAndLabels(
            data['val']), names, name='val') if data.get('val') else None
        if data.get('train'):
            data['train'] = WANDB_ARTIFACT_PREFIX + str(Path(project) / 'train')
        if data.get('val'):
            data['val'] = WANDB_ARTIFACT_PREFIX + str(Path(project) / 'val')
        path = data_file if overwrite_config else '_wandb.'.join(data_file.rsplit('.', 1))  # updated data.yaml path
        data.pop('download', None)
        with open(path, 'w') as f:
            yaml.dump(data, f)

        if self.job_type == 'Training':  # builds correct artifact pipeline graph
            self.wandb_run.use_artifact(self.val_artifact)
            self.wandb_run.use_artifact(self.train_artifact)
            self.val_artifact.wait()
            self.val_table = self.val_artifact.get('val')
            self.map_val_table_path()
        else:
            self.wandb_run.log_artifact(self.train_artifact)
            self.wandb_run.log_artifact(self.val_artifact)
        return path

    def map_val_table_path(self):
        self.val_table_map = {}
        print("Mapping dataset")
        for i, data in enumerate(tqdm(self.val_table.data)):
            self.val_table_map[data[3]] = data[0]

    def create_dataset_table(self, dataset, class_to_id, name='dataset'):
        # TODO: Explore multiprocessing to slpit this loop parallely| This is essential for speeding up the the logging
        artifact = wandb.Artifact(name=name, type="dataset")
        img_files = tqdm([dataset.path]) if isinstance(dataset.path, str) and Path(dataset.path).is_dir() else None
        img_files = tqdm(dataset.img_files) if not img_files else img_files
        for img_file in img_files:
            if Path(img_file).is_dir():
                artifact.add_dir(img_file, name='data/images')
                labels_path = 'labels'.join(dataset.path.rsplit('images', 1))
                artifact.add_dir(labels_path, name='data/labels')
            else:
                artifact.add_file(img_file, name='data/images/' + Path(img_file).name)
                label_file = Path(img2label_paths([img_file])[0])
                artifact.add_file(str(label_file),
                                  name='data/labels/' + label_file.name) if label_file.exists() else None
        table = wandb.Table(columns=["id", "train_image", "Classes", "name"])
        class_set = wandb.Classes([{'id': id, 'name': name} for id, name in class_to_id.items()])
        for si, (img, labels, paths, shapes) in enumerate(tqdm(dataset)):
            height, width = shapes[0]
            labels[:, 2:] = (xywh2xyxy(labels[:, 2:].view(-1, 4))) * torch.Tensor([width, height, width, height])
            box_data, img_classes = [], {}
            for cls, *xyxy in labels[:, 1:].tolist():
                cls = int(cls)
                box_data.append({"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": cls,
                                 "box_caption": "%s" % (class_to_id[cls]),
                                 "scores": {"acc": 1},
                                 "domain": "pixel"})
                img_classes[cls] = class_to_id[cls]
            boxes = {"ground_truth": {"box_data": box_data, "class_labels": class_to_id}}  # inference-space
            table.add_data(si, wandb.Image(paths, classes=class_set, boxes=boxes), json.dumps(img_classes),
                           Path(paths).name)
        artifact.add(table, name)
        return artifact

    def log_training_progress(self, predn, path, names):
        if self.val_table and self.result_table:
            class_set = wandb.Classes([{'id': id, 'name': name} for id, name in names.items()])
            box_data = []
            total_conf = 0
            for *xyxy, conf, cls in predn.tolist():
                if conf >= 0.25:
                    box_data.append(
                        {"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                         "class_id": int(cls),
                         "box_caption": "%s %.3f" % (names[cls], conf),
                         "scores": {"class_score": conf},
                         "domain": "pixel"})
                    total_conf = total_conf + conf
            boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
            id = self.val_table_map[Path(path).name]
            self.result_table.add_data(self.current_epoch,
                                       id,
                                       wandb.Image(self.val_table.data[id][1], boxes=boxes, classes=class_set),
                                       total_conf / max(1, len(box_data))
                                       )

    def log(self, log_dict):
        if self.wandb_run:
            for key, value in log_dict.items():
                self.log_dict[key] = value

    def end_epoch(self, best_result=False):
        if self.wandb_run:
            wandb.log(self.log_dict)
            self.log_dict = {}
            if self.result_artifact:
                train_results = wandb.JoinedTable(self.val_table, self.result_table, "id")
                self.result_artifact.add(train_results, 'result')
                wandb.log_artifact(self.result_artifact, aliases=['latest', 'epoch ' + str(self.current_epoch),
                                                                  ('best' if best_result else '')])
                self.result_table = wandb.Table(["epoch", "id", "prediction", "avg_confidence"])
                self.result_artifact = wandb.Artifact("run_" + wandb.run.id + "_progress", "evaluation")

    def finish_run(self):
        if self.wandb_run:
            if self.log_dict:
                wandb.log(self.log_dict)
            wandb.run.finish()
```

## models/__init__.py

```python
# init```

## models/common.py

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from utils.general import non_max_suppression


##### basic ####

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class ReOrg(nn.Module):
    def __init__(self):
        super(ReOrg, self).__init__()

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Shortcut(nn.Module):
    def __init__(self, dimension=0):
        super(Shortcut, self).__init__()
        self.d = dimension

    def forward(self, x):
        return x[0]+x[1]


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class ConvCheckpoint(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvCheckpoint, self).__init__()
        self.conv = Conv(c1, c2, k, s, p, g, act)

    def forward(self, x):
        x = checkpoint.checkpoint(self.conv, x)
        return x

##### end of basic #####


##### cspnet #####

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

##### end of cspnet #####


##### yolor #####

class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x

##### end of yolor #####


##### repvgg #####

class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d( c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels = conv.in_channels,
                              out_channels = conv.out_channels,
                              kernel_size = conv.kernel_size,
                              stride=conv.stride,
                              padding = conv.padding,
                              dilation = conv.dilation,
                              groups = conv.groups,
                              bias = True,
                              padding_mode = conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")

        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])

        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm)):
            # print(f"fuse: rbr_identity == BatchNorm2d or SyncBatchNorm")
            identity_conv_1x1 = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.groups,
                    bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            # print(f"fuse: rbr_identity != BatchNorm2d, rbr_identity = {self.rbr_identity}")
            bias_identity_expanded = torch.nn.Parameter( torch.zeros_like(rbr_1x1_bias) )
            weight_identity_expanded = torch.nn.Parameter( torch.zeros_like(weight_1x1_expanded) )


        #print(f"self.rbr_1x1.weight = {self.rbr_1x1.weight.shape}, ")
        #print(f"weight_1x1_expanded = {weight_1x1_expanded.shape}, ")
        #print(f"self.rbr_dense.weight = {self.rbr_dense.weight.shape}, ")

        self.rbr_dense.weight = torch.nn.Parameter(self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None

##### end of repvgg #####


##### yolov5 #####

class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)

##### end of yolov5 ######


##### CBNet #####

class CBLinear(nn.Module):
    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):  # ch_in, ch_outs, kernel, stride, padding, groups
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs

class CBFuse(nn.Module):
    def __init__(self, idx):
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode='nearest') for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out

##### end of CBNet #####


##### DynamicDet #####

def sigmoid(logits, hard=False, threshold=0.5):
    y_soft = logits.sigmoid()
    if hard:
        indices = (y_soft < threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


class AdaptiveRouter(nn.Module):
    def __init__(self, features_channels, out_channels, reduction=4):
        super(AdaptiveRouter, self).__init__()
        self.inp = sum(features_channels)
        self.oup = out_channels
        self.reduction = reduction
        self.layer1 = nn.Conv2d(self.inp, self.inp//self.reduction, kernel_size=1, bias=True)
        self.layer2 = nn.Conv2d(self.inp//self.reduction, self.oup, kernel_size=1, bias=True)

    def forward(self, xs, thres=0.5):
        xs = [x.mean(dim=(2, 3), keepdim=True) for x in xs]
        xs = torch.cat(xs, dim=1)
        xs = self.layer1(xs)
        xs = F.relu(xs, inplace=True)
        xs = self.layer2(xs).flatten(1)
        if self.training:
            xs = sigmoid(xs, hard=False, threshold=thres)
        else:
            xs = xs.sigmoid()
        return xs

##### end of DynamicDet #####
```

## models/yolo.py

```python
import math
import argparse
import logging
from pathlib import Path
import yaml
import sys
from copy import deepcopy

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)
import torch
from torch.nn.modules.batchnorm import _BatchNorm
from models.common import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class IDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    def fuse(self):
        print("IDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1,c2,_,_ = self.m[i].weight.shape
            c1_,c2_, _,_ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1,c2),self.ia[i].implicit.reshape(c2_,c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1,c2, _,_ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0,1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=z.device)
        box @= convert_matrix
        return (box, score)


class Model(nn.Module):
    def __init__(self, cfg, ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        assert isinstance(cfg, str)
        self.yaml_file = Path(cfg).name
        with open(cfg) as f:
            self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        self.dynamic = self.yaml.get('dynamic', False)
        if self.dynamic:
            router_channels = self.yaml['router_channels']
            reduction = self.yaml.get('router_reduction', 4)
            self.router = AdaptiveRouter(router_channels, 1, reduction=reduction)
            self.router_ins = self.yaml['router_ins']
            self.dy_thres = 0.5
            self.get_score = False

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        self.model_b, self.save_b, self.model_b2, self.save_b2, self.model_h, self.save_h, self.model_h2, self.save_h2 = parse_model(deepcopy(self.yaml), ch_b=[ch])  # model, savelist
        self.keep_input = self.yaml.get('keep_input', False)
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model_h[-1]  # Detect()
        m2 = self.model_h2[-1]  # Detect()
        if isinstance(m, IDetect):
            s = 256  # 2x min stride
            if self.dynamic:
                m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[0][:m.anchors.shape[0]]])  # forward
            else:
                m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[:m.anchors.shape[0]]])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            if self.dynamic:
                m2.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[0][:m2.anchors.shape[0]]])  # forward
            else:
                m2.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[:m2.anchors.shape[0]]])  # forward
            check_anchor_order(m2)
            m2.anchors /= m2.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())
        # Init weights, biases
        initialize_weights(self)
        # self.initialize_cblinear()
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        if self.keep_input:
            input_x = x

        y, dt = [], []  # outputs
        outs = []
        for m in self.model_b:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                c = isinstance(m, IDetect)
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                for _ in range(10):
                    m(x.copy() if c else x)
                t = time_synchronized()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run

            y.append(x if m.i in self.save_b else None)  # save output
        assert len(y) == self.yaml['n_first_layers']

        if self.dynamic:
            score = self.router([y[i] for i in self.router_ins]) # 'score' denotes the (1 - difficulty score)

            if not hasattr(self, 'get_score'):
                self.get_score = False
            if self.get_score:
                return score

        need_second = self.training or (not self.dynamic) or score[:, 0] < self.dy_thres
        need_first_head = self.training or (self.dynamic and score[:, 0] >= self.dy_thres)

        if need_second:
            for m in self.model_b2:
                if m.f == 'input':
                    x = input_x
                elif m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

                if profile:
                    c = isinstance(m, IDetect)
                    o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                    for _ in range(10):
                        m(x.copy() if c else x)
                    t = time_synchronized()
                    for _ in range(10):
                        m(x.copy() if c else x)
                    dt.append((time_synchronized() - t) * 100)
                    print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

                x = m(x)  # run

                y.append(x if m.i in self.save_b2 else None)  # save output

        if need_first_head:
            for m in self.model_h:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

                if profile:
                    c = isinstance(m, IDetect)
                    o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                    for _ in range(10):
                        m(x.copy() if c else x)
                    t = time_synchronized()
                    for _ in range(10):
                        m(x.copy() if c else x)
                    dt.append((time_synchronized() - t) * 100)
                    print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

                x = m(x)  # run

                y.append(x if m.i in self.save_h else None)  # save output

            outs.extend(x)

        if need_second:
            for m in self.model_h2:
                if isinstance(m.f, int) and m.f > 0:
                    cur_f = m.f + self.yaml['n_first_layers']
                else:
                    cur_f = m.f
                if cur_f != -1:  # if not from previous layer
                    x = y[cur_f] if isinstance(cur_f, int) else [x if j == -1 else y[j] for j in cur_f]  # from earlier layers

                if profile:
                    c = isinstance(m, IDetect)
                    o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                    for _ in range(10):
                        m(x.copy() if c else x)
                    t = time_synchronized()
                    for _ in range(10):
                        m(x.copy() if c else x)
                    dt.append((time_synchronized() - t) * 100)
                    print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

                x = m(x)  # run

                y.append(x if m.i in self.save_h2 else None)  # save output

            outs.extend(x)

        if profile:
            print('%.1fms total' % sum(dt))
        if self.training and self.dynamic:
            return outs, score
        else:
            return outs

    def close_all_bn(self):
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model_h[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        m = self.model_h2[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model_h[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))
        m = self.model_h2[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for model in [self.model_b, self.model_b2, self.model_h, self.model_h2]:
            for m in model.modules():
                if isinstance(m, RepConv):
                    #print(f" fuse_repvgg_block")
                    m.fuse_repvgg_block()
                elif type(m) is Conv and hasattr(m, 'bn'):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.fuseforward  # update forward
                elif isinstance(m, IDetect):
                    m.fuse()
                    m.forward = m.fuseforward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model_h[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model_h[-1].i + 1  # index
            self.model_h.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model_h = self.model_h[:-1]  # remove
        present = type(self.model_h2[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model_h2[-1].i + 1  # index
            self.model_h2.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model_h2 = self.model_h2[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch_b):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    layers_b, save_b, c2 = [], [], ch_b[-1]  # layers, savelist, ch_b out

    for i, (f, n, m, args) in enumerate(d['backbone']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, ConvCheckpoint, RepConv, SPPCSPC]:
            c1, c2 = ch_b[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [SPPCSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch_b[f]]
        elif m is Concat:
            c2 = sum([ch_b[x] for x in f])
        elif m is Shortcut:
            c2 = ch_b[f[0]]
        elif m is IDetect:
            args.append([ch_b[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            c2 = ch_b[f] * 4
        else:
            c2 = ch_b[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save_b.extend(x % i for x in ([f] if isinstance(f, (int, str)) else f) if x != -1)  # append to savelist
        layers_b.append(m_)
        if i == 0:
            ch_b = []
        ch_b.append(c2)

    layers_b2, save_b2 = [], []  # layers, savelist
    ch_b2 = []

    for i, (f, n, m, args) in enumerate(d['dual_backbone']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        chs = []
        for x in ([f] if isinstance(f, (int, str)) else f):
            if isinstance(x, str):
                continue
            if x >= 0:
                chs.append(ch_b)
            else:
                chs.append(ch_b2)

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, ConvCheckpoint, RepConv, SPPCSPC]:
            if f == 'input':
                c1, c2 = 3, args[0]
            else:
                assert len(chs) == 1
                c1, c2 = chs[0][f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [SPPCSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            assert len(chs) == 1
            args = [chs[0][f]]
        elif m is Concat:
            c2 = sum([ch[x] for x, ch in zip(f, chs)])
        elif m is Shortcut:
            assert len(chs) == 1
            c2 = chs[0][f[0]]
        elif m is IDetect:
            args.append([ch[x] for x, ch in zip(f, chs)])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            assert len(chs) == 1
            c2 = chs[0][f] * 4
        elif m is CBLinear:
            c2 = args[0]
            assert len(chs) == 1
            c1 = chs[0][f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = chs[-1][f[-1]]
        else:
            assert len(chs) == 1
            c2 = chs[0][f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        for x in ([f] if isinstance(f, (int, str)) else f):  # append to savelist
            if isinstance(x, str):
                continue
            if x >= 0:
                save_b.extend([x])
            elif x != -1:
                save_b2.extend([x % i])
        layers_b2.append(m_)
        ch_b2.append(c2)

    layers_h, save_h, ch_h = [], [], []
    for i, (f, n, m, args) in enumerate(d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        chs = []
        for x in ([f] if isinstance(f, (int, str)) else f):
            if isinstance(x, str):
                continue
            if x >= 0:
                chs.append(ch_b)
            else:
                chs.append(ch_h)

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, ConvCheckpoint, RepConv, SPPCSPC]:
            assert len(chs) == 1
            c1, c2 = chs[0][f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [SPPCSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            assert len(chs) == 1
            args = [chs[0][f]]
        elif m is Concat:
            c2 = sum([ch[x] for x, ch in zip(f, chs)])
        elif m is Shortcut:
            assert len(chs) == 1
            c2 = chs[0][f[0]]
        elif m is IDetect:
            args.append([ch[x] for x, ch in zip(f, chs)])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            assert len(chs) == 1
            c2 = chs[0][f] * 4
        else:
            assert len(chs) == 1
            c2 = chs[0][f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        for x in ([f] if isinstance(f, (int, str)) else f):  # append to savelist
            if isinstance(x, str):
                continue
            if x >= 0:
                save_b.extend([x])
            elif x != -1:
                save_h.extend([x % i])
        layers_h.append(m_)
        ch_h.append(c2)

    layers_h2, save_h2, ch_h2 = [], [], []
    for i, (f, n, m, args) in enumerate(d['head2']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        chs = []
        for x in ([f] if isinstance(f, (int, str)) else f):
            if isinstance(x, str):
                continue
            if x >= 0:
                chs.append(ch_b2)
            else:
                chs.append(ch_h2)

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, ConvCheckpoint, RepConv, SPPCSPC]:
            assert len(chs) == 1
            c1, c2 = chs[0][f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [SPPCSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            assert len(chs) == 1
            args = [chs[0][f]]
        elif m is Concat:
            c2 = sum([ch[x] for x, ch in zip(f, chs)])
        elif m is Shortcut:
            assert len(chs) == 1
            c2 = chs[0][f[0]]
        elif m is IDetect:
            args.append([ch[x] for x, ch in zip(f, chs)])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            assert len(chs) == 1
            c2 = chs[0][f] * 4
        else:
            assert len(chs) == 1
            c2 = chs[0][f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        for x in ([f] if isinstance(f, (int, str)) else f):  # append to savelist
            if isinstance(x, str):
                continue
            if x >= 0:
                save_b.extend([x])
            elif x != -1:
                save_h2.extend([x % i])
        layers_h2.append(m_)
        ch_h2.append(c2)

    save_b.extend(d['b1_save'])
    save_b2.extend(d['b2_save'])

    return nn.Sequential(*layers_b), sorted(save_b), nn.Sequential(*layers_b2), sorted(save_b2), \
            nn.Sequential(*layers_h), sorted(save_h), nn.Sequential(*layers_h2), sorted(save_h2)
```

## hyp/hyp.finetune.dynamic.adam.yaml

```
lr0: 0.00001  # initial learning rate
lrf: 1  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.005  # optimizer weight decay 5e-4
warmup_epochs: 0.01  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.01  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.3  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 0.7  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.2  # image translation (+/- fraction)
scale: 0.9  # image scale (+/- gain)
shear: 0.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.0  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 0.0  # image mosaic (probability)
mixup: 0.0  # image mixup (probability)
copy_paste: 0.0  # image copy paste (probability)
paste_in: 0.0  # image copy paste (probability), use 0 for faster training
```

## hyp/hyp.scratch.p5.yaml

```
lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.1  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.3  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 0.7  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.2  # image translation (+/- fraction)
scale: 0.9  # image scale (+/- gain)
shear: 0.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.0  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 1.0  # image mosaic (probability)
mixup: 0.15  # image mixup (probability)
copy_paste: 0.0  # image copy paste (probability)
paste_in: 0.15  # image copy paste (probability), use 0 for faster training
```

## hyp/hyp.scratch.p6.yaml

```
lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.2  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.3  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 0.7  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.2  # image translation (+/- fraction)
scale: 0.9  # image scale (+/- gain)
shear: 0.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.0  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 1.0  # image mosaic (probability)
mixup: 0.15  # image mixup (probability)
copy_paste: 0.0  # image copy paste (probability)
paste_in: 0.15  # image copy paste (probability), use 0 for faster training
```

## cfg/dy-yolov7x-step1.yaml

```
# parameters
nc: 80  # number of classes
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple

# anchors
anchors:
  - [12,16, 19,36, 40,28]  # P3/8
  - [36,75, 76,55, 72,146]  # P4/16
  - [142,110, 192,243, 459,401]  # P5/32

# yolov7 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [40, 3, 1]],  # 0

   [-1, 1, Conv, [80, 3, 2]],  # 1-P1/2
   [-1, 1, Conv, [80, 3, 1]],

   [-1, 1, Conv, [160, 3, 2]],  # 3-P2/4
   [-1, 1, Conv, [64, 1, 1]],
   [-2, 1, Conv, [64, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [320, 1, 1]],  # 13-loc1

   [-1, 1, MP, []],
   [-1, 1, Conv, [160, 1, 1]],
   [-3, 1, Conv, [160, 1, 1]],
   [-1, 1, Conv, [160, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 18-P3/8
   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [640, 1, 1]],  # 28-loc2

   [-1, 1, MP, []],
   [-1, 1, Conv, [320, 1, 1]],
   [-3, 1, Conv, [320, 1, 1]],
   [-1, 1, Conv, [320, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 33-P4/16
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [1280, 1, 1]],  # 43-loc3

   [-1, 1, MP, []],
   [-1, 1, Conv, [640, 1, 1]],
   [-3, 1, Conv, [640, 1, 1]],
   [-1, 1, Conv, [640, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 48-P5/32
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [1280, 1, 1]],  # 58
  ]

n_first_layers: 59

b1_save: [2, 13, 28, 43, 58]
b2_save: [36, 52, 68]

keep_input: True

dual_backbone:
   [[2, 1, CBLinear, [[80]]], # 0
   [13, 1, CBLinear, [[80, 160]]], # 1
   [28, 1, CBLinear, [[80, 160, 320]]], # 2
   [43, 1, CBLinear, [[80, 160, 320, 640]]], # 3
   [58, 1, CBLinear, [[80, 160, 320, 640, 1280]]], # 4

   [input, 1, Conv, [40, 3, 1]],  # 5

   [-1, 1, Conv, [80, 3, 2]],  # 6-P1/2
   [[-7, -6, -5, -4, -3, -1], 1, CBFuse, [[0, 0, 0, 0, 0]]], # 7
   [-1, 1, Conv, [80, 3, 1]],

   [-1, 1, Conv, [160, 3, 2]],  # 9-P2/4
   [[-9, -8, -7, -6, -1], 1, CBFuse, [[1, 1, 1, 1]]], # 10
   [-1, 1, Conv, [64, 1, 1]],
   [-2, 1, Conv, [64, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [320, 1, 1]],  # 20-loc1

   [-1, 1, MP, []],
   [-1, 1, Conv, [160, 1, 1]],
   [13, 1, Conv, [160, 1, 1]],
   [-1, 1, Conv, [160, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 25-P3/8
   [[-24, -23, -22, -1], 1, CBFuse, [[2, 2, 2]]], # 26
   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [640, 1, 1]],  # 36-loc2

   [-1, 1, MP, []],
   [-1, 1, Conv, [320, 1, 1]],
   [-3, 1, Conv, [320, 1, 1]],
   [-1, 1, Conv, [320, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 41-P4/16
   [[-39, -38, -1], 1, CBFuse, [[3, 3]]], # 42

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [1280, 1, 1]],  # 52-loc3

   [-1, 1, MP, []],
   [-1, 1, Conv, [640, 1, 1]],
   [-3, 1, Conv, [640, 1, 1]],
   [-1, 1, Conv, [640, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 57-P5/32
   [[-54, -1], 1, CBFuse, [[4]]], # 58

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [1280, 1, 1]],  # 68
  ]

# yolov7 head
head:
  [[58, 1, SPPCSPC, [640]], # 0

   [-1, 1, Conv, [320, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [43, 1, Conv, [320, 1, 1]], # route backbone P4
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [320, 1, 1]], # 14

   [-1, 1, Conv, [160, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [28, 1, Conv, [160, 1, 1]], # route backbone P3
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [160, 1, 1]], # 28

   [-1, 1, MP, []],
   [-1, 1, Conv, [160, 1, 1]],
   [-3, 1, Conv, [160, 1, 1]],
   [-1, 1, Conv, [160, 3, 2]],
   [[-1, -3, -19], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [320, 1, 1]], # 43

   [-1, 1, MP, []],
   [-1, 1, Conv, [320, 1, 1]],
   [-3, 1, Conv, [320, 1, 1]],
   [-1, 1, Conv, [320, 3, 2]],
   [[-1, -3, -48], 1, Concat, [1]],

   [-1, 1, Conv, [512, 1, 1]],
   [-2, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [640, 1, 1]], # 58

   [-31, 1, Conv, [320, 3, 1]], # 59
   [-17, 1, Conv, [640, 3, 1]], # 60
   [-3, 1, Conv, [1280, 3, 1]], # 61

   [[-3, -2, -1], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5)
  ]

# yolov7 head
head2:
  [[68, 1, SPPCSPC, [640]], # 0

   [-1, 1, Conv, [320, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [52, 1, Conv, [320, 1, 1]], # route backbone P4
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [320, 1, 1]], # 14

   [-1, 1, Conv, [160, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [36, 1, Conv, [160, 1, 1]], # route backbone P3
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [160, 1, 1]], # 28

   [-1, 1, MP, []],
   [-1, 1, Conv, [160, 1, 1]],
   [-3, 1, Conv, [160, 1, 1]],
   [-1, 1, Conv, [160, 3, 2]],
   [[-1, -3, -19], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [320, 1, 1]], # 43

   [-1, 1, MP, []],
   [-1, 1, Conv, [320, 1, 1]],
   [-3, 1, Conv, [320, 1, 1]],
   [-1, 1, Conv, [320, 3, 2]],
   [[-1, -3, -48], 1, Concat, [1]],

   [-1, 1, Conv, [512, 1, 1]],
   [-2, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [640, 1, 1]], # 58

   [-31, 1, Conv, [320, 3, 1]], # 59
   [-17, 1, Conv, [640, 3, 1]], # 60
   [-3, 1, Conv, [1280, 3, 1]], # 61

   [[-3, -2, -1], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5)
  ]
```

## cfg/dy-yolov7-w6-step2.yaml

```
# parameters
nc: 80  # number of classes
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple

# anchors
anchors:
  - [ 19,27,  44,40,  38,94 ]  # P3/8
  - [ 96,68,  86,152,  180,137 ]  # P4/16
  - [ 140,301,  303,264,  238,542 ]  # P5/32
  - [ 436,615,  739,380,  925,792 ]  # P6/64

# yolov7 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, ReOrg, []],  # 0
   [-1, 1, Conv, [64, 3, 1]],  # 1-P1/2

   [-1, 1, Conv, [128, 3, 2]],  # 2-P2/4
   [-1, 1, Conv, [64, 1, 1]],
   [-2, 1, Conv, [64, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [128, 1, 1]],  # 10-loc1

   [-1, 1, Conv, [256, 3, 2]],  # 11-P3/8
   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]],  # 19-loc2

   [-1, 1, Conv, [512, 3, 2]],  # 20-P4/16
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]],  # 28-loc3

   [-1, 1, Conv, [768, 3, 2]],  # 29-P5/32
   [-1, 1, Conv, [384, 1, 1]],
   [-2, 1, Conv, [384, 1, 1]],
   [-1, 1, Conv, [384, 3, 1]],
   [-1, 1, Conv, [384, 3, 1]],
   [-1, 1, Conv, [384, 3, 1]],
   [-1, 1, Conv, [384, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [768, 1, 1]],  # 37-loc4

   [-1, 1, Conv, [1024, 3, 2]],  # 38-P6/64
   [-1, 1, Conv, [512, 1, 1]],
   [-2, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [1024, 1, 1]],  # 46
  ]


n_first_layers: 47

b1_save: [0, 1, 10, 19, 28, 37, 46]
b2_save: [25, 35, 45, 55]

dynamic: True
router_ins: [1, 10, 19, 28, 37, 46]
router_channels: [64, 128, 256, 512, 768, 1024]


# keep_input: True

dual_backbone:
  [[10, 1, CBLinear, [[128]]], # 0
   [19, 1, CBLinear, [[128, 256]]], # 1
   [28, 1, CBLinear, [[128, 256, 512]]], # 2
   [37, 1, CBLinear, [[128, 256, 512, 768]]], # 3
   [46, 1, CBLinear, [[128, 256, 512, 768, 1024]]], # 4

   [0, 1, Conv, [64, 3, 1]],  # 5-P1/2

   [-1, 1, Conv, [128, 3, 2]],  # 6-P2/4
   [[-7, -6, -5, -4, -3, -1], 1, CBFuse, [[0, 0, 0, 0, 0]]], # 7
   [-1, 1, Conv, [64, 1, 1]],
   [-2, 1, Conv, [64, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [128, 1, 1]],  # 15-loc1

   [-1, 1, Conv, [256, 3, 2]],  # 16-P3/8
   [[-16, -15, -14, -13, -1], 1, CBFuse, [[1, 1, 1, 1]]], # 17
   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]],  # 25-loc2

   [-1, 1, Conv, [512, 3, 2]],  # 26-P4/16
   [[-25, -24, -23, -1], 1, CBFuse, [[2, 2, 2]]], # 27
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]],  # 35-loc3

   [-1, 1, Conv, [768, 3, 2]],  # 36-P5/32
   [[-34, -33, -1], 1, CBFuse, [[3, 3]]], # 37
   [-1, 1, Conv, [384, 1, 1]],
   [-2, 1, Conv, [384, 1, 1]],
   [-1, 1, Conv, [384, 3, 1]],
   [-1, 1, Conv, [384, 3, 1]],
   [-1, 1, Conv, [384, 3, 1]],
   [-1, 1, Conv, [384, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [768, 1, 1]],  # 45-loc4

   [-1, 1, Conv, [1024, 3, 2]],  # 46-P6/64
   [[-43, -1], 1, CBFuse, [[4]]], # 47
   [-1, 1, Conv, [512, 1, 1]],
   [-2, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [1024, 1, 1]],  # 55
  ]

# yolov7 head
head:
  [[46, 1, SPPCSPC, [512]], # 0

   [-1, 1, Conv, [384, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [37, 1, Conv, [384, 1, 1]], # route backbone P5
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [384, 1, 1]],
   [-2, 1, Conv, [384, 1, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [384, 1, 1]], # 12

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [28, 1, Conv, [256, 1, 1]], # route backbone P4
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]], # 24

   [-1, 1, Conv, [128, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [19, 1, Conv, [128, 1, 1]], # route backbone P3
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [128, 1, 1]], # 36

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, -14], 1, Concat, [1]],  # cat

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]], # 46

   [-1, 1, Conv, [384, 3, 2]],
   [[-1, -36], 1, Concat, [1]],  # cat

   [-1, 1, Conv, [384, 1, 1]],
   [-2, 1, Conv, [384, 1, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [384, 1, 1]], # 56

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, -58], 1, Concat, [1]],  # cat

   [-1, 1, Conv, [512, 1, 1]],
   [-2, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]], # 66

   [-31, 1, Conv, [256, 3, 1]],
   [-22, 1, Conv, [512, 3, 1]],
   [-13, 1, Conv, [768, 3, 1]],
   [-4, 1, Conv, [1024, 3, 1]],

   [[-4, -3, -2, -1], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5, P6)
  ]

# yolov7 head
head2:
  [[55, 1, SPPCSPC, [512]], # 0

   [-1, 1, Conv, [384, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [45, 1, Conv, [384, 1, 1]], # route backbone P5
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [384, 1, 1]],
   [-2, 1, Conv, [384, 1, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [384, 1, 1]], # 12

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [35, 1, Conv, [256, 1, 1]], # route backbone P4
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]], # 24

   [-1, 1, Conv, [128, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [25, 1, Conv, [128, 1, 1]], # route backbone P3
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [128, 1, 1]], # 36

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, -14], 1, Concat, [1]],  # cat

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]], # 46

   [-1, 1, Conv, [384, 3, 2]],
   [[-1, -36], 1, Concat, [1]],  # cat

   [-1, 1, Conv, [384, 1, 1]],
   [-2, 1, Conv, [384, 1, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [384, 1, 1]], # 56

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, -58], 1, Concat, [1]],  # cat

   [-1, 1, Conv, [512, 1, 1]],
   [-2, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]], # 66

   [-31, 1, Conv, [256, 3, 1]],
   [-22, 1, Conv, [512, 3, 1]],
   [-13, 1, Conv, [768, 3, 1]],
   [-4, 1, Conv, [1024, 3, 1]],

   [[-4, -3, -2, -1], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5, P6)
  ]
```

## cfg/dy-yolov7-step2.yaml

```
# parameters
nc: 80  # number of classes
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple

# anchors
anchors:
  - [12,16, 19,36, 40,28]  # P3/8
  - [36,75, 76,55, 72,146]  # P4/16
  - [142,110, 192,243, 459,401]  # P5/32

# yolov7 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [32, 3, 1]],  # 0

   [-1, 1, Conv, [64, 3, 2]],  # 1-P1/2
   [-1, 1, Conv, [64, 3, 1]],

   [-1, 1, Conv, [128, 3, 2]],  # 3-P2/4
   [-1, 1, Conv, [64, 1, 1]],
   [-2, 1, Conv, [64, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]],  # 11-loc1

   [-1, 1, MP, []],
   [-1, 1, Conv, [128, 1, 1]],
   [-3, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 16-P3/8
   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]],  # 24-loc2

   [-1, 1, MP, []],
   [-1, 1, Conv, [256, 1, 1]],
   [-3, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 29-P4/16
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [1024, 1, 1]],  # 37-loc3

   [-1, 1, MP, []],
   [-1, 1, Conv, [512, 1, 1]],
   [-3, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [512, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 42-P5/32
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [1024, 1, 1]],  # 50
  ]

n_first_layers: 51

b1_save: [2, 11, 24, 37, 50]
b2_save: [32, 46, 60]

keep_input: True

dynamic: True
router_ins: [2, 11, 24, 37, 50]
router_channels: [64, 256, 512, 1024, 1024]

dual_backbone:
   [[2, 1, CBLinear, [[64]]], # 0
   [11, 1, CBLinear, [[64, 128]]], # 1
   [24, 1, CBLinear, [[64, 128, 256]]], # 2
   [37, 1, CBLinear, [[64, 128, 256, 512]]], # 3
   [50, 1, CBLinear, [[64, 128, 256, 512, 1024]]], # 4

   [input, 1, Conv, [32, 3, 1]],  # 5

   [-1, 1, Conv, [64, 3, 2]],  # 6-P1/2
   [[-7, -6, -5, -4, -3, -1], 1, CBFuse, [[0, 0, 0, 0, 0]]], # 7
   [-1, 1, Conv, [64, 3, 1]],

   [-1, 1, Conv, [128, 3, 2]],  # 9-P2/4
   [[-9, -8, -7, -6, -1], 1, CBFuse, [[1, 1, 1, 1]]], # 10
   [-1, 1, Conv, [64, 1, 1]],
   [-2, 1, Conv, [64, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]],  # 18-loc1

   [-1, 1, MP, []],
   [-1, 1, Conv, [128, 1, 1]],
   [-3, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 23-P3/8
   [[-22, -21, -20, -1], 1, CBFuse, [[2, 2, 2]]], # 24
   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]],  # 32-loc2

   [-1, 1, MP, []],
   [-1, 1, Conv, [256, 1, 1]],
   [-3, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 37-P4/16
   [[-35, -34, -1], 1, CBFuse, [[3, 3]]], # 38
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [1024, 1, 1]],  # 46-loc3

   [-1, 1, MP, []],
   [-1, 1, Conv, [512, 1, 1]],
   [-3, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [512, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 51-P5/32
   [[-48, -1], 1, CBFuse, [[4]]], # 52
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [1024, 1, 1]],  # 60-loc4
  ]

# yolov7 head
head:
  [[50, 1, SPPCSPC, [512]], # 0

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [37, 1, Conv, [256, 1, 1]], # route backbone P4
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]], # 12

   [-1, 1, Conv, [128, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [24, 1, Conv, [128, 1, 1]], # route backbone P3
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [128, 1, 1]], # 24

   [-1, 1, MP, []],
   [-1, 1, Conv, [128, 1, 1]],
   [-3, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 2]],
   [[-1, -3, -17], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]], # 37

   [-1, 1, MP, []],
   [-1, 1, Conv, [256, 1, 1]],
   [-3, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 2]],
   [[-1, -3, -42], 1, Concat, [1]],

   [-1, 1, Conv, [512, 1, 1]],
   [-2, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]], # 50

   [-27, 1, RepConv, [256, 3, 1]], # 51
   [-15, 1, RepConv, [512, 3, 1]], # 52
   [-3, 1, RepConv, [1024, 3, 1]], # 53

   [[-3, -2, -1], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5)
  ]

# yolov7 head
head2:
  [[60, 1, SPPCSPC, [512]], # 0

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [46, 1, Conv, [256, 1, 1]], # route backbone P4
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]], # 12

   [-1, 1, Conv, [128, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [32, 1, Conv, [128, 1, 1]], # route backbone P3
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [128, 1, 1]], # 24

   [-1, 1, MP, []],
   [-1, 1, Conv, [128, 1, 1]],
   [-3, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 2]],
   [[-1, -3, -17], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]], # 37

   [-1, 1, MP, []],
   [-1, 1, Conv, [256, 1, 1]],
   [-3, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 2]],
   [[-1, -3, -42], 1, Concat, [1]],

   [-1, 1, Conv, [512, 1, 1]],
   [-2, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]], # 50

   [-27, 1, RepConv, [256, 3, 1]], # 51
   [-15, 1, RepConv, [512, 3, 1]], # 52
   [-3, 1, RepConv, [1024, 3, 1]], # 53

   [[-3, -2, -1], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5)
  ]
```

## cfg/dy-yolov7-step1.yaml

```
# parameters
nc: 80  # number of classes
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple

# anchors
anchors:
  - [12,16, 19,36, 40,28]  # P3/8
  - [36,75, 76,55, 72,146]  # P4/16
  - [142,110, 192,243, 459,401]  # P5/32

# yolov7 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [32, 3, 1]],  # 0

   [-1, 1, Conv, [64, 3, 2]],  # 1-P1/2
   [-1, 1, Conv, [64, 3, 1]],

   [-1, 1, Conv, [128, 3, 2]],  # 3-P2/4
   [-1, 1, Conv, [64, 1, 1]],
   [-2, 1, Conv, [64, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]],  # 11-loc1

   [-1, 1, MP, []],
   [-1, 1, Conv, [128, 1, 1]],
   [-3, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 16-P3/8
   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]],  # 24-loc2

   [-1, 1, MP, []],
   [-1, 1, Conv, [256, 1, 1]],
   [-3, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 29-P4/16
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [1024, 1, 1]],  # 37-loc3

   [-1, 1, MP, []],
   [-1, 1, Conv, [512, 1, 1]],
   [-3, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [512, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 42-P5/32
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [1024, 1, 1]],  # 50
  ]

n_first_layers: 51

b1_save: [2, 11, 24, 37, 50]
b2_save: [32, 46, 60]

keep_input: True

dual_backbone:
   [[2, 1, CBLinear, [[64]]], # 0
   [11, 1, CBLinear, [[64, 128]]], # 1
   [24, 1, CBLinear, [[64, 128, 256]]], # 2
   [37, 1, CBLinear, [[64, 128, 256, 512]]], # 3
   [50, 1, CBLinear, [[64, 128, 256, 512, 1024]]], # 4

   [input, 1, Conv, [32, 3, 1]],  # 5

   [-1, 1, Conv, [64, 3, 2]],  # 6-P1/2
   [[-7, -6, -5, -4, -3, -1], 1, CBFuse, [[0, 0, 0, 0, 0]]], # 7
   [-1, 1, Conv, [64, 3, 1]],

   [-1, 1, Conv, [128, 3, 2]],  # 9-P2/4
   [[-9, -8, -7, -6, -1], 1, CBFuse, [[1, 1, 1, 1]]], # 10
   [-1, 1, Conv, [64, 1, 1]],
   [-2, 1, Conv, [64, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]],  # 18-loc1

   [-1, 1, MP, []],
   [-1, 1, Conv, [128, 1, 1]],
   [-3, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 23-P3/8
   [[-22, -21, -20, -1], 1, CBFuse, [[2, 2, 2]]], # 24
   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]],  # 32-loc2

   [-1, 1, MP, []],
   [-1, 1, Conv, [256, 1, 1]],
   [-3, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 37-P4/16
   [[-35, -34, -1], 1, CBFuse, [[3, 3]]], # 38
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [1024, 1, 1]],  # 46-loc3

   [-1, 1, MP, []],
   [-1, 1, Conv, [512, 1, 1]],
   [-3, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [512, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 51-P5/32
   [[-48, -1], 1, CBFuse, [[4]]], # 52
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [1024, 1, 1]],  # 60-loc4
  ]

# yolov7 head
head:
  [[50, 1, SPPCSPC, [512]], # 0

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [37, 1, Conv, [256, 1, 1]], # route backbone P4
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]], # 12

   [-1, 1, Conv, [128, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [24, 1, Conv, [128, 1, 1]], # route backbone P3
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [128, 1, 1]], # 24

   [-1, 1, MP, []],
   [-1, 1, Conv, [128, 1, 1]],
   [-3, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 2]],
   [[-1, -3, -17], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]], # 37

   [-1, 1, MP, []],
   [-1, 1, Conv, [256, 1, 1]],
   [-3, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 2]],
   [[-1, -3, -42], 1, Concat, [1]],

   [-1, 1, Conv, [512, 1, 1]],
   [-2, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]], # 50

   [-27, 1, RepConv, [256, 3, 1]], # 51
   [-15, 1, RepConv, [512, 3, 1]], # 52
   [-3, 1, RepConv, [1024, 3, 1]], # 53

   [[-3, -2, -1], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5)
  ]

# yolov7 head
head2:
  [[60, 1, SPPCSPC, [512]], # 0

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [46, 1, Conv, [256, 1, 1]], # route backbone P4
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]], # 12

   [-1, 1, Conv, [128, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [32, 1, Conv, [128, 1, 1]], # route backbone P3
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [128, 1, 1]], # 24

   [-1, 1, MP, []],
   [-1, 1, Conv, [128, 1, 1]],
   [-3, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 2]],
   [[-1, -3, -17], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]], # 37

   [-1, 1, MP, []],
   [-1, 1, Conv, [256, 1, 1]],
   [-3, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 2]],
   [[-1, -3, -42], 1, Concat, [1]],

   [-1, 1, Conv, [512, 1, 1]],
   [-2, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]], # 50

   [-27, 1, RepConv, [256, 3, 1]], # 51
   [-15, 1, RepConv, [512, 3, 1]], # 52
   [-3, 1, RepConv, [1024, 3, 1]], # 53

   [[-3, -2, -1], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5)
  ]
```

## cfg/dy-yolov7x-step2.yaml

```
# parameters
nc: 80  # number of classes
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple

# anchors
anchors:
  - [12,16, 19,36, 40,28]  # P3/8
  - [36,75, 76,55, 72,146]  # P4/16
  - [142,110, 192,243, 459,401]  # P5/32

# yolov7 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [40, 3, 1]],  # 0

   [-1, 1, Conv, [80, 3, 2]],  # 1-P1/2
   [-1, 1, Conv, [80, 3, 1]],

   [-1, 1, Conv, [160, 3, 2]],  # 3-P2/4
   [-1, 1, Conv, [64, 1, 1]],
   [-2, 1, Conv, [64, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [320, 1, 1]],  # 13-loc1

   [-1, 1, MP, []],
   [-1, 1, Conv, [160, 1, 1]],
   [-3, 1, Conv, [160, 1, 1]],
   [-1, 1, Conv, [160, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 18-P3/8
   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [640, 1, 1]],  # 28-loc2

   [-1, 1, MP, []],
   [-1, 1, Conv, [320, 1, 1]],
   [-3, 1, Conv, [320, 1, 1]],
   [-1, 1, Conv, [320, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 33-P4/16
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [1280, 1, 1]],  # 43-loc3

   [-1, 1, MP, []],
   [-1, 1, Conv, [640, 1, 1]],
   [-3, 1, Conv, [640, 1, 1]],
   [-1, 1, Conv, [640, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 48-P5/32
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [1280, 1, 1]],  # 58
  ]

n_first_layers: 59

b1_save: [2, 13, 28, 43, 58]
b2_save: [36, 52, 68]

keep_input: True

dynamic: True
router_ins: [2, 13, 28, 43, 58]
router_channels: [80, 320, 640, 1280, 1280]


dual_backbone:
   [[2, 1, CBLinear, [[80]]], # 0
   [13, 1, CBLinear, [[80, 160]]], # 1
   [28, 1, CBLinear, [[80, 160, 320]]], # 2
   [43, 1, CBLinear, [[80, 160, 320, 640]]], # 3
   [58, 1, CBLinear, [[80, 160, 320, 640, 1280]]], # 4

   [input, 1, Conv, [40, 3, 1]],  # 5

   [-1, 1, Conv, [80, 3, 2]],  # 6-P1/2
   [[-7, -6, -5, -4, -3, -1], 1, CBFuse, [[0, 0, 0, 0, 0]]], # 7
   [-1, 1, Conv, [80, 3, 1]],

   [-1, 1, Conv, [160, 3, 2]],  # 9-P2/4
   [[-9, -8, -7, -6, -1], 1, CBFuse, [[1, 1, 1, 1]]], # 10
   [-1, 1, Conv, [64, 1, 1]],
   [-2, 1, Conv, [64, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [320, 1, 1]],  # 20-loc1

   [-1, 1, MP, []],
   [-1, 1, Conv, [160, 1, 1]],
   [13, 1, Conv, [160, 1, 1]],
   [-1, 1, Conv, [160, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 25-P3/8
   [[-24, -23, -22, -1], 1, CBFuse, [[2, 2, 2]]], # 26
   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [640, 1, 1]],  # 36-loc2

   [-1, 1, MP, []],
   [-1, 1, Conv, [320, 1, 1]],
   [-3, 1, Conv, [320, 1, 1]],
   [-1, 1, Conv, [320, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 41-P4/16
   [[-39, -38, -1], 1, CBFuse, [[3, 3]]], # 42

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [1280, 1, 1]],  # 52-loc3

   [-1, 1, MP, []],
   [-1, 1, Conv, [640, 1, 1]],
   [-3, 1, Conv, [640, 1, 1]],
   [-1, 1, Conv, [640, 3, 2]],
   [[-1, -3], 1, Concat, [1]],  # 57-P5/32
   [[-54, -1], 1, CBFuse, [[4]]], # 58

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [1280, 1, 1]],  # 68
  ]

# yolov7 head
head:
  [[58, 1, SPPCSPC, [640]], # 0

   [-1, 1, Conv, [320, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [43, 1, Conv, [320, 1, 1]], # route backbone P4
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [320, 1, 1]], # 14

   [-1, 1, Conv, [160, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [28, 1, Conv, [160, 1, 1]], # route backbone P3
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [160, 1, 1]], # 28

   [-1, 1, MP, []],
   [-1, 1, Conv, [160, 1, 1]],
   [-3, 1, Conv, [160, 1, 1]],
   [-1, 1, Conv, [160, 3, 2]],
   [[-1, -3, -19], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [320, 1, 1]], # 43

   [-1, 1, MP, []],
   [-1, 1, Conv, [320, 1, 1]],
   [-3, 1, Conv, [320, 1, 1]],
   [-1, 1, Conv, [320, 3, 2]],
   [[-1, -3, -48], 1, Concat, [1]],

   [-1, 1, Conv, [512, 1, 1]],
   [-2, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [640, 1, 1]], # 58

   [-31, 1, Conv, [320, 3, 1]], # 59
   [-17, 1, Conv, [640, 3, 1]], # 60
   [-3, 1, Conv, [1280, 3, 1]], # 61

   [[-3, -2, -1], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5)
  ]

# yolov7 head
head2:
  [[68, 1, SPPCSPC, [640]], # 0

   [-1, 1, Conv, [320, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [52, 1, Conv, [320, 1, 1]], # route backbone P4
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [320, 1, 1]], # 14

   [-1, 1, Conv, [160, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [36, 1, Conv, [160, 1, 1]], # route backbone P3
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [160, 1, 1]], # 28

   [-1, 1, MP, []],
   [-1, 1, Conv, [160, 1, 1]],
   [-3, 1, Conv, [160, 1, 1]],
   [-1, 1, Conv, [160, 3, 2]],
   [[-1, -3, -19], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [320, 1, 1]], # 43

   [-1, 1, MP, []],
   [-1, 1, Conv, [320, 1, 1]],
   [-3, 1, Conv, [320, 1, 1]],
   [-1, 1, Conv, [320, 3, 2]],
   [[-1, -3, -48], 1, Concat, [1]],

   [-1, 1, Conv, [512, 1, 1]],
   [-2, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [[-1, -3, -5, -7, -8], 1, Concat, [1]],
   [-1, 1, Conv, [640, 1, 1]], # 58

   [-31, 1, Conv, [320, 3, 1]], # 59
   [-17, 1, Conv, [640, 3, 1]], # 60
   [-3, 1, Conv, [1280, 3, 1]], # 61

   [[-3, -2, -1], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5)
  ]
```

## cfg/dy-yolov7-w6-step1.yaml

```
# parameters
nc: 80  # number of classes
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple

# anchors
anchors:
  - [ 19,27,  44,40,  38,94 ]  # P3/8
  - [ 96,68,  86,152,  180,137 ]  # P4/16
  - [ 140,301,  303,264,  238,542 ]  # P5/32
  - [ 436,615,  739,380,  925,792 ]  # P6/64

# yolov7 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, ReOrg, []],  # 0
   [-1, 1, Conv, [64, 3, 1]],  # 1-P1/2

   [-1, 1, Conv, [128, 3, 2]],  # 2-P2/4
   [-1, 1, Conv, [64, 1, 1]],
   [-2, 1, Conv, [64, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [128, 1, 1]],  # 10-loc1

   [-1, 1, Conv, [256, 3, 2]],  # 11-P3/8
   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]],  # 19-loc2

   [-1, 1, Conv, [512, 3, 2]],  # 20-P4/16
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]],  # 28-loc3

   [-1, 1, Conv, [768, 3, 2]],  # 29-P5/32
   [-1, 1, Conv, [384, 1, 1]],
   [-2, 1, Conv, [384, 1, 1]],
   [-1, 1, Conv, [384, 3, 1]],
   [-1, 1, Conv, [384, 3, 1]],
   [-1, 1, Conv, [384, 3, 1]],
   [-1, 1, Conv, [384, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [768, 1, 1]],  # 37-loc4

   [-1, 1, Conv, [1024, 3, 2]],  # 38-P6/64
   [-1, 1, Conv, [512, 1, 1]],
   [-2, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [1024, 1, 1]],  # 46
  ]


n_first_layers: 47

b1_save: [0, 10, 19, 28, 37, 46]
b2_save: [25, 35, 45, 55]

# keep_input: True

dual_backbone:
  [[10, 1, CBLinear, [[128]]], # 0
   [19, 1, CBLinear, [[128, 256]]], # 1
   [28, 1, CBLinear, [[128, 256, 512]]], # 2
   [37, 1, CBLinear, [[128, 256, 512, 768]]], # 3
   [46, 1, CBLinear, [[128, 256, 512, 768, 1024]]], # 4

   [0, 1, Conv, [64, 3, 1]],  # 5-P1/2

   [-1, 1, Conv, [128, 3, 2]],  # 6-P2/4
   [[-7, -6, -5, -4, -3, -1], 1, CBFuse, [[0, 0, 0, 0, 0]]], # 7
   [-1, 1, Conv, [64, 1, 1]],
   [-2, 1, Conv, [64, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [128, 1, 1]],  # 15-loc1

   [-1, 1, Conv, [256, 3, 2]],  # 16-P3/8
   [[-16, -15, -14, -13, -1], 1, CBFuse, [[1, 1, 1, 1]]], # 17
   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]],  # 25-loc2

   [-1, 1, Conv, [512, 3, 2]],  # 26-P4/16
   [[-25, -24, -23, -1], 1, CBFuse, [[2, 2, 2]]], # 27
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]],  # 35-loc3

   [-1, 1, Conv, [768, 3, 2]],  # 36-P5/32
   [[-34, -33, -1], 1, CBFuse, [[3, 3]]], # 37
   [-1, 1, Conv, [384, 1, 1]],
   [-2, 1, Conv, [384, 1, 1]],
   [-1, 1, Conv, [384, 3, 1]],
   [-1, 1, Conv, [384, 3, 1]],
   [-1, 1, Conv, [384, 3, 1]],
   [-1, 1, Conv, [384, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [768, 1, 1]],  # 45-loc4

   [-1, 1, Conv, [1024, 3, 2]],  # 46-P6/64
   [[-43, -1], 1, CBFuse, [[4]]], # 47
   [-1, 1, Conv, [512, 1, 1]],
   [-2, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [-1, 1, Conv, [512, 3, 1]],
   [[-1, -3, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [1024, 1, 1]],  # 55
  ]

# yolov7 head
head:
  [[46, 1, SPPCSPC, [512]], # 0

   [-1, 1, Conv, [384, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [37, 1, Conv, [384, 1, 1]], # route backbone P5
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [384, 1, 1]],
   [-2, 1, Conv, [384, 1, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [384, 1, 1]], # 12

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [28, 1, Conv, [256, 1, 1]], # route backbone P4
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]], # 24

   [-1, 1, Conv, [128, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [19, 1, Conv, [128, 1, 1]], # route backbone P3
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [128, 1, 1]], # 36

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, -14], 1, Concat, [1]],  # cat

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]], # 46

   [-1, 1, Conv, [384, 3, 2]],
   [[-1, -36], 1, Concat, [1]],  # cat

   [-1, 1, Conv, [384, 1, 1]],
   [-2, 1, Conv, [384, 1, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [384, 1, 1]], # 56

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, -58], 1, Concat, [1]],  # cat

   [-1, 1, Conv, [512, 1, 1]],
   [-2, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]], # 66

   [-31, 1, Conv, [256, 3, 1]],
   [-22, 1, Conv, [512, 3, 1]],
   [-13, 1, Conv, [768, 3, 1]],
   [-4, 1, Conv, [1024, 3, 1]],

   [[-4, -3, -2, -1], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5, P6)
  ]

# yolov7 head
head2:
  [[55, 1, SPPCSPC, [512]], # 0

   [-1, 1, Conv, [384, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [45, 1, Conv, [384, 1, 1]], # route backbone P5
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [384, 1, 1]],
   [-2, 1, Conv, [384, 1, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [384, 1, 1]], # 12

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [35, 1, Conv, [256, 1, 1]], # route backbone P4
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]], # 24

   [-1, 1, Conv, [128, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [25, 1, Conv, [128, 1, 1]], # route backbone P3
   [[-1, -2], 1, Concat, [1]],

   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [128, 1, 1]], # 36

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, -14], 1, Concat, [1]],  # cat

   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]], # 46

   [-1, 1, Conv, [384, 3, 2]],
   [[-1, -36], 1, Concat, [1]],  # cat

   [-1, 1, Conv, [384, 1, 1]],
   [-2, 1, Conv, [384, 1, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [-1, 1, Conv, [192, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [384, 1, 1]], # 56

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, -58], 1, Concat, [1]],  # cat

   [-1, 1, Conv, [512, 1, 1]],
   [-2, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]], # 66

   [-31, 1, Conv, [256, 3, 1]],
   [-22, 1, Conv, [512, 3, 1]],
   [-13, 1, Conv, [768, 3, 1]],
   [-4, 1, Conv, [1024, 3, 1]],

   [[-4, -3, -2, -1], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5, P6)
  ]
```

## data/coco.yaml

```
# COCO 2017 dataset http://cocodataset.org

# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: ./coco/train2017.txt  # 118287 images
val: ./coco/val2017.txt  # 5000 images
test: ./coco/test-dev2017.txt  # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794

# number of classes
nc: 80

# class names
names: [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
```

