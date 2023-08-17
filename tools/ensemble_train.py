import logging

import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import os

import torch
import tqdm
import time
import glob
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils

import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter

from pcdet.models import load_data_to_gpu
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model

from ensemble_utils.ensemble_model import Ensemble
from ensemble_utils.ensemble_setting import ensemble_cfg

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_list', type=str, default=None, help='specify the config for training')
    parser.add_argument('--ckpt_list', type=str, default=None, help='checkpoint to start from')

    parser.add_argument('--batch_size', type=int, default=4, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=2, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--model_choose', type=str, default='attn', help='[weig, attn, swim] to choose')
    parser.add_argument('--ckpt_name', type=str, default=None, help='ensemble ckpt load for experiment')
    parser.add_argument('--save_name', type=str, default=None, help='ensemble ckpt save for experiment')
    parser.add_argument('--id', type=str, default='0', help='choose gpu id')

    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=3, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False,
                        help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=10, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', action='store_true', help='use mix precision training')

    args = parser.parse_args()

    args.cfg_list = ensemble_cfg['cfg_list']
    args.ckpt_list = ensemble_cfg['ckpt_list']

    # 除了模型的配置外其余设置均使用第一个yaml文件来进行配置
    cfg_from_yaml_file(args.cfg_list[0], cfg)
    # cfg.TAG = Path(args.cfg_file).stem
    # cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    # if args.set_cfgs is not None:
    #     cfg_from_list(args.set_cfgs, cfg)
    np.random.seed(1024)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.id
    return args, cfg


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False,
                    use_logger_to_record=False, logger=None, logger_iter_interval=50, cur_epoch=None,
                    total_epochs=None, ckpt_save_dir=None, ckpt_save_time_interval=300, show_gpu_stat=False,
                    use_amp=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    ckpt_save_cnt = 1
    start_it = accumulated_iter % total_it_each_epoch

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=optim_cfg.get('LOSS_SCALE_FP16', 2.0 ** 16))

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train',
                         dynamic_ncols=True, bar_format="{l_bar}{bar:20}{r_bar}")
        # pbar = tqdm(total=total_it_each_epoch, desc='train')
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()
        losses_m = common_utils.AverageMeter()

    end = time.time()
    for cur_it in range(start_it, total_it_each_epoch):
        try:
            batch_dict = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch_dict = next(dataloader_iter)
            print('new iters')

        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter, cur_epoch)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        optimizer.zero_grad()

        load_data_to_gpu(batch_dict)
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = model(batch_dict)     # 重新整合结果构建损失

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)       # 梯度裁剪
        scaler.step(optimizer)
        scaler.update()

        accumulated_iter += 1

        cur_forward_time = time.time() - data_timer
        cur_batch_time = time.time() - end
        end = time.time()

        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            disp_dict = {}
            batch_size = batch_dict.get('batch_size', None)

            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            losses_m.update(loss.item(), batch_size)

            disp_dict.update({
                'loss': loss.item(), 'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})',
                'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })

            if use_logger_to_record:
                if accumulated_iter % logger_iter_interval == 0 or cur_it == start_it or cur_it + 1 == total_it_each_epoch:
                    trained_time_past_all = tbar.format_dict['elapsed']
                    second_each_iter = pbar.format_dict['elapsed'] / max(cur_it - start_it + 1, 1.0)

                    trained_time_each_epoch = pbar.format_dict['elapsed']
                    remaining_second_each_epoch = second_each_iter * (total_it_each_epoch - cur_it)
                    remaining_second_all = second_each_iter * (
                                (total_epochs - cur_epoch) * total_it_each_epoch - cur_it)

                    logger.info(
                        'Train: {:>4d}/{} ({:>3.0f}%) [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        'LR: {lr:.3e}  '
                        f'Time cost: {tbar.format_interval(trained_time_each_epoch)}/{tbar.format_interval(remaining_second_each_epoch)} '
                        f'[{tbar.format_interval(trained_time_past_all)}/{tbar.format_interval(remaining_second_all)}]  '
                        'Acc_iter {acc_iter:<10d}  '
                        'Data time: {data_time.val:.2f}({data_time.avg:.2f})  '
                        'Forward time: {forward_time.val:.2f}({forward_time.avg:.2f})  '
                        'Batch time: {batch_time.val:.2f}({batch_time.avg:.2f})'.format(
                            cur_epoch + 1, total_epochs, 100. * (cur_epoch + 1) / total_epochs,
                            cur_it, total_it_each_epoch, 100. * cur_it / total_it_each_epoch,
                            loss=losses_m,
                            lr=cur_lr,
                            acc_iter=accumulated_iter,
                            data_time=data_time,
                            forward_time=forward_time,
                            batch_time=batch_time
                        )
                    )

                    if show_gpu_stat and accumulated_iter % (3 * logger_iter_interval) == 0:
                        # To show the GPU utilization, please install gpustat through "pip install gpustat"
                        gpu_info = os.popen('gpustat').read()
                        logger.info(gpu_info)
            else:
                pbar.update()
                pbar.set_postfix(dict(total_it=accumulated_iter))
                tbar.set_postfix(disp_dict)
                # tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                # for key, val in tb_dict.items():
                #     tb_log.add_scalar('train/' + key, val, accumulated_iter)

            # save intermediate ckpt every {ckpt_save_time_interval} seconds
            # time_past_this_epoch = pbar.format_dict['elapsed']
            # if time_past_this_epoch // ckpt_save_time_interval >= ckpt_save_cnt:
            #     ckpt_name = ckpt_save_dir / 'latest_model'
            #     save_checkpoint(
            #         checkpoint_state(model, optimizer, cur_epoch, accumulated_iter), filename=ckpt_name,
            #     )
            #     logger.info(f'Save latest model to {ckpt_name}')
            #     ckpt_save_cnt += 1

        # log to console and tensorboard
        # if rank == 0:
        #     disp_dict = {}
        #     batch_size = batch_dict.get('batch_size', None)
        #
        #     data_time.update(avg_data_time)
        #     forward_time.update(avg_forward_time)
        #     batch_time.update(avg_batch_time)
        #     losses_m.update(loss.item(), batch_size)
        #
        #     disp_dict.update({
        #         'loss': loss.item(),
        #         'lr': cur_lr,
        #         'd_time': data_time.val,
        #         'f_time': forward_time.val,
        #         'b_time': batch_time.val
        #     })
        #     postfix_str = ", ".join([f"{k}={v:.4f}" for k, v in disp_dict.items()])
        #
        #     pbar.update()
        #     pbar.set_postfix(dict(total_it=accumulated_iter))
        #     tbar.set_postfix(disp_dict)
        #     # tbar.set_postfix_str(postfix_str)

    if rank == 0:
        pbar.close()

    return accumulated_iter


def train_ensemble(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                    start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, ckpt_save_name, train_sampler=None,
                    lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                    merge_all_iters_to_one_epoch=False, use_amp=False,
                    use_logger_to_record=False, logger=None, logger_iter_interval=None,
                   ckpt_save_time_interval=None, show_gpu_stat=False, cfg=None):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0),
                     bar_format="{l_bar}{bar:10}{r_bar}{postfix}") as tbar:
    # with tqdm(range(total_epochs), desc='epochs') as tbar:
        total_it_each_epoch = len(train_loader)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            # if train_sampler is not None:
            #     train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter,
                optim_cfg=optim_cfg,
                rank=rank,
                tbar=tbar,
                tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                cur_epoch=cur_epoch,
                total_epochs=total_epochs,
                use_logger_to_record=use_logger_to_record,
                logger=logger,
                logger_iter_interval=logger_iter_interval,
                ckpt_save_dir=ckpt_save_dir,
                ckpt_save_time_interval=ckpt_save_time_interval,
                show_gpu_stat=show_gpu_stat,
                use_amp=use_amp
            )

            # After training, Save the model, default name is merge_net.pth
            save_name = ckpt_save_name if ckpt_save_name is not None else 'merge_net.pth'
            ckpt_save_path = ckpt_save_dir / save_name
            torch.save(model.mergenet.state_dict(), ckpt_save_path)
            logger.info('\n ********* save model at epoch{} in {}********** \n'.format(cur_epoch, ckpt_save_path))


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True
    # print('*'*20, 'dist_train: ', dist_train, '*'*20)

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    # output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir = cfg.ROOT_DIR / 'output' / 'ensemble_model' / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        logger.info('Training with a single process')

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    # if cfg.LOCAL_RANK == 0:
    #     os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    logger.info("----------- Create dataloader & network & optimizer -----------")
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )

    # ensemble model
    ckpt_list, cfg_list = args.ckpt_list, args.cfg_list
    model_choose = args.model_choose
    model = Ensemble(cfg_list, ckpt_list, train_set, logger, dist_train, model_name=model_choose)
    # model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model.ensemble_train = True     # 训练模式
    model.cuda()

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    ckpt_name = args.ckpt_name
    merge_ckpt_path = Path(os.getcwd()).parent / 'output/ensemble_model/default/ckpt' / ckpt_name \
        if ckpt_name is not None else None
    use_merge_ckpt_ = ckpt_name is not None and merge_ckpt_path.exists()
    if use_merge_ckpt_:
        merge_ckpt = torch.load(merge_ckpt_path)
        model.mergenet.load_state_dict(merge_ckpt)
    logger.info('********' * 10)
    logger.info('load merge net success') if use_merge_ckpt_ else logger.info('not use merge net ckpt')
    logger.info('********' * 10)

    # if args.pretrained_model is not None:
    #     model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    # model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    model.mergenet.train()
    if dist_train:
        model.mergenet = nn.parallel.DistributedDataParallel(
            model.mergenet, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model.mergenet)

    # 构建优化器和学习调度器
    cfg.OPTIMIZATION.LR = 0.01
    # cfg.OPTIMIZATION.OPTIMIZER = 'sgd'  # adam / sgd / adam_onecycle
    optimizer = build_optimizer(model.mergenet, cfg.OPTIMIZATION)   # 只需要优化指定部分

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s**********************' % (args.extra_tag))
    train_ensemble(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        ckpt_save_name=args.save_name,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        logger=logger,
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record,
        show_gpu_stat=not args.wo_gpu_stat,
        use_amp=args.use_amp,
        cfg=cfg
    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()
    logger.info('**********************End training %s**********************' % (args.extra_tag))

    logger.info('**********************Start evaluation %s**********************' % (args.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    # test_set, test_loader, sampler = build_dataloader(
    #     dataset_cfg=cfg.DATA_CONFIG,
    #     class_names=cfg.CLASS_NAMES,
    #     batch_size=args.batch_size,
    #     dist=dist_train, workers=args.workers, logger=logger, training=False
    # )

    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval,
                           0)  # Only evaluate the last args.num_epochs_to_eval epochs

    # model.ensemble_train = False     # 训练模式
    model.mergenet.eval()       # 验证模式
    with torch.no_grad():
        from ensemble_test import eval_ckpt
        eval_ckpt(cfg, args, model, dataloader=test_loader, epoch_id='Ensemble', logger=logger,
                  dist_test=dist_train, result_dir=eval_output_dir)
    # repeat_eval_ckpt(
    #     model.module if dist_train else model,
    #     test_loader, args, eval_output_dir, logger, ckpt_dir,
    #     dist_test=dist_train
    # )
    logger.info('**********************End evaluation %s**********************' % (args.extra_tag))


if __name__ == '__main__':
    main()
