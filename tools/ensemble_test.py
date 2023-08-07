import torchvision.ops.boxes

import _init_path
import argparse
import datetime
import glob
import os
import re
from pathlib import Path
import tqdm
import pickle
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from collections import defaultdict

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file, merge_new_config
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from tools.eval_utils.eval_utils import statistics_info

from ensemble_utils.ensemble_model import Ensemble
from ensemble_utils.ensemble_setting import ensemble_cfg


# class Ensemble(torch.nn.ModuleList):
#     def __init__(self, cfg_list, ckpt_list, dataset, logger, dist_test=False):
#         super(Ensemble, self).__init__()
#         assert len(cfg_list) == len(ckpt_list), "Number dont match between cfg_list and ckpt_list"
#         self.cfg_list = cfg_list
#         self.ckpt_list = ckpt_list
#         self.dataset = dataset
#         self.logger = logger
#         self.dist_test = dist_test
#         self.load_model()
#
#     def load_model(self):
#         for cfg_path, ckpt_path in zip(self.cfg_list, self.ckpt_list):
#             model_cfg = self.load_model_cfg(cfg_path)
#
#             model = build_network(model_cfg=model_cfg, num_class=len(self.dataset.class_names), dataset=self.dataset)
#             with torch.no_grad():
#                 model.load_params_from_file(filename=ckpt_path,
#                                             logger=self.logger,
#                                             to_cpu=self.dist_test,
#                                             pre_trained_path=None)
#                 model.cuda()
#             self.append(model)
#
#         print("load {} model success".format(len(self)))
#
#     def load_model_cfg(self, cfg_path):
#         import yaml
#         from easydict import EasyDict
#
#         config = EasyDict()
#         with open(cfg_path, 'r') as f:
#             new_config = yaml.safe_load(f)
#             merge_new_config(config=config, new_config=new_config)
#
#         return config.MODEL
#
#     def xywh2xyxy(self, x):
#         # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
#         y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#         y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
#         y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
#         y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
#         y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
#         return y
#
#     def xyxy2xywh(self, x):
#         y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#         y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
#         y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
#         y[:, 2] = x[:, 2] - x[:, 0]        # width
#         y[:, 3] = x[:, 3] - x[:, 1]        # height
#         return y
#
#     def statistic_recall_info(self,  ret_dict_list):
#         new_ret_dict = defaultdict(int)     # init with zero
#         key_list = ret_dict_list[0].keys()  # each ret_dict should be use the same keys
#         model_nums = len(ret_dict_list)
#         for key in key_list:
#             for ret_dict in ret_dict_list:
#                 new_ret_dict[key] += ret_dict[key]
#             new_ret_dict[key] = round(new_ret_dict[key] / model_nums)
#         return new_ret_dict
#
#     def non_max_suppression(self, ensemble_dict_list, cond_thres=0.2, iou_thresh=0.3):
#         new_pred_dicts = []
#         for ensemble_dict in ensemble_dict_list:
#             boxes = ensemble_dict['pred_boxes']
#             scores = ensemble_dict['pred_scores']
#             labels = ensemble_dict['pred_labels']
#
#             info = torch.cat([boxes, scores[:, None], labels[:, None]], dim=-1)
#
#             cond_mask = info[:, -2] > cond_thres
#             info = info[cond_mask]
#
#             # max_w, max_l, max_h, min_lwh = 5.0, 3.0, 3.0, 0.05
#             # size_mask = torch.cat(
#             #     [info[:, 3:4] < max_w, info[:, 4:5] < max_l, info[:, 5:6] < max_h, info[:, 3:6] > min_lwh], dim=-1
#             # )
#             # size_mask = size_mask.all(-1)
#             # info = info[size_mask]
#
#             sort_index = info[:, -2].argsort(descending=True)
#             info = info[sort_index]
#
#             box_xywl = info[:, [0, 1, 3, 4]]        # xywl
#             box_xyxy = self.xywh2xyxy(box_xywl)     # xyxy
#
#             # SHIFT_NUM = 100
#             info[:, [0, 1, 3, 4]] = box_xyxy + 100.
#
#             # merge-nms
#             res = []
#             labels = info[:, -1]
#             for c in labels.unique():
#                 boxes = info[labels == c][:, [0, 1, 3, 4]]
#                 dc = info[labels == c]
#                 while len(dc):
#                     if len(dc) == 1:
#                         res.append(dc)
#                         break
#                     i = torchvision.ops.box_iou(boxes[:1], boxes).squeeze(0) > iou_thresh  # i = True/False的集合
#                     scores = dc[i, -2:-1]  # score
#                     boxes[0, :4] = (scores * boxes[i, :4]).sum(0) / scores.sum()  # 重叠框位置信息求解平均值 (k, 4)
#                     dc[:1][:, [0, 1, 3, 4]] = boxes[0, :4]
#                     res.append(dc[:1])
#                     dc = dc[i == 0]
#                     boxes = boxes[i == 0]
#
#             info = torch.cat(res, dim=0)
#
#             box_xyxy = info[:, [0, 1, 3, 4]] - 100.
#             info[:, [0, 1, 3, 4]] = self.xyxy2xywh(box_xyxy)
#
#             # keep the same type with origin
#             pred_dict = dict()
#             pred_dict['pred_boxes'] = info[:, :9]
#             pred_dict['pred_scores'] = info[:, -2]
#             pred_dict['pred_labels'] = info[:, -1].type(torch.int64)
#             new_pred_dicts.append(pred_dict)
#
#         return new_pred_dicts
#
#     def nms(self, ensemble_dict_list, cond_thres=0.2, nms_thresh=0.3):
#         new_pred_dicts = []
#         for ensemble_dict in ensemble_dict_list:
#             boxes = ensemble_dict['pred_boxes']
#             scores = ensemble_dict['pred_scores']
#             labels = ensemble_dict['pred_labels']
#
#             info = torch.cat([boxes, scores[:, None], labels[:, None]], dim=-1)
#
#             cond_mask = info[:, -2] > cond_thres
#             info = info[cond_mask]
#
#             # max_w, max_l, max_h, min_lwh = 5.0, 3.0, 3.0, 0.05
#             # size_mask = torch.cat(
#             #     [info[:, 3:4] < max_w, info[:, 4:5] < max_l, info[:, 5:6] < max_h, info[:, 3:6] > min_lwh], dim=-1
#             # )
#             # size_mask = size_mask.all(-1)
#             # info = info[size_mask]
#
#             sort_index = info[:, -2].argsort(descending=True)
#             info = info[sort_index]
#
#             box_xywl = info[:, [0, 1, 3, 4]]        # xywl
#             box_xyxy = self.xywh2xyxy(box_xywl)     # xyxy
#
#             # use hard batch nms
#             nms_index = torchvision.ops.boxes.batched_nms(box_xyxy, info[:, -2], info[:, -1], nms_thresh)
#             info = info[nms_index]
#
#             # keep the same type with origin
#             pred_dict = dict()
#             pred_dict['pred_boxes'] = info[:, :9].type(boxes.dtype)
#             pred_dict['pred_scores'] = info[:, -2].type(scores.dtype)
#             pred_dict['pred_labels'] = info[:, -1].type(labels.dtype)
#             new_pred_dicts.append(pred_dict)
#
#         return new_pred_dicts
#
#     def post_process(self, ensemble_pred_dict, ret_dict_list):
#         # 1) 对多模型的recall进行平均处理
#         ret_dict = self.statistic_recall_info(ret_dict_list)
#
#         # 2) 对多模型的预测结果使用nms来进行评估(2d / 3d)
#         pred_dicts = self.non_max_suppression(ensemble_pred_dict)
#
#         return pred_dicts, ret_dict
#
#     def forward(self, batch_dict):
#         pred_dict_list, ret_dict_list = [], []
#
#         # save the result from different model
#         for model in self:
#             pred_dict, ret_dict = model(batch_dict)
#             pred_dict_list.append(pred_dict)
#             ret_dict_list.append(ret_dict)
#
#         # ensemble the result from different model
#         batch_size = batch_dict['batch_size']
#         ensemble_pred_dict = []
#         for frame in range(batch_size):        # 遍历点云帧
#             ensemble_dict = defaultdict(list)
#             # 将每个模型的对应帧场景预测结果进行存储
#             for pred_dict in pred_dict_list:
#                 ensemble_dict['pred_boxes'].append(pred_dict[frame]['pred_boxes'])
#                 ensemble_dict['pred_scores'].append(pred_dict[frame]['pred_scores'])
#                 ensemble_dict['pred_labels'].append(pred_dict[frame]['pred_labels'])
#                 # if 'pred_ious' in pred_dict[frame]:       # not use
#                 #     ensemble_dict['pred_ious'].append(pred_dict[frame]['pred_ious'])
#             # 将多模型的预测结果进行拼接操作
#             ensemble_dict['pred_boxes'] = torch.cat(ensemble_dict['pred_boxes'], dim=0)
#             ensemble_dict['pred_scores'] = torch.cat(ensemble_dict['pred_scores'], dim=0)
#             ensemble_dict['pred_labels'] = torch.cat(ensemble_dict['pred_labels'], dim=0)
#             ensemble_pred_dict.append(ensemble_dict)
#
#         # return nms result
#         pred_dicts, ret_dict = self.post_process(ensemble_pred_dict, ret_dict_list)
#         return pred_dicts, ret_dict

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_list', type=str, default=None, help='specify the config for training')
    parser.add_argument('--ckpt_list', type=str, default=None, help='checkpoint to start from')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt_name', type=str, default=None, help='ensemble ckpt load for experiment')
    parser.add_argument('--model_choose', type=str, default='weig', help='[weight, attn, swim] to choose')
    parser.add_argument('--id', type=str, default='6', help='choose gpu id')

    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')

    args = parser.parse_args()

    args.cfg_list = ensemble_cfg['cfg_list']
    args.ckpt_list = ensemble_cfg['ckpt_list']

    # 除了模型的配置外其余设置均使用第一个yaml文件来进行配置
    cfg_from_yaml_file(args.cfg_list[0], cfg)
    # cfg.TAG = Path(args.cfg_file).stem
    # cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.id
    return args, cfg


def eval_ckpt(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        # if getattr(args, 'infer_time', False):
        #     inference_time = time.time() - start_time
        #     infer_time_meter.update(inference_time * 1000)
        #     # use ms to measure inference time
        #     disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)   # add current batch point recall infos
        annos = dataset.generate_prediction_dicts(      # 更多的也只是进行了字典信息的重新组建
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: {:.4f} second [{:.4f} / {:d}]).'.format(
        sec_per_example, (time.time() - start_time), len(dataloader.dataset)
    ))

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    # 通过gt点数和各阈值预测点数来计算召回率
    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    # from det pred get metrics
    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        # eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


def main():
    args, cfg = parse_config()

    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / 'ensemble_model' / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'
    eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    # 普通配置正常按第一个来进行测试
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    # ensemble model
    ckpt_list = args.ckpt_list if args.ckpt_list is not None else output_dir / 'ckpt'
    cfg_list = args.cfg_list
    model_choose = args.model_choose
    model = Ensemble(cfg_list, ckpt_list, test_set, logger, dist_test, model_name=model_choose)
    # model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    # load model
    ckpt_name = args.ckpt_name
    merge_ckpt_path = Path(os.getcwd()).parent / 'output/ensemble_model/default/ckpt' / ckpt_name \
        if ckpt_name is not None else None
    use_merge_ckpt_ = ckpt_name is not None and merge_ckpt_path.exists()
    if use_merge_ckpt_:
        merge_ckpt = torch.load(merge_ckpt_path)
        model.mergenet.load_state_dict(merge_ckpt)
    print('********' * 10)
    print('load merge net success') if use_merge_ckpt_ else print('not use merge net ckpt')
    print('********' * 10)

    # model.ensemble_train = False  # 验证模式
    model.mergenet.eval()
    with torch.no_grad():
        eval_ckpt(cfg, args, model, dataloader=test_loader, epoch_id='Ensemble', logger=logger,
                  dist_test=dist_test, result_dir=eval_output_dir)


if __name__ == '__main__':
    main()




