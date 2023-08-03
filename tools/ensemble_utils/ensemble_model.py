import os.path
import pickle

import numpy as np
import torch
import torch.nn as nn
import torchvision

from einops.layers.torch import Rearrange
from collections import defaultdict
from pathlib import Path

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file, merge_new_config
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from tools.eval_utils.eval_utils import statistics_info

from tools.ensemble_utils.ensemble_tools import xywh2xyxy, xyxy2xywh, EnsembleCoder
from tools.ensemble_utils.ensemble_setting import ensemble_cfg
from pcdet.utils import box_utils
from tools.ensemble_utils import loss_utils
from tools.ensemble_utils import merge_model
import torch.nn.functional as F


class Ensemble(nn.Module):
    def __init__(self, cfg_list, ckpt_list, dataset, logger, dist_test=False, loss='mse', model_name='swim'):
        super(Ensemble, self).__init__()
        self.ensemble_train = True

        # init merge net
        self.box_size = 7
        self.cfg = ensemble_cfg
        self.max_merge_nums = self.cfg['MAX_MERGE_NUMS']  # hyperparameter
        print('max merge nums = {}'.format(self.max_merge_nums))

        assert model_name in ['swim', 'attn', 'weig'], "No such {} model to choose.".format(model_name)
        if model_name == 'swim':
            self.mergenet = merge_model.SwimMergeNet(max_c=self.max_merge_nums, depth=3, init_values=True, reg_token=True)
        elif model_name == 'attn':
            self.mergenet = merge_model.AttentionMergeNet(max_c=self.max_merge_nums, depth=3)
        elif model_name == 'weig':
            self.mergenet = merge_model.WeightMergeNet(max_c=self.max_merge_nums)
        self.mergenet.cuda()

        # load mutil model
        assert len(cfg_list) == len(ckpt_list), "Number dont match between cfg_list and ckpt_list"
        self.cfg_list = cfg_list
        self.ckpt_list = ckpt_list
        self.dataset = dataset
        self.logger = logger
        self.dist_test = dist_test
        self.model_list = nn.ModuleList()
        self.load_model()

        # coder
        use_boxcoder = False
        self.boxcoder = EnsembleCoder(code_size=self.cfg['BOX_SIZE']) if use_boxcoder else None

        # loss init
        box_weight = self.cfg['BOX_WEIGHT']
        self.loss_compute = loss_utils.WeightedMSELoss(code_weights=box_weight) if loss == 'mse' \
            else loss_utils.WeightedSmoothL1Loss(code_weights=box_weight)

    def load_model(self):
        # 加载多个3d检测模型
        for cfg_path, ckpt_path in zip(self.cfg_list, self.ckpt_list):
            model_cfg = self.load_model_cfg(cfg_path)

            model = build_network(model_cfg=model_cfg, num_class=len(self.dataset.class_names), dataset=self.dataset)
            with torch.no_grad():
                model.load_params_from_file(filename=ckpt_path,
                                            logger=self.logger,
                                            to_cpu=self.dist_test,
                                            pre_trained_path=None)
                model.cuda()

            # freeze model weights
            for name, parameter in model.named_parameters():
                parameter.requires_grad = False

            self.model_list.append(model)

        self.model_list.eval()     # 集成的模型全部设置为验证模式

        # 加载融合模型
        # ckpt_name = 'merge_net.pth'
        # merge_ckpt_path = Path(os.getcwd()).parent / 'output/ensemble_model/default/ckpt' / ckpt_name
        # use_merge_ckpt_ = merge_ckpt_path.exists() and self.use_merge_ckpt
        # if use_merge_ckpt_:
        #     merge_ckpt = torch.load(merge_ckpt_path)
        #     self.mergenet.load_state_dict(merge_ckpt)
        #
        # print('********' * 10)
        # print("load {} model success".format(len(self.model_list)))
        # print('load merge net success') if use_merge_ckpt_ else None
        # print('********' * 10)

    def load_model_cfg(self, cfg_path):
        import yaml
        from easydict import EasyDict

        config = EasyDict()
        with open(cfg_path, 'r') as f:
            new_config = yaml.safe_load(f)
            merge_new_config(config=config, new_config=new_config)

        return config.MODEL

    def statistic_recall_info(self,  ret_dict_list):
        new_ret_dict = defaultdict(int)     # init with zero
        key_list = ret_dict_list[0].keys()  # each ret_dict should be use the same keys
        model_nums = len(ret_dict_list)
        for key in key_list:
            for ret_dict in ret_dict_list:
                new_ret_dict[key] += ret_dict[key]
            new_ret_dict[key] = round(new_ret_dict[key] / model_nums)
        return new_ret_dict

    def non_max_suppression(self, ensemble_dict_list, cond_thres=0.2, iou_thresh=0.3):
        new_pred_dicts = []
        for ensemble_dict in ensemble_dict_list:
            # concat into tensor
            boxes = ensemble_dict['pred_boxes']
            scores = ensemble_dict['pred_scores']
            labels = ensemble_dict['pred_labels']
            info = torch.cat([boxes, scores[:, None], labels[:, None]], dim=-1)

            # filter the low confindece
            cond_mask = info[:, -2] > cond_thres
            info = info[cond_mask]

            # Sort from largest to smallest
            sort_index = info[:, -2].argsort(descending=True)
            info = info[sort_index]

            # collect the close boxes
            res = []
            labels = info[:, -1]
            for c in labels.unique():
                boxes = info[labels == c][:, :7]    # has been sorted
                dc = info[labels == c]
                while len(dc):
                    if len(dc) <= 1:
                        res.append(dc)
                        break

                    # get the match thresh
                    iou = box_utils.boxes3d_nearest_bev_iou(boxes[:1], boxes).view(-1)
                    iou_mask = (iou > iou_thresh)
                    boxes_overlap = boxes[iou_mask]  # (k2, 11) or (0, 11)

                    # get the max iou box (top 1)
                    max_iou_index = iou.argsort(descending=True)[:1]
                    boxes_max = boxes[max_iou_index]

                    # get input box
                    count = boxes_overlap.shape[0]      # overlap boxes numbers
                    input_boxes = torch.zeros([self.max_merge_nums, self.box_size], device=info.device)  # (k, d)
                    if count > 1:
                        input_boxes[:count] = boxes_overlap[:self.max_merge_nums]
                    else:
                        input_boxes[:1] = boxes_max

                    # get merge box with encoder and decoder
                    if getattr(self, 'boxcoder'):
                        encode_boxes = self.boxcoder.encode_torch(input_boxes)
                        merge_box = self.mergenet(encode_boxes)  # (1, d)
                        output_boxes = self.boxcoder.decode_torch(merge_box)
                    else:
                        merge_box = self.mergenet(input_boxes)
                        output_boxes = merge_box
                    # clamp and save (the clamp operate will borken the gradient)
                    # output_boxes[:, 3:6] = torch.clamp(output_boxes[:, 3:6], min=1e-5)

                    dc[:1][:, :7] = output_boxes
                    res.append(dc[:1])

                    # filter the match iou sample
                    dc = dc[iou_mask == 0]
                    boxes = boxes[iou_mask == 0]

            info = torch.cat(res, dim=0)

            # Avoid setting the box size < 0, boxed size in 345 and keep gradiant
            min_boxes = info[:, 3:6].min(dim=0)[0] - 1e-1
            min_boxes = min_boxes.repeat(info.shape[0], 1)
            mask = info[:, 3:6] <= 0
            info[:, 3:6][mask] -= min_boxes[mask]

            # eval the result
            assert (info[:, 3:6] > 0).all(), "boxes size must be > 0"
            new_pred_dicts.append(info)

        return new_pred_dicts

    def ensemble_pred_result(self, pred_dict_list, batch_size):
        # ensemble the result from different model
        ensemble_pred_dict = []
        for frame in range(batch_size):  # 遍历点云帧
            ensemble_dict = defaultdict(list)
            # 将每个模型的对应帧场景预测结果进行存储
            for pred_dict in pred_dict_list:
                ensemble_dict['pred_boxes'].append(pred_dict[frame]['pred_boxes'])
                ensemble_dict['pred_scores'].append(pred_dict[frame]['pred_scores'])
                ensemble_dict['pred_labels'].append(pred_dict[frame]['pred_labels'])
                # if 'pred_ious' in pred_dict[frame]:       # not use
                #     ensemble_dict['pred_ious'].append(pred_dict[frame]['pred_ious'])
            # 将多模型的预测结果进行拼接操作
            ensemble_dict['pred_boxes'] = torch.cat(ensemble_dict['pred_boxes'], dim=0)
            ensemble_dict['pred_scores'] = torch.cat(ensemble_dict['pred_scores'], dim=0)
            ensemble_dict['pred_labels'] = torch.cat(ensemble_dict['pred_labels'], dim=0)
            ensemble_pred_dict.append(ensemble_dict)
        return ensemble_pred_dict

    def forward(self, batch_dict):
        # ensemble each model result
        pred_dict_list, ret_dict_list = [], []
        for model in self.model_list:
            pred_dict, ret_dict = model(batch_dict)
            pred_dict_list.append(pred_dict)
            ret_dict_list.append(ret_dict)
        ensemble_pred_dict = self.ensemble_pred_result(pred_dict_list, batch_size=batch_dict['batch_size'])

        if self.mergenet.training:
            merge_loss = self.get_loss(ensemble_pred_dict, batch_dict['gt_boxes'])   # 重新对整合的内容构造损失
            return merge_loss
        else:
            pred_dicts, ret_dict = self.post_process(ensemble_pred_dict, ret_dict_list)
            return pred_dicts, ret_dict


    def get_loss(self, ensemble_pred_dict, gt_dicts, iou_thresh=0.1):
        """
        params:
            pred_infos: [boxes, scores, labels]
            gt_dicts:   [boxes, labels]
        """
        pred_infos = self.non_max_suppression(ensemble_pred_dict)
        batch_size = len(pred_infos)
        reg_batch_loss = 0
        for i in range(batch_size):
            gt_info = gt_dicts[i]       # (N, 10)
            pred_info = pred_infos[i]   # (M, 11)

            # filter the empty gt
            n = gt_info.any(-1).sum()
            gt_info = gt_info[:n]

            reg_class_loss = 0
            use_label_num = 0
            unique_label = pred_info[:, -1].unique()
            for c in unique_label:
                pred_box = pred_info[pred_info[:, -1] == c][:, :7]      # (m, 7)
                gt_box = gt_info[gt_info[:, -1] == c][:, :7]            # (n, 7)
                if pred_box.shape[0] == 0 or gt_box.shape[0] == 0:
                    continue

                # get the match pred boxes and gt boxes, both shape is (k, 7)
                iou = box_utils.boxes3d_nearest_bev_iou(pred_box, gt_box)  # (m, n)
                iou_n = (iou > iou_thresh).any(-1)     # iou thresh
                pred_match_box = pred_box[iou_n]
                max_iou_index = iou.argmax(-1)
                gt_match_box = gt_box[max_iou_index][iou_n]

                # compute the loss
                loss = self.loss_compute(pred_match_box, gt_match_box)
                # loss = F.mse_loss(pred_match_box, gt_match_box)

                # filter the nan or zero result
                if loss > 0:
                    reg_class_loss += loss
                    use_label_num += 1

            # compute the batch loss if use_label_num else
            reg_batch_loss += reg_class_loss / use_label_num if use_label_num else 0

        reg_loss = reg_batch_loss / batch_size
        return reg_loss


    def get_training_loss(self, pred_dicts, gt_dicts):
        BOX_SIZE = self.cfg['BOX_SIZE']

        batch_size = len(pred_dicts)
        pred_lists = []
        for pred_dict in pred_dicts:
            pred = torch.cat(
                [pred_dict['pred_boxes'], pred_dict['pred_scores'][:, None], pred_dict['pred_labels'][:, None]], dim=-1
            )
            # sort confidence
            sort_index = pred[:, -2].argsort(descending=True)  # 从大到小排序
            pred = pred[sort_index]
            pred_lists.append(pred)

        reg_loss_total = 0
        for batch in range(batch_size):
            gt_frame = gt_dicts[batch]
            pred_frame = pred_lists[batch]

            # 过滤掉用0填充的gt部分
            n = gt_frame.any(-1).sum()
            gt_frame = gt_frame[:n]

            reg_loss_frame = 0
            # 对每个gt进行样本分配并进行损失计算, 分配的标准是高过thresh或者最高的iou值
            for gt in gt_frame:
                # get the same label pred
                gt_box, gt_label = gt[:7], gt[-1]
                label_mask = pred_frame[:, -1] == gt_label
                pred_frame_label = pred_frame[label_mask]  # (k1, 11)
                if pred_frame_label.shape[0] == 0:
                    continue

                # get the match thresh
                iou = box_utils.boxes3d_nearest_bev_iou(gt_box[None, :], pred_frame_label[:, :7])
                # iou = iou3d_nms_utils.boxes_iou3d_gpu(gt_box[None, :], pred_frame_label[:, :7])
                iou_mask = (iou > 0.6).view(-1)
                pred_frame_label_match = pred_frame_label[iou_mask]  # (k2, 11) or (0, 11)

                # get the max iou box
                max_iou_index = iou.argmax(dim=-1)
                pred_frame_label_max_iou = pred_frame_label[max_iou_index]

                # get input box
                count = pred_frame_label_match.shape[0]
                overlap_boxes = torch.zeros([self.max_merge_nums, BOX_SIZE], device=gt.device)  # (Max, 7)
                if count > 1:   # 如果超出了范围最多也只选择前max_merge_nums个来进行融合
                    overlap_boxes[:count] = pred_frame_label_match[:self.max_merge_nums, :7]  # (k, 7) to overlap
                else:
                    overlap_boxes[:1] = pred_frame_label_max_iou[:, :7]  # (1, 7) to overlap

                # encoder and decoder to refine
                overlap_boxes = self.boxcoder.encode_torch(overlap_boxes)
                merge_box = self.mergenet(overlap_boxes)
                box_decode = self.boxcoder.decode_torch(merge_box)

                loss = F.mse_loss(box_decode.squeeze(0), gt_box)
                # loss = F.smooth_l1_loss(box_decode.squeeze(0), gt_box)
                reg_loss_frame += loss

            # print('reg_loss_frame: ', reg_loss_frame / n)
            reg_loss_total += reg_loss_frame / n

        reg_loss = reg_loss_total / batch_size
        # print('reg_loss_batch', reg_loss_total)
        # print('reg_loss', reg_loss)
        return reg_loss


    def post_process(self, ensemble_pred_dict, ret_dict_list):
        # 1) 对多模型的recall进行平均处理
        ret_dict = self.statistic_recall_info(ret_dict_list)

        # 2) 对多模型的预测结果使用nms来进行评估(2d / 3d)
        pred_infos = self.non_max_suppression(ensemble_pred_dict)
        pred_dicts = self.collect_pred_data(pred_infos)
        return pred_dicts, ret_dict

    def collect_pred_data(self, pred_infos):
        pred_dicts = []
        for info in pred_infos:
            pred_dict = dict()
            pred_dict['pred_boxes'] = info[:, :9]
            pred_dict['pred_scores'] = info[:, -2]
            pred_dict['pred_labels'] = info[:, -1].type(torch.int64)
            pred_dicts.append(pred_dict)
        return pred_dicts



if __name__ == '__main__':
    box = torch.rand(7, 4)
    print(box)
    model = MergeNet(max_c=7)
    out = model(box)
    print(out)