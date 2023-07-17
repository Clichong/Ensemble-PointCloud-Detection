import pickle

import torch
import torchvision
import numpy as np


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]        # width
    y[:, 3] = x[:, 3] - x[:, 1]        # height
    return y


def non_max_suppression(self, ensemble_dict_list, cond_thres=0.2, iou_thresh=0.3):
    new_pred_dicts = []
    for ensemble_dict in ensemble_dict_list:
        boxes = ensemble_dict['pred_boxes']
        scores = ensemble_dict['pred_scores']
        labels = ensemble_dict['pred_labels']

        info = torch.cat([boxes, scores[:, None], labels[:, None]], dim=-1)

        cond_mask = info[:, -2] > cond_thres
        info = info[cond_mask]

        # max_w, max_l, max_h, min_lwh = 5.0, 3.0, 3.0, 0.05
        # size_mask = torch.cat(
        #     [info[:, 3:4] < max_w, info[:, 4:5] < max_l, info[:, 5:6] < max_h, info[:, 3:6] > min_lwh], dim=-1
        # )
        # size_mask = size_mask.all(-1)
        # info = info[size_mask]

        sort_index = info[:, -2].argsort(descending=True)
        info = info[sort_index]

        box_xywl = info[:, [0, 1, 3, 4]]  # xywl
        box_xyxy = xywh2xyxy(box_xywl)  # xyxy

        # SHIFT_NUM = 100
        info[:, [0, 1, 3, 4]] = box_xyxy + 100.

        # merge-nms
        res = []
        labels = info[:, -1]
        for c in labels.unique():
            boxes = info[labels == c][:, [0, 1, 3, 4]]
            dc = info[labels == c]
            while len(dc):
                if len(dc) == 1:
                    res.append(dc)
                    break
                # boxes or scores has been sorted
                i = torchvision.ops.box_iou(boxes[:1], boxes).squeeze(0) > iou_thresh  # i = True/False的集合

                # weight merge
                # scores = dc[i, -2:-1]  # score
                # boxes[0, :4] = (scores * boxes[i, :4]).sum(0) / scores.sum()  # 重叠框位置信息求解平均值
                # dc[:1][:, [0, 1, 3, 4]] = boxes[0, :4]

                # learnable network merge
                count = i.sum()  # overlap boxes numbers
                merge_box = boxes[:1]  # (1, d)
                if count > 1:
                    overlap_boxes = torch.zeros([self.max_merge_nums, 4], device=info.device)  # (k, d)
                    overlap_boxes[:count] = boxes[i][:self.max_merge_nums]
                    merge_box = self.mergenet(overlap_boxes)  # (1, d)

                # dc[:1][:, [0, 1, 3, 4]] = self.mergenet(boxes[i][:self.max_merge_nums])    # select topK boxes to merge
                dc[:1][:, [0, 1, 3, 4]] = merge_box
                res.append(dc[:1])
                dc = dc[i == 0]
                boxes = boxes[i == 0]

        info = torch.cat(res, dim=0)

        box_xyxy = info[:, [0, 1, 3, 4]] - 100.
        info[:, [0, 1, 3, 4]] = xyxy2xywh(box_xyxy)

        # keep the same type with origin
        pred_dict = dict()
        pred_dict['pred_boxes'] = info[:, :9]
        pred_dict['pred_scores'] = info[:, -2]
        pred_dict['pred_labels'] = info[:, -1].type(torch.int64)
        new_pred_dicts.append(pred_dict)

    return new_pred_dicts


def nms(ensemble_dict_list, cond_thres=0.2, nms_thresh=0.3):
    new_pred_dicts = []
    for ensemble_dict in ensemble_dict_list:
        boxes = ensemble_dict['pred_boxes']
        scores = ensemble_dict['pred_scores']
        labels = ensemble_dict['pred_labels']

        info = torch.cat([boxes, scores[:, None], labels[:, None]], dim=-1)

        cond_mask = info[:, -2] > cond_thres
        info = info[cond_mask]

        # max_w, max_l, max_h, min_lwh = 5.0, 3.0, 3.0, 0.05
        # size_mask = torch.cat(
        #     [info[:, 3:4] < max_w, info[:, 4:5] < max_l, info[:, 5:6] < max_h, info[:, 3:6] > min_lwh], dim=-1
        # )
        # size_mask = size_mask.all(-1)
        # info = info[size_mask]

        sort_index = info[:, -2].argsort(descending=True)
        info = info[sort_index]

        box_xywl = info[:, [0, 1, 3, 4]]        # xywl
        box_xyxy = xywh2xyxy(box_xywl)     # xyxy

        # use hard batch nms
        nms_index = torchvision.ops.boxes.batched_nms(box_xyxy, info[:, -2], info[:, -1], nms_thresh)
        info = info[nms_index]

        # keep the same type with origin
        pred_dict = dict()
        pred_dict['pred_boxes'] = info[:, :9].type(boxes.dtype)
        pred_dict['pred_scores'] = info[:, -2].type(scores.dtype)
        pred_dict['pred_labels'] = info[:, -1].type(labels.dtype)
        new_pred_dicts.append(pred_dict)

    return new_pred_dicts


class EnsembleCoder(object):
    def __init__(self, code_size=7):
        super().__init__()
        self.code_size = code_size
        self.coder_base = None

    def encode_torch(self, boxes):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:
        """
        n = boxes.any(-1).sum()

        # boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)
        boxes_encode = torch.zeros_like(boxes)

        self.coder_base = boxes[:n].mean(dim=0)
        boxes_encode[:n] = boxes[:n] - self.coder_base

        return boxes_encode

    def decode_torch(self, box_encodings):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:
        """
        return box_encodings + self.coder_base


if __name__ == '__main__':
    from pcdet.utils import box_utils
    from tools.ensemble_utils.ensemble_model import MergeNet
    import torch.nn.functional as F
    from pcdet.ops.iou3d_nms import iou3d_nms_utils

    with open('../gt_dicts.pkl', 'rb') as f:
        gt_dicts = pickle.load(f)
    with open('../pred_dicts.pkl', 'rb') as f:
        pred_dicts = pickle.load(f)

    MAX_MERGE_NUMS = 10
    BOX_SIZE = 7
    batch_size = len(pred_dicts)
    model = MergeNet(max_c=MAX_MERGE_NUMS)
    model.cuda()

    boxcoder = EnsembleCoder(code_size=BOX_SIZE)

    pred_lists = []
    for pred_dict in pred_dicts:
        pred = torch.cat(
            [pred_dict['pred_boxes'], pred_dict['pred_scores'][:, None], pred_dict['pred_labels'][:, None]], dim=-1
        )
        # sort confidence
        sort_index = pred[:, -2].argsort(descending=True)   # 从大到小排序
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
            pred_frame_label = pred_frame[label_mask]      # (k1, 11)
            # if pred_frame_label.shape[0] == 0:
            #     continue

            # get the match thresh
            iou = box_utils.boxes3d_nearest_bev_iou(gt_box[None, :], pred_frame_label[:, :7])
            # iou = iou3d_nms_utils.boxes_iou3d_gpu(gt_box[None, :], pred_frame_label[:, :7])
            iou_mask = (iou > 0.6).view(-1)
            pred_frame_label_match = pred_frame_label[iou_mask]    # (k2, 11) or (0, 11)

            # get the max iou box
            max_iou_index = iou.argmax(dim=-1)
            pred_frame_label_max_iou = pred_frame_label[max_iou_index]

            # get input box
            count = pred_frame_label_match.shape[0]
            overlap_boxes = torch.zeros([MAX_MERGE_NUMS, BOX_SIZE], device=gt.device)  # (Max, 7)
            if count > 1:
                overlap_boxes[:count] = pred_frame_label_match[:MAX_MERGE_NUMS, :7]   # (k, 7) to overlap
            else:
                overlap_boxes[:1] = pred_frame_label_max_iou[:, :7]     # (1, 7) to overlap

            # encoder and decoder to refine
            box_encode = boxcoder.encode_torch(overlap_boxes)
            merge_box = model(box_encode)
            box_decode = boxcoder.decode_torch(merge_box)

            loss = F.mse_loss(merge_box.squeeze(0), gt_box)
            # loss = F.smooth_l1_loss(box_decode.squeeze(0), gt_box)
            reg_loss_frame += loss

        # print('reg_loss_frame: ', reg_loss_frame / n)
        reg_loss_total += reg_loss_frame / n

    reg_loss = reg_loss_total / batch_size
    # print('reg_loss_batch', reg_loss_total)
    print('reg_loss', reg_loss)

