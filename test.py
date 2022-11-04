import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch
import argparse
import numpy as np
import utils as trans

from tqdm import tqdm
from torchvision.ops import nms
from torch.utils.data import DataLoader

from wlip import Wlip
from wsddn import WSDDN
from draw import draw_boxes
from my_dataset import myDataSet


class_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                 'train', 'tvmonitor']

def evaluate(
    dataloader, net, num_images):
    """ Evaluate a given dataset using a given net.
    # Arguments
        dataset          : the dataset to evaluate.
        net              : The net to evaluate.
    # Returns
        all_scores       : dict, {'000001.jpg': 1828x20 tensor,..., '000014.jpg': 3091x20 tensor}
        all_proposals    : dict, {'000001.jpg': 1828x4 tensor,..., '000014.jpg': 3091x4 tensor}
        temp_groundtruth : dict, {'000001.jpg': [[[],..., [], [[8, 12, 352, 498]], [], [], [], [], []]],
                                 ...,
                                 '000014.jpg': [[[],..., [[314, 8, 344, 65], [331, 4, 361, 61], [357, 8, 401, 61]],..., []]]}
        all_groundtruth  : list, [[[], [], [], [], [], [], [], [], [], [], [], [[48, 240, 195, 371]], [], [], [[8, 12, 352, 498]], [], [], [], [], []],
                                 ...,
                                 [[185, 194, 500, 316], [416, 180, 500, 222], [163, 197, 267, 244]], [], [], [], [], [], [], [], [[314, 8, 344, 65], [331, 4, 361, 61], [357, 8, 401, 61]], [], [], [], [], []]]
    """
    with torch.no_grad():
        index = 0

        all_groundtruths = []
        all_detections = [[[] for _ in range(num_images)] for _ in range(20)]

        for data in tqdm(dataloader):

            image, target, box_proposals, groundtruths = data

            if len(box_proposals) == 0:
                continue

            scores = net(image.cuda(), box_proposals.cuda())
            scores = scores.cpu()

            _, _, detections = get_detections(box_proposals[0], scores)

            for cls_idx in range(0, len(detections)):
                all_detections[cls_idx][index] = detections[cls_idx]

            all_groundtruths.append(groundtruths)

            index += 1

        # all_groundtruths = list(map(list, zip(*all_groundtruths)))
        all_groundtruths = get_groundtruths(all_groundtruths, num_images)

        return all_detections, all_groundtruths


def get_groundtruths(
        gt, num_images):

    all_detections = [[[] for _ in range(num_images)] for _ in range(20)]

    for i, each_images in enumerate(gt):

        for j, each_class in enumerate(each_images):
            if len(each_class) == 0:
                all_detections[j][i] = np.empty(shape=(0, 4), dtype=np.float32)

            else:
                all_detections[j][i] = np.vstack([k.numpy() for k in each_class])
    return all_detections


def get_detections(
    proposals, scores, score_threshold=0.1, iou_threshold=0.4, max_detections=100):
    """ Connect detections and scores.
    # Arguments
        proposals       : (N, 4) ndarray of float, proposals[i] = [xi_min, yi_min, xi_max, yi_max].
        scores          : (N, 1) ndarray of float, scores for each proposal.
        score_threshold : The score confidence threshold to use.
        iou_threshold   : discard all overlapping boxes with IoU > iou_threshold.
        max_detections  : The maximum number of detections to use per image.
    # Returns
        all_detections  : list, all_detections[i] = [[x1_min, y1_min, x1_max, y1_max, score1],..., [xn_min, yn_min, xn_max, yn_max, scoren]]
    """
    num_classes = 20
    cls_boxes = [[] for _ in range(num_classes)]
    # print(scores.shape)

    # for each class
    for j in range(0, num_classes):
        # np.where:return the index which meets conditions
        # score_threshold = 0.1
        # 对于评分矩阵的某一列（即某张图片中的某一类物品）
        # 将评分大于 0.1 的行坐标记录在 inds 中

        # Records indexes with a score greater than 0.1
        inds = np.where(scores[:, j] > score_threshold)[0]
        # print('inds:')
        # print(inds.shape)
        scores_j = scores[inds, j]

        # 根据 inds ，筛选出评分大于 0.1 的 proposal
        boxes_j = proposals[inds, :]

        # boxes_j 是 nx4 的二维数组，n 即为符合要求的 proposal 的数量，4 即为相应 proposal 的坐标
        # scores_j 是 n 的数组，使用 np.newaxis 将其调整为 nx1 的二维数组
        # 使用 np.hstack 在水平方向拼接
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)

        boxes_j = torch.tensor(boxes_j)
        scores_j = torch.tensor(scores_j)

        # print('boxes:')
        # print(boxes_j.shape)
        # print(boxes_j.dtype)
        # print(boxes_j.device)
        #
        # print('scores:')
        # print(scores_j.shape)
        # print(scores_j.dtype)
        # print(scores_j.device)

        # iou_threshold = 0.4
        keep = nms(boxes_j, scores_j, iou_threshold)
        # 'keep' is the index of the retained proposal after nms processing
        nms_dets = dets_j[keep.numpy(), :]

        # 经过 scores 、nms 双重筛选后的 proposal
        # 其形式为4个坐标 + 最终评分
        cls_boxes[j] = nms_dets

    data = []
    # 经过 append 处理后，data 的形式为 [[s1], [s2],..., [sn]]
    for j in range(0, num_classes):
        data.append(cls_boxes[j][:, -1])

    # image_score 的形式为 [s1, s2,...,sn]
    image_scores = np.hstack(data)

    if len(image_scores) > max_detections:
        # image_thresh 为 image_scores 中第100个大的元素
        image_thresh = np.sort(image_scores)[-max_detections]

        for j in range(0, num_classes):
            # 每张图片最多保留100个 proposal
            keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
            cls_boxes[j] = cls_boxes[j][keep, :]

    '''
    cls_box = [ # class 1
                [[x1, y1, w1, h1, s1],..., [xn, yn, wn, hn, sn]]
                ,..., 
                # class n
                [[x1, y1, w1, h1, s1],..., [xn, yn, wn, hn, sn]]
                ]

    im_results = [
                    [x1_1, y1_1, w1_1, h1_1, s1_1]
                    ,..., 
                    [x1_n, y1_n, w1_n, h1_n, s1_n]
                    ,..., 
                    [xn_1, yn_1, wn_1, hn_1, sn_1]
                    ,..., 
                    [xn_n, yn_n, wn_n, hn_n, sn_n]
                ]
    '''
    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]

    return scores, boxes, cls_boxes


def compute_ap(recall, precision):
    """ Compute the average precision
    # Arguments
        recall    : list
        precision : list
    # Returns
        ap        : int, The average precision
    """
    # correct AP calculation
    # first append sentinel values at the end recall首位拼接上0，1；precision首位拼接上0，0
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope 将precision逆序比较，从大到小排列
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value 求出recall变化的点坐标
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec 求PR曲线下的面积即AP
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_iou(
    a, b):
    """ Compute the iou
    # Arguments
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float
    # Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def distinguish_samples(
    image_detections, image_annotations, false_positives, true_positives, scores, iou_threshold=0.5):
    """ Distinguish positive and negative samples in all_detections
    # Arguments
        image_detections  : All detections in this image
        image_annotations : All annotations in this image
        iou_threshold     : The threshold used to consider when a detection is positive or negative.
    # Returns
        false_positives	  : int， number of false positives
        true_positives    : int， number of true positives
    """
    if image_detections.shape[0] == 0:
        return true_positives, false_positives, scores

    detected_annotations = []
    # 迭代每一张图片中预测的边界框
    for d in image_detections:

        scores = np.append(scores, d[4])

        if image_annotations.shape[0] == 0:  # 如果真实值bbox为0,则属于错检FP
            false_positives = np.append(false_positives, 1)
            true_positives = np.append(true_positives, 0)
            continue
        # 计算一个预测框与真实框bbox的重叠率IOU (1,115)
        overlaps = compute_iou(np.expand_dims(d[:4], axis=0), image_annotations)
        assigned_annotation = np.argmax(overlaps, axis=1)  # 获得重叠率最大的index
        max_overlap = overlaps[0, assigned_annotation]  # 获得最大的重叠率max_overlap
        # 如果最大的重叠率大于阈值,且真实值bbox的index没有被标记detected_annotations,则为TP,否则都为FP
        # if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
        if max_overlap >= 0.4:
            false_positives = np.append(false_positives, 0)
            true_positives = np.append(true_positives, 1)
            detected_annotations.append(assigned_annotation)
        else:
            false_positives = np.append(false_positives, 1)
            true_positives = np.append(true_positives, 0)
        # draw_boxes('./data/vOCdevkit/VOC2007/JPEGlmages/000001.jpg', [d[:4], image_annotations])
    return true_positives, false_positives, scores


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test WSDDN')
    parser.add_argument("--model", default='Wlip', help="model")
    # 4951
    parser.add_argument("--num_images", type=int, default=4951, help="Images count")
    parser.add_argument('--root_path', default='./', help='dataset')
    parser.add_argument('--state_path', default='./weight/wlip_4_60.pt', help='dataset')

    return parser.parse_args()


def main():

    with torch.no_grad():

        args = parse_args()

        model = args.model
        num_images = args.num_images
        root_path = args.root_path
        state_path = args.state_path
        num_classes = 20

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using {} device.".format(device))

        if model == 'Wlip':
            net = Wlip()
        else:
            net = WSDDN()

        net.to(device)
        net.eval()
        net.load_state_dict(torch.load(state_path))

        test_ds = myDataSet(root_path, "test.txt", num_images)
        test_dl = DataLoader(test_ds, num_workers=4, batch_size=1)

        all_detections, all_groundtruths = evaluate(test_dl, net, num_images)

        # iterate through each image.
        average_precisions = {}
        precisions = {}

        # 迭代每一个分类label
        for label in range(num_classes):

            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            # false_positives, true_positives, scores 为 array([], dtype=float64)

            num_annotations = 0.0

            # print(all_annotations)

            # 迭代每一张图片
            for i in range(num_images):
                detections = all_detections[label][i]  # 预测框的位置x1、y1、x2、y2、score
                # annotations = all_annotations[i][label]  # 真实框的位置x1y1x2y2
                annotations = all_groundtruths[label][i]  # 真实框的位置x1y1x2y2

                # print('annotations:')
                # print(annotations)

                num_annotations += annotations.shape[0]  # 真实框bbox的个数
                # num_annotations += annotations.shape  # 真实框bbox的个数
                detected_annotations = []
                # 迭代每一张图片中预测的边界框
                true_positives, false_positives, scores = distinguish_samples(
                    detections, annotations, false_positives, true_positives, scores)

            # no annotations -> AP for this class is 0 (is this correct?)如果预测框数量为0,则ap直接为0
            if num_annotations == 0:
                average_precisions[label] = 0, 0
                precisions[label] = 0
                continue

            # sort by score 对得分进行逆序排序
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives 按行进行梯度累加和
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            if true_positives.size == 0:
                average_precisions[label] = 0, 0
                precisions[label] = 0
                continue

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            precisions[label] = true_positives[-1] / np.maximum(true_positives[-1] + false_positives[-1], np.finfo(np.float64).eps)

            # compute average precision
            average_precision = compute_ap(recall, precision)
            average_precisions[label] = average_precision, num_annotations


        print('\nAP:')
        sum_AP = 0.0
        for label in range(num_classes):
            label_name = class_labels[label]
            print('{}: {}'.format(label_name, average_precisions[label][0]))
            sum_AP += average_precisions[label][0]

        print('\nmAP: {}'.format(sum_AP / 20))

        print('\nPrecision:')
        for label in range(num_classes):
            label_name = class_labels[label]
            print('{}: {}'.format(label_name, precisions[label]))


if __name__ == '__main__':
    main()
