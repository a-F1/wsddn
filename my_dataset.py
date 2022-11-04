import os
import cv2
import torch
import random
import pickle

import numpy as np
import xml.etree.ElementTree as ET

from utils import pre_process, sort_proposals, \
    draw_boxes, clip_boxes_to_image, filter_small_boxes, unique_boxes
from torch.utils.data import Dataset



class myDataSet(Dataset):

    def __init__(self, voc_root, txt_name, num_images=10):

        # self.root = '.\data\VOCdevkit\VOC2007'
        self.root = os.path.join(voc_root, "data", "VOCdevkit", "VOC2007")
        # self.img_root = '.\data\VOCdevkit\VOC2007\JPEGImages'
        self.img_root = os.path.join(self.root, "JPEGImages")
        # self.annotations_root = '.\data\VOCdevkit\VOC2007\Annotations'
        self.annotations_root = os.path.join(self.root, "Annotations")

        # read trainval.txt or test.txt file
        # self.root = '.\data\VOCdevkit\VOC2007\ImageSets\Main\trainval.txt'
        txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_path)

        with open(txt_path) as read:
            self.xml_list = [line.strip() for line in read.readlines() if len(line.strip()) > 0]
        # Load the specified number of images
        self.xml_list = self.xml_list[:num_images]

        # read class_labels
        self.class_labels = {"aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4, "bus": 5, "car": 6,
                             "cat": 7, "chair": 8, "cow": 9, "diningtable": 10, "dog": 11, "horse": 12,
                             "motorbike": 13, "person": 14, "pottedplant": 15, "sheep": 16, "sofa": 17,
                             "train": 18, "tvmonitor": 19,}
        self.num_classes = len(self.class_labels)

        # self.images = '.\data\VOCdevkit\VOC2007\JPEGImages\000005.jpg'
        self.images = [os.path.join(self.img_root, x + ".jpg") for x in self.xml_list]
        # annotations = '.\data\VOCdevkit\VOC2007\Annotations\000005.xml'
        self.annotations = [os.path.join(self.annotations_root, x + ".xml") for x in self.xml_list]
        assert (len(self.images) == len(self.annotations))

        # load selectivesearch_data
        if txt_name == 'trainval.txt':
            ss_data = './data/selective_search_data/voc_2007_trainval.pkl'
        else:
            ss_data = './data/selective_search_data/voc_2007_test.pkl'
        # print(ss_data)

        with open(ss_data, 'rb') as f:
            self.proposals = pickle.load(f)

        sort_proposals(self.proposals, 'indexes')

        self.txt_name = txt_name
        self.num_images = num_images
        self.scales = [480, 576, 688, 864, 1200]

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, i):

        img = cv2.imread(self.images[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = self.get_boxes(i)
        gt_boxes, gt_labels = self.get_annotations(i)

        # if self.txt_name == "trainval.txt":
        scaled_img, scaled_proposals, scaled_gt = pre_process(
            img,
            boxes,
            # wsddn
            # random.choice(self.scales),
            # wlip
            224,
            random.choice([False, True]),
            gt_boxes,
            gt_labels,
            self.txt_name
        )
        annotations = self.get_groundtruths(scaled_gt, gt_labels)
        labels = self.get_class_labels(gt_labels)

        if self.num_images < 10:
            draw_boxes(scaled_img, scaled_gt, self.xml_list[i])

        return scaled_img, labels, scaled_proposals, annotations

    def get_class_labels(self, target):

        classification_labels = torch.zeros(self.num_classes)

        for label in target:
            classification_labels[label] = 1

        return classification_labels

    def get_boxes(self, i):

        min_proposal_size = 20
        boxes = self.proposals['boxes'][i]
        root = ET.parse(self.annotations[i]).getroot()
        size = root.find('size')

        height = int(size.find('height').text)
        width = int(size.find('width').text)
        boxes = clip_boxes_to_image(boxes, height, width)

        # Remove duplicate boxes and very small boxes
        keep = unique_boxes(boxes)
        boxes = boxes[keep, :]
        keep = filter_small_boxes(boxes, min_proposal_size)
        boxes = boxes[keep, :]

        return boxes.astype(np.float32)

    def get_annotations(self, i):
        xml = ET.parse(self.annotations[i])

        boxes = []
        labels = []

        for obj in xml.findall("object"):
            if obj.find("difficult").text != "1":
                bndbox = obj.find("bndbox")
                boxes.append(
                    [
                        int(bndbox.find(tag).text)
                        for tag in ("xmin", "ymin", "xmax", "ymax")
                    ]
                )
                boxes[-1][0] -= 1
                boxes[-1][1] -= 1
                labels.append(self.class_labels[obj.find("name").text])

        boxes = np.stack(boxes).astype(np.float32)
        labels = np.stack(labels).astype(np.int32)
        return boxes, labels

    def get_groundtruths(self, groundtruths, labels):

        all_annotations = [[] for i in range(20)]

        for i in range(len(labels)):
            index = labels[i]
            all_annotations[index].append(groundtruths[i])

        return all_annotations