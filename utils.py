import os
import cv2
import numpy as np

from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import Normalize
from albumentations import BboxParams, Compose, HorizontalFlip, LongestMaxSize, Resize


def pre_process(img, boxes, max_dim=None, xflip=False, gt_boxes=None, gt_labels=None, mode="trainval.txt"):

    if mode == "trainval.txt":
        aug = Compose(
            [
                # wsddn
                # LongestMaxSize(max_size=max_dim),

                # wlip
                Resize(224, 224),
                HorizontalFlip(p=float(xflip)),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ],
            bbox_params=BboxParams(format="pascal_voc", label_fields=["gt_labels"])
        )
        augmented = aug(
            image=img, bboxes=boxes, gt_labels=np.full(len(boxes), fill_value=1)
        )
        augmented_gt = aug(image=img, bboxes=gt_boxes, gt_labels=gt_labels)

        img = augmented["image"].numpy().astype(np.float32)
        boxes = np.asarray(augmented["bboxes"]).astype(np.float32)
        gt_boxes = np.asarray(augmented_gt["bboxes"]).astype(np.float32)

    else:
        aug = Compose(
            [
                # wsddn
                # LongestMaxSize(max_size=480),

                # wlip
                Resize(224, 224),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ],
            bbox_params=BboxParams(format="pascal_voc", label_fields=["gt_labels"])
        )
        augmented = aug(
            image=img, bboxes=boxes, gt_labels=np.full(len(boxes), fill_value=1)
        )
        augmented_gt = aug(image=img, bboxes=gt_boxes, gt_labels=gt_labels)

        img = augmented["image"].numpy().astype(np.float32)
        boxes = np.asarray(augmented["bboxes"]).astype(np.float32)
        gt_boxes = np.asarray(augmented_gt["bboxes"]).astype(np.float32)

    return img, boxes, gt_boxes


def draw_boxes(img, boxes, name, box_color = (0, 0, 255)):

    img_maxValue = img.max()
    img = img * 255 / img_maxValue
    img = np.uint8(img)
    img = img.transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes = boxes.astype(int)

    for i in boxes:
        # i = i.astype(int)
        cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), color=box_color, thickness=2)

    path = os.path.join('/home/fcy/wsddn/data/processed', name + ".jpg")
    cv2.imwrite(path, img)


def sort_proposals(proposals, id_field):
    """Sort proposals by the specified id field."""
    order = np.argsort(proposals[id_field])
    fields_to_sort = ['boxes', id_field, 'scores']
    for k in fields_to_sort:
        proposals[k] = [proposals[k][i] for i in order]


def clip_boxes_to_image(boxes, height, width):
    """Clip an array of boxes to an image with the given height and width."""
    boxes[:, [0, 2]] = np.minimum(width - 1., np.maximum(0., boxes[:, [0, 2]]))
    boxes[:, [1, 3]] = np.minimum(height - 1., np.maximum(0., boxes[:, [1, 3]]))
    return boxes


def filter_small_boxes(boxes, min_size):
    """Keep boxes with width and height both greater than min_size."""
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((w > min_size) & (h > min_size))[0]
    return keep


def unique_boxes(boxes, scale=1.0):
    """Return indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)