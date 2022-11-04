import torch
import cv2
import numpy as np
from random import choice

__all__ = ["Compose", "ToTensor", "ToArray", "Normalize", "Resize", "HorizontalFlip"]

class_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                'train', 'tvmonitor']

class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, to_transform):
        for t in self.transforms:
            to_transform = t(to_transform)
        return to_transform

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):

        """
        Convert a ``numpy.ndarray`` to tensor.
        Converts a numpy.ndarray (H x W x C) in the range
        [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        """

        def __call__(self, to_transform):

            channel_swap = (2, 0, 1)
            to_transform['img'] = to_transform['img'].transpose(channel_swap)

            to_transform['img'] = torch.tensor(to_transform['img'])

            to_transform['proposals'] = torch.tensor(to_transform['proposals'])

            return to_transform

        def __repr__(self):
            return self.__class__.__name__ + '()'


class ToArray(object):
    """ Get the ground truth annotations.
    # Arguments
        to_transform['gt'] : list, [[x1_min, y1_min, x1_max, y1_max, name1],..., [xn_min, yn_min, xn_max, yn_max, namen]]
    # Returns
        all_annotations    : list, all_annotations[i] = [[x1_min, y1_min, x1_max, y1_max],..., [xn_min, yn_min, xn_max, yn_max]]
    """

    def __call__(self, to_transform):

        all_annotations = [[] for i in range(20)]

        for i in range(len(to_transform['gt'])):
            ind = class_labels.index(to_transform['gt'][i][4])
            all_annotations[ind].append(to_transform['gt'][i][:4])

        to_transform['gt'] = all_annotations

        return to_transform

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    def __init__(self, mean, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, to_transform):
        to_transform['img'] = to_transform['img'].astype(np.float32, copy=False)
        to_transform['img'] -= np.array([[self.mean]])

        return to_transform

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Resize(object):
    def __init__(self, interpolation=cv2.INTER_LINEAR):
        self.interpolation = interpolation

    def __call__(self, to_transform):
        im_shape = to_transform['img'].shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        im_scale = float(576) / float(im_size_max)
        to_transform['img'] = cv2.resize(to_transform['img'], None, None, fx=im_scale, fy=im_scale,
                                         interpolation=self.interpolation)

        if to_transform['proposals'] is not None:
            rois = to_transform['proposals'].astype(np.float, copy=False) * im_scale
            levels = np.zeros((rois.shape[0], 1), dtype=np.int)

            rois_blob = np.hstack((levels, rois))

            to_transform['proposals'] = rois_blob.astype(np.float32, copy=False)

        return to_transform

    def __repr__(self):
        interpolate_str = self.interpolation
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class HorizontalFlip(object):
    def __call__(self, to_transform):
        if choice([True, False]):
            to_transform['img'] = to_transform['img'][:, ::-1, :]

            im_width = to_transform['img'].shape[1]
            to_transform['proposals'] = flip_boxes(to_transform['proposals'], im_width)

        return to_transform

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def flip_boxes(boxes, im_width):
    """Flip boxes horizontally."""
    boxes_flipped = boxes.copy()
    boxes_flipped[:, 0::4] = im_width - boxes[:, 2::4] - 1
    boxes_flipped[:, 2::4] = im_width - boxes[:, 0::4] - 1
    return boxes_flipped