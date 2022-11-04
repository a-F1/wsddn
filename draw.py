import cv2
from numpy import random


def draw_boxes(img_name, boxes):
    img = cv2.imread(img_name)
    box_color_1 = (0, 0, 255)
    box_color_2 = (225, 0, 0)
    box_color_3 = (0, 128, 0)
    box_color_4 = (0, 255, 255)
    # for i in boxes[:3]:
    #     # i = i.astype(int)
    #     cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), color=box_color_1, thickness=2)
    # for i in boxes[3:6]:
    #     # i = i.astype(int)
    #     cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), color=box_color_2, thickness=2)
    cv2.rectangle(img, (boxes[0][0], boxes[0][1]), (boxes[0][2], boxes[0][3]), color=box_color_1, thickness=2)
    cv2.rectangle(img, (boxes[1][0], boxes[1][1]), (boxes[1][2], boxes[1][3]), color=box_color_2, thickness=2)
    cv2.rectangle(img, (boxes[2][0], boxes[2][1]), (boxes[2][2], boxes[2][3]), color=box_color_3, thickness=2)
    cv2.rectangle(img, (boxes[3][0], boxes[3][1]), (boxes[3][2], boxes[3][3]), color=box_color_4, thickness=2)
    return img


if __name__ == '__main__':
    # boxes = [[0, 66, 60, 153],
    #          [0, 0, 338, 317],
    #          [48, 240, 195, 371],
    #          [8, 12, 352, 498]]

    boxes = [
                # the detection of dog
                [42, 104, 64, 266],
                # the groundtruth of dog
                [45, 229, 187, 356],
                # the detection of person
                [86, 67, 105, 92],
                # the groundtruth of person
                [7, 11, 338, 478]
    ]
    image = draw_boxes('C:/fcy/wsddn/data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000001.jpg', boxes)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    retval = cv2.imwrite('C:/fcy/wsddn/data/result/t000004.jpg', image)
