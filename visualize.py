import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader


from wlip import Wlip
from wsddn import WSDDN
from my_dataset import myDataSet
from test import get_detections

class_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                 'train', 'tvmonitor']

def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test WSDDN')
    parser.add_argument("--model", default='Wlip', help="model")
    # 4951
    parser.add_argument("--num_images", type=int, default=20, help="Images count")
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

        print(model)
        if model == 'Wlip':
            net = Wlip()
        else:
            net = WSDDN()

        net.to(device)
        net.eval()
        net.load_state_dict(torch.load(state_path))

        test_ds = myDataSet(root_path, "test.txt", num_images)
        test_dl = DataLoader(test_ds, num_workers=4, batch_size=1)

        with torch.no_grad():

            k = 0
            for m, data in enumerate(test_dl):

                image, target, box_proposals, groundtruths = data

                scores = net(image.cuda(), box_proposals.cuda())
                scores = scores.cpu()

                img = image[0].numpy()
                img = np.uint8(img)  # float32-->uint8
                img = img.transpose(1, 2, 0)  # mat_shape: (982, 814，3)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                _, _, detections = get_detections(box_proposals[0], scores)

                for idx, gt_i in enumerate(groundtruths):

                    if len(gt_i) == 0:
                        continue

                    for gt_box in gt_i:

                        gt_p1 = (int(gt_box[0][0]), int(gt_box[0][1]))
                        gt_p2 = (int(gt_box[0][2]), int(gt_box[0][3]))
                        det_color = (0, 255, 0)

                        img = cv2.rectangle(img, gt_p1, gt_p2, (255, 0, 0), 2)

                for idx, det_i in enumerate(detections):

                    if len(det_i) == 0:
                        continue

                    for det_box in det_i:

                        det_p1 = (int(det_box[0]), int(det_box[1]))
                        det_p2 = (int(det_box[2]), int(det_box[3]))

                        img = cv2.rectangle(img, det_p1, det_p2, (0, 255, 0), 2)
                        text = "{:s} {:2d}%".format(class_labels[idx], int(det_box[4]*100))

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        thickness = 1

                        (text_width, text_height) = cv2.getTextSize(text, font, font_scale, thickness)[0]

                        text_offset_x = det_p1[0]
                        text_offset_y = det_p1[1] + text_height

                        # make the coords of the box with a small padding of two pixels
                        box_coords = (
                        (text_offset_x, text_offset_y + 2), (text_offset_x + text_width + 2, text_offset_y - text_height))

                        cv2.rectangle(img, box_coords[0], box_coords[1], det_color, cv2.FILLED)

                        cv2.putText(img, text, (text_offset_x, text_offset_y), font, font_scale, (0, 0, 0), thickness,
                                    cv2.LINE_AA)

                path = os.path.join('/home/fcy/wsddn/data/processed', 'result_{:d}'.format(m + 1) + ".jpg")
                cv2.imwrite(path, img)


if __name__ == '__main__':
    main()