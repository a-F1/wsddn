import os
import torch
import torch.nn.functional as F

from torch import nn
from torchvision.models import vgg16
from torchvision.ops import roi_pool


class Fan_Net(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Fan_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# class SPPLayer(nn.Module):
#     def __init__(self, dim_in, spatial_scale):
#         super().__init__()
#         self.dim_in = dim_in
#
#         res = 7
#         self.spp = RoIPool(output_size=(res, res), spatial_scale=spatial_scale)
#
#         self.spatial_scale = spatial_scale
#         self.dim_out = hidden_dim = 4096
#
#         roi_size = 7
#         self.fc6 = nn.Linear(dim_in * roi_size**2, hidden_dim)
#         self.fc7 = nn.Linear(hidden_dim, hidden_dim)
#
#     def forward(self, x, rois):
#         x = self.spp(x, rois) # [roi_num,512,7,7]
#         batch_size = x.size(0)
#         x = F.relu(self.fc6(x.view(batch_size, -1)), inplace=True) # [roi_num,4096]
#         x = F.relu(self.fc7(x), inplace=True) # [roi_num,4096]
#         return x


class WSDDN(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = vgg16(pretrained=False)
        state_path = "./weight/vgg16_pre.pth"
        self.base.load_state_dict(torch.load(state_path))
        self.roi_output_size = (7, 7)

        self.features = self.base.features[:-1]
        self.fcs = self.base.classifier[:-1]

        # self.spp = SPPLayer(512, 1. / 8.)

        self.fc8_c = nn.Linear(4096, 20)
        self.fc8_d = nn.Linear(4096, 20)

    # delete batch_scores
    def forward(self, imgs, boxes):
        boxes = [boxes[0]]

        out = self.features(imgs)
        out = roi_pool(out, boxes, self.roi_output_size, 1.0 / 16)
        out = out.view(len(boxes[0]), -1)
        out = self.fcs(out)  # [4000, 4096]

        classification_scores = F.softmax(self.fc8_c(out), dim=1)
        detection_scores = F.softmax(self.fc8_d(out), dim=0)
        combined_scores = classification_scores * detection_scores
        return combined_scores

    @staticmethod
    def calculate_loss(combined_scores, target):
        image_level_scores = torch.sum(combined_scores, dim=0)
        image_level_scores = torch.clamp(image_level_scores, min=0.0, max=1.0)
        loss = F.binary_cross_entropy(image_level_scores, target, reduction="sum")
        return loss

        # target = target.repeat(combined_scores.shape[0], 1)
        # combined_scores = torch.clamp(combined_scores, min=0.0, max=1.0)
        # loss = F.binary_cross_entropy(combined_scores, target, reduction="sum")
        # return loss

        # cls_score = combined_scores.clamp(1e-6, 1 - 1e-6)
        # labels = target.clamp(0, 1)
        # loss = -labels * torch.log(cls_score) - (1 - labels) * torch.log(1 - cls_score)
        #
        # return loss.mean()