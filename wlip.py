import clip
import torch
import torch.nn.functional as F

from torch import nn
from collections import OrderedDict
from torchvision.models import vgg16
from torchvision.ops import roi_pool
from torch.utils.checkpoint import checkpoint


# class LayerNorm(nn.LayerNorm):
#     """Subclass torch's LayerNorm to handle fp16."""
#
#     def forward(self, x: torch.Tensor):
#         orig_type = x.dtype
#         ret = super().forward(x.type(torch.float32))
#         return ret.type(orig_type)
#
#
# class QuickGELU(nn.Module):
#
#     def forward(self, x: torch.Tensor):
#         return x * torch.sigmoid(1.702 * x)
#
#
# class ResidualAttentionBlock(nn.Module):
#     def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
#         super().__init__()
#
#         self.attn = nn.MultiheadAttention(d_model, n_head)
#         self.ln_1 = LayerNorm(d_model)
#         self.mlp = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(d_model * 4, d_model))
#         ]))
#         self.ln_2 = LayerNorm(d_model)
#         self.attn_mask = attn_mask
#
#     def attention(self, x: torch.Tensor):
#         self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
#         return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
#
#     def forward(self, x: torch.Tensor):
#         x = x + self.attention(self.ln_1(x))
#         x = x + self.mlp(self.ln_2(x))
#         return x
#
#
# class Transformer(nn.Module):
#     def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
#         super().__init__()
#         self.width = width
#         self.layers = layers
#         self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
#
#     def forward(self, x: torch.Tensor):
#         return self.resblocks(x)
#
#
# class VisionTransformer(nn.Module):
#     def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.output_dim = output_dim
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
#
#         scale = width ** -0.5
#         self.class_embedding = nn.Parameter(scale * torch.randn(width))
#         self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
#         self.ln_pre = LayerNorm(width)
#
#         self.transformer = Transformer(width, layers, heads)
#
#         self.ln_post = LayerNorm(width)
#         self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
#
#     def forward(self, x: torch.Tensor):
#         x = self.conv1(x)  # shape = [*, width, grid, grid]
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#         x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#         x = x + self.positional_embedding.to(x.dtype)
#         x = self.ln_pre(x)
#
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#
#         x = self.ln_post(x[:, 0, :])
#
#         if self.proj is not None:
#             x = x @ self.proj
#
#         return x


class_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow',
                 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


class Wlip(nn.Module):
    def __init__(self, use_checkpoint=False):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.roi_output_size = (7, 7)

        """ Image Encoder """
        model, _ = clip.load("RN50")
        self.features = model.visual
        # print(model)

        """ Text Encoder """
        # self.prompt = model.encode_text

        # self.features = VisionTransformer(224, 32, 768, 12, 12, 512)
        #
        # vit_state_path = "./weight/ViT-B-32.pt"
        # model = torch.jit.load(vit_state_path, map_location=torch.device('cpu')).eval()
        # vit = "visual.proj" in model.state_dict()
        # self.features.load_state_dict(model.state_dict(), strict=False)

        # vit_state_path = "./weight/ViT-B-32.pt"
        # self.features = build_model(vit_state_path)

        """ fc5 """
        self.fc5 = nn.Linear(2048, 512)

        """ fc6 and fc7 """
        # self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=0.5)
        # self.fc6 = nn.Linear(25088, 4096)
        # self.fc7 = nn.Linear(4096, 4096)
        self.base = vgg16(pretrained=False)
        vgg_state_path = "./weight/vgg16_pre.pth"
        self.base.load_state_dict(torch.load(vgg_state_path))
        self.fcs = self.base.classifier[:-1]

        """ Text Encoder """
        # self.ln_pro = nn.Linear(512, 1455)

        self.fc8_c = nn.Linear(4096, 20)
        self.fc8_d = nn.Linear(4096, 20)

        self.features_in_hook = []
        self.features_out_hook = []

        layer_name = 'visual.layer4.2.relu3'
        for (name, module) in model.named_modules():
            # print(name)
            if name == layer_name:
                module.register_forward_hook(hook=self.hook)
                # module.register_backward_hook(hook=self.hook)

    def seg0(self, y):
        y = y.permute(0, 3, 2, 1)
        y = self.fc5(y)
        y = y.permute(0, 3, 2, 1)
        return y

    def seg1(self, y, boxes):
        y = roi_pool(y, boxes, self.roi_output_size, 0.5)
        y = y.view(len(boxes[0]), -1)
        return y

    def seg2(self, y):
        y = self.fcs(y)
        return y

    def seg3(self, y):
        y1 = self.fc8_c(y)
        y2 = self.fc8_d(y)
        return y1, y2

    def seg4(self, y1, y2):
        classification_scores = F.softmax(y1, dim=1)
        detection_scores = F.softmax(y2, dim=0)
        combined_scores = classification_scores * detection_scores
        return combined_scores

    # delete batch_scores
    def forward(self, imgs, boxes):
        boxes = [boxes[0]]

        """ wsddn """
        # out = self.features(imgs)

        """ Image Encoder """
        out = self.features(imgs)

        # out = self.features_out_hook[0].to(torch.float32)
        out = self.features_out_hook[0].float()
        # print("out 是否存在 nan:", torch.isnan(out).any())
        # print(out.requires_grad)

        """ Text Encoder """
        # text = clip.tokenize(class_labels).to(device)
        # text_features = self.prompt(text).to(torch.float32)
        #
        # c_out = self.fc8_c(out)
        # d_out = self.ln_pro(text_features).T

        # out = self.fc6(out)
        # out = self.relu(out)
        # out = self.dropout(out)
        # out = self.fc7(out)
        # out = self.relu(out)
        # out = self.dropout(out)

        if self.use_checkpoint:
            """ fc5 """
            # print("1:{}".format(torch.cuda.memory_allocated(0)))
            out = checkpoint(out.permute, (0, 3, 2, 1))
            out = checkpoint(self.fc5, out)
            out = checkpoint(out.permute, (0, 3, 2, 1))

            """ SPP """
            out = checkpoint(roi_pool, out, boxes, self.roi_output_size, 1.0 / 32)
            out = checkpoint(out.view, (len(boxes[0]), -1))

            """ fc6 and fc7"""
            out = checkpoint(self.fcs, out) # [1455, 4096]

            """ fc8 """
            classification_scores = checkpoint(F.softmax, self.fc8_c(out), 1)
            detection_scores = checkpoint(F.softmax, self.fc8_d(out), 0)
            out = classification_scores * detection_scores

        else:
            """ fc5 """
            out = out.permute(0, 3, 2, 1)
            out = self.fc5(out)
            out = out.permute(0, 3, 2, 1)

            # print(len(boxes[0]))

            out = roi_pool(out, boxes, self.roi_output_size, 1.0 / 4)
            out = out.view(len(boxes[0]), -1)

            out = self.fcs(out)  # [1455, 4096]
            classification_scores = F.softmax(self.fc8_c(out), dim=1)
            detection_scores = F.softmax(self.fc8_d(out), dim=0)
            out = classification_scores * detection_scores

        # print(len(self.features_out_hook))

        self.features_in_hook = []
        self.features_out_hook = []

        return out

    def hook(self, module, fea_in, fea_out):
        self.features_in_hook.append(fea_in)
        self.features_out_hook.append(fea_out)

    @staticmethod
    def calculate_loss(combined_scores, target):
        image_level_scores = torch.sum(combined_scores, dim=0)
        image_level_scores = torch.clamp(image_level_scores, min=0.0, max=1.0)
        # print("score 是否存在 nan:", torch.isnan(image_level_scores).any())
        loss = F.binary_cross_entropy(image_level_scores, target, reduction="sum")
        # print("loss 是否存在 nan:", torch.isnan(loss).any())
        # print(loss)
        return loss


# def convert_weights(model: nn.Module):
#     """Convert applicable model parameters to fp16"""
#
#     def _convert_weights_to_fp16(l):
#         if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
#             l.weight.data = l.weight.data.half()
#             if l.bias is not None:
#                 l.bias.data = l.bias.data.half()
#
#         if isinstance(l, nn.MultiheadAttention):
#             for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
#                 tensor = getattr(l, attr)
#                 if tensor is not None:
#                     tensor.data = tensor.data.half()
#
#         for name in ["text_projection", "proj"]:
#             if hasattr(l, name):
#                 attr = getattr(l, name)
#                 if attr is not None:
#                     attr.data = attr.data.half()
#
#     model.apply(_convert_weights_to_fp16)
#
#
# def build_model(model_path):
#     with open(model_path, 'rb') as opened_file:
#         state_dict = torch.jit.load(opened_file, map_location="cpu").eval().state_dict()
#
#     vit = "visual.proj" in state_dict
#
#     if vit:
#         vision_width = state_dict["visual.conv1.weight"].shape[0]
#         vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
#         vision_heads = vision_width // 64
#         vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
#         grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
#         image_resolution = vision_patch_size * grid_size
#     else:
#         counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
#         vision_layers = tuple(counts)
#         vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
#         vision_heads = vision_width // 64
#         output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
#         vision_patch_size = None
#         assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
#         image_resolution = output_width * 32
#
#     embed_dim = state_dict["text_projection"].shape[1]
#     context_length = state_dict["positional_embedding"].shape[0]
#     vocab_size = state_dict["token_embedding.weight"].shape[0]
#     transformer_width = state_dict["ln_final.weight"].shape[0]
#     transformer_heads = transformer_width // 64
#     transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
#
#     for key in ["input_resolution", "context_length", "vocab_size"]:
#         if key in state_dict:
#             del state_dict[key]
#
#     model = VisionTransformer(image_resolution, vision_patch_size, vision_width, vision_layers, vision_heads, 256)
#     model.load_state_dict(state_dict)
#     return model.eval()