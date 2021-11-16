#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from collections import OrderedDict
from pathlib import Path

import cv2
import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo
from PIL import Image
from torchvision.transforms.functional import to_tensor, normalize
from torchvision.utils import save_image

__all__ = ["FaceParser", "SEMANTIC_REGION"]

resnet18_url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"

CelebAMaskHQ_LABELS = {
    0: "background",
    1: "skin",
    2: "l_brow",
    3: "r_brow",
    4: "l_eye",
    5: "r_eye",
    6: "eye_g",
    7: "l_ear",
    8: "r_ear",
    9: "ear_r",
    10: "nose",
    11: "mouth",
    12: "u_lip",
    13: "l_lip",
    14: "neck",
    15: "neck_l",
    16: "cloth",
    17: "hair",
    18: "hat",
}

CelebAMaskHQ_LABELS_21 = {
    0: "background",
    1: "l_brow",
    2: "r_brow",
    3: "l_eye",
    4: "r_eye",
    5: "eye_g",
    6: "l_ear",
    7: "r_ear",
    8: "ear_r",
    9: "nose",
    10: "mouth",
    11: "u_lip",
    12: "l_lip",
    13: "neck",
    14: "neck_l",
    15: "cloth",
    16: "hair",
    17: "hat",
    18: "face_up",
    19: "face_middle",
    20: "face_down"
}
SEMANTIC_REGION = OrderedDict(
    [
        ("background", (0,)),  # 0
        ("brow", (1, 2)),  # 1
        ("eye", (3, 4)),  # 2
        ("glass", (5,)),  # 3
        ("ear", (6, 7, 8)),  # 4
        ("nose", (9,)),  # 5
        ("mouth", (10,)),  # 6
        ("lips", (11, 12)),  # 7
        ("neck", (13, 14)),  # 8
        ("cloth", (15,)),  # 9
        ("hair", (16,)),  # 10
        ("hat", (17,)),  # 11
        ("face_up", (18,)),  # 12
        ("face_middle", (19,)),  # 13
        ("face_down", (20,)),  # 14
    ]
)

REGION = {
    "BROW": (2, 3),
    "EYE_REGION": (2, 3, 4, 5, 6),
    "EYE": (5, 4),
    "EAR": (7, 8, 9),
    "MOUTH": (11, 12, 13),
    "NECK": (14, 15),
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out


def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum - 1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)
        # self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x)  # 1/8
        feat16 = self.layer3(feat8)  # 1/16
        feat32 = self.layer4(feat16)  # 1/32
        return feat8, feat16, feat32

    def init_weight(self):
        state_dict = modelzoo.load_url(resnet18_url)
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if "fc" in k:
                continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode="nearest")

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode="nearest")
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode="nearest")
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up  # x8, x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


### This is not used, since I replace this with the resnet feature with the same size
class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.n_classes = n_classes
        ## here self.sp is deleted
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)  # here return res3b1 feature
        feat_sp = feat_res8  # use res3b1 feature to replace spatial path feature
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode="bilinear", align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode="bilinear", align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode="bilinear", align_corners=True)
        return feat_out, feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, FeatureFusionModule) or isinstance(child, BiSeNetOutput):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path="vis_results/parsing_map_on_im.jpg"):
    # Colors for all 20 parts
    part_colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 0, 85],
        [255, 0, 170],
        [0, 255, 0],
        [85, 255, 0],
        [170, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [0, 85, 255],
        [0, 170, 255],
        [255, 255, 0],
        [255, 255, 85],
        [255, 255, 170],
        [255, 0, 255],
        [255, 85, 255],
        [255, 170, 255],
        [0, 255, 255],
        [85, 255, 255],
        [170, 255, 255],
    ]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] + ".png", vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im


def valid_mask(mask_19):
    # skin, brow, nose
    has_skin = (mask_19[:, 1].sum(dim=[-1, -2]) > 0).sum() == mask_19.size(0)
    has_nose = (mask_19[:, 10].sum(dim=[-1, -2]) > 0).sum() == mask_19.size(0)
    has_r_brow = (mask_19[:, 3].sum(dim=[-1, -2]) > 0).sum() == mask_19.size(0)
    has_l_brow = (mask_19[:, 2].sum(dim=[-1, -2]) > 0).sum() == mask_19.size(0)
    return has_skin & has_nose & (has_r_brow | has_l_brow)


def _region_middle(region_coordinates):
    region_max = region_coordinates.amax(dim=[1, 2, 3])
    region_coordinates[region_coordinates == 0] = region_coordinates.max() + 1
    region_min = region_coordinates.amin(dim=[1, 2, 3])
    return (0.5 * region_min + 0.5 * region_max).to(torch.long)


def _rectangle_mask(mask_size, r_min, r_max, device="cuda"):
    mask = torch.zeros(mask_size)
    if isinstance(r_max, int):
        r_max = [r_max] * mask.size()[0]
    if isinstance(r_min, int):
        r_min = [r_min] * mask.size()[0]

    for i in range(mask.size()[0]):
        mask[i, :, int(r_min[i]): int(r_max[i]), :] = 1
    return mask.to(device)


def reorganize(mask_19, compact_mask=True):
    if not valid_mask(mask_19):
        return None
    device = mask_19.device

    aym = torch.meshgrid(torch.arange(mask_19.size(-2)), torch.arange(mask_19.size(-1)))[0].to(device=device)

    # brow middle
    eye_region_min = _region_middle((mask_19[:, (2, 3)].sum(dim=1, keepdim=True) > 0).float() * aym)
    # nose middle
    eye_region_max = _region_middle(mask_19[:, (10,)] * aym)

    skin = mask_19[:, 1:2]
    face_middle = (skin * _rectangle_mask(skin.size(), eye_region_min, eye_region_max, device=device)).clip_(0, 1)
    face_down = (skin * _rectangle_mask(skin.size(), eye_region_max, mask_19.size(-2), device=device)).clip_(0, 1)
    face_up = (skin * _rectangle_mask(skin.size(), 0, eye_region_min, device=device)).clip_(0, 1)

    mask_21 = torch.cat([mask_19[:, [0] + list(range(2, 19))], face_up, face_middle, face_down], dim=1)

    if compact_mask:
        return torch.cat([mask_21[:, ids].sum(dim=1, keepdim=True) for ids in SEMANTIC_REGION.values()], dim=1).clip(
            0, 1
        )
    return mask_21


class FaceParser:
    def __init__(self, model_path, device="cuda"):
        self.net = BiSeNet(n_classes=len(CelebAMaskHQ_LABELS)).to(device)
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()

        self.device = torch.device(device)

    def load_image(self, path):
        img = to_tensor(Image.open(path))
        img = img.unsqueeze_(dim=0).to(self.device)
        return img

    def normalize_image(self, tensor, pre_normalize=False, resize=True):
        """
        assume img has range [0, 1]
        """
        if pre_normalize:
            tensor = (tensor + 1) / 2
        if resize:
            tensor = F.interpolate(tensor, (512, 512))
        return normalize(tensor.to(self.device), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def batch_run(self, images, pre_normalize=False, image_repr=True, resize=True, compact_mask=True):
        with torch.no_grad():
            device = images.device
            image_size = images.size()[-2:]
            images = self.normalize_image(images, pre_normalize, resize=resize)
            out = self.net(images)[0]
            if resize:
                out = F.interpolate(out, image_size, mode="nearest")
            out = out.argmax(dim=1).to(device, torch.long)
            if not image_repr:
                out = torch.stack([out == i for i in range(self.net.n_classes)], dim=1).to(device)
                out = reorganize(out, compact_mask=compact_mask)
            return out

    def image_run(self, image_path, output_masked_image=False):
        image = self.normalize_image(self.load_image(image_path))
        parsing = self.batch_run(image, image_repr=True)

        image_path = Path(image_path)

        Image.fromarray(np.uint8(parsing.squeeze(0).cpu().numpy())).save(
            image_path.parent / f"{image_path.stem}_parsing.png"
        )

        if output_masked_image:
            image = F.interpolate(self.load_image(image_path), (512, 512))
            parsing = reorganize(
                torch.stack([parsing == i for i in range(self.net.n_classes)], dim=1), compact_mask=True
            )
            for i in range(parsing.size(1)):
                masked_image = parsing[:, i: i + 1] * image
                save_image(
                    masked_image,
                    image_path.parent / f"{image_path.stem}_parsing_{i}.png",
                    normalize=True,
                    range=(0, 1),
                )


if __name__ == "__main__":
    fire.Fire(FaceParser)
