import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torchvision.transforms as transforms
import torchaudio
import kornia.augmentation as K
import numpy as np

from tqdm import tqdm

import sys
sys.path.append('..')
from config import init_args, params
from utils import utils, torch_utils
import models

def conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=None):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01),
        nn.LeakyReLU(inplace=True),
    )

class CameraEncoder(nn.Module):
    def __init__(self, args, pr):
        super(CameraEncoder, self).__init__()
        self.img_augmentation = K.ColorJiggle(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, same_on_batch=False, p=0.9) if args.color_jitter else nn.Identity()
        self.img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def transform_img(self, imgs, augment=False):
        # import pdb; pdb.set_trace()
        if augment:
            imgs = self.img_augmentation(imgs)
        imgs = self.img_transform(imgs).detach()
        return imgs


class CameraATTNNet(CameraEncoder):
    def __init__(self, args, pr):
        super(CameraATTNNet, self).__init__(args, pr)
        backbone = args.vision_backbone
        weights = "IMAGENET1K_V1" if args.imagenet_pretrain else None 
        if backbone == 'resnet18':
            self.backbone = self.get_truncated_resnet(torchvision.models.resnet18(weights=weights))
        elif backbone == 'resnet34':
            self.backbone = self.get_truncated_resnet(torchvision.models.resnet34(weights=weights))
        elif backbone == 'resnet50':
            self.backbone = self.get_truncated_resnet(torchvision.models.resnet50(weights=weights))
        else:
            raise NotImplementedError
        
        backbone_downsample_rate = 16
        attn_in_channels = int((pr.img_size[0] / backbone_downsample_rate) * (pr.img_size[1] / backbone_downsample_rate))
    
        self.attn_convs = nn.Sequential(
            conv2d(in_channels=attn_in_channels, out_channels=128, kernel_size=3, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
        )

        attn_conv_downsample_rate = 8
        attn_out_size = int(np.ceil(pr.img_size[0] / backbone_downsample_rate / attn_conv_downsample_rate) * np.ceil(pr.img_size[1] / backbone_downsample_rate / attn_conv_downsample_rate))
        
        self.linear = nn.Linear(attn_out_size * 128, pr.visual_feature_dim)

        # init weights
        for m in self.attn_convs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
        nn.init.trunc_normal_(self.linear.weight, mean=0.0, std=0.01)

        if args.add_geometric:
            out_dim = 1
            self.rot_head = nn.Sequential(
                nn.Linear(pr.visual_feature_dim, int(pr.visual_feature_dim // 2)),
                nn.ReLU(True),
                nn.Linear(int(pr.visual_feature_dim // 2), out_dim)
            )
            for m in self.rot_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01)
            

    def forward(self, img_1, img_2, augment=False, return_angle=False, backbone=False, correlation=False):
        # import pdb; pdb.set_trace()
        ''' 
            img_1: (N, C, H, W)
            img_2: (N, C, H, W)
        '''
        if backbone: 
            im_feature1 = self.forward_backbone(self, img_1, augment)
            return im_feature1

        if correlation:
            x = self.forward_correlation(img_1, img_2, return_angle=return_angle)
            return x

        img_1 = self.transform_img(img_1, augment)
        img_2 = self.transform_img(img_2, augment)
        im_feature1 = self.backbone(img_1)
        im_feature2 = self.backbone(img_2)
        x = self.forward_correlation(im_feature1, im_feature2, return_angle=return_angle)
        return x

    def forward_backbone(self, img, augment):
        img = self.transform_img(img, augment)
        im_feature = self.backbone(img)
        return im_feature

    def forward_correlation(self, im_feature1, im_feature2, return_angle=False):
        aff = self.compute_corr_softmax(im_feature1, im_feature2)
        x = self.attn_convs(aff)
        x = torch.flatten(x, 1)
        x = F.relu(self.linear(x))
        if return_angle:
            pred = self.rot_head(x).squeeze(-1)

            return x, pred
        return x

    def get_truncated_resnet(self, resnet):
        return nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

    def compute_corr_softmax(self, im_feature1, im_feature2):
        _, _, h1, w1 = im_feature1.size()
        _, _, h2, w2 = im_feature2.size()
        im_feature2 = im_feature2.transpose(2, 3)
        im_feature2_vec = im_feature2.contiguous().view(im_feature2.size(0), im_feature2.size(1), -1)
        im_feature2_vec = im_feature2_vec.transpose(1, 2)
        im_feature1_vec = im_feature1.contiguous().view(im_feature1.size(0), im_feature1.size(1), -1)
        corrfeat = torch.matmul(im_feature2_vec, im_feature1_vec)
        corrfeat = corrfeat.view(corrfeat.size(0), h2*w2, h1, w1)
        corrfeat  = F.softmax(corrfeat, dim=1)
        return corrfeat


# --------------------------- Vision Classification Network ------------------------------- #

class VisionSFMNet(nn.Module):
    # Vision Relative Angle Net
    def __init__(self, args, pr, net):
        super(VisionSFMNet, self).__init__()
        self.pr = pr
        self.args = args
        self.vision_net = net
        self.azimuth_loss_type = args.azimuth_loss_type

        if self.azimuth_loss_type == 'classification':
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
            num_classes = pr.num_classes
        elif self.azimuth_loss_type == 'regression':
            self.criterion = nn.L1Loss(reduction='mean')
            num_classes = 1
        
        self.fc = nn.Linear(pr.visual_feature_dim, num_classes)
        nn.init.trunc_normal_(self.fc.weight, mean=0.0, std=0.01)

        if args.freeze_camera:
            for param in self.vision_net.parameters():
                param.requires_grad = False
            tqdm.write('Freezed the vision backbone.')

    def forward(self, inputs, loss=False, evaluate=False):
        # import pdb; pdb.set_trace()
        img_1 = inputs['img_1']
        img_2 = inputs['img_2']
        visual_feat = self.vision_net(img_1, img_2, augment=(not evaluate))
        preds = self.fc(visual_feat)

        if loss:
            loss = self.calc_loss(preds, inputs)
            return loss
        if evaluate:
            output = self.evaluate(preds, inputs)
            return output

        return preds

    def calc_loss(self, preds, inputs):
        if self.azimuth_loss_type == 'classification':
            target = inputs['relative_camera1_angle_bin']
            loss = self.criterion(preds, target.long())
        elif self.azimuth_loss_type == 'regression':
            # normalize the prediction to angle range
            preds = torch.tanh(preds.squeeze(-1)) * 180.0
            target = inputs['relative_camera1_angle']
            loss = self.criterion(preds, target.long())
        loss = loss.view(1, -1)
        return loss

    def evaluate(self, preds, inputs):
        loss = self.calc_loss(preds, inputs)
        output = {
            'Loss': loss.view(1, -1)
        }
        if self.azimuth_loss_type == 'classification':
            target = inputs['relative_camera1_angle_bin']
            top_1_acc = torch_utils.calc_acc(preds, target, k=1)
            output['Top-1 acc'] = top_1_acc.view(1, -1)
        
        return output

    def score_model_performance(self, res):
        if self.azimuth_loss_type == 'classification':
            score = res['Top-1 acc']
        elif self.azimuth_loss_type == 'regression':
            score = 1 / res['Loss']
        return score



# CUDA_VISIBLE_DEVICES=0 python camera_net.py
if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    args = init_args()
    fn = getattr(params, 'audiosfm_replica')
    pr = fn()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpus = torch.cuda.device_count()
    gpu_ids = list(range(gpus))
    net = CameraATTNNet(args, pr).to(device)
    net = nn.DataParallel(net, device_ids=gpu_ids)
    im1 = torch.rand(16, 3, 240, 320).to(device)
    im2 = torch.rand(16, 3, 240, 320).to(device)
    out = net(im1, im2)