'''
    code modified from https://github.com/facebookresearch/2.5D-Visual-Sound/tree/main/models
'''

from locale import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torchaudio


import sys
sys.path.append('..')
from config import init_args, params


def unet_conv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[downconv, downnorm, downrelu])
    else:
        return nn.Sequential(*[downconv])


def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d, kernel_size=4):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv])


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


class AudioCondUNet(nn.Module):
    def __init__(self, args, pr, ngf=64):
        super(AudioCondUNet, self).__init__()
        #initialize layers
        input_nc = args.unet_input_nc
        output_nc = args.unet_output_nc
        n_view = args.n_view
        self.args = args
        self.pr = pr
        self.no_vision = args.no_vision
        self.no_cond_audio = args.no_cond_audio
        self.mono2binaural = args.mono2binaural
        
        self.audionet_convlayer1 = unet_conv(input_nc, ngf, outermost=True)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)

        self.audio_visual_feat_dim = ngf * 8
        cond_feat_dim = 0
        if self.no_vision and not self.no_cond_audio:
            cond_feat_dim = pr.audio_feature_dim
        elif not self.no_vision and self.no_cond_audio:
            cond_feat_dim = pr.visual_feature_dim
        elif not self.no_vision and not self.no_cond_audio:
            cond_feat_dim = pr.visual_feature_dim + pr.audio_feature_dim
        
        self.audio_visual_feat_dim += cond_feat_dim

        self.audionet_upconvlayer1 = unet_upconv(self.audio_visual_feat_dim, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, outermost=True)


    def forward(self, input_audio, cond_feats):
        '''
            input_audio: (N, C, F, T), C = 1 or 2
            cond_feats: (N, C), C = X, all the conditional features
        '''
        # import pdb; pdb.set_trace()
        audio_conv1feature = self.audionet_convlayer1(input_audio)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        audioVisual_feature = audio_conv5feature
        if not self.no_cond_audio or not self.no_vision:
            cond_feats = cond_feats.view(cond_feats.size(0), -1, 1, 1).repeat(1, 1, audio_conv5feature.shape[-2], audio_conv5feature.shape[-1])
            audioVisual_feature = torch.cat((cond_feats, audioVisual_feature), dim=1)
        
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        prediction = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature), dim=1))
        
        if self.mono2binaural:
            mask_prediction = torch.sigmoid(prediction) * 2 - 1
            spec_diff_real = input_audio[:, 0, :, :] * mask_prediction[:, 0, :, :] - input_audio[:, 1, :, :] * mask_prediction[:, 1, :, :]
            spec_diff_img = input_audio[:, 0, :, :] * mask_prediction[:, 1, :, :] + input_audio[:, 1, :, :] * mask_prediction[:, 0, :, :]
            prediction = torch.cat((spec_diff_real.unsqueeze(1), spec_diff_img.unsqueeze(1)), dim=1)
        else:
            prediction = torch.sigmoid(prediction) * 2 - 1
        return prediction




# CUDA_VISIBLE_DEVICES=0 python audio_unet.py

if __name__ == "__main__":
    import pdb; pdb.set_trace()
    args = init_args()
    fn = getattr(params, 'slfm_hm3d')
    pr = fn()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpus = torch.cuda.device_count()
    gpu_ids = list(range(gpus))
    net = AudioCondUNet(args, pr).to(device)
    net = nn.DataParallel(net, device_ids=gpu_ids)
    spec_input = torch.rand(16, 2, 256, 256).to(device)
    visual_input = torch.rand(16, 256).to(device)
    out = net(spec_input, visual_input)
