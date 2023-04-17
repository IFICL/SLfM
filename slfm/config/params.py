import os

import numpy as np

import sys
sys.path.append('..')
from utils import utils

Params = utils.Params


def base(name):
    pr = Params(
        frame_rate = 10,
        samp_sr = 16000,
        clip_length = 2.55,
        cond_clip_length = 2.55,
        log_spec = True,
        f_min=0,
        f_max=None,
        log_offset=1e-5,
        n_mel = 96,
        hop_length = 160,
        win_length=400,
        n_fft=512,
        spec_min=-100.,
        spec_max = 100.,
        num_samples = 0,
        cutoff_freq = 80,
        # mono = True,
        seed=2022,
        # seed=2023,
        img_size=(240, 320),
        crop_size=224,
        fov=60,
        flip_prob = 0.5,
        gamma=0.3,
        num_classes=32,
        visual_feature_dim = 512,
        audio_feature_dim=512,
        fused_feature_dim=512,
        objective=None,
        net=None,
        dataloader=None,
        loss=None,
        format='mel',
        lr_milestones = [20, 40, 60, 80, 100],
        list_train = ' ',
        list_val = ' ',
        list_test = ' ',
        audiobase_path='data/AI-Habitat/data-split/FMA',
        list_vis = ' ',
        tau=0.05,
        # transformer part
        nhead=6,
        feedforward_dim=1536,
        dropout=0.1,
        nlayers=12,
        scaling_rate=(0.5, 1.5)
    )

    return pr


# --------------------- SLfM: HM3D ------------------------ # 

def slfm_hm3d(**kwargs):
    pr = base('slfm_hm3d', **kwargs)
    pr.clip_length = 2.55
    pr.cond_clip_length = 2.55
    pr.num_samples = int(round(pr.samp_sr * pr.clip_length))
    pr.visual_feature_dim = 512
    pr.audio_feature_dim = 512
    pr.dataloader = 'SLfMbaseDataset'
    pr.net = 'SLfMNet'
    pr.generative_net = 'AudioCondUNet'
    pr.vision_net = 'CameraATTNNet'
    pr.audio_net = 'CondAudioEncoder'
    pr.vision_cls_net = 'VisionSFMNet'
    pr.audio_cls_net = 'AudioLoCNet'
    pr.geometric_net = 'SLfMGeoNet'
    pr.num_classes = 64
    pr.list_train = 'data/AI-Habitat/data-split/hm3d-4view-rotation/train.csv'
    pr.list_val = 'data/AI-Habitat/data-split/hm3d-4view-rotation/val.csv'
    pr.list_test = 'data/AI-Habitat/data-split/hm3d-4view-rotation/test.csv'
    pr.audiobase_path = 'data/AI-Habitat/data-split/FMA'
    return pr

