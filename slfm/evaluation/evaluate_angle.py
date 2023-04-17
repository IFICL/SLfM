import argparse
import numpy as np
import os
import sys
import time
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(sys.path[0]))
from config import init_args, params
# from data import *
import data
import models
from models import *
from utils import utils, torch_utils

from functs import update_param, visualize_prediction, visualize_error



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_net(args, pr):
    if args.input == 'vision':
        net = models.__dict__[pr.vision_net](args, pr)
    elif args.input == 'audio' or 'binaural':
        net = models.__dict__[pr.audio_net](args, pr)

    if args.weights:
        resume = './checkpoints/' + args.weights
        if os.path.exists(resume):
            net, _ = torch_utils.load_model(resume, net, strict=True, model_name=args.input)
        else:
            tqdm.write('No checkpoint find. Use Random instead.')
    return net


def logit2angle(args, logit):
    angle_range = 2 if args.filter_sound else 1
    if args.activation == 'tanh':
        theta_s = torch.tanh(logit)
    elif args.activation == 'clamp':
        theta_s = torch.clamp(logit, min=-1., max=1.)
    elif args.activation == 'sigmoid':
        theta_s = torch.sigmoid(logit)
        theta_s = theta_s * 2. - 1.
    theta_s = theta_s * 180.0 / angle_range

    if not args.filter_sound:
        # we map the sound angle outside of (-90, 90) back to (-90, 90)
        theta_s[theta_s <= -90] = -180 - theta_s[theta_s <= -90]
        theta_s[theta_s >= 90] = 180 - theta_s[theta_s >= 90]

    return theta_s


def rot2theta(args, rot):
    # activation each prediction
    if args.activation == 'tanh':
        theta = torch.tanh(rot)
    elif args.activation == 'clamp':
        theta = torch.clamp(rot, min=-1., max=1.)
    elif args.activation == 'sigmoid':
        theta = torch.sigmoid(rot)
        theta = theta * 2. - 1.
    # fit the angle prediction into corresponding range
    if args.finer_rotation:
        theta = theta * 180.0 / 2       # Finer range: [-90, 90]
    else:
        theta = theta * 180.0           # Full range: [-180, 180]
    return theta


def predict(args, pr, net, batch, device):
    # import pdb; pdb.set_trace()
    inputs = {}
    if args.input == 'vision':
        inputs['img_1'] = batch['img_1'].to(device)
        inputs['img_2'] = batch['img_2'].to(device)
        gt_angle = batch['relative_camera1_angle'].to(device)
        _, pred_angle = net(inputs['img_2'], inputs['img_1'], return_angle=True)
        
        pred_angle = rot2theta(args, pred_angle) * pr.rotation_correctness
        gt_angle = gt_angle

    elif args.input == 'audio':
        inputs['audio_1'] = batch['audio_1'].to(device)
        gt_angle = batch['angle_between_source1_camera1'].to(device)
        _, pred_angle = net(inputs['audio_1'], return_angle=True)
        pred_angle = logit2angle(args, pred_angle)
        gt_angle = gt_angle

    elif args.input == 'audio2vision':
        # import pdb; pdb.set_trace()
        inputs['audio_1'] = batch['audio_1'].to(device)
        inputs['audio_2'] = batch['audio_2'].to(device)
        _, pred_angle_1 = net(inputs['audio_1'], return_angle=True)
        _, pred_angle_2 = net(inputs['audio_2'], return_angle=True)
        pred_angle_1 = logit2angle(args, pred_angle_1)
        pred_angle_2 = logit2angle(args, pred_angle_2)
        pred_angle_diff = -(pred_angle_2 - pred_angle_1)
        pred_angle_diff[pred_angle_diff >= 180] -= 360
        pred_angle_diff[pred_angle_diff < -180] += 360
        gt_angle = batch['relative_camera1_angle'].to(device)
        pred_angle = pred_angle_diff

    return {
        'pred': pred_angle,
        'gt': gt_angle.float()
    }


def evaluate_angle_difference_by_dot_product(res):
    # import pdb; pdb.set_trace()
    pred = res['pred'] / 180 * np.pi
    gt = res['gt'] / 180 * np.pi
    pred_vec = [torch.cos(pred).unsqueeze(-1), torch.sin(pred).unsqueeze(-1)]
    pred_vec = torch.cat(pred_vec, dim=-1)
    gt_vec = [torch.cos(gt).unsqueeze(-1), torch.sin(gt).unsqueeze(-1)]
    gt_vec = torch.cat(gt_vec, dim=-1)
    dot_product = (pred_vec * gt_vec).sum(-1)
    dot_product = torch.clamp(dot_product, min=-1, max=1)
    angle_diff = torch.acos(dot_product) / np.pi * 180.0
    # angle_diff = angle_diff.mean()
    return angle_diff


def prompt_rotation_direction(args, pr, net, device='cuda'):
    # import pdb; pdb.set_trace()
    pr.dataloader = 'SLfMbaseDataset'
    dataset, data_loader = torch_utils.get_dataloader(args, pr, split='val', shuffle=False, drop_last=False, batch_size=12)
    data_loader.dataset.rng = np.random.default_rng(pr.seed)
    net.eval()
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            # import pdb; pdb.set_trace()
            img_1 = batch['img_1'].to(device)
            img_2 = batch['img_2'].to(device)
            gt_angle = batch['relative_camera1_angle'].to(device)
            _, pred_angle = net(img_2, img_1, return_angle=True)
            pred_angle = rot2theta(args, pred_angle)
            acc = (torch.sign(pred_angle) == torch.sign(gt_angle)).float().mean()
            pr.rotation_correctness = 1 if acc >=0.5 else -1
            tqdm.write(f'Rotation correctness: {pr.rotation_correctness}')
            break
    torch.cuda.empty_cache()
    return


def inference(args, pr, net, data_loader, device='cuda'):
    # import pdb; pdb.set_trace()
    data_loader.dataset.rng = np.random.default_rng(pr.seed)
    net.eval()
    res = {}
    final_res = {}
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"{args.exp} - Evaluation"):
            # import pdb; pdb.set_trace()
            out = predict(args, pr, net, batch, device)
            for key in out.keys():
                if key not in res.keys():
                    res[key] = torch.tensor([]).to(device)
                res[key] = torch.cat([res[key], out[key]], dim=0)

        angle_diff = evaluate_angle_difference_by_dot_product(res)
        
        final_res['mean angle difference'] = angle_diff.mean().item()
        final_res['median angle difference'] = angle_diff.median().item()
        visualize_prediction(args, res)
        visualize_error(args, res, angle_diff)
    torch.cuda.empty_cache()
    return final_res


def test(args, device):
    # save dir
    gpus = torch.cuda.device_count()
    gpu_ids = list(range(gpus))
    # ----- get parameters ----- #
    fn = getattr(params, args.setting)
    pr = fn()
    update_param(args, pr)
    pr.seed = 2023
    if args.setting in ['slfm_hm3d']:
        pr.list_test = 'data/AI-Habitat/data-split/hm3d-4view-rotation-filterangle/test.csv'
    pr.rotation_correctness = 1.0
    # ----- make dirs for checkpoints ----- #
    sys.stdout = utils.LoggerOutput(os.path.join('results', args.exp, 'log.txt'))
    os.makedirs('./results/' + args.exp, exist_ok=True)
    # ------------------------------------- #
    tqdm.write('{}'.format(args)) 
    tqdm.write('{}'.format(pr))
    # ------------------------------------ #
    # ----- Dataset and Dataloader ----- #
    test_dataset, test_loader = torch_utils.get_dataloader(args, pr, split='test', shuffle=False, drop_last=False)
    # --------------------------------- #
    
    # ----- Network and Loading weights ----- #
    net = build_net(args, pr).to(device)
    net = nn.DataParallel(net, device_ids=gpu_ids)
    # --------------------- #

    #  --------- Inference ------------ #
    if args.input == 'vision':
        prompt_rotation_direction(args, pr, net, device)

    res = inference(args, pr, net, test_loader, device)
    tqdm.write('{}'.format(res))


# Usage: CUDA_VISIBLE_DEVICES=1 python evaluation/evaluate_angle.py --exp='test' --setting='slfm_hm3d' --vision_backbone='resnet18' --audio_backbone='resnet18' --batch_size=12 --num_workers=4  --n_source=1 --n_view=2 --online_render  --add_geometric --input='audio' --activation='tanh' --weights='HM3D-Rotation-Geometric-New/EXP2.0-UNetMtoB-Online-N=1-2View-Freeze-Tanh-g1b1s0/audio_best.pth.tar'


if __name__ == '__main__':
    parser = init_args(return_parser=True)
    parser.add_argument('--input', type=str, default='audio', required=False)

    args = parser.parse_args()
    test(args, DEVICE)