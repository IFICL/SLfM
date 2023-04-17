import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import argparse
import numpy as np
import time
import csv
from tqdm import tqdm
from collections import OrderedDict
import soundfile as sf
import imageio
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
import seaborn as sns
import shutil

from config import init_args, params
# from data import *
import data
import models
from models import *
from utils import utils, torch_utils



def build_audio_net(args, pr):
    net_audio = models.__dict__[pr.audio_net](args, pr)
    if args.weights_audio:
        resume = './checkpoints/' + args.weights_audio
        net_audio, _ = torch_utils.load_model(resume, net_audio, strict=True, model_name='audio')
    return net_audio


def build_vision_net(args, pr):
    net_vision = models.__dict__[pr.vision_net](args, pr)
    if args.weights_vision:
        resume = './checkpoints/' + args.weights_vision
        net_vision, _ = torch_utils.load_model(resume, net_vision, strict=True, model_name='vision')
    return net_vision


def update_param(args, pr):
    for attr in vars(pr).keys():
        if attr in vars(args).keys():
            attr_args = getattr(args, attr)
            if attr_args is not None:
                setattr(pr, attr, attr_args)


def predict(args, pr, net, batch, device, evaluate=False, loss=False, inference=False):
    inputs = {}
    inputs['img_1'] = batch['img_1'].to(device)
    inputs['img_2'] = batch['img_2'].to(device)
    inputs['audio_1'] = batch['audio_1'].to(device)
    if 'audio_2' in batch.keys():
        inputs['audio_2'] = batch['audio_2'].to(device)
    out = net(inputs, evaluate=evaluate, loss=loss, inference=inference)
    return out


def visualize_prediction(args, res):
    '''
        We visualize the prediction for each sample in an ascending order
    '''
    sns.set_style('white')
    save_folder = os.path.join('./results', args.exp)
    save_path = os.path.join(save_folder, f'{args.input}_pred_visualization.png')

    preds = res['pred'].data.cpu().numpy()
    targets = res['gt'].data.cpu().numpy()
    
    idx_list = np.argsort(targets)
    preds = preds[idx_list]
    targets = targets[idx_list]
    x = np.arange(len(targets))

    names = ['Target', 'Pred']
    res = [targets, preds]
    colors = ['#6495ed', '#ff726f']
    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=150)

    for idx in range(len(names)):
        ax1.scatter(x, res[idx], label=names[idx], marker='o', s=1, c=colors[idx])
    
    ax1.legend(ncol=1, fontsize=14, frameon=False)
    ax1.set_xlabel('Samples', fontsize=14)
    ax1.set_ylabel('Angle (degree)', fontsize=14)
    ax1.set_title('Prediction vs. Ground Truth', fontsize=14)
    plt.margins(0.03, tight=True)
    plt.savefig(save_path, bbox_inches='tight')


def visualize_error(args, res, angle_diff):
    '''
        We visualize the prediction error for each angle bin in an ascending order
    '''
    sns.set_style('white')
    save_folder = os.path.join('./results', args.exp)
    save_path = os.path.join(save_folder, f'{args.input}_error_visualization.png')

    preds = res['pred'].data.cpu().numpy()
    targets = res['gt'].data.cpu().numpy()
    angle_diff = angle_diff.data.cpu().numpy()
    idx_list = np.argsort(targets)
    preds = preds[idx_list]
    targets = targets[idx_list]
    angle_diff = angle_diff[idx_list]

    bin_width = 5
    if args.input == 'audio':
        angle_range = 90 if args.filter_sound else 180
    else:
        angle_range = 180
    x = np.arange(start=-angle_range, stop=angle_range, step=bin_width)
    bin_names = []
    average_errs = []
    for i in x:
        bin_name = f'{i}~{i+bin_width}'
        inds = ((targets >= i) & (targets < i+5)).nonzero()[0]
        if np.any(inds):
            average_err = angle_diff[inds].mean()
        else:
            average_err = -0.1
        bin_names.append(bin_name)
        average_errs.append(average_err)

    fig, ax1 = plt.subplots(figsize=(20, 6), dpi=150)
    ax1.bar(bin_names, average_errs, width=0.5, color='#ff726f')
    ax1.set_xlabel('Angle bin', fontsize=14)
    ax1.set_ylabel('Average Err (degree)', fontsize=14)
    ax1.set_title('Prediction Error vs. Angle bin', fontsize=14)
    plt.xticks(rotation=45)
    plt.margins(0.03, tight=True)
    plt.savefig(save_path, bbox_inches='tight')



