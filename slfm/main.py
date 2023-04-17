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

from config import init_args, params
# from data import *
import data
import models
from models import *
from utils import utils, torch_utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_param(args, pr):
    # Change network to geometric net
    if args.add_geometric:
        pr.net = pr.geometric_net
    
    for attr in vars(pr).keys():
        if attr in vars(args).keys():
            attr_args = getattr(args, attr)
            if attr_args is not None:
                setattr(pr, attr, attr_args)


def validation(args, pr, net, data_loader, device='cuda', status='Val'):
    # import pdb; pdb.set_trace()
    # re-init the seed for same sampling
    data_loader.dataset.rng = np.random.default_rng(pr.seed)
    net.eval()
    res = {}
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"{args.exp} - {status}"):
            out = predict(args, pr, net, batch, device, evaluate=True)
            for key in out.keys():
                if key not in res.keys():
                    res[key] = torch.tensor([]).to(device)
                res[key] = torch.cat([res[key], out[key].view(1, -1)], dim=0)

    for key in res.keys():
        res[key] = torch.mean(res[key]).item()
    torch.cuda.empty_cache()
    # net.train()
    return res

def evaluation(args, device):
    gpus = torch.cuda.device_count()
    gpu_ids = list(range(gpus))

    # ----- get parameters for audio ----- #
    fn = getattr(params, args.setting)
    pr = fn()
    update_param(args, pr)
    # ----- make dirs for checkpoints ----- #
    checkpoint_path = 'checkpoints'
    sys.stdout = utils.LoggerOutput(os.path.join(checkpoint_path, args.exp, 'log_eval.txt'))
    os.makedirs(os.path.join(checkpoint_path, args.exp), exist_ok=True)

    # ------------------------------------- #
    tqdm.write('{}'.format(args))
    tqdm.write('{}'.format(pr))
    # ------------------------------------ #

    # ----- Network and Loading weights ----- #
    torch_utils.init_random_seed(pr)

    net_vision = build_vision_net(args, pr)
    net_audio = build_audio_net(args, pr)
    
    args.weights = f'{args.exp}/pretext_best.pth.tar' # path to the best model
    if not os.path.exists('./checkpoints/' + args.weights):
        tqdm.write('No best checkpoint found. Use Random Instead')
        args.weights = ''
    net = build_pretext_net(args, pr, (net_vision, net_audio)).to(device)
    net = nn.DataParallel(net, device_ids=gpu_ids)
    # --------------------- #

    # ----- Dataset and Dataloader ----- #
    test_dataset, test_loader = torch_utils.get_dataloader(args, pr, split='test', shuffle=False, drop_last=True)
    # --------------------------------- #

    #  --------- Evaluation ------------ #
    tqdm.write('Start Evaluation on Test set...')
    res = validation(args, pr, net, test_loader, device, status='Eval')
    tqdm.write(f"Evaluation results: {res}\n")


def predict(args, pr, net, batch, device, evaluate=False, loss=False):
    inputs = {}
    for i in range(1, args.n_view + 1):
        inputs[f'img_{i}'] = batch[f'img_{i}'].to(device)
        inputs[f'audio_{i}'] = batch[f'audio_{i}'].to(device)
    
    if args.use_gt_rotation:
        for i in range(1, args.n_view):
            inputs[f'relative_camera{i}_angle'] = batch[f'relative_camera{i}_angle'].to(device)

    out = net(inputs, loss=loss, evaluate=evaluate)
    return out


def build_audio_net(args, pr):
    net_audio = models.__dict__[pr.audio_net](args, pr)
    if args.weights_audio:
        resume = os.path.join('./checkpoints', args.weights_audio)
        net_audio, _ = torch_utils.load_model(resume, net_audio, strict=True, model_name='audio')
    return net_audio


def build_vision_net(args, pr):
    net_vision = models.__dict__[pr.vision_net](args, pr)
    if args.weights_vision:
        resume = os.path.join('./checkpoints', args.weights_vision)
        net_vision, _ = torch_utils.load_model(resume, net_vision, strict=True, model_name='vision')
    return net_vision


def build_pretext_net(args, pr, nets):
    net = models.__dict__[pr.net](args, pr, nets)
    if args.weights:
        resume = os.path.join('./checkpoints', args.weights)
        strict = False if args.add_geometric else True
        net, _ = torch_utils.load_model(resume, net, strict=strict, model_name='pretext')
    return net


def save_model(args, pr, epoch, step, net_vision, net_audio, net_pretext, optimizer, best_info, res):
    latest_score = net_pretext.module.score_model_performance(res)
    # current_best_score = best_info['score']
    model_zip = zip(['vision', 'audio', 'pretext', 'optimizer'], [net_vision, net_audio, net_pretext, optimizer])
    for name, net in model_zip:
        if (epoch + 1) % args.save_step == 0:
            save_path = os.path.join('./checkpoints', args.exp, f'{name}_latest.pth.tar')
            torch.save({
                    'epoch': epoch + 1,
                    'step': step,
                    'state_dict': net.state_dict(),
            }, save_path)
        
        if (epoch + 1) % args.valid_step == 0:
            if latest_score > best_info['score']:
                save_path = os.path.join('./checkpoints', args.exp, f'{name}_best.pth.tar')
                torch.save({
                        'epoch': epoch + 1,
                        'step': step,
                        'state_dict': net.state_dict(),
                    }, save_path)

    if latest_score > best_info['score']:
        best_info['score'] = latest_score
        best_info['epoch'] = epoch + 1
        best_info['res'] = res
    
    return best_info


def train(args, device):
    gpus = torch.cuda.device_count()
    gpu_ids = list(range(gpus))

    # ----- get parameters for audio ----- #
    fn = getattr(params, args.setting)
    pr = fn()
    update_param(args, pr)
    # ----- make dirs for checkpoints ----- #
    sys.stdout = utils.LoggerOutput(os.path.join('checkpoints', args.exp, 'log.txt'))
    os.makedirs('./checkpoints/' + args.exp, exist_ok=True)

    writer = SummaryWriter(os.path.join('./checkpoints', args.exp, 'visualization'))
    # ------------------------------------- #
    
    tqdm.write('{}'.format(args)) 
    tqdm.write('{}'.format(pr))
    # ------------------------------------ #

    # ----- Network and Loading weights ----- #
    torch_utils.init_random_seed(pr)

    net_vision = build_vision_net(args, pr)
    net_audio = build_audio_net(args, pr)
    net = build_pretext_net(args, pr, (net_vision, net_audio)).to(device)

    optimizer = torch_utils.make_optimizer(net, args)
    net = nn.DataParallel(net, device_ids=gpu_ids)
    # --------------------- #

    # ----- Dataset and Dataloader ----- #
    train_dataset, train_loader = torch_utils.get_dataloader(args, pr, split='train', shuffle=True, drop_last=True)
    val_dataset, val_loader = torch_utils.get_dataloader(args, pr, split='val', shuffle=False, drop_last=True)
    # --------------------------------- #
    
    #  --------- Random or resume validation ------------ #
    res = validation(args, pr, net, val_loader, device)
    writer.add_scalars('SLfM' + '/validation', res, args.start_epoch)
    tqdm.write("Beginning, Validation results: {}".format(res))
    tqdm.write('\n')

    best_info = {
        'score': net.module.score_model_performance(res),
        'epoch': args.start_epoch,
        'res': res
    }

    # ----------------- Training ---------------- #
    # import pdb; pdb.set_trace()
    
    VALID_STEP = args.valid_step
    for epoch in range(args.start_epoch, args.epochs):
        net.train()
        running_loss = 0.0
        torch_utils.adjust_learning_rate(optimizer, epoch, args, pr)
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"{args.exp} - Training"):
            out = predict(args, pr, net, batch, device, loss=True)
            loss = out.mean()      
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 1 == 0:
                tqdm.write("Epoch: {}/{}, step: {}/{}, loss: {}".format(epoch+1, args.epochs, step+1, len(train_loader), loss))
                running_loss += loss.item()

            current_step = epoch * len(train_loader) + step + 1
            BOARD_STEP = 10
            if (step+1) % BOARD_STEP == 0:
                writer.add_scalar('SLfM' + '/training loss', running_loss / BOARD_STEP, current_step)
                running_loss = 0.0
            
        # torch.cuda.empty_cache()
        # ----------- Validtion -------------- #
        if (epoch + 1) % VALID_STEP == 0:
            res = validation(args, pr, net, val_loader, device)
            writer.add_scalars('SLfM' + '/validation', res, epoch + 1)
            tqdm.write("Epoch: {}/{}, Validation results: {}".format(epoch + 1, args.epochs, res))
            # tqdm.write('\n')

        # ---------- Save model ----------- #
        best_info = save_model(args, pr, epoch, current_step, net_vision, net_audio, net, optimizer, best_info, res)
        tqdm.write(f'Current Best model: {best_info}.')
        # --------------------------------- #
    torch.cuda.empty_cache()
    tqdm.write('Training Complete!')
    tqdm.write(f'Best model: {best_info}.')
    tqdm.write('\n')

    writer.close()


if __name__ == '__main__':
    args = init_args()
    if not args.test_mode:
        train(args, DEVICE)

    if args.eval:
        evaluation(args, DEVICE)
