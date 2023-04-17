from collections import OrderedDict
import os
import numpy as np
import random
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append('..')
import data





# ---------------------------------------------------- #
def binary_acc(pred, target, thred):
    pred = pred > thred
    acc = np.sum(pred == target) / target.shape[0]
    return acc

def calc_acc(prob, labels, k):
    # import pdb; pdb.set_trace()
    pred = torch.argsort(prob, dim=-1, descending=True)[..., :k]
    top_k_acc = torch.sum(pred == labels.view(-1, 1)).float() / labels.size(0)
    return top_k_acc

# ---------------------------------------------------- #

def get_dataloader(args, pr, split='train', shuffle=False, drop_last=False, batch_size=None):
    data_loader = getattr(data, pr.dataloader)
    if split == 'train':
        read_list = pr.list_train
    elif split == 'val':
        read_list = pr.list_val
    elif split == 'test':
        read_list = pr.list_test
    dataset = data_loader(args, pr, read_list, split=split)
    batch_size = batch_size if batch_size else args.batch_size
    
    if args.save_audio:
        for i in range(20):
            dataset.getitem_test(i)
        exit()

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=drop_last)
    
    return dataset, loader


# ---------------------------------------------------- #
def make_optimizer(model, args):
    '''
    Args:
        model: NN to train
    Returns:
        optimizer: pytorch optmizer for updating the given model parameters.
    '''
    # import pdb; pdb.set_trace()
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=False
        )
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            # weight_decay=0,
            # weight_decay=args.weight_decay,
        )
    
    args.start_epoch = 0

    if args.weights_optim: 
        resume = './checkpoints/' + args.weights_optim
        if os.path.isfile(resume):
            checkpoint = torch.load(resume)
            optim_state = checkpoint['state_dict']
            optimizer.load_state_dict(optim_state)
            args.start_epoch = checkpoint['epoch']
            tqdm.write(f"=> loaded optimizer weights from '{resume}' (epoch {checkpoint['epoch']})")
        else:
            tqdm.write(f"=> no checkpoint found at {resume}")
            sys.exit()
    return optimizer


def adjust_learning_rate(optimizer, epoch, args, pr):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.schedule == 'cos':  # cosine lr schedule
        lr *= 0.5 * (1. + np.cos(np.pi * epoch / args.epochs))
    elif args.schedule == 'step':  # stepwise lr schedule
        for milestone in pr.lr_milestones:
            lr *= pr.gamma if epoch >= milestone else 1.
    elif args.schedule == 'none':  # no lr schedule
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# ---------------------------------------------------- #
def load_model(cp_path, net, device=None, strict=True, model_name=''): 
    if not device:
        device = torch.device('cpu')
    if os.path.isfile(cp_path): 
        print("=> loading checkpoint '{}'".format(cp_path))
        checkpoint = torch.load(cp_path, map_location=device)

        # check if there is module
        if list(checkpoint['state_dict'].keys())[0][:7] == 'module.': 
            state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items(): 
                name = k[7:]
                state_dict[name] = v
        else: 
            state_dict = checkpoint['state_dict']
        net.load_state_dict(state_dict, strict=strict) 

        if model_name != '':
            model_name += ' '

        print(f"=> loaded {model_name}weights from '{cp_path}' (epoch {checkpoint['epoch']})")
        start_epoch = checkpoint['epoch']
    else: 
        print(f"=> no checkpoint found at {cp_path}")
        start_epoch = 0
        sys.exit()
    
    return net, start_epoch

def save_model(args, pr, epoch, step, net, optimizer, best_score, latest_score):
    if (epoch + 1) % args.save_step == 0:
        path = os.path.join('./checkpoints', args.exp, 'checkpoint_latest.pth.tar')
        torch.save(
            {'epoch': epoch + 1,
            'step': step,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, path)
    
    if (epoch + 1) % args.valid_step == 0:
        if latest_score > best_score:
            torch.save(
                {'epoch': epoch + 1,
                'step': step,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                },
                os.path.join('./checkpoints', args.exp, 'checkpoint_best.pth.tar'))
            best_score = latest_score
    
    return best_score


def init_random_seed(pr):
    # init random seed
    random.seed(pr.seed)
    np.random.seed(pr.seed)
    torch.manual_seed(pr.seed)
