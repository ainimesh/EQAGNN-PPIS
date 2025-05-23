import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os, time
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

# Paths
import warnings
from tqdm import tqdm
from MHA_GGNNS.evalution import compute_roc, compute_aupr, compute_mcc, micro_score, acc_score, compute_performance








#! ---------------------------------------------------------------------------------------------
class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#! ---------------------------------------------------------------------------------------------

# training for one epoch

def train_epoch(model, train_loader, optimizer, scheduler, epoch, all_epochs, device, print_freq=100):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        y_pred = model(batch)

        
