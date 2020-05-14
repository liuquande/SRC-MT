import os
import sys
# from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import pandas as pd

import torch
from torch.nn import functional as F

from utils.metrics import compute_AUCs, compute_metrics, compute_metrics_test
from utils.metric_logger import MetricLogger

def epochVal(model, dataLoader, loss_fn, args):
    training = model.training
    model.eval()

    meters = MetricLogger()

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    
    gt_study   = {}
    pred_study = {}
    studies    = []
    
    with torch.no_grad():
        for i, (study, image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            _, output = model(image)
            # _, output = model(image)

            loss = loss_fn(output, label.clone())
            meters.update(loss=loss)
            
            output = F.softmax(output, dim=1)

            for i in range(len(study)):
                if study[i] in pred_study:
                    assert torch.equal(gt_study[study[i]], label[i])
                    pred_study[study[i]] = torch.max(pred_study[study[i]], output[i])
                else:
                    gt_study[study[i]] = label[i]
                    pred_study[study[i]] = output[i]
                    studies.append(study[i])

            # gt = torch.cat((gt, label), 0)
            # pred = torch.cat((pred, output), 0)
        
        for study in studies:
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)
        
        AUROCs = compute_AUCs(gt, pred, competition=True)
    
    model.train(training)

    return meters.loss.global_avg, AUROCs


def epochVal_metrics(model, dataLoader):
    training = model.training
    model.eval()

    meters = MetricLogger()

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    
    gt_study   = {}
    pred_study = {}
    studies    = []
    
    with torch.no_grad():
        for i, (study, _, image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            _, output = model(image)
            output = F.softmax(output, dim=1)
            # _, output = model(image)

            for i in range(len(study)):
                if study[i] in pred_study:
                    assert torch.equal(gt_study[study[i]], label[i])
                    pred_study[study[i]] = torch.max(pred_study[study[i]], output[i])
                else:
                    gt_study[study[i]] = label[i]
                    pred_study[study[i]] = output[i]
                    studies.append(study[i])

            # gt = torch.cat((gt, label), 0)
            # pred = torch.cat((pred, output), 0)
        
        for study in studies:
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)

        AUROCs, Accus, Senss, Specs = compute_metrics(gt, pred, competition=True)
    
    model.train(training)

    return AUROCs, Accus, Senss, Specs

def epochVal_metrics_test(model, dataLoader):
    training = model.training
    model.eval()

    meters = MetricLogger()

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    
    gt_study   = {}
    pred_study = {}
    studies    = []
    
    with torch.no_grad():
        for i, (study, _, image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            _, output = model(image)
            output = F.softmax(output, dim=1)
            # _, output = model(image)

            for i in range(len(study)):
                if study[i] in pred_study:
                    assert torch.equal(gt_study[study[i]], label[i])
                    pred_study[study[i]] = torch.max(pred_study[study[i]], output[i])
                else:
                    gt_study[study[i]] = label[i]
                    pred_study[study[i]] = output[i]
                    studies.append(study[i])

            # gt = torch.cat((gt, label), 0)
            # pred = torch.cat((pred, output), 0)
        
        for study in studies:
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)

        AUROCs, Accus, Senss, Specs, pre, F1 = compute_metrics_test(gt, pred, competition=True)
    
    model.train(training)

    return AUROCs, Accus, Senss, Specs, pre,F1