#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSSModel - PyTorch Implementation for CIFAR-10 Classification
============================================================

This project implements a deep learning model for image classification on the CIFAR-10 dataset.
The model uses modern techniques such as:
- RFAConv (Receptive Field Attention Convolution) for feature extraction
- SCSA (Spatial and Channel Self-Attention) for attention mechanism
- 5-Fold Cross Validation for robust training
- Mixup data augmentation for improved generalization
- Test Time Augmentation (TTA) for better inference

Author: Jackksonns
GitHub: https://github.com/Jackksonns

Files:
- training_process_optimization.py: Main training script with 5-fold cross validation
- model.py: Model architecture definitions (RFAConv, SCSA modules)
- utils.py: Utility functions
- torch_utils: Additional PyTorch utility functions
- train.py: Normal training script

Training Pipeline:
1. Loads CIFAR-10 dataset with data augmentation
2. Performs 5-fold cross-validation
3. Trains model with Mixup augmentation
4. Uses Cosine Annealing LR scheduler
5. Saves best models based on validation accuracy
6. Ensemble prediction using Test Time Augmentation (TTA)

"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union, Tuple
from model import *
import time
import torchvision.transforms as transforms
import torchvision
from utils import *
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import cv2
from scipy import stats
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torchvision import transforms, datasets
import ttach as tta
import torch_utils as tu



class SSSModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rfaconv = RFAConv(3,32,5,1)
        self.scsa = SCSA(dim=32, head_num=8, window_size=7)
        self.global_avgpool = nn.AdaptiveAvgPool2d(2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128,10)

    def forward(self, x):
        # print(f"input shape: {x.shape}")
        x = self.rfaconv(x)
        # print(f"after block rfaconv: {x.shape}")
        x = self.scsa(x)
        # print(f"after block scsa: {x.shape}")
        x = self.global_avgpool(x)
        # print(f"after global_avgpool: {x.shape}")
        x = self.flatten(x)
        # print(f"after flatten: {x.shape}")
        x = self.linear(x)
        # print(f"after linear: {x.shape}")
        return x


SSSModel = SSSModel()

# 超参数设置
FOLD = 5
EPOCHS = 50
MIXUP = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_dir = './checkpoints'
os.makedirs(save_dir, exist_ok=True)

# 定义数据增强
train_transform1 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
test_transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

# 加载完整 CIFAR-10 训练集（用于五折交叉验证）
full_train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform1)
# KFold 划分
kf = KFold(n_splits=FOLD, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(full_train_ds)):
    print(f'Start Fold{fold}...')
    # 构建当前 fold 的 train/val 子集
    train_subset = Subset(full_train_ds, train_idx)
    val_subset = Subset(
        datasets.CIFAR10(root='./data', train=True, download=False, transform=test_transform1),
        val_idx
    )

    # 创建 DataLoader
    train_dl = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4)
    val_dl   = DataLoader(val_subset,   batch_size=64, shuffle=False, num_workers=4)
    n_train, n_val = len(train_subset), len(val_subset)

    #重新初始化模型、优化器、损失函数、调度器和 Mixup 函数
    # Mixup 增强函数
    mixup_fn = tu.Mixup(prob=0.1, switch_prob=0.0, onehot=True,
                       label_smoothing=0.05, num_classes=10)
    model = SSSModel
    model.to(device)
    optimizer = torch.optim.Adam(SSSModel.parameters(), lr=1e-3)
    # 损失函数
    loss_fn = tu.SoftTargetCrossEntropy() if MIXUP else tu.LabelSmoothing(0.1)
    loss_fn_test = F.cross_entropy
    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(train_dl), eta_min=3e-4 / 20
    )
    scaler = torch.cuda.amp.GradScaler()

    model_name = f'5fold_test_fold{fold}'
    train_losses, val_losses = [], []
    train_accus, val_accus   = [], []
    best_accu = 0
    best_loss = float('inf')
    lrs = []

    for epoch in range(EPOCHS):
        t1 = time.time()
        val_accu = 0
        train_accu = 0
        train_losses_tmp = []

        # Train
        model.train()
        for x, y in train_dl:
            if MIXUP:
                # 如果开启 Mixup，则对数据做混合增强
                x, y = mixup_fn(x, y)
            x, y = x.to(device), y.to(device)
            # Forward
            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = loss_fn(pred, y)
            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()
            # Statistics 统计学习率、训练损失和训练准确率
            lrs.append(optimizer.param_groups[0]['lr'])
            train_losses_tmp.append(loss.item())
            pred_labels = torch.argmax(pred.data, dim=1)
            y_labels = torch.argmax(y.data, dim=1) if MIXUP else y.data
            train_accu += (pred_labels == y_labels).float().sum()
        train_losses.append(np.mean(train_losses_tmp))
        train_accu /= n_train
        train_accus.append(train_accu.item())

        t2 = time.time()

        # Validation
        val_losses_tmp = []
        model.eval()
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logit = model(x)
                val_loss = loss_fn_test(logit, y)
                val_losses_tmp.append(val_loss.item())
                pred = torch.argmax(logit.data, dim=1)
                val_accu += (pred == y.data).float().sum()
        val_loss = np.mean(val_losses_tmp)
        val_losses.append(val_loss)
        val_accu /= n_val
        val_accus.append(val_accu.item())

        t3 = time.time()
        print(
            'fold', fold,
            'epoch', epoch,
            'train_loss', train_losses[epoch],
            'val_loss', val_losses[epoch],
            'val_accu', val_accu.item(),
            'train_accu', train_accu.item(),
            'train time', t2 - t1,
            'val time', t3 - t2,
            'lr[0]', lrs[-1]
        )

        # 保存最优模型
        if save_dir is not None:
            if val_accu == best_accu:
                if val_loss < best_loss:
                    # checkpoint = {"model": model.state_dict()}
                    torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}_best.pth'))
                    print(f'Stored a new best model in {save_dir}')
                    best_loss = val_loss
            elif val_accu > best_accu:
                # checkpoint = {"model": model.state_dict()}
                torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}_best.pth'))
                print(f'Stored a new best model in {save_dir}')
                best_accu = val_accu

test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform1)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

# 准备 TTA Wrapper
tta_transforms = tta.aliases.d4_transform()
# 如果不想用 D4，可以改成 flip_transform、crop_transform 等：
# tta_transforms = tta.aliases.flip_transform()

# 批量载入模型并收集每个模型的预测
model_dir = './checkpoints'
model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
all_model_preds = []  # 将存放 shape=(n_samples,) 的各模型预测

for fname in sorted(model_files):
    # 构建模型结构并加载权重
    model = SSSModel()
    # model.fc = torch.nn.Linear(model.fc.in_features, 10)
    state = torch.load(os.path.join(model_dir, fname), map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # 包装 TTA
    tta_model = tta.ClassificationTTAWrapper(model, tta_transforms, merge_mode='mean')

    # 对测试集做推理，收集预测标签
    preds = []
    with torch.no_grad():
        for images, _ in test_dl:
            images = images.to(device)
            logits = tta_model(images)
            batch_pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
            preds.append(batch_pred.cpu().numpy())
    preds = np.concatenate(preds, axis=0)  # (10000,)
    all_model_preds.append(preds)

# 转成数组，shape = (n_models, n_samples)
all_model_preds = np.stack(all_model_preds, axis=0)

# 多数投票融合
#    对每个样本，取出现次数最多的标签
final_preds = stats.mode(all_model_preds, axis=0)[0].squeeze()  # (10000,)

# 计算并打印 ensemble 准确率
true_labels = np.array(test_ds.targets)  # (10000,)
accuracy = (final_preds == true_labels).mean()
print(f'Ensemble accuracy on CIFAR-10 test set: {accuracy * 100:.2f}%')
