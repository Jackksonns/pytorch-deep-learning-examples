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
# print(SSSModel)

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SSSModel.to(device)

optimizer = torch.optim.Adam(SSSModel.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

#training parameters
epochs = 10


def train():
    step = 0
    start_time = time.time()
    SSSModel.train()
    for epoch in range(epochs):
        print('epoch:', epoch)
        for data in train_loader:
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            # print('before ada2',imgs.shape)
            # imgs = input_adaptation2(imgs)
            # print('after ada2', imgs.shape)

            outputs = SSSModel(imgs)
            # print('outputs shape:', outputs.shape)
            # print('labels shape:', labels.shape)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            end_time = time.time()
            if step % 500 == 0:
                print('spend', time.time() - start_time)
                print(f'step: {step}, train loss: {loss.item()}')

        SSSModel.eval()
        total_loss, total_acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = SSSModel(imgs)
                total_loss += loss_fn(outputs, targets).item() * imgs.size(0)
                total_acc  += (outputs.argmax(1)==targets).sum().item()
                n += imgs.size(0)
                #就是先算求和，走完测试循环再除
        avg_loss = total_loss / n
        avg_acc  = total_acc  / n
        print(f'epoch {epoch} | test loss: {avg_loss:.4f}, test acc: {avg_acc:.4f}')
            
        torch.save(SSSModel.state_dict(), f'model_{epoch}.pth')

train()