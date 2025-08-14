import typing as t
import torch
import torch.nn as nn
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import warnings
import torch
import torch.nn as nn
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import warnings
import numpy as np
import torchvision
import torchvision.transforms as transforms

def input_adaptation1(img):
    img = img.unsqueeze(0)
    conv_layer = nn.Conv2d(3, 32, kernel_size=1)
    img = conv_layer(img)
    return img

def input_adaptation2(img):
    img = img.unsqueeze(0)
    # conv_layer = nn.Conv2d(3, 32, kernel_size=1)
    # img = conv_layer(img)
    return img