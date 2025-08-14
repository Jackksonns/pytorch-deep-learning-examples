from model import *
import math
import torch
from torch import nn

model=nn.Sequential(InceptionTransformerEncoder(10, 512, 8, 2048, 6, 0.1, 1000), InceptionTransformerDecoder(10, 512, 8, 2048, 6, 0.1, 1000, 3))


