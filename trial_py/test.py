import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from importlib import import_module
import torch.nn.functional as F
#!apt-get -y install libgl1-mesa-glx
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from time import time
import torch.utils.data as data
from torchvision import transforms  
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter

from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR


input = torch.randn(2, 3)
print(input)
m=torch.nn.Softmax(dim=-1)
print(m(input))

# b=[2,2,2]

