import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, ToTensor, Normalize
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from time import time
import torch.utils.data as data
from torchvision import transforms  
import matplotlib.pyplot as plt
import seaborn as sns


from torchvision.models import resnet50
model = resnet50(pretrained=True)
#model
num_classes = 18
model.fc = nn.Linear(2048,18)
# Freeze except fc parts
model.conv1.requires_grad_(False)
model.bn1.requires_grad_(False)
model.layer1.requires_grad_(False)
model.layer1.requires_grad_(False)
model.layer2.requires_grad_(False)
model.layer3.requires_grad_(False)
model.layer4.requires_grad_(False)
#for param, weight in model.named_parameters():
#    print(f"param {param:20} required gradient? -> {weight.requires_grad}")

#fc init
import torch.nn.init as init
model.fc.weight.data.normal_(0,0.01)
model.fc.weight.data.zero_()