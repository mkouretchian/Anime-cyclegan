#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 09:48:22 2021

@author: roji
"""


import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0
LAMBDA_CYCLE = 10
NUM_WORKERS = 2
NUM_EPOCHS = 200
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_P = "genp.pth.tar"
CHECKPOINT_GEN_A = 'gena.pth.tar'
CHECKPOINT_CRITIC_P = "criticp.pth.tar"
CHECKPOINT_CRITIC_A = "critica.pth.tar"

transforms = A.Compose(
    [
    A.Resize(width = 128, height = 128),
    A.HorizontalFlip(p = 0.5),
    A.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5], max_pixel_value=255),
    ToTensorV2(),
    ],
    additional_targets = {"image0" : "image"}
    
    )