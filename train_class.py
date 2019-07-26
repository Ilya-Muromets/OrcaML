from utils.datasets import *
from utils.classifier import *
import os
import time
import torch


train = SpectrogramLoader(os.getcwd() + "/data/train/")
val = SpectrogramLoader(os.getcwd() + "/data/val/")

AMRF = AutoClassMRF(batchsize=128, num_epochs=100, num_workers=16, model_name="medium_orca")
AMRF.fit(train, val)
