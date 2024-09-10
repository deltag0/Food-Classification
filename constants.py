import torch
import os
from pathlib import Path

BATCH_SIZE = 64
EPOCHS = 30
DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = Path(DIR + "\\train")
TEST_DIR = Path(DIR + "\\test")
DEVICE = "cpu"
MODEL_NAME = "AlexNet"  # Model name can be AlexNet OR VGGnet
