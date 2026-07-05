import os

# =====================================================
# Project Root
# =====================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =====================================================
# Detect Environment
# =====================================================

IS_KAGGLE = os.path.exists("/kaggle/input")
IS_COLAB = os.path.exists("/content")

# =====================================================
# Dataset
# =====================================================

if IS_KAGGLE:

    CASIA_PATH = "/kaggle/input/datasets/ntl0601/casia-webface/casia-webface"

    LFW_PATH = "/kaggle/input/datasets/atulanandjha/lfwpeople/lfw_funneled"

elif IS_COLAB:

    # Sau này sẽ sửa nếu dùng Colab
    CASIA_PATH = "/content/datasets/casia-webface"
    LFW_PATH = "/content/datasets/lfw_funneled"

else:

    CASIA_PATH = os.path.join(ROOT_DIR, "datasets", "casia-webface")
    LFW_PATH = os.path.join(ROOT_DIR, "datasets", "lfw_funneled")

# =====================================================
# Image
# =====================================================

IMAGE_SIZE = 112
CHANNELS = 3

# =====================================================
# Training
# =====================================================

BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 30

# =====================================================
# Device
# =====================================================

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# Optimizer
# =====================================================

LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4

# =====================================================
# Embedding
# =====================================================

EMBEDDING_SIZE = 512

# =====================================================
# Checkpoint
# =====================================================

if IS_KAGGLE:
    CHECKPOINT_DIR = "/kaggle/working/checkpoint"
    LOG_DIR = "/kaggle/working/logs"
else:
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoint")
    LOG_DIR = os.path.join(ROOT_DIR, "logs")