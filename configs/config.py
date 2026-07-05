import os
import torch

# =====================================================
# Project Root
# =====================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =====================================================
# Dataset
# =====================================================

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

# Windows nên để 0 để tránh lỗi DataLoader.
# Khi chuyển sang Colab/Kaggle có thể tăng lên 2 hoặc 4.
NUM_WORKERS = 0

EPOCHS = 30

# Tự động chọn GPU nếu có, ngược lại dùng CPU.
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

CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoint")

# =====================================================
# Logs
# =====================================================

LOG_DIR = os.path.join(ROOT_DIR, "logs")