# src/dinoseg/__init__.py

"""
DINOv3-based segmentation for ISIC2018 skin lesion dataset.

This package provides:
- Pre-trained DINOv3 encoders for segmentation
- Baseline UNet architecture
- Training utilities and metrics
- Data loaders for ISIC2018
"""

__version__ = "0.1.0"
__author__ = "Abdoulaye Diallo"

from .models import UNet, Dinov3UNet, count_params
from .training import Trainer, TrainerConfig, EarlyStopping
from .utils.metrics import DiceCoef, IoU, SigmoidThreshold
from .utils.seed import set_seed

__all__ = [
    # Models
    "UNet",
    "Dinov3UNet",
    "count_params",
    # Training
    "Trainer",
    "TrainerConfig",
    "EarlyStopping",
    # Utils
    "DiceCoef",
    "IoU",
    "SigmoidThreshold",
    "set_seed",
]
