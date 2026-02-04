"""MIL Toolbox - Multiple Instance Learning training and visualization toolkit."""

from .data import FoldManager, FoldInfo, WSIDataset, DummyWSIDataset, MILDataModule
from .models import MILModel
from .train import CrossValidationTrainer
from .inference import (
    MILPredictor,
    SlideEmbeddingCalculator,
)

__version__ = "0.0.1"

__all__ = [
    # Data
    "FoldManager",
    "FoldInfo",
    "WSIDataset",
    "DummyWSIDataset",
    "MILDataModule",
    # Models
    "MILModel",
    # Training
    "CrossValidationTrainer",
    # Inference
    "MILPredictor",
    "SlideEmbeddingCalculator",
]
