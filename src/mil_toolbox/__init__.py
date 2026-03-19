"""MIL Toolbox - Multiple Instance Learning training and visualization toolkit."""

from .data import FoldManager, FoldInfo, WSIDataset, DummyWSIDataset, EmbeddingDataset, MILDataModule
from .models import MILModel, LinearProbeModel
from .train import CrossValidationTrainer
from .inference import (
    MILPredictor,
    SlideEmbeddingCalculator,
    TITANAggregator,
)

__version__ = "0.0.1"

__all__ = [
    # Data
    "FoldManager",
    "FoldInfo",
    "WSIDataset",
    "DummyWSIDataset",
    "EmbeddingDataset",
    "MILDataModule",
    # Models
    "MILModel",
    "LinearProbeModel",
    # Training
    "CrossValidationTrainer",
    # Inference
    "MILPredictor",
    "SlideEmbeddingCalculator",
    "TITANAggregator",
]
