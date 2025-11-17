"""
ED-Copilot Model Architecture
Módulos para o modelo multimodal de predição em Emergency Department
"""

from .linearization import FeatureLinearizer
from .dataset import EDCopilotDataset
from .ed_copilot_sft import EDCopilotSFT

__all__ = [
    'FeatureLinearizer',
    'EDCopilotDataset',
    'EDCopilotSFT',
]