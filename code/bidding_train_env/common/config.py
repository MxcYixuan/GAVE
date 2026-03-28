"""Configuration management for GAVE project.

This module provides default configurations and utilities for managing
training and evaluation parameters.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    n_ctx: int = 1024
    n_embd: int = 512
    n_layer: int = 8
    n_head: int = 16
    n_inner: int = 1024
    activation_function: str = "relu"
    n_position: int = 1024
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""
    step_num: int = 10000
    save_step: int = 5000
    batch_size: int = 128
    learning_rate: float = 0.0001
    time_dim: int = 8
    hidden_size: int = 512
    expectile: float = 0.99
    loss_report: int = 100
    budget_rate: float = 1.0


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    device: str = "cuda:0"
    state_dim: int = 16
    act_dim: int = 1
    max_ep_len: int = 96
    scale: int = 2000
    K: int = 20


def get_default_config(
    data_dir: Optional[str] = None,
    device: str = "cuda:0",
    test_csv: str = "period-7.csv"
) -> Dict[str, Any]:
    """Get default configuration for training and evaluation.

    Args:
        data_dir: Path to data directory. Defaults to parent of code directory.
        device: Device to use for training.
        test_csv: Test data filename.

    Returns:
        Configuration dictionary.
    """
    if data_dir is None:
        code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(os.path.dirname(code_dir), "data")

    return {
        "training": TrainingConfig().__dict__,
        "model": ModelConfig().__dict__,
        "evaluation": EvaluationConfig().__dict__,
        "data": {
            "dir": os.path.join(data_dir, "trajectory/trajectory_data.csv"),
            "test_csv": os.path.join(data_dir, "traffic", test_csv),
        },
        "device": device,
        "save_dir": None,  # Will be set dynamically
    }


def merge_config(base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge override config into base config.

    Args:
        base_config: Base configuration.
        overrides: Override values.

    Returns:
        Merged configuration.
    """
    config = base_config.copy()
    for key, value in overrides.items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    return config