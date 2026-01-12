"""
PiDeeL Configuration Module

Central configuration for all paths and settings.
All paths are relative to the repository root.
"""
import os
from pathlib import Path
import torch

# =============================================================================
# Repository Root Detection
# =============================================================================
# Get the absolute path to the repository root (where this config.py lives)
REPO_ROOT = Path(__file__).parent.absolute()

# =============================================================================
# Directory Paths
# =============================================================================
DATA_DIR = REPO_ROOT / "pideel_data" / "targeted"
LOGS_DIR = REPO_ROOT / "logs"
MODELS_DIR = REPO_ROOT / "models"
PLOTS_DIR = REPO_ROOT / "plots"
FIGURES_DIR = REPO_ROOT / "figures"
SCRIPTS_DIR = REPO_ROOT / "scripts"
TESTS_DIR = REPO_ROOT / "tests"

# =============================================================================
# Device Configuration
# =============================================================================
# Automatically use GPU if available, otherwise fall back to CPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Helper Functions
# =============================================================================
def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_log_path(task: str, filename: str = "c_indices.txt") -> Path:
    """Get the path for a log file, creating directories if needed."""
    log_dir = LOGS_DIR / task
    ensure_dir(log_dir)
    return log_dir / filename


def get_model_path(task: str, filename: str) -> Path:
    """Get the path for a model file, creating directories if needed."""
    model_dir = MODELS_DIR / task
    ensure_dir(model_dir)
    return model_dir / filename


def get_plot_path(task: str, filename: str) -> Path:
    """Get the path for a plot file, creating directories if needed."""
    plot_dir = PLOTS_DIR / task
    ensure_dir(plot_dir)
    return plot_dir / filename
