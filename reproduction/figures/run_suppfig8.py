#!/usr/bin/env python3
"""
Supplementary Figure 8: Loss Function Comparison

Compares PiDeeL with different loss functions (DeepHit, PC-Hazard, DeepSurv).
Requires logs from test29 (DeepHit) and test30 (PC-Hazard).
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
LOG_CHECKS = [
    ("test_logs/test29/4layer/pathway/c_indices.txt", "test_scripts/test29/4layer/pathway/main.py"),
    ("test_logs/test30/4layer/pathway/c_indices.txt", "test_scripts/test30/4layer/pathway/main.py"),
]

def check_and_run_experiments():
    missing = []
    for log_path, script_path in LOG_CHECKS:
        if not (REPO_ROOT / log_path).exists():
            missing.append((log_path, script_path))
    
    if missing:
        print(f"Missing {len(missing)} log files. Running experiments...")
        for log_path, script_path in missing:
            full_script = REPO_ROOT / script_path
            print(f"\n{'='*60}\nRunning: {script_path}\n{'='*60}\n")
            try:
                subprocess.run([sys.executable, str(full_script)], check=True, cwd=str(full_script.parent))
                print(f"✓ Generated: {log_path}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed: {e}")
                return False
    else:
        print("✓ All required logs exist")
    return True

def plot_figure():
    print(f"\n{'='*60}\nGenerating Supplementary Figure 8\n{'='*60}\n")
    script = REPO_ROOT / "figures" / "suppfig8_loss_comparison.py"
    try:
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(script.parent))
        print("\n✓ Supp Fig 8 generated!")
        print("  Output: figures/suppfig8_loss_comparison.png, suppfig8_loss_comparison.pdf")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60 + "\nSupp Fig 8 Runner: Loss Function Comparison\n" + "="*60)
    sys.exit(0 if (check_and_run_experiments() and plot_figure()) else 1)
