#!/usr/bin/env python3
"""
Supplementary Figure 9: DeepHit Loss Comparison

Checks for required logs from test29 (DeepHit loss experiments).
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
LOG_CHECKS = [
    ("test_logs/test29/2layer/no_pathway/c_indices.txt", "test_scripts/test29/2layer/no_pathway/main.py"),
    ("test_logs/test29/3layer/no_pathway/c_indices.txt", "test_scripts/test29/3layer/no_pathway/main.py"),
    ("test_logs/test29/4layer/no_pathway/c_indices.txt", "test_scripts/test29/4layer/no_pathway/main.py"),
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
    print(f"\n{'='*60}\nGenerating Supplementary Figure 9\n{'='*60}\n")
    script = REPO_ROOT / "figures" / "suppfig9_deephit.py"
    try:
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(script.parent))
        print("\n✓ Supp Fig 9 generated!")
        print("  Output: figures/suppfig9_deephit.png, suppfig9_deephit.pdf")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60 + "\nSupp Fig 9 Runner: DeepHit Comparison\n" + "="*60)
    sys.exit(0 if (check_and_run_experiments() and plot_figure()) else 1)
