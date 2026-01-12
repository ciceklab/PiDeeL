#!/usr/bin/env python3
"""
Supplementary Figure 7: Undersampled Dataset Comparison

Checks for required logs from test28 (undersampled to 50, 100, 200 samples).
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
LOG_CHECKS = [
    ("test_logs/test28/2layer/no_pathway/c_indices_50.txt", "test_scripts/test28/2layer/no_pathway/main.py"),
    ("test_logs/test28/2layer/no_pathway/c_indices_100.txt", "test_scripts/test28/2layer/no_pathway/main100.py"),
    ("test_logs/test28/2layer/no_pathway/c_indices_200.txt", "test_scripts/test28/2layer/no_pathway/main200.py"),
    ("test_logs/test28/2layer/pathway/c_indices_50.txt", "test_scripts/test28/2layer/pathway/main.py"),
    ("test_logs/test28/2layer/pathway/c_indices_100.txt", "test_scripts/test28/2layer/pathway/main100.py"),
    ("test_logs/test28/2layer/pathway/c_indices_200.txt", "test_scripts/test28/2layer/pathway/main200.py"),
    ("test_logs/test28/3layer/no_pathway/c_indices_50.txt", "test_scripts/test28/3layer/no_pathway/main.py"),
    ("test_logs/test28/3layer/no_pathway/c_indices_100.txt", "test_scripts/test28/3layer/no_pathway/main100.py"),
    ("test_logs/test28/3layer/no_pathway/c_indices_200.txt", "test_scripts/test28/3layer/no_pathway/main200.py"),
    ("test_logs/test28/3layer/pathway/c_indices_50.txt", "test_scripts/test28/3layer/pathway/main.py"),
    ("test_logs/test28/3layer/pathway/c_indices_100.txt", "test_scripts/test28/3layer/pathway/main100.py"),
    ("test_logs/test28/3layer/pathway/c_indices_200.txt", "test_scripts/test28/3layer/pathway/main200.py"),
    ("test_logs/test28/4layer/no_pathway/c_indices_50.txt", "test_scripts/test28/4layer/no_pathway/main.py"),
    ("test_logs/test28/4layer/no_pathway/c_indices_100.txt", "test_scripts/test28/4layer/no_pathway/main100.py"),
    ("test_logs/test28/4layer/no_pathway/c_indices_200.txt", "test_scripts/test28/4layer/no_pathway/main200.py"),
    ("test_logs/test28/4layer/pathway/c_indices_50.txt", "test_scripts/test28/4layer/pathway/main.py"),
    ("test_logs/test28/4layer/pathway/c_indices_100.txt", "test_scripts/test28/4layer/pathway/main100.py"),
    ("test_logs/test28/4layer/pathway/c_indices_200.txt", "test_scripts/test28/4layer/pathway/main200.py"),
]

def check_and_run_experiments():
    script_to_logs = {}
    for log_path, script_path in LOG_CHECKS:
        if not (REPO_ROOT / log_path).exists():
            if script_path not in script_to_logs:
                script_to_logs[script_path] = []
            script_to_logs[script_path].append(log_path)
    
    if script_to_logs:
        print(f"Missing logs for {len(script_to_logs)} experiments. Running...")
        for script_path, logs in script_to_logs.items():
            full_script = REPO_ROOT / script_path
            print(f"\n{'='*60}\nRunning: {script_path}\n{'='*60}\n")
            try:
                subprocess.run([sys.executable, str(full_script)], check=True, cwd=str(full_script.parent))
                print(f"✓ Generated {len(logs)} log files")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed: {e}")
                return False
    else:
        print("✓ All required logs exist")
    return True

def plot_figure():
    print(f"\n{'='*60}\nGenerating Supplementary Figure 7\n{'='*60}\n")
    script = REPO_ROOT / "figures" / "suppfig7_undersampled.py"
    try:
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(script.parent))
        print("\n✓ Supp Fig 7 generated!")
        print("  Output: figures/suppfig7_undersampled.png, suppfig7_undersampled.pdf")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60 + "\nSupp Fig 7 Runner: Undersampled Datasets\n" + "="*60)
    sys.exit(0 if (check_and_run_experiments() and plot_figure()) else 1)
