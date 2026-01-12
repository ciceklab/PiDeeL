#!/usr/bin/env python3
"""
Supplementary Figure 3: Single vs Multitask Learning

Requires metrics from test5 (single task) and test7 (multitask) experiments.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
LOG_CHECKS = [
    # test5 single task AUROC/AUPR
    ("test_logs/test5/2layer/pathway/auroc.txt", "test_scripts/test5/2layer/pathway/main.py"),
    ("test_logs/test5/2layer/pathway/aupr.txt", "test_scripts/test5/2layer/pathway/main.py"),
    ("test_logs/test5/3layer/pathway/auroc.txt", "test_scripts/test5/3layer/pathway/main.py"),
    ("test_logs/test5/3layer/pathway/aupr.txt", "test_scripts/test5/3layer/pathway/main.py"),
    ("test_logs/test5/4layer/pathway/auroc.txt", "test_scripts/test5/4layer/pathway/main.py"),
    ("test_logs/test5/4layer/pathway/aupr.txt", "test_scripts/test5/4layer/pathway/main.py"),
    # test7 multitask C-indices and metrics
    ("test_logs/test7/2layer/pathway/c_indices.txt", "test_scripts/test7/2layer/pathway/main.py"),
    ("test_logs/test7/2layer/pathway/auroc.txt", "test_scripts/test7/2layer/pathway/main.py"),
    ("test_logs/test7/2layer/pathway/aupr.txt", "test_scripts/test7/2layer/pathway/main.py"),
    ("test_logs/test7/3layer/pathway/c_indices.txt", "test_scripts/test7/3layer/pathway/main.py"),
    ("test_logs/test7/3layer/pathway/auroc.txt", "test_scripts/test7/3layer/pathway/main.py"),
    ("test_logs/test7/3layer/pathway/aupr.txt", "test_scripts/test7/3layer/pathway/main.py"),
    ("test_logs/test7/4layer/pathway/c_indices.txt", "test_scripts/test7/4layer/pathway/main.py"),
    ("test_logs/test7/4layer/pathway/auroc.txt", "test_scripts/test7/4layer/pathway/main.py"),
    ("test_logs/test7/4layer/pathway/aupr.txt", "test_scripts/test7/4layer/pathway/main.py"),
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
    print(f"\n{'='*60}\nGenerating Supplementary Figure 3\n{'='*60}\n")
    script = REPO_ROOT / "figures" / "suppfig3_multitask.py"
    try:
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(script.parent))
        print("\n✓ Supp Fig 3 generated!")
        print("  Output: figures/suppfig3_multitask.png, suppfig3_multitask.pdf")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60 + "\nSupp Fig 3 Runner: Multitask Learning\n" + "="*60)
    sys.exit(0 if (check_and_run_experiments() and plot_figure()) else 1)
