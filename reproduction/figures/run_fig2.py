#!/usr/bin/env python3
"""
Figure 2: Main Comparison Runner

Checks for required logs (baselines, main PiDeeL models, DeepHit, PC-Hazard)
and generates them if missing, then plots Figure 2.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
LOG_CHECKS = [
    # Baseline models
    ("logs/baseline/coxph/c_indices.txt", "scripts/baseline/coxph/main.py"),
    ("logs/baseline/cwgb/c_indices.txt", "scripts/baseline/cwgb/main.py"),
    ("logs/baseline/rf/c_indices.txt", "scripts/baseline/rf/main.py"),
    
    # 4-layer models (main PiDeeL vs DeepSurv)
    ("logs/4layer/no_pathway/c_indices.txt", "scripts/4layer/no_pathway/main.py"),
    ("logs/4layer/pathway/c_indices.txt", "scripts/4layer/pathway/main.py"),

    # DeepHit (from test29)
    ("test_logs/test29/4layer/no_pathway/c_indices.txt", "test_scripts/test29/4layer/no_pathway/main.py"),
    
    # PC-Hazard (from test30)
    ("test_logs/test30/4layer/no_pathway/c_indices.txt", "test_scripts/test30/4layer/no_pathway/main.py"),
]

def check_and_run_experiments():
    """Check for missing logs and run corresponding experiments."""
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
            print(f"\n{'='*60}")
            print(f"Running: {script_path}")
            print(f"{'='*60}\n")
            try:
                subprocess.run(
                    [sys.executable, str(full_script)],
                    check=True,
                    cwd=str(full_script.parent)
                )
                print(f"✓ Generated logs for {Path(script_path).parent.name}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to run {script_path}")
                print(f"  Error: {e}")
                return False
    else:
        print("✓ All required logs exist")
    
    return True

def plot_figure():
    """Run the figure plotting script."""
    print(f"\n{'='*60}")
    print("Generating Figure 2: Main Comparison")
    print(f"{'='*60}\n")
    
    script = REPO_ROOT / "figures" / "fig2_main_comparison.py"
    try:
        subprocess.run(
            [sys.executable, str(script)],
            check=True,
            cwd=str(script.parent)
        )
        print("\n✓ Figure 2 generated successfully!")
        print(f"  Output: figures/fig2_main_comparison.png")
        print(f"  Output: figures/fig2_main_comparison.pdf")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to generate figure")
        print(f"  Error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Figure 2 Runner: Main Comparison")
    print("="*60)
    
    if check_and_run_experiments():
        if plot_figure():
            sys.exit(0)
    
    sys.exit(1)
