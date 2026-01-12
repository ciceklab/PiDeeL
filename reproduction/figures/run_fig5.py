#!/usr/bin/env python3
"""
Figure 5: Dropout Probability Comparison

Checks for required logs from test13 (dropout experiments) and generates them if missing,
then plots the figure comparing different dropout probabilities.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
LOG_CHECKS = [
    ("test_logs/test13/2layer/no_pathway/0.5_c_indices.txt", "test_scripts/test13/2layer/no_pathway/main.py"),
    ("test_logs/test13/2layer/no_pathway/0.6_c_indices.txt", "test_scripts/test13/2layer/no_pathway/main.py"),
    ("test_logs/test13/2layer/no_pathway/0.7_c_indices.txt", "test_scripts/test13/2layer/no_pathway/main.py"),
    ("test_logs/test13/2layer/no_pathway/0.8_c_indices.txt", "test_scripts/test13/2layer/no_pathway/main.py"),
    ("test_logs/test13/2layer/no_pathway/0.91_c_indices.txt", "test_scripts/test13/2layer/no_pathway/main.py"),
    ("test_logs/test13/3layer/no_pathway/0.5_c_indices.txt", "test_scripts/test13/3layer/no_pathway/main.py"),
    ("test_logs/test13/3layer/no_pathway/0.6_c_indices.txt", "test_scripts/test13/3layer/no_pathway/main.py"),
    ("test_logs/test13/3layer/no_pathway/0.7_c_indices.txt", "test_scripts/test13/3layer/no_pathway/main.py"),
    ("test_logs/test13/3layer/no_pathway/0.8_c_indices.txt", "test_scripts/test13/3layer/no_pathway/main.py"),
    ("test_logs/test13/3layer/no_pathway/0.91_c_indices.txt", "test_scripts/test13/3layer/no_pathway/main.py"),
    ("test_logs/test13/4layer/no_pathway/0.5_c_indices.txt", "test_scripts/test13/4layer/no_pathway/main.py"),
    ("test_logs/test13/4layer/no_pathway/0.6_c_indices.txt", "test_scripts/test13/4layer/no_pathway/main.py"),
    ("test_logs/test13/4layer/no_pathway/0.7_c_indices.txt", "test_scripts/test13/4layer/no_pathway/main.py"),
    ("test_logs/test13/4layer/no_pathway/0.8_c_indices.txt", "test_scripts/test13/4layer/no_pathway/main.py"),
    ("test_logs/test13/4layer/no_pathway/0.91_c_indices.txt", "test_scripts/test13/4layer/no_pathway/main.py"),
]

def check_and_run_experiments():
    """Check for missing logs and run corresponding experiments."""
    # Group by script to avoid running same script multiple times
    script_to_logs = {}
    for log_path, script_path in LOG_CHECKS:
        full_log = REPO_ROOT / log_path
        if not full_log.exists():
            if script_path not in script_to_logs:
                script_to_logs[script_path] = []
            script_to_logs[script_path].append(log_path)
    
    if script_to_logs:
        print(f"Missing log files for {len(script_to_logs)} experiments. Running...")
        for script_path, logs in script_to_logs.items():
            full_script = REPO_ROOT / script_path
            print(f"\n{'='*60}")
            print(f"Running: {script_path}")
            print(f"Will generate: {', '.join([Path(l).name for l in logs])}")
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
    print("Generating Figure 5: Dropout Probability Comparison")
    print(f"{'='*60}\n")
    
    script = REPO_ROOT / "figures" / "fig5_dropout.py"
    try:
        subprocess.run(
            [sys.executable, str(script)],
            check=True,
            cwd=str(script.parent)
        )
        print("\n✓ Figure 5 generated successfully!")
        print(f"  Output: figures/fig5_dropout.png")
        print(f"  Output: figures/fig5_dropout.pdf")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to generate figure")
        print(f"  Error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Figure 5 Runner: Dropout Probability Comparison")
    print("="*60)
    
    if check_and_run_experiments():
        if plot_figure():
            sys.exit(0)
    
    sys.exit(1)
