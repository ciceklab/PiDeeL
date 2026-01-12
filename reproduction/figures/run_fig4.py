#!/usr/bin/env python3
"""
Figure 4: Random Connections Comparison

Checks for required logs from test1, test2, test14 and generates them if missing,
then plots the figure comparing PiDeeL with randomly connected and shuffled networks.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
LOG_CHECKS = [
    ("test_logs/test1/2layer/no_pathway/c_indices.txt", "test_scripts/test1/2layer/no_pathway/main.py"),
    ("test_logs/test1/3layer/no_pathway/c_indices.txt", "test_scripts/test1/3layer/no_pathway/main.py"),
    ("test_logs/test1/4layer/no_pathway/c_indices.txt", "test_scripts/test1/4layer/no_pathway/main.py"),
    ("test_logs/test2/2layer/no_pathway/c_indices.txt", "test_scripts/test2/2layer/no_pathway/main.py"),
    ("test_logs/test2/3layer/no_pathway/c_indices.txt", "test_scripts/test2/3layer/no_pathway/main.py"),
    ("test_logs/test2/4layer/no_pathway/c_indices.txt", "test_scripts/test2/4layer/no_pathway/main.py"),
    ("test_logs/test14/2layer/no_pathway/c_indices.txt", "test_scripts/test14/2layer/no_pathway/main.py"),
    ("test_logs/test14/3layer/no_pathway/c_indices.txt", "test_scripts/test14/3layer/no_pathway/main.py"),
    ("test_logs/test14/4layer/no_pathway/c_indices.txt", "test_scripts/test14/4layer/no_pathway/main.py"),
]

def check_and_run_experiments():
    """Check for missing logs and run corresponding experiments."""
    missing = []
    for log_path, script_path in LOG_CHECKS:
        full_log = REPO_ROOT / log_path
        if not full_log.exists():
            missing.append((log_path, script_path))
    
    if missing:
        print(f"Missing {len(missing)} log files. Running experiments...")
        for log_path, script_path in missing:
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
                print(f"✓ Generated: {log_path}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to generate {log_path}")
                print(f"  Error: {e}")
                return False
    else:
        print("✓ All required logs exist")
    
    return True

def plot_figure():
    """Run the figure plotting script."""
    print(f"\n{'='*60}")
    print("Generating Figure 4: Random Connections Comparison")
    print(f"{'='*60}\n")
    
    script = REPO_ROOT / "figures" / "fig4_random_connections.py"
    try:
        subprocess.run(
            [sys.executable, str(script)],
            check=True,
            cwd=str(script.parent)
        )
        print("\n✓ Figure 4 generated successfully!")
        print(f"  Output: figures/fig4_random_connections.png")
        print(f"  Output: figures/fig4_random_connections.tiff")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to generate figure")
        print(f"  Error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Figure 4 Runner: Random Connections Comparison")
    print("="*60)
    
    if check_and_run_experiments():
        if plot_figure():
            sys.exit(0)
    
    sys.exit(1)
