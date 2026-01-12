#!/usr/bin/env python3
"""
Supplementary Figure 11: Main DeepSurv vs PiDeeL Comparison

This figure uses the main experiment logs (no test experiments needed).
Checks that main logs exist and plots the comparison.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
LOG_CHECKS = [
    ("logs/2layer/pathway/c_indices.txt", "scripts/2layer/pathway/main.py"),
    ("logs/2layer/no_pathway/c_indices.txt", "scripts/2layer/no_pathway/main.py"),
    ("logs/3layer/pathway/c_indices.txt", "scripts/3layer/pathway/main.py"),
    ("logs/3layer/no_pathway/c_indices.txt", "scripts/3layer/no_pathway/main.py"),
    ("logs/4layer/pathway/c_indices.txt", "scripts/4layer/pathway/main.py"),
    ("logs/4layer/no_pathway/c_indices.txt", "scripts/4layer/no_pathway/main.py"),
]

def check_and_run_experiments():
    """Check for missing logs and run corresponding experiments."""
    missing = []
    for log_path, script_path in LOG_CHECKS:
        full_log = REPO_ROOT / log_path
        if not full_log.exists():
            missing.append((log_path, script_path))
    
    if missing:
        print(f"Missing {len(missing)} log files. Running main experiments...")
        print("TIP: You can run 'python run_all.py' to generate all main logs")
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
    print("Generating Supplementary Figure 11: DeepSurv vs PiDeeL")
    print(f"{'='*60}\n")
    
    script = REPO_ROOT / "figures" / "suppfig11_main_comparison.py"
    try:
        subprocess.run(
            [sys.executable, str(script)],
            check=True,
            cwd=str(script.parent)
        )
        print("\n✓ Supplementary Figure 11 generated successfully!")
        print(f"  Output: figures/suppfig11_main_comparison.png")
        print(f"  Output: figures/suppfig11_main_comparison.pdf")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to generate figure")
        print(f"  Error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Supplementary Figure 11 Runner: Main Comparison")
    print("="*60)
    
    if check_and_run_experiments():
        if plot_figure():
            sys.exit(0)
    
    sys.exit(1)
