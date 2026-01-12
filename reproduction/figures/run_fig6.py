#!/usr/bin/env python3
"""
Figure 6: External Validation

NOTE: This figure requires external validation data and trained models.
It will check for required models but cannot automatically generate the external test data.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

def check_requirements():
    """Check if external validation data and models exist."""
    print("Checking requirements for Figure 6...")
    
    missing = []
    
    # Check for external test data
    test_data = REPO_ROOT / "test_scripts" / "test31" / "quant_test.pickle"
    if not test_data.exists():
        missing.append(f"External test data: {test_data}")
    
    # Check for trained models (example paths - actual paths may vary)
    model_paths = [
        "test_scripts/test31/PiDeeL_2layer.pth",
        "test_scripts/test31/PiDeeL_3layer.pth",
        "test_scripts/test31/PiDeeL_4layer.pth",
    ]
    
    for model_path in model_paths:
        full_path = REPO_ROOT / model_path
        if not full_path.exists():
            missing.append(f"Model: {model_path}")
    
    if missing:
        print("\n⚠️  Missing required files:")
        for item in missing:
            print(f"  - {item}")
        print("\nFigure 6 requires:")
        print("  1. External validation data (quant_test.pickle)")
        print("  2. Trained PiDeeL models for each layer")
        print("  3. DeepHit model from test29")
        print("  4. PC-Hazard model from test30")
        print("\nPlease provide these files before running this figure.")
        return False
    
    print("✓ All required files found")
    return True

def plot_figure():
    """Run the figure plotting script."""
    print(f"\n{'='*60}\nGenerating Figure 6: External Validation\n{'='*60}\n")
    
    # Note: This script requires a --layer argument
    print("NOTE: This script requires a --layer argument (2, 3, or 4)")
    print("Running with layer=4 as default...\n")
    
    script = REPO_ROOT / "figures" / "fig6_external_validation.py"
    try:
        subprocess.run(
            [sys.executable, str(script), "--layer", "4"],
            check=True,
            cwd=str(script.parent)
        )
        print("\n✓ Figure 6 generated successfully!")
        print("  Output: figures/fig6_external_validation.png, fig6_external_validation.pdf")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to generate figure")
        print(f"  Error: {e}")
        print("\nTry running manually with:")
        print(f"  cd figures && python fig6_external_validation.py --layer [2|3|4]")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Figure 6 Runner: External Validation")
    print("="*60)
    
    if check_requirements():
        if plot_figure():
            sys.exit(0)
    
    sys.exit(1)
