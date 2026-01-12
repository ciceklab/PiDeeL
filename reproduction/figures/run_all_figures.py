#!/usr/bin/env python3
"""
Master Figure Runner

Generates all paper figures by checking for required logs and running
necessary experiments before plotting each figure.

Usage:
    python run_all_figures.py              # Generate all figures
    python run_all_figures.py --figure 4   # Generate specific figure
"""
import subprocess
import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Define all figures and their dependencies
FIGURES = {
    "fig2": {
        "name": "Figure 2: Main Comparison",
        "script": "fig2_main_comparison.py",
        "runner": "run_fig2.py",
        "outputs": ["fig2_main_comparison.png", "fig2_main_comparison.pdf"],
    },
    "fig4": {
        "name": "Figure 4: Random Connections",
        "script": "fig4_random_connections.py",
        "runner": "run_fig4.py",
        "outputs": ["fig4_random_connections.png", "fig4_random_connections.tiff"],
    },
    "fig5": {
        "name": "Figure 5: Dropout Comparison",
        "script": "fig5_dropout.py",
        "runner": "run_fig5.py",
        "outputs": ["fig5_dropout.png", "fig5_dropout.pdf"],
    },
    "fig6": {
        "name": "Figure 6: External Validation",
        "script": "fig6_external_validation.py",
        "runner": "run_fig6.py",  # Complex, requires external data
        "outputs": ["fig6_external_validation.png", "fig6_external_validation.pdf"],
        "note": "Requires external validation data (quant_test.pickle) and trained models"
    },
    "suppfig2": {
        "name": "Supplementary Figure 2: AUROC/AUPR",
        "script": "suppfig2_auroc_aupr.py",
        "runner": "run_suppfig2.py",  # Requires test5, test22
        "outputs": ["suppfig2_auroc_aupr.png", "suppfig2_auroc_aupr.pdf"],
        "note": "Requires AUROC/AUPR metrics from test5 and test22"
    },
    "suppfig3": {
        "name": "Supplementary Figure 3: Multitask Learning",
        "script": "suppfig3_multitask.py",
        "runner": "run_suppfig3.py",  # Requires test5, test7
        "outputs": ["suppfig3_multitask.png", "suppfig3_multitask.pdf"],
        "note": "Requires single/multitask comparison from test7"
    },
    "suppfig6": {
        "name": "Supplementary Figure 6: Full Spectrum",
        "script": "suppfig6_full_spectrum.py",
        "runner": "run_suppfig6.py",  # Requires test27
        "outputs": ["suppfig6_full_spectrum.png", "suppfig6_full_spectrum.pdf"],
        "note": "Requires full spectrum comparison from test27"
    },
    "suppfig7": {
        "name": "Supplementary Figure 7: Undersampled Data",
        "script": "suppfig7_undersampled.py",
        "runner": "run_suppfig7.py",  # Requires test28
        "outputs": ["suppfig7_undersampled.png", "suppfig7_undersampled.pdf"],
        "note": "Requires undersampled datasets (50/100/200) from test28"
    },
    "suppfig8": {
        "name": "Supplementary Figure 8: Loss Comparison",
        "script": "suppfig8_loss_comparison.py",
        "runner": "run_suppfig8.py",  # Requires test29, test30
        "outputs": ["suppfig8_loss_comparison.png", "suppfig8_loss_comparison.pdf"],
        "note": "Requires DeepHit (test29) and PC-Hazard (test30) models"
    },
    "suppfig9": {
        "name": "Supplementary Figure 9: DeepHit",
        "script": "suppfig9_deephit.py",
        "runner": "run_suppfig9.py",  # Requires test29
        "outputs": ["suppfig11_main_comparison.png", "suppfig11_main_comparison.pdf"],
        "note": "Requires DeepHit comparison from test29"
    },
    "suppfig10": {
        "name": "Supplementary Figure 10: PC-Hazard",
        "script": "suppfig10_pchazard.py",
        "runner": "run_suppfig10.py",  # Requires test30
        "outputs": ["suppfig10_pchazard.png", "suppfig10_pchazard.pdf"],
        "note": "Requires PC-Hazard comparison from test30"
    },
    "suppfig11": {
        "name": "Supplementary Figure 11: Main Comparison",
        "script": "suppfig11_main_comparison.py",
        "runner": "run_suppfig11.py",
        "outputs": ["suppfig11_main_comparison.png", "suppfig11_main_comparison.pdf"],
    },
}

def run_figure(fig_key):
    """Run a specific figure generation."""
    fig = FIGURES[fig_key]
    print(f"\n{'='*70}")
    print(f"{fig['name']}")
    print(f"{'='*70}")
    
    # Check if there's a runner script
    if fig["runner"]:
        runner_path = REPO_ROOT / "figures" / fig["runner"]
        try:
            result = subprocess.run(
                [sys.executable, str(runner_path)],
                check=True,
                cwd=str(runner_path.parent)
            )
            print(f"✓ {fig['name']} generated successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"✗ Failed to generate {fig['name']}")
            return False
    else:
        # No runner, just try to plot directly
        script_path = REPO_ROOT / "figures" / fig["script"]
        if "note" in fig:
            print(f"⚠️  Note: {fig['note']}")
        
        print(f"\nAttempting to generate figure...")
        try:
            subprocess.run(
                [sys.executable, str(script_path)],
                check=True,
                cwd=str(script_path.parent)
            )
            print(f"✓ {fig['name']} generated successfully")
            for output in fig["outputs"]:
                print(f"  Output: figures/{output}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to generate {fig['name']}")
            print(f"  This figure may require additional test experiments or data")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_figures.py              # Generate all figures
  python run_all_figures.py --figure 4   # Generate Figure 4
  python run_all_figures.py --figure suppfig11  # Generate Supp Fig 11
  python run_all_figures.py --list       # List all available figures
        """
    )
    parser.add_argument(
        "--figure",
        type=str,
        help="Specific figure to generate (e.g., 'fig4', 'suppfig11')"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available figures"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Figures:")
        print("="*70)
        for key, fig in FIGURES.items():
            status = "✓ Has runner" if fig["runner"] else "⚠️  Direct plot only"
            print(f"{key:12s} - {fig['name']:45s} {status}")
            if "note" in fig:
                print(f"             Note: {fig['note']}")
        return 0
    
    if args.figure:
        fig_key = args.figure.lower()
        if fig_key not in FIGURES:
            print(f"Error: Unknown figure '{args.figure}'")
            print(f"Available: {', '.join(FIGURES.keys())}")
            return 1
        
        success = run_figure(fig_key)
        return 0 if success else 1
    
    # Generate all figures
    print("="*70)
    print("Generating All Figures")
    print("="*70)
    
    successful = []
    failed = []
    
    for fig_key in FIGURES.keys():
        if run_figure(fig_key):
            successful.append(fig_key)
        else:
            failed.append(fig_key)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Successful: {len(successful)}/{len(FIGURES)}")
    if successful:
        print(f"  {', '.join(successful)}")
    print(f"Failed: {len(failed)}/{len(FIGURES)}")
    if failed:
        print(f"  {', '.join(failed)}")
    
    print(f"\nFigures saved to: {REPO_ROOT / 'figures'}")
    
    return 0 if len(failed) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
