# PiDeeL: metabolic pathway-informed deep learning model for survival analysis and pathological classification of gliomas

[![DOI](https://img.shields.io/badge/DOI-10.1093%2Fbioinformatics%2Fbtad684-blue)](https://doi.org/10.1093/bioinformatics/btad684)

Official implementation of **PiDeeL** (Pathway-Informed Deep Learning Model), a deep learning framework that incorporates metabolic pathway information for improved survival prediction in glioma patients using HRMAS NMR spectroscopy data.

![system overview](https://github.com/ciceklab/PiDeeL/blob/main/system_fig.png)

## Paper

**PiDeeL: metabolic pathway-informed deep learning model for survival analysis and pathological classification of gliomas**

Published in *Bioinformatics*, Volume 39, Issue 11, November 2023

📄 [Read the paper](https://academic.oup.com/bioinformatics/article/39/11/btad684/7413171)

## Overview

PiDeeL integrates biological metabolic pathway knowledge into the neural network architecture to predict patient survival outcomes. By constraining the network's weights according to metabolite-pathway relationships, PiDeeL achieves better interpretability and performance.


## Installation

1. Clone the repo
   ```bash
   git clone https://github.com/ciceklab/PiDeeL/
   cd PiDeeL
   ```

2. Create conda environment
   ```bash
   conda env create --name PiDeeL --file PiDeeL.yml
   conda activate PiDeeL
   ```

### Key Dependencies

- PyTorch 2.0+
- pycox (for survival analysis)
- scikit-survival
- scikit-learn
- pandas, numpy, matplotlib
- shap

## Repository Structure

```
PiDeeL/
├── README.md
├── PiDeeL.yml                    # Conda environment file
├── system_fig.png
│
├── run/                          # Inference with pretrained models
│   ├── predict.py                # Main prediction script
│   ├── model.py                  # Model architecture
│   ├── model_utils.py            # Utility functions
│   ├── PiDeeL_2layer.pth         # Pretrained 2-layer model
│   ├── PiDeeL_3layer.pth         # Pretrained 3-layer model
│   ├── PiDeeL_4layer.pth         # Pretrained 4-layer model
│   └── sample_quant.pickle       # Sample input data
│
└── reproduction/                 # Full experiment reproduction
    ├── config.py                 # Central configuration
    ├── scripts/                  # Training scripts
    │   ├── load_targeted_data.py
    │   ├── model_utils.py
    │   ├── 1layer/, 2layer/, 3layer/, 4layer/
    │   └── baseline/             # Cox-PH, CWGB, RF
    ├── figures/                  # Paper figure generation
    │   ├── run_fig2.py, run_fig4.py, ...
    │   └── run_all_figures.py
    ├── models/                   # Saved model weights
    ├── logs/                     # C-Index results
    ├── plots/                    # Training loss plots
    └── pideel_data/              # Data directory
```

---

## Prediction Using Pretrained PiDeeL

Use pretrained models to predict survival risk scores for new samples.

1. Navigate to the run directory:
   ```bash
   cd run/
   ```

2. Prepare your input data:
   - Use the provided `sample_quant.pickle` as a template
   - Or use the automated metabolite quantification pipeline from [Cakmakci et al.](https://github.com/ciceklab/targeted_brain_tumor_margin_assessment) to quantify your HRMAS NMR spectroscopy data

3. Run prediction:
   ```bash
   python predict.py --layer 4 --dev gpu
   ```
   
   Options:
   - `--layer`: Select model architecture (2, 3, or 4)
   - `--dev`: Device to use (`gpu` or `cpu`)

4. Output: Risk scores printed to terminal

---

## Reproducing Paper Results

### Quick Start: Interactive Notebook

The easiest way to reproduce the main comparison figure is using the Jupyter notebook at the repo root:

```bash
jupyter notebook reproduce_main_comparison.ipynb
```

**By default**, the notebook uses pretrained model logs included in the repository to generate Figure 2 immediately.

**To retrain models from scratch**, set the `RETRAIN` flag in the notebook:
```python
RETRAIN = True  # Set to True to retrain all models instead of using pretrained logs
```

### Data Setup

The preprocessed samples are already included in the repository under `reproduction/pideel_data/targeted/`. No additional download is required to reproduce the results.

If you want access to the raw HRMAS NMR spectroscopy data, you can download it from Zenodo:
```
https://zenodo.org/record/7228791
```

### Training Models

Navigate to the reproduction directory:
```bash
cd reproduction/
```

**PiDeeL and DeepSurv models:**
```bash
# 2-layer
python scripts/2layer/no_pathway/main.py  # DeepSurv
python scripts/2layer/pathway/main.py     # PiDeeL

# 3-layer
python scripts/3layer/no_pathway/main.py
python scripts/3layer/pathway/main.py

# 4-layer
python scripts/4layer/no_pathway/main.py
python scripts/4layer/pathway/main.py
```

**Baseline models:**
```bash
python scripts/baseline/coxph/main.py  # Cox Proportional Hazards
python scripts/baseline/cwgb/main.py   # Component-wise Gradient Boosting
python scripts/baseline/rf/main.py     # Random Survival Forest
```

### Generating Figures

```bash
cd reproduction/figures/

# Generate all figures
python run_all_figures.py

# Or generate individual figures
python run_fig2.py      # Main comparison (Fig. 2)
python run_fig4.py      # Random connections ablation
python run_fig5.py      # Dropout analysis
python run_fig6.py      # External validation
```

### Results

- C-Index results: `reproduction/logs/*/c_indices.txt`
- Generated figures: `reproduction/figures/`
- Training plots: `reproduction/plots/`

---

## GPU/CPU Support

The code automatically detects GPU availability:
- **GPU available**: Uses CUDA for training (recommended)
- **CPU only**: Falls back to CPU (slower but functional)

No code changes needed - device selection is automatic via `config.py`.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{kaynar2023pideel,
  title={PiDeeL: metabolic pathway-informed deep learning model for survival analysis and pathological classification of gliomas},
  author={Kaynar, Gun and Cakmakci, Doruk and Bund, Caroline and Todeschi, Julien and Namer, Izzie Jacques and Cicek, A Ercument},
  journal={Bioinformatics},
  volume={39},
  number={11},
  pages={btad684},
  year={2023},
  publisher={Oxford University Press}
}

```

## License

Distributed under the MIT License.

## Contact

Gun Kaynar - [gunkaynar.com](https://gunkaynar.com/)

A. Ercument Cicek - http://ciceklab.cs.bilkent.edu.tr/ercument

## Acknowledgements

This work was supported by grants from BPI France (ExtempoRMN Project), Hôpitaux Universitaires de Strasbourg, Bruker BioSpin, Univ. de Strasbourg and the Centre National de la Recherche Scientifique; also by TUBA GEBIP, Bilim Akademisi BAGEP and TUSEB Research Incentive awards to AEC.