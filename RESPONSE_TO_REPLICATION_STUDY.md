# Response to "Reproducibility and Environmental Efficiency of Metabolomics Cancer Modeling" (Claire Jean-Quartier, Niklas Tscheppe, Stefan Millonig, Lena Klambauer, Andreas Holzinger, Sarah Stryeck, Fleur Jeanquartier; Applied Sciences; 2026)


**Authors of the original PiDeeL study:**
Gun Kaynar, Doruk Cakmakci, Caroline Bund, Julien Todeschi, Izzie Jacques Namer, A. Ercument Cicek

**Date:** April 2026

---

## Summary

Jean-Quartier et al. (2026) published a paper in *Applied Sciences* (MDPI) titled "Reproducibility and Environmental Efficiency of Metabolomics Cancer Modeling" ([link](https://www.mdpi.com/2076-3417/16/2/588)), which claims to be a reproducibility study of our PiDeeL model published in *Bioinformatics* (Oxford University Press, 2023). Their central claim is that they "were not able to reproduce the results in a satisfying manner" and that the reproduction shows a c-index of ~61% instead of the reported 68.7%.

This claim is false. We have fully reproduced every figure and result from the original paper. The reproduction materials, including a self-contained Jupyter notebook, are available in this repository. The authors' failure stems from their own technical errors and misconfigurations, not from any issue with PiDeeL.

---

## Our Reproduction Materials

We prepared and published the following within two days of learning about the Jean-Quartier et al. manuscript:

- **Full reproduction pipeline:** [`reproduction/`](https://github.com/ciceklab/PiDeeL/tree/main/reproduction) — reproduces all figures from the paper.
- **Self-contained Jupyter notebook:** [`reproduce_main_comparison.ipynb`](https://github.com/ciceklab/PiDeeL/blob/main/reproduce_main_comparison.ipynb) — reproduces Figure 2 (the exact figure that Jean-Quartier et al. claim they could not replicate). This notebook can use pretrained logs to generate the figure instantly, or retrain all models from scratch by setting `RETRAIN = True`.
- **Preprocessed data included:** As of January 2026, the preprocessed metabolite concentration data is included directly in the repository under `reproduction/pideel_data/targeted/`, eliminating the need to run the quantification pipeline from raw data. Previously, only the raw HRMAS NMR FID signals were available on Zenodo.

Anyone can verify our results by running the notebook. The data is included in the repository. No additional downloads or preprocessing are necessary.

---

## What Went Wrong in Jean-Quartier et al.

### 1. They Likely Failed at Data Preprocessing

This is the most critical issue. At the time Jean-Quartier et al. conducted their study, the PiDeeL repository provided raw HRMAS NMR FID signals via Zenodo, not the preprocessed metabolite concentration vectors that PiDeeL uses as input. Converting raw FID signals into quantified concentrations for 37 metabolites requires running a separate metabolite quantification pipeline ([Cakmakci et al.](https://github.com/ciceklab/targeted_brain_tumor_margin_assessment)), which involves NMR spectrum preprocessing and automated metabolite-specific quantification, which is a non-trivial step.

The replication authors explicitly stated in their email to us: "we are computer science students, and do not know a lot about the medical side of the model." If the data preprocessing was performed incorrectly or not performed at all, or performed with different parameters, then every downstream result (every model, every c-index, every figure) would be wrong, regardless of whether the code itself ran without errors.

Their paper does not describe how they handled the data preprocessing step, nor does it verify that their input data matched the expected metabolite concentration format. A c-index drop from 68.7% to ~61% is consistent with models trained on improperly preprocessed input data.

Since January 2026, we have included the preprocessed data directly in the repository to eliminate this as a source of error for future use.

### 2. They Knew About a CUDA Error and Did Not Wait for Our Response

In their email to us dated July 9, 2024, the replication authors explicitly identified a CUDA device mismatch:

> *"During the training of the pathological classification models, we have encountered the following: `RuntimeError: CUDA error: invalid device ordinal`. This might be caused by [...] where "cuda:1" is used multiple times versus "cuda:0" which is used in the rest of the program. Is this intentional?"*

Their published paper then acknowledges (Section 4.4): "further communication with the PiDeeL authors was attempted, but because of the time frame of this project, it was not possible to wait for a reply and/or fixes from their side."

They identified a CUDA device configuration error, emailed us about it, did not wait for our response, attempted to fix it themselves, and then published a paper concluding that our results are not reproducible.

### 3. They Did Not Contact Us About the Discrepant Results

The email thread shows that the replication authors contacted us about setup issues (missing directories, typos, path problems) and we responded promptly, fixing every reported issue. Their paper acknowledges: "The authors of the PiDeeL project were quick in responding our questions and updated the PiDeeL repository to fix most of the issues."

Yet when they obtained a c-index of ~61% instead of 68.7%, the core finding of their paper, they never contacted us. They did not email us to ask: "We are getting different numbers; could our setup be wrong?" Instead, they published a paper claiming our results are irreproducible. If they had sent a single email, we would have identified the problem immediately.

### 4. They Ran on Windows Subsystem for Linux (WSL), Not Native Linux

Their paper states: "attempts to run the reproduced script on other operating systems resulted in errors, the final emissions evaluation using CodeCarbon was performed on two Windows machines using Windows Subsystem Linux (WSL)." Their discussion section admits: "we were only able to compute on virtual environments rather than native. This could in fact explain differences between models."

WSL is not equivalent to native Linux for GPU-accelerated deep learning workloads. CUDA behavior, memory management, and random number generation can differ under WSL. The authors acknowledge this limitation yet still published a conclusion of non-reproducibility.

### 5. The Issues They Found Were Minor Setup/Documentation Issues, Not Scientific Errors

The issues reported in their correspondence with us were:
- A typo in the README (`--layer 3` instead of `--layer 4`)
- Missing placeholder files for empty Git directories
- A path separator issue on Windows
- An `os.system()` call that should have been `os.chdir()`

These are trivial documentation and packaging matters that exist in virtually every research software repository. None of them affect the scientific validity of PiDeeL or its reported results. We fixed all of them within days of being notified. These are the kinds of issues resolved through GitHub issues or emails, not through a published paper claiming non-reproducibility.

### 6. They Made Code Modifications Without Understanding the Pipeline

The authors admit (Section 4.4) to making their own code changes to work around the CUDA error and other issues. When you modify a research codebase to work around errors you have explicitly stated you do not understand, and then obtain different results, the scientific conclusion is that your modifications likely introduced the discrepancy, not that the original results are wrong.

---

## The Actual Results

Our reproduction notebook generates the following results, which match the original paper:

- **PiDeeL 4-layer median c-index: ~68.7%** (matching the reported value)
- **PiDeeL outperforms DeepSurv and all baselines**, consistent with all claims in the original paper.

These results can be independently verified by anyone with a standard Linux machine and a GPU (or even CPU. It will just take longer).

---

## On the Venue and Review Quality

The Jean-Quartier et al. paper was published in *Applied Sciences* (MDPI), a journal with well-documented concerns about editorial standards. The paper went from submission (November 19, 2025) to acceptance (December 24, 2025) in **35 days**, including the holiday period,a timeline that raises serious questions about the depth of peer review.

A competent reviewer should have:
1. Asked whether the authors verified that their preprocessed input data was correct, given the non-trivial quantification pipeline required.
2. Asked whether the authors contacted the original PiDeeL team about the c-index discrepancy.
3. Noted that running deep learning code on WSL with CUDA errors does not constitute a valid reproduction attempt.
4. Recognized that a c-index drop from 68.7% to 61% with acknowledged CUDA device errors, potential data preprocessing issues, and ad hoc code modifications is almost certainly a setup problem.

None of this appears to have happened.

---

## On the "Carbon Emissions" Component

Roughly half of the Jean-Quartier et al. paper is devoted to measuring CO2 emissions of training PiDeeL using CodeCarbon. While sustainability in ML is a legitimate topic, this portion of the paper has no bearing on the reproducibility claims and appears to serve as padding to justify a full publication. The emission measurements themselves are acknowledged by the authors to be inaccurate (Section 5.2: "the total execution duration calculated by CodeCarbon was found to be much too low to be plausible").

---

## Timeline of Events

| Date | Event |
|------|-------|
| November 2023 | PiDeeL published in *Bioinformatics* (Oxford University Press). Raw FID data released on Zenodo. |
| May 23, 2024 | Replication authors first contact us about setup issues |
| May 24, 2024 | We respond and begin fixing reported issues |
| July 9, 2024 | Replication authors send follow-up identifying CUDA device error, state they lack domain knowledge, ask additional questions |
| *(They do not wait for our reply — their paper states this was due to "the time frame of this project")* | |
| *(No contact from anyone about the discrepant c-index results — ever)* | |
| November 19, 2025 | Jean-Quartier et al. submitted to *Applied Sciences* without informing us |
| December 24, 2025 | Paper accepted (35 days from submission, over the holiday period) |
| January 6, 2026 | Paper published claiming PiDeeL results are not reproducible |
| January 12, 2026 | We publish full reproduction materials on GitHub, including preprocessed data |
| January 12, 2026 | We contact the handling editor and editor-in-chief of *Applied Sciences* requesting the opportunity to respond |
| January 26, 2026 | Follow-up email to the handling editor — no response |
| April 6, 2026 | Third email to the handling editor |
| April 7, 2026 | Handling editor responds: "I do not recall any earlier e-mails. Please deal directly with MDPI." |
| April 7, 2026 | Formal complaint filed with MDPI editorial office |

---

## Our Request to the Journal

We contacted the handling editor of *Applied Sciences* on January 12, 2026, requesting the opportunity to publish a brief response. The editor-in-chief was CC'd. We received no reply. We sent follow-up emails on January 26 and April 6, 2026. On April 7, the handling editor responded stating he did not recall our earlier emails and directed us to deal with MDPI directly.

We have filed a formal complaint with MDPI's editorial office requesting a correction, the opportunity to publish a response, or retraction.

We are publishing this response on GitHub so that the scientific community has access to the facts while the editorial process is ongoing.

---

All results from the original PiDeeL paper (Bioinformatics, 2023) are fully reproducible. We invite anyone to verify this using the materials in this repository.

---

## How to Reproduce Our Results

```bash
# Clone the repository
git clone https://github.com/ciceklab/PiDeeL.git
cd PiDeeL

# Create the conda environment
conda env create --name PiDeeL --file PiDeeL.yml
conda activate PiDeeL

# Run the reproduction notebook
jupyter notebook reproduce_main_comparison.ipynb
```

Set `RETRAIN = True` in the notebook to retrain all models from scratch and generate Figure 2.

The preprocessed data is included in the repository under `reproduction/pideel_data/targeted/`. No external downloads or preprocessing steps are needed.
