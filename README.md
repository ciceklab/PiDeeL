# PiDeel

## About The Project
The source code of the pre-print: Pathway-informed deep learning model for survival analysis and pathological classification of gliomas

![alt text]([http://url/to/img.png](https://github.com/ciceklab/PiDeeL/blob/main/system_figure.png))
## Getting Started
To reproduce the results discussed in the pre-print, please follow the steps below:

### Installation
1. Clone the repo
   ```
   git clone https://github.com/ciceklab/PiDeeL/
   ```
2. Create conda environment
   ```
   conda env create --name PiDeeL --file = PiDeeL.yml
   ```
3. Download the dataset
   https://zenodo.org/record/7228791
   \\
   Extract the zip into /reproduce/data/ folder

4. Download the pyNMR library

   https://github.com/bennomeier/pyNMR/tree/c58d1500dc7c540dcd2aaf28bdf8a660e7f496ff
   Move the files to /reproduce/reproduction_scripts/pNNMR_lib
   
3. Run the run_reproduction.py
   ```
   conda activate PiDeel
   python run_reproduction.py
   ```
   
## License

Distributed under the MIT License.

## Contact

Gun Kaynar - http://gun.kaynar.bilkent.edu.tr/
A. Ercument Cicek - http://ciceklab.cs.bilkent.edu.tr/ercument


## Acknowledgements
This work was supported by grants from BPI France (ExtempoRMN Project), Hôpitaux Universitaires de Strasbourg, Bruker BioSpin, Univ. de Strasbourg and the Centre National de la Recherche Scientifique; also by TUBA GEBIP, Bilim Akademisi BAGEP and TUSEB Research Incentive awards to AEC.

