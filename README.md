# PiDeeL

## About The Project
The source code of the pre-print: Pathway-informed deep learning model for survival analysis and pathological classification of gliomas


![alt text](https://github.com/ciceklab/PiDeeL/blob/main/system_figure.png)


## Getting Started
To reproduce the results discussed in the pre-print, please follow the steps below:

### Installation
1. Clone the repo
   ```
   git clone https://github.com/ciceklab/PiDeeL/
   ```
2. Create conda environment
   ```
   conda env create --name PiDeeL --file PiDeeL.yml
   ```
### Reproduction

1. Download the dataset
    ```
   https://zenodo.org/record/7228791
   
   Extract the zip into /reproduce/data/ folder
   ```

3. Download the pyNMR library
   ```
   https://github.com/bennomeier/pyNMR/tree/c58d1500dc7c540dcd2aaf28bdf8a660e7f496ff
   Move the files to /reproduce/reproduction_scripts/pNNMR_lib
   ```
4. Set the path
   ```
   Open hyper_config.py
   Change the ultimate_path variable to path/to/PiDeeL 
   ```
5. Run the reproduction scripts
   ```
   conda activate PiDeel
   python run_reproduction.py
   ```
6. Get the reproduced figures
   
   Figures are under reproduce/reproduction_scripts/figures/
   
### Prediction using pretrained PiDeeL
1. Go to /run
   
   You can use the sample metabolite quantification features as input, or you can use the automated metabolite quantification pipeline from [Cakmakci et al.](https://github.com/ciceklab/targeted_brain_tumor_margin_assessment) to quantify your HRMAS NMR spectroscopy data.
   
2. Select the parameters.
   ```
   --layer: select a pretrained model among 2-layer, 3-layer and 4-layer PiDeeL (2, 3, 4)
   --dev: select the device to use the model. (gpu or cpu) 
   ```
3. Run the command below with the sample arguments
   ```
   python predict.py --layer 3 --dev gpu
   ```
4. See the output of the script printed on the terminal. The values correspond to the risk scores of the samples.
<img width="432" alt="Screenshot 2023-08-01 at 00 29 17" src="https://github.com/ciceklab/PiDeeL/assets/45332095/2578085e-b33e-4c38-a543-80669fdce0a6">


## License

Distributed under the MIT License.

## Contact

Gun Kaynar - http://gun.kaynar.bilkent.edu.tr/

A. Ercument Cicek - http://ciceklab.cs.bilkent.edu.tr/ercument


## Acknowledgements
This work was supported by grants from BPI France (ExtempoRMN Project), Hôpitaux Universitaires de Strasbourg, Bruker BioSpin, Univ. de Strasbourg and the Centre National de la Recherche Scientifique; also by TUBA GEBIP, Bilim Akademisi BAGEP and TUSEB Research Incentive awards to AEC.

