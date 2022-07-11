# HYDROsim-paper

Code supply for the paper 'Hydraulic Head Change Predictions in Groundwater Models using a Probabilistic Neural Network'.

Full training data set available on Zenodo 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6817606.svg)](https://doi.org/10.5281/zenodo.6817606)

This repository contains a python file for simulation hydraulic head changes from well pumping in the San Pedro River Basin groundwater model. The FloPy model can be downloaded from https://github.com/modflowpy/flopy. Also included are the python file for training the neural network 'SanPedroModelSpyder.py'. Training data can be found in the Zeniodo link.

## Use of the Repository

Download the repository and unzip. Use the 'environment.yml' for python environment set-up. Run


### conda env create --file environment.yml

from the anaconda command prompt within the repository.

In order to run the 'Data_sampling.py' script, please download the groundwater model from https://github.com/modflowpy/flopy and specify the folder in the script. Also get the .exe files for MODFLOW 2005 from https://www.usgs.gov/software/modflow-2005-usgs-three-dimensional-finite-difference-ground-water-model and add them to PATH.

