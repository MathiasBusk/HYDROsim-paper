# HYDROsim-paper

Code supply for the paper 'Implementing a Neural Network for Decision Support in Groundwater Management'.

Try the Network_train notebook in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MathiasBusk/HYDROsim-paper/blob/main/Network_train.ipynb)

Try the Test_case notebook in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MathiasBusk/HYDROsim-paper/blob/main/Test_case.ipynb)

This repository contains a python file for simulation hydraulic head changes from well pumping in the San Pedro River Basin groundwater model. The FloPy model can be downloaded from https://github.com/modflowpy/flopy. Also included are two notebooks on training the neural network 'Network_train.ipynb' and testing it 'Test_case.ipynb'. The trained network is attached in the file 'my_model.h5' along with a normal scaler 'std_scaler.bin'. The network training notebook trains a new network on a smaller data set 'Train_sub.zip' (needs to be un-zipepd to a .csv file), but with the same structure as the one in the article. The 'Test_case.ipynb' notebook runs the pre-trained neural network, but feel free to train your own network and tests its abilities on some of the attached simulation examples from MODFLOW in the 'Well_data_examps' folder.

## Use of the Repository

Either open the notebooks in google colab or 
Download the repository and unzip. Use the 'environment.yml' for python environment set-up. Run


### conda env create --file environment.yml


from the anaconda command prompt within the repository. Make sure to unpack the 'Train_sub.zip' file containing the data for 'Network_train.ipynb'. 

In order to run the 'Data_sampling.py' script, please download the groundwater model from https://github.com/modflowpy/flopy and specify the folder in the script. Also get the .exe files for MODFLOW 2005 from https://www.usgs.gov/software/modflow-2005-usgs-three-dimensional-finite-difference-ground-water-model and add them to PATH.

