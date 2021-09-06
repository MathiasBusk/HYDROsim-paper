# HYDROsim-paper

Code supply for the paper 'Implementing a Neural Network for Decision Support in Groundwater Management'.

This repository contains a python file for simulation hydraulic head changes from well pumping in the San Pedro River Basin groundwater model. The FloPy model can be downloaded from https://github.com/modflowpy/flopy. Also included are two notebooks on training the neural network 'Network_train.ipynb' and testing it 'Test_case.ipynb'. The trained network is attached in the file 'my_model.h5' along with a normal scaler 'std_scaler.bin'. The network training notebook trains a new network on a smaller data set 'Train_sub.zip' (needs to be un-zipepd to a .csv file), but with the same structure as the one in the article. The 'Test_case.ipynb' notebook runs the pre-trained neural network, but feel free to train your own network and tests its abilities on some of the attached simulation examples from MODFLOW in the 'Well_data_examps' folder.

## Use of the Repository

Download the repository and unzip. Use the 'environment.yml' for python environment set-up. Run


### conda env create --file environment.yml


from the anaconda command prompt within the repository. 

Make sure to unpack the 'Train_sub.zip' file containing the data for 'Network_train.ipynb'. 

