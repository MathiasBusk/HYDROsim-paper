c# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 14:32:59 2021

@author: DAHL

-----Generate data for training Neural Network-----

The script runs the MODFLOW Groundwater model of the San Pedro River Basin and simulates 
a well for groundwater extraction. The hydraulic head values are saved as a .npy file.

Specify:
    base_pth: The MODFLOW files compatible with FloPy https://github.com/modflowpy/flopy/tree/develop/examples/groundwater_paper/uspb/flopy
    data_pth: Folder to work in
    Q: Pumping rate of the well
    row_number: List of rows to simulate wells in
    col_number: List of cols to simulate wells in

""" 


import os
import sys
import numpy as np
#import scipy.ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
import flopy
import time


base_pth = os.path.join("flopy") #Specify your folder path for the groundwater model
data_pth = os.path.join("data_folder") #Work folder


Q = -200
row_number = [180]
col_number = [180]



print(sys.version) 
print('numpy version: {}'.format(np.__version__))
print('matplotlib version: {}'.format(mpl.__version__))
print('flopy version: {}'.format(flopy.__version__))
plt.close('all')


ml = flopy.modflow.Modflow.load(
    "DG.nam",
    exe_name="mf2005dbl", #Make sure that you include a MODFLOW .exe file. Files can be downloaded at https://www.usgs.gov/software/modflow-2005-usgs-three-dimensional-finite-difference-ground-water-model
    verbose=True,
    model_ws=base_pth,
)


nrow, ncol = ml.dis.nrow, ml.dis.ncol
ibound = ml.bas6.ibound[3, :, :]


# create base model and run
ml.model_ws = data_pth
ml.exe_name = 'mf2005dbl'
ml.write_input()
ml.run_model(silent=True)
idx = 1
t1 = time.time()


hedObj = flopy.utils.HeadFile(os.path.join(base_pth, "DG.hds"))
h_0 = hedObj.get_data(kstpkper=(0,0))




for i in range(len(row_number)):
    t1 = time.time()
    print('\nrow {} - col {}, sim = {} / {}\n'.format(row_number[i], col_number[i],idx,len(row_number)))


    wd = {0: [[3, row_number[i], col_number[i],Q]]}
    ml.remove_package("WEL")
    wel = flopy.modflow.ModflowWel(model=ml, stress_period_data=wd)
    wel.write_file()
    ml.run_model(silent=True)

    hedObj = flopy.utils.HeadFile(os.path.join(data_pth, "DG.hds"), precision='double')
    h = hedObj.get_data(kstpkper=(0,0))
    np.save('San_Pedro_head_'+np.str(row_number[i])+'_'+np.str(col_number[i])+'.npy',h )
    idx += 1
    
    t2 = time.time()
    
    print(f'This took {round(((t2-t1)),2)} s')


