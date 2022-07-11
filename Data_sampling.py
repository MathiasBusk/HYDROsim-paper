
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
    rows: List of rows to simulate wells in
    cols: List of cols to simulate wells in

""" 


import os
import sys
import numpy as np
#import scipy.ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
import flopy
import time
from tqdm import trange


base_pth = os.path.join("flopy") #Specify your folder path for the groundwater model
data_pth = os.path.join("data_folder") #Work folder

pump_rates = np.load('pump_rates.npy')
Q = pump_rates
data = np.load('samples_hk_over1_1000.npy')

rows = data[:,-2]
cols = data[:,-1]

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
h_0 = hedObj.get_data(kstpkper=(0,0),mflay=3)

time_save = []
for i in trange(1000):
    t1 = time.time()
    print('\nrow {} - col {},  sim = {} / {}\n'.format(rows[i], cols[i],idx,len(rows)))

    pump = -Q[i]*(i+1)
    wd = {0: [[3, rows[i], cols[i],pump]],1: [[3, rows[i], cols[i],pump]],2: [[3, rows[i], cols[i],pump]]}
    ml.remove_package("WEL")
    wel = flopy.modflow.ModflowWel(model=ml, stress_period_data=wd)
    wel.write_file()
    ml.model_ws = data_pth
    ml.write_input()
    ml.exe_name = 'mf2005dbl'
    ml.run_model(silent=True)

    hedObj = flopy.utils.HeadFile(os.path.join(data_pth, "DG.hds"), precision='double')
    h = hedObj.get_data(kstpkper=(0,0),mflay=3)
    #np.save('San_Pedro_head_response_'+str(rows[i])+'_'+str(cols[i])+'_'+str(pump)+'.npy',h )
    idx += 1
    
    t2 = time.time()
    time_save.append(t2-t1)
    print(f'This took {round(((t2-t1)),2)} s')
    



