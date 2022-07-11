# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:11:20 2022

@author: DAHL

Training of the neural network. Pre-trained network is saved on the Github page.
This script requires a folder with .npy files of MODFLOW simulations in the PATH variable to train with. This is available
at DOI 10.5281/zenodo.6817606.
Network and standard scaler are saved as:
    model_new_one.h5
    std_scaler_new_one.bin

"""

# Import
import pandas as pd 
import scipy.ndimage
import scipy.signal
import scipy.stats
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import os
from sklearn.preprocessing import StandardScaler             
from sklearn.metrics import accuracy_score
import time
import glob
import joblib
np.random.seed(42)
import skfmm
import random
import pylab as pl
import datetime
from tqdm import trange
from IPython.display import clear_output
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Dropout
from tensorflow import keras
import tensorflow_probability as tfp



class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();


def load_data(files_nr):
    Path =  r'*\*\*\data_pump_1000'
    Path2 =  r'*\*\*\steady_state'
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(Path):
        for file in f:
            if '.npy' in file:
                files.append(os.path.join(r, file))
                
    length = files_nr
    data = np.empty((length,440,320))
    
    a = np.linspace(0,999,1000)
    a = list(a)
    a = [int(x) for x in a]
    b = random.sample(a,files_nr)
    b = np.sort(b)
    b = list(b)
    filess = [files[x] for x in b]
   
    for i in range(len(filess)):
        dats = np.load(filess[i])
        data[i,:,:] = dats[:,:]
    
    data0 = os.path.join(Path2,'head_no_pump.npy')
    data0 = np.load(data0)
    data0 = data0[3,:,:]
    
    return data, data0, filess, b

def well_dist(row,col):    
    xmax = 80000
    ymax = 110000
    X, Y = np.meshgrid(np.linspace(0,xmax,320), np.linspace(ymax,0,440))
    phi =  -1* np.ones_like(X)
    phi[row,col] = 1

    d_well = skfmm.distance(phi,dx=250) 
    d_well = -d_well
    return d_well

def smooth_hk():
    result = np.load('outside.npy')
    data_hk = np.loadtxt('hyd_kon')
    data_hk = data_hk[::-1]
    hyk = np.flipud((data_hk))
    hyk +=0.1
    hk_smooth = scipy.ndimage.filters.gaussian_filter(hyk,(7,7))
    hk_smooth[result[0,:],result[1,:]] = -2
    return hk_smooth
    


def travel_time(row,col,hk_smooth):
    data_hk = np.loadtxt('hyd_kon')
    data_hk = data_hk[::-1]
    hyk = np.flipud((data_hk))

    xmax = 80000
    ymax = 110000
    X, Y = np.meshgrid(np.linspace(0,320,320), np.linspace(440,0,440))
    phi =  np.ones_like(hyk)*hyk
    phi[phi == 0] = 0
    phi[phi != 0] = -1
    phi[row,col] = 1

    speed=np.ones_like(phi)*hk_smooth
    t = skfmm.travel_time(phi, speed,dx=250)
    return t


def data_processing(data,data0,files_nr,b):
    data_str = np.loadtxt('str.txt')
    pump_rates = np.load('pump_rates.npy')
    pump_rates = pump_rates[b]
    row = (data_str[:,1])
    col = (data_str[:,2])
    
    row = row.astype(int)
    col = col.astype(int)
    
    xmax = 80000
    ymax = 110000
    X, Y = np.meshgrid(np.linspace(0,xmax,320), np.linspace(ymax,0,440))
    phi =  -1* np.ones_like(X)
    phi[row,col] = 1
    
    d = skfmm.distance(phi,dx=250) 
    d = -d
    d_boundary = np.load('d_boundary.npy')
    row_nr = np.empty((440,320))
    row_nr.fill(0)
    col_nr = np.empty((440,320))
    r = -1
    c = 0
    for i in range(440):
        r += 1
        c = 0
        for j in range(320):
            row_nr[i,j] = r
            col_nr[i,j] = c
            c+=1

    df = np.load('samples_hk_over1_1000.npy')
    df = df[b]
    rows = df[:,-2]
    cols = df[:,-1]
    data_hk = np.loadtxt('hyd_kon')
    data_hk = data_hk[::-1]
    hyk = np.flipud((data_hk))
    hk_smooth = smooth_hk()
    head_differ = []
    dista = []
    times = []
    dista_well = []
    head0s = []
    hyks = []
    hyks_log = []
    pump = []
    rowss = []
    colss = []
    dist_bound = []
    pumps_row = []
    pumps_col = []
    for i in trange(files_nr):
      
        t = travel_time(int(rows[i]),int(cols[i]),hk_smooth)
        d_well = well_dist(int(rows[i]),int(cols[i]))
    
        hyd_diff = data0-data[i,:,:]
        head_diff = np.reshape(hyd_diff,(320*440))
        dist = np.reshape(d,(320*440))
        dist_b = np.reshape(d_boundary,(320*440))
        time = np.reshape(t,(320*440))
        dist_well = np.reshape(d_well,(320*440))
        head0 = np.reshape(data0,(320*440))
        hykk = np.reshape(hyk,(320*440))
        hykk_l = np.reshape(np.log10(hyk),(320*440))
        pump_row = np.ones((320*440)) * rows[i]
        pump_col = np.ones((320*440)) * cols[i]
        
        row = np.reshape(row_nr,(320*440))
        col = np.reshape(col_nr,(320*440))
        
        indexs = np.where(head0 != -999)

        pum = pump_rates[i]*np.ones((320*440,1))
        pump.append(pum[indexs])
    
        
        head_differ.append(head_diff[indexs])
        dista.append(dist[indexs])
        dist_bound.append(dist_b[indexs])
        times.append(time[indexs])
        dista_well.append(dist_well[indexs])
        head0s.append(head0[indexs])
        hyks.append(hykk[indexs])
        hyks_log.append(hykk_l[indexs])
        pumps_row.append(pump_row[indexs])
        pumps_col.append(pump_col[indexs])
        
        rowss.append(row[indexs])
        colss.append(col[indexs])
        
    head_differ = np.array(head_differ)
    dista = np.array(dista)
    dist_bound = np.array(dist_bound)
    times = np.array(times)
    dista_well = np.array(dista_well)
    head0s = np.array(head0s)
    hyks = np.array(hyks)
    hyks_log = np.array(hyks_log)
    pumps_row = np.array(pumps_row)
    pumps_col = np.array(pumps_col)
    
    rowss = np.array(rowss)
    colss = np.array(colss)
    pumps = np.array(pump)
    
    head_diff = np.reshape(head_differ,(len(rowss)*len(indexs[0])))
    
    dist = np.reshape(dista,(len(rowss)*len(indexs[0])))
    dist_bounds = np.reshape(dist_bound,(len(rowss)*len(indexs[0])))
    time = np.reshape(times,(len(rowss)*len(indexs[0])))
    dist_well = np.reshape(dista_well,(len(rowss)*len(indexs[0])))
    head0 = np.reshape(head0s,(len(rowss)*len(indexs[0])))
    hykss = np.reshape(hyks,(len(rowss)*len(indexs[0])))
    hykss_log = np.reshape(hyks_log,(len(rowss)*len(indexs[0])))
    row = np.reshape(rowss,(len(rowss)*len(indexs[0])))
    col = np.reshape(colss,(len(rowss)*len(indexs[0])))
    pumpi = np.reshape(pumps,(len(pumps)*len(indexs[0])))
    pump_roww = np.reshape(pumps_row,(len(pumps_row)*len(indexs[0])))
    pump_coll = np.reshape(pumps_col,(len(pumps_col)*len(indexs[0])))
    
    data_set= pd.DataFrame(head_diff)

    data_set.columns = ["head_diff"]
    
    data_set['head']=head0
    data_set['dist']=dist
    data_set['time']=time
    data_set['dist_well']=dist_well
    data_set['h_cond']=hykss
    data_set['h_cond_log']=hykss_log
    data_set['pump_rate']=pumpi
    data_set['pump_row'] = pump_roww
    data_set['pump_col'] = pump_coll
    data_set['row']=row
    data_set['col']=col
    data_set['dist_boundary']=dist_bounds


    data_set.drop(data_set.loc[data_set['head']==-999].index, inplace=True)
 
    data_set = data_set[(data_set[['time']] != 0).all(axis=1)]
    data_set = data_set[(data_set[['head_diff']] < 200).all(axis=1)]
    data_set = data_set[(data_set[['head_diff']] > 0.0000001).all(axis=1)]
    min_bin = data_set['dist_well'].min()
    max_bin = data_set['dist_well'].max()
    bins2 = np.linspace(min_bin-0.001,max_bin+0.1,500)
    
    qc = pd.qcut(data_set['head_diff'], q=20, precision=0)
    min_bin = data_set['head_diff'].min()
    max_bin = data_set['head_diff'].max()
    bins = np.linspace(min_bin-0.0001,max_bin+0.0001,100)
    data_set['binned']=qc
    data_set['binned2']=pd.cut(data_set['dist_well'], bins=bins2)
    data_set.sort_values('binned')
    uni_bin2 = np.unique(data_set.iloc[:,-1])
   
    
    data_collection = data_set.copy()   
    qc = pd.qcut(data_collection['head_diff'], q=20, precision=0)
    data_collection['binned']=qc
    uni_bin = np.unique(data_collection.iloc[:,-2])
    s=(data_collection['binned']).value_counts()
    
    data_collections = data_collection.iloc[0:0,:].copy()
    k=1
    for i in range(len(s)-1,-1,-1):
        df = data_collection[data_collection['binned'] == uni_bin[i]]
        n = int(np.round(len(df)*k,0))
        df_samp = df.sample(n=n)
        data_collections = pd.concat([data_collections, df_samp])
        k = k*0.9

    return data_collections

def Training_data(data_collection):
    validation = np.load('validation_set_pump.npy',allow_pickle=True)
    X_train, X_test, y_train, y_test = train_test_split(data_collection.iloc[:,1:10], data_collection.iloc[:,0], test_size=0.2, random_state=1)
    X_val = validation[:,1:10]
    y_val = validation[:,0]
    scaler = StandardScaler() 
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    joblib.dump(scaler, 'std_scaler_new_one.bin', compress=True)
    return X_train,X_test, X_val, y_train, y_test, y_val
    
def Training(X_train,X_test, y_train, y_test):
    tfd = tfp.distributions
    negloglik = lambda y, p_y: -p_y.log_prob(y)
    logs_base_dir = "summaries_head" 
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=500)
    inputA = Input(shape=(9,),name='inputA')
    
    interpB = Dense(75, activation='relu')(inputA)#),kernel_regularizer=tf.keras.regularizers.l2(0.00001), activity_regularizer=tf.keras.regularizers.l2(0.00001))(inputA)
    interp1B = Dense(75, activation='relu')(interpB)#,kernel_regularizer=tf.keras.regularizers.l2(0.00001), activity_regularizer=tf.keras.regularizers.l2(0.00001))(interpB)
    interp2B = Dense(75, activation='relu')(interp1B)#),kernel_regularizer=tf.keras.regularizers.l2(0.00001), activity_regularizer=tf.keras.regularizers.l2(0.00001))(interp1B)
    
    output = Dense(1+1, activation='linear')(interp2B)
    outputs =  tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:])))(output)
    
    model = Model(inputs=inputA, outputs=outputs)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=20000,
        decay_rate=0.95)
    model.compile(optimizer = keras.optimizers.Adam(
        learning_rate=0.001), loss=negloglik)
    logdir = os.path.join(logs_base_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
    # summarize layers
    print(model.summary())
    t1 = time.time()


    results=model.fit(X_train,y_train,batch_size=2000,epochs=1000,validation_data=(X_test,y_test),callbacks = [tensorboard_callback, callback])
    t2 = time.time()
    print('This took {} seconds'.format(round(t2-t1,2)))
    time_tot = t2-t1
    model.save("model_new_one.h5")
    print("Saved model to disk")
    return model, results, time_tot

def evaluate(X_val, y_val,files_nr):
    import time
    t1 = time.time()
    y_hat = model(X_val)
    mean = y_hat.mean()
    stddev = y_hat.stddev()
    t2 = time.time()
    print('This took {} seconds'.format(round(t2-t1,2)))
    y_testi = y_val

    fig, ax = plt.subplots(figsize=(15,6))
    ax.scatter(y_testi[::10], mean[::10],s=15)
       
    err = np.array(stddev)
    err = err.flatten()
    y_testi = np.array(y_testi)
    mean = np.array(mean)
    mean= mean.flatten()
    print(err.shape)
    ax.errorbar(x=y_testi[::10], y=mean[::10], yerr=err[::10], fmt="o")
    ax.plot([y_testi.min(), y_testi.max()], [y_testi.min(), y_testi.max()], 'k--', lw=4)

    ax.set_xlabel('Measured head change [m]')
    ax.set_ylabel('Predicted head change [m]')
    plt.title(f'Validation performance with {files_nr} simulations in training set')
    
    fig, ax = plt.subplots(figsize=(15,6))
    ax.scatter(y_testi[::10], mean[::10],s=15)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.plot([y_testi.min(), y_testi.max()], [y_testi.min(), y_testi.max()], 'k--', lw=4)

    ax.set_xlabel('Measured head change [m]')
    ax.set_ylabel('Predicted head change [m]')
    plt.title(f'Validation performance with {files_nr} simulations in training set')

    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import median_absolute_error
    MSE = mean_squared_error(y_testi,mean) #Mean square of the residuals
    print("MSE: {}" .format(round((MSE), 4))) #Root mean square error
    print("RMSE: {}" .format(round(np.sqrt(MSE), 4))) #Root mean square error
    print("NRMSE: {}" .format(round(np.sqrt(MSE)/(np.max(y_testi)-np.min(y_testi)), 5))) #Root mean square error
    return MSE, mean, err, y_testi, y_hat

if __name__ == "__main__":

    files_nr = [1000]
    loss = []
    mse = []
    epoch_nr = []
    train_time = []
    means = []
    errs = []
    loglike = []
    for i in range(len(files_nr)):
        data, data0, filess,b = load_data(files_nr[i])
        data_collection = data_processing(data,data0,files_nr[i],b) 
        X_train,X_test, X_val, y_train, y_test, y_val = Training_data(data_collection)
        model, results, time_tot = Training(X_train,X_test, y_train, y_test)
        MSE, mean, err, y_testi, y_hat = evaluate(X_val,y_val,files_nr[i])
        loss.append(results.history['loss'][-1])
        mse.append(MSE)
        epoch_nr.append(len(results.history['loss']))
        train_time.append(time_tot)
        means.append(mean)
        errs.append(err)
        log_like = []
        for i in range(len(err)):
           # log_like.append(-1/2*np.log(2*np.pi*err[i]**2)-1/(2*err[i]**2)*(y_val[i]-mean[i])**2)
           log_like.append( -1/2*( (y_val[i]-mean[i])**2 )/( err[i]**2 )) 
        
        loglike.append(np.sum(log_like))
       #loglike.append(np.mean(log_like))


    np.save('means.npy',means)
    np.save('loss.npy',loss)
    np.save('mse.npy',mse)
    np.save('epoch_nr.npy',epoch_nr)
    np.save('errs.npy',errs)
    np.save('train_time.npy',train_time)
    
    
