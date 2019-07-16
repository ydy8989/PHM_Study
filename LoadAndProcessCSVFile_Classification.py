import random
random.seed(1234)
# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd 
    
sampleRate = 10  
TimeStepSize = 60 # every 600 second
n_skipSample = 30 # skip a lot of samples
#------------------------------------------------------------------------------
# drop the rows having nan values in ttf_data
def cutoff(sensor_data, faults_data, ttf_data, column):
    # cut off the tail of the data set that with NaN ttf
    temp = faults_data[faults_data['fault_name'] == column]
    last_failure = temp['time'].values[-1]
    array = np.asarray(sensor_data['time'])
    closest_ind = (np.abs(array - last_failure)).argmin()
    if ((array[closest_ind] - last_failure) != np.abs(array[closest_ind] - last_failure)):
        ind = closest_ind + 1
    elif ((array[closest_ind] - last_failure) == 0):
        ind = closest_ind + 1
    else:
        ind = closest_ind
    sensor_data = sensor_data[:ind]
    ttf_data = ttf_data[:ind]
    faults_data = faults_data[faults_data['fault_name'] == column]
    return sensor_data, ttf_data, faults_data
    
#------------------------------------------------------------------------------
# Shift dataset
def series_to_supervised(data, y, n_in=100, n_skip = 30, dropnan=True):
    data_col = []
    for i in range (0, n_in):
        data_col.append(data.shift(i).ix[::n_skip,])
    result = pd.concat(data_col, axis = 1)
    label = y[::n_skip]
    #pd.concat(y_col.append(y.shift(1)), axis = 1)
    
    if dropnan:
        result = result[n_in:]
        label = label[n_in:]
    return result, label
    
#------------------------------------------------------------------------------
# Read in Data
def loadAndProcessRawData(sensorFilePath, faultsFilePath, ttfFilePath):      
    #파일 불러옴
    sensor_data = pd.read_csv(sensorFilePath)
    faults_data = pd.read_csv(faultsFilePath)
    ttf_data = pd.read_csv(ttfFilePath)
    #노필요 컬럼 드랍시킴
    sensor_data = sensor_data.drop(['Tool'], axis = 1)
    sensor_data = sensor_data.drop(['Lot'], axis = 1)
    #그냥 인덱스 재설정? 정도가되겠다
    sensor_data.index = range(0,len(sensor_data))
    ttf_data.index = range(0,len(ttf_data))
    
    #그냥 앞으로 당긴거, fault time 기준으로.
    sensor_fault1, ttf_fault1, faults_fault1 = cutoff(sensor_data, faults_data, \
                        ttf_data, 'FlowCool Pressure Dropped Below Limit')    
    
    sensor_fault1 = sensor_fault1.fillna(method = 'ffill')
    sensor_fault1['recipe'] = sensor_fault1['recipe'] + 200
    label = ttf_fault1['TTF_FlowCool Pressure Dropped Below Limit']
    #label.plot()
    
    w0 = 86400 # fail within one day
    w1 = 604800 # fail within one week
    label_tmp = label.copy()

    label_tmp.loc[label <= w0] = 2   
    label_tmp.loc[(label < w1) & (label > w0)] = 1
    label_tmp.loc[label >= w1] = 0
    label_tmp.value_counts()
    label = label_tmp
#    label_tmp.value_counts()
    #------------------------------------------------------------------------------
    
    # down sample to data 
        #여기다가 tumbling function 삽입
    df_select, y_select = sensor_fault1.ix[::sampleRate], label.ix[::sampleRate]

    df, y = series_to_supervised(df_select, y_select, TimeStepSize, n_skipSample, True)
    
    # scale the date for better performance
    #df_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
    #y_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
    #feature = df_scaler.fit_transform(df)
    #label = y_scaler.fit_transform(y)
    
    return df, y