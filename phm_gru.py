import random
random.seed(1234)
# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
#!pip install seaborn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, GRU
import keras.callbacks
from keras import backend as K
from keras.models import Model
from keras.layers import Activation
from keras.layers import Masking

from keras.optimizers import RMSprop
from LoadAndProcessCSVFile_Classification import TimeStepSize
from LoadAndProcessCSVFile_Classification import loadAndProcessRawData

def Error(y_pred, y_real):
    y_pred = np.nan_to_num(y_pred, copy = True)
    y_real = np.nan_to_num(y_real, copy = True)
    temp = np.exp(-0.001 * y_real) * np.abs(y_real - y_pred)
    error = np.sum(temp)
    return error

def customLoss(y_pred, y_real):
    return K.sum(K.exp(-0.001 * y_real) * K.abs(y_real - y_pred))
def  activate ( ab ) :
    a = K.exp (ab [:, 0 ])
    b = K.softplus (ab [:, 1 ])

    a = K.reshape (a, (K.shape (a) [ 0 ], 1 ))
    b = K.reshape (b, (K.shape (b) [ 0 ], 1 ))

    return K.concatenate ((a, b), axis = 1 )

def weibull_loglik_discrete(y_true, ab_pred, name=None):
    y_ = y_true[:, 0]
    u_ = y_true[:, 1]
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]

    hazard0 = K.pow((y_ + 1e-35) / a_, b_)
    hazard1 = K.pow((y_ + 1) / a_, b_)

    return -1 * K.mean(u_ * K.log(K.exp(hazard1 - hazard0) - 1.0) - hazard1)
    
    
#------------------------------------------------------------------------------
# Read in Data
df = pd.DataFrame();
y = pd.DataFrame();


for i in range(1,2):

    sensorFilePath = '0{}_M01_DC_train.csv'.format(i)
    faultsFilePath = 'train_faults\\0{}_M01_train_fault_data.csv'.format(i)
    ttfFilePath = 'train_ttf\\0{}_M01_DC_train.csv'.format(i)
    df_tmp, y_tmp = loadAndProcessRawData(sensorFilePath, faultsFilePath, ttfFilePath)

    df = df.append(df_tmp)

    y = [y,y_tmp]
    y = pd.concat(y)
#    sensorFilePath = './data/train\\0{}_M02_DC_train.csv'.format(i,i)
#    faultsFilePath = './data/train\\train_faults\\0{}_M02_train_fault_data.csv'.format(i,i)
#    ttfFilePath = './data/train\\train_ttf\\0{}_M02_DC_train.csv'.format(i,i)
#    df_tmp, y_tmp = loadAndProcessRawData(sensorFilePath, faultsFilePath, ttfFilePath)
#
#    df = df.append(df_tmp)
#
#    y = [y,y_tmp]
#    y = pd.concat(y)

#%%
# scale data for better performance
df_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
y_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
feature = df_scaler.fit_transform(df)
label = y_scaler.fit_transform(y)


#------------------------------------------------------------------------------
# split data for train, validate, and test
x, X_test, y, y_test = train_test_split(feature,label,test_size=0.2,train_size=0.8)
X_train, X_valid, y_train, y_valid = train_test_split(x,y,test_size = 0.1,train_size =0.9)

#------------------------------------------------------------------------------
# LSTM
X_train = X_train.reshape((X_train.shape[0], TimeStepSize, 22))
X_valid = X_valid.reshape((X_valid.shape[0], TimeStepSize, 22))
X_all = feature.reshape((feature.shape[0], TimeStepSize, 22))

#------------------------------------------------------------------------------
# Train

model = Sequential()
model.add(GRU(10, return_sequences=True,  input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(GRU(10, return_sequences=True))
model.add(Dropout(0.3))
model.add(GRU(10))
model.add(Dense(1))
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss=customLoss, optimizer='adam')
# Early stopping
es = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')
history = model.fit(X_train, y_train, epochs=100, batch_size=256, \
                    validation_data=(X_valid, y_valid), verbose=2, shuffle=False)

#------------------------------------------------------------------------------
# Visualize
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

yhat = model.predict(X_all)
y_pred = y_scaler.inverse_transform(yhat)
y_real = y_scaler.inverse_transform(label)
plt.figure()
#t=np.arange(len(yhat))/len(label)*max(ttf_fault1['time'])/3600
#scale = 1/len(label)*max(ttf_fault1['TTF_FlowCool Pressure Dropped Below Limit'])/3600;
plt.plot(y_real[:,0],label="Real Data")
plt.plot(y_pred[:,0]*500,label="Predicted")
plt.xlabel("Time (hour)")
plt.ylabel("Remaining Life (hour)")
plt.title("Predicted Remaining Life v.s. Real Remaining Life")
plt.legend();
plt.show()


#%%

model.layers
layer_outputs = [layer.output for layer in model.layers]

layer_outputs
model.input



activation_model = Model(inputs=model.input, outputs=layer_outputs)
activation_model.summary()
activations = activation_model.predict(X_test)
learned_feature = activations[4]
#min(activations[4][7])
#max(activations[4][7])
learned_Label = activations[5]#신뢰도임. 
learned_Label = np.argmax(learned_Label, axis=1)
len(activations)
activations[4].shape
activations[5].shape

#output layer feature간의 상관관계를 보여주는 scatter
df.columns.values[:22]
df.runnum

plt.figure(figsize = (10,6))
plt.scatter(learned_feature[learned_Label==2,0], learned_feature[learned_Label==2,20], marker='^',c='r', label='High Risk')
plt.scatter(learned_feature[learned_Label==1,0], learned_feature[learned_Label==1,20], marker='s',c='b', label='Middle Risk')
plt.scatter(learned_feature[learned_Label==0,0], learned_feature[learned_Label==0,20], marker='+',c='g', label='Low Risk')
plt.legend()
#
#for i in learned_feature[:10]:
#    plt.figure()
#    plt.plot(i)