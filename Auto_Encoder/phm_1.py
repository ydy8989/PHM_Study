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
    
#------------------------------------------------------------------------------
# Read in Data
df = pd.DataFrame();
y = pd.DataFrame();
#root = 'C:\\Users\\hbee\\Desktop\\proj1\\data\\phm_data\\train'

for i in range(1,2):
    i = 1
    sensorFilePath = 'C:\\Users\\hbee\\Desktop\\proj1\\data\\phm_data\\train\\0{}_M01_DC_train.csv'.format(i)
    faultsFilePath = 'C:\\Users\\hbee\\Desktop\\proj1\\data\\phm_data\\train\\train_faults\\0{}_M01_train_fault_data.csv'.format(i)
    ttfFilePath = 'C:\\Users\\hbee\\Desktop\\proj1\\data\\phm_data\\train\\train_ttf\\0{}_M01_DC_train.csv'.format(i) 
    df_tmp, y_tmp = loadAndProcessRawData(sensorFilePath, faultsFilePath, ttfFilePath)

    df = df.append(df_tmp)
    y = [y,y_tmp]
    y = pd.concat(y)
#    sensorFilePath = 'C:\\Users\\hbee\\Desktop\\proj1\\data\\phm_data\\train\\0{}_M02_DC_train.csv'.format(i,i)
#    faultsFilePath = 'C:\\Users\\hbee\\Desktop\\proj1\\data\\phm_data\\train\\train_faults\\0{}_M02_train_fault_data.csv'.format(i,i)
#    ttfFilePath = 'C:\\Users\\hbee\\Desktop\\proj1\\data\\phm_data\\train\\train_ttf\\0{}_M02_DC_train.csv'.format(i,i) 
#    df_tmp, y_tmp = loadAndProcessRawData(sensorFilePath, faultsFilePath, ttfFilePath)
#
#    df = df.append(df_tmp)
#
#    y = [y,y_tmp]
#    y = pd.concat(y)

X1 = df.values

y1 = y.values.reshape((y.shape[0],)).astype('int64')
def tsne_plot(x1, y1, name="graph.png"):
#    x1 = X.copy()
#    y1 = Y.copy()
    
    tsne = TSNE(n_components=3, random_state=0)
    X_t = tsne.fit_transform(x1)

    plt.figure(figsize=(12, 8))
    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth='1', alpha=0.8, label='label0')
    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='r', linewidth='1', alpha=0.8, label='label1')
    plt.scatter(X_t[np.where(y1 == 2), 0], X_t[np.where(y1 == 2), 1], marker='o', color='b', linewidth='1', alpha=0.8, label='label2')
    
    plt.legend(loc='best');
#    plt.savefig(name);
    plt.show();
tsne_plot(X1, y1, "original.png")


#%%
#------------------------------------------------------------------------------
# scale data for better performance
df_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
y_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
feature = df_scaler.fit_transform(df)




#label = y_scaler.fit_transform(y)
label = keras.utils.to_categorical(y)

#------------------------------------------------------------------------------
# split data for train, validate, and test
x, X_test, y, y_test = train_test_split(feature,label,test_size=0.2,train_size=0.8)
X_train, X_valid, y_train, y_valid = train_test_split(x,y,test_size = 0.1,train_size =0.9)
label.shape
#------------------------------------------------------------------------------
# Gated Recurrent Unit
X_train = X_train.reshape((X_train.shape[0], TimeStepSize, 22))
X_valid = X_valid.reshape((X_valid.shape[0], TimeStepSize, 22))
X_test = X_test.reshape((X_test.shape[0], TimeStepSize, 22))
X_all = feature.reshape((feature.shape[0], TimeStepSize, 22))

#------------------------------------------------------------------------------
# Train
#%%
model = Sequential()
model.add(GRU(22, return_sequences=True,  input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(GRU(22, return_sequences=True))
model.add(Dropout(0.3))
model.add(GRU(22))
model.add(Dense(3, activation='softmax'))
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#model.compile(loss=customLoss, optimizer='adam')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Early stopping
es = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')
history = model.fit(X_train,y_train, epochs=50, batch_size=256, \
                    validation_data=(X_valid, y_valid), verbose=2, shuffle=False)
#%%
#------------------------------------------------------------------------------

# Visualize
model.summary()

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validate')
plt.legend()
plt.show()

df_plot = pd.DataFrame({'epochs':history.epoch, 'accuracy': history.history['acc'], 'validation_accuracy': history.history['val_acc']})
g = sns.pointplot(x="epochs", y="accuracy", data=df_plot, fit_reg=False)
g = sns.pointplot(x="epochs", y="validation_accuracy", data=df_plot, fit_reg=False, color='green')

predicted = model.predict(X_test)
predicted = np.argmax(predicted, axis=1)
y = np.argmax(y_test, axis=1)
print( 'matching score is ', accuracy_score(y, predicted))

print(y)
print(predicted)


#loss function print하기...

plt.figure(figsize = (13,7))
plt.plot(y[:500], label = 'train')

plt.figure(figsize = (13,7))
plt.plot(predicted[:500])
plt.legend()
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

#컬럼간의 상관관계를 보여주는 scatter
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