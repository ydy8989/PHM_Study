from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

sns.set(style = 'whitegrid')
np.random.seed(201)

#data load
data = pd.read_csv('03_M02_DC_train.csv')
ttf = pd.read_csv('./train_ttf/03_M02_DC_train.csv')
faults = pd.read_csv('./train_faults/03_M02_train_fault_data.csv')

data = data.drop(['Tool','Lot'], axis = 1)



########################### 
#           EDA           
########################### 
data



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
#    faults_data = faults_data[faults_data['fault_name'] == column]
    return sensor_data, ttf_data
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
data = data.fillna(method = 'ffill')
data.recipe = data.recipe + 200
#여기까지는 거의 필수 단계임... ttf랑 fault 차이의 간극을 줄이는 작업   
sensor_fault1, ttf_fault1 = cutoff(data, faults, ttf, 'FlowCool Pressure Dropped Below Limit') 
#%%
#for i in range(0,200000,100):
#    sensor_fault1.ix[i:i+600,17].plot(grid = True, figsize = (10, 3))
#    plt.show()

#%%

#%%

#그래프 그려보면 [1850000:2000000].plot(subplots = True, figsize = (10,15)) recipe가 0임... 이상.
#%%
sensor_fault1 = sensor_fault1.fillna(method = 'ffill')
dropcol = [2,7,9,12,13]
dropcolname = list(sensor_fault1.columns[dropcol])
sensor_fault1 = sensor_fault1.drop(dropcolname, axis = 1)
selcol = [0,14,15,16]
selcolname = list(sensor_fault1.columns[selcol])

sensor2 = sensor_fault1.ix[:,selcol]

plt.figure(figsize=(15,15))
sns.heatmap(data = sensor2.corr(), annot=True, 
fmt = '.2f', linewidths=.5, cmap='Blues')

len(sensor_fault1.columns)



#%%
label = ttf_fault1['TTF_FlowCool Pressure Dropped Below Limit']

#print(sensor_fault1.shape, label.shape)
label.plot(figsize = (15,5))


#하루 이내에 고장은 1 아닌건 다 0으로 레이블링 하는 함수.
w0 = 86400 * 2
label_tmp = label.copy()
label_tmp.loc[label<=w0] = 1
label_tmp.loc[label > w0] = 0
label = label_tmp

#sensor_fault1.ix[:,18].describe()
#label_1 = sensor_fault1[label==1].ix[:,3]
import matplotlib.gridspec as gridspec
plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(sensor_fault1.columns[:11]):#[16,18,19,20]
    ax = plt.subplot(gs[i])
    sns.distplot(sensor_fault1[cn][label == 1])
    sns.distplot(sensor_fault1[cn][label == 0])
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.show()
#sensor_fault1[sensor_fault1.columns[19]][label == 0].value_counts().sum()

#%%




#생각보다 없는 하루 직전 에러들.... 눈으로 확인하시길
#plt.figure(figsize=(12, 8))
#plt.scatter(label_tmp.index[label_tmp==1], label_tmp[label_tmp==1], marker = 'o', color='g', linewidth='1', alpha=0.8, label='high risk')
#plt.scatter(label_tmp.index[label_tmp!=1], label_tmp[label_tmp!=1], marker = 'o', color='r', linewidth='1', alpha=0.8, label='low risk')
#

#data
#label


#다운 샘플링
    #다운샘플링 하고나서
    #일단 오른쪽으로 쉬프팅 후 셔플 : 윈도우 섞기임
sampleRate = 5
down_data, down_label = data.ix[::sampleRate], label.ix[::sampleRate]
df, y = series_to_supervised(down_data, down_label, 1,1)


print(df.shape)
print(y.shape)
y.value_counts()

def tsne_plot(x1, y1, name="graph.png"):
#    x1 = X.copy()
#    y1 = Y.copy()
    
    tsne = TSNE(n_components=2, random_state=0)
    X_t = tsne.fit_transform(x1)

    plt.figure(figsize=(12, 8))
    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth='1', alpha=0.8, label='Non Fraud')
    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='r', linewidth='1', alpha=0.8, label='Fraud')

    plt.legend(loc='best');
#    plt.savefig(name);
    plt.show();
    
#tsne_plot(df,y)

#어차피 ttf에 없는 타임라인에 대해서 df에서 마지막을 자를 필요가 없음. 
#왜냐하면 인덱스로 호출할거라;
non_fraud = df.loc[y[y == 0].index].sample(frac = 1)
fraud = df.loc[y[y==1].index]
main_df = non_fraud.append(fraud).sample(frac = 1).reset_index(drop = True)



input_layer = Input(shape=(main_df.shape[1],)) #인풋 레이어를 컬럼갯수로 하겠다는 마인드.

## encoding part
encoded = Dense(22, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoded = Dense(10, activation='relu')(encoded)
encoded = Dense(7, activation='relu')(encoded)

## decoding part
decoded = Dense(10, activation='tanh')(encoded)
decoded = Dense(22, activation='tanh')(decoded)

## output layer
output_layer = Dense(main_df.shape[1], activation='relu')(decoded)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer="adadelta", loss="mse")


x_scale = preprocessing.MinMaxScaler().fit_transform(non_fraud.values)
x_fraud = preprocessing.MinMaxScaler().fit_transform(fraud.values)
#x_scale.shape
#x_norm, x_fraud = x_scale[y == 0], x_scale[y == 1]

autoencoder.fit(x_scale, x_scale, 
                batch_size = 256, epochs = 50, 
                shuffle = True, validation_split = 0.20)

hidden_representation = Sequential()
hidden_representation.add(autoencoder.layers[0])
hidden_representation.add(autoencoder.layers[1])
hidden_representation.add(autoencoder.layers[2])
hidden_representation.add(autoencoder.layers[3])

norm_hid_rep = hidden_representation.predict(x_scale)
fraud_hid_rep = hidden_representation.predict(x_fraud)

rep_x = np.append(norm_hid_rep, fraud_hid_rep, axis = 0)
y_n = np.zeros(norm_hid_rep.shape[0])
y_f = np.ones(fraud_hid_rep.shape[0])
rep_y = np.append(y_n, y_f)
#tsne_plot(rep_x, rep_y, "latent_representation.png")


train_x, val_x, train_y, val_y = train_test_split(rep_x, rep_y, test_size=0.25)
clf = LogisticRegression(solver="lbfgs").fit(train_x, train_y)
pred_y = clf.predict(val_x)

print ("")
print ("Classification Report: ")
print (classification_report(val_y, pred_y))

print ("")
print ("Accuracy Score: ", accuracy_score(val_y, pred_y))

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(val_y, pred_y)
LABELS = ["Normal", "Break"]
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
