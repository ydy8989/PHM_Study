#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Masking
from keras.layers import BatchNormalization

from keras import backend as k
from keras import callbacks

from sklearn.preprocessing import normalize

import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

from six.moves import xrange
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.layers import LSTM,GRU
from keras.layers import Lambda
from keras.layers.wrappers import TimeDistributed

from keras.optimizers import RMSprop,adam
from keras.callbacks import History

import wtte.weibull as weibull
import wtte.wtte as wtte

from wtte.wtte import WeightWatcher

np.random.seed(2)
pd.set_option("display.max_rows",1000)

#%%




# In[3]:


"""
    Discrete log-likelihood for Weibull hazard function on censored survival data
    y_true is a (samples, 2) tensor containing time-to-event (y), and an event indicator (u)
    ab_pred is a (samples, 2) tensor containing predicted Weibull alpha (a) and beta (b) parameters
    For math, see https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf (Page 35)
"""
def weibull_loglik_discrete(y_true, ab_pred, name=None):
    y_ = y_true[:, 0]
    u_ = y_true[:, 1]
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]

    hazard0 = k.pow((y_ + 1e-35) / a_, b_)
    hazard1 = k.pow((y_ + 1) / a_, b_)

    return -1 * k.mean(u_ * k.log(k.exp(hazard1 - hazard0) - 1.0) - hazard1)

"""
    Not used for this model, but included in case somebody needs it
    For math, see https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf (Page 35)
"""
def weibull_loglik_continuous(y_true, ab_pred, name=None):
    y_ = y_true[:, 0]
    u_ = y_true[:, 1]
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]

    ya = (y_ + 1e-35) / a_
    return -1 * k.mean(u_ * (k.log(b_) + b_ * k.log(ya)) - k.pow(ya, b_))


"""
    Custom Keras activation function, outputs alpha neuron using exponentiation and beta using softplus
"""
def activate(ab):
    a = k.exp(ab[:, 0])
    b = k.softplus(ab[:, 1])

    a = k.reshape(a, (k.shape(a)[0], 1))
    b = k.reshape(b, (k.shape(b)[0], 1))

    return k.concatenate((a, b), axis=1)


# In[4]:


"""
    Load and parse engine data files into:
       - an (engine/day, observed history, sensor readings) x tensor, where observed history is 100 days, zero-padded
         for days that don't have a full 100 days of observed history (e.g., first observed day for an engine)
       - an (engine/day, 2) tensor containing time-to-event and 1 (since all engines failed)
    There are probably MUCH better ways of doing this, but I don't use Numpy that much, and the data parsing isn't the
    point of this demo anyway.
"""
pass


# In[5]:


id_col = 'unit_number'
time_col = 'time'
feature_cols = [ 'op_setting_1', 'op_setting_2', 'op_setting_3'] + ['sensor_measurement_{}'.format(x) for x in range(1,22)]
column_names = [id_col, time_col] + feature_cols


# In[6]:


np.set_printoptions(suppress=True, threshold=10000)

train_orig = pd.read_csv('train.csv', header=None, names=column_names)
test_x_orig = pd.read_csv('test_x.csv', header=None, names=column_names)
test_y_orig = pd.read_csv('test_y.csv', header=None, names=['T'])


# In[7]:
#
#
#test_x_orig.set_index(['unit_number', 'time'], verify_integrity=True)
#
#
## In[8]:
#
#
#train_orig.set_index(['unit_number', 'time'], verify_integrity=True)


# In[8]:


# help(normalize)


# In[9]:
#노말라이제이션 cell


from sklearn import pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#컴바인 데이터 : 아래로 merge 
all_data_orig = pd.concat([train_orig, test_x_orig])
# all_data = all_data[feature_cols]
# all_data[feature_cols] = normalize(all_data[feature_cols].values)

scaler=pipeline.Pipeline(steps=[
#     ('z-scale', StandardScaler()),
     ('minmax', MinMaxScaler(feature_range=(-1, 1))),
     ('remove_constant', VarianceThreshold())
])

all_data = all_data_orig.copy()
all_data.info()
#인덱스로 만든 unit_number랑 time 빼고 나머지는 다 스케일링 하긴 하는데, 분산 기준치보다 낮은 피쳐(컬럼)은 삭제
all_data = np.concatenate([all_data[['unit_number', 'time']], scaler.fit_transform(all_data[feature_cols])], axis=1)
all_data.shape #>>> 컬럼 7ㄱㅐ 삭제됨.


# In[10]:

'''
정규화 시점에 대한 고민 필요
스플릿 전 vs 후
'''

# then split them back out
train = all_data[0:train_orig.shape[0], :]
test = all_data[train_orig.shape[0]:, :]


# Make engine numbers and days zero-indexed, for everybody's sanity
train[:, 0:2] -= 1
test[:, 0:2] -= 1


# In[11]:


import tqdm
from tqdm import tqdm

# TODO: replace using wtte data pipeline routine
def build_data(engine, time, x, max_time, is_test, mask_value):
#    engine=train[:, 0]
#    time=train[:, 1]
#    x=train[:, 2:]
#    max_time=max_time
#    is_test=False
#    mask_value=mask_value

    # y[0] will be days remaining, y[1] will be event indicator, always 1 for this data
    out_y = []
    
    #피쳐 갯수
    d = x.shape[1]

    # A full history of sensor readings to date for each x
    out_x = []
    
    #우리꺼는 stage 기준이면, 조금 더 다르게 해야함. 100개가 아니라 134개? 일걸
    n_engines=100
    for i in tqdm(range(n_engines)):
        # When did the engine fail? (Last day + 1 for train data, irrelevant for test.)
        max_engine_time = int(np.max(time[engine == i])) + 1 # 해당 엔진넘버에서의 맥스 시간, 그러니깐 고장 직전 시간을 봄
        #테스트 true면, 한칸짜리로 걍 연습하는거임...
        if is_test:
            start = max_engine_time - 1
        else:
            start = 0

        this_x = []

        for j in range(start, max_engine_time):
            engine_x = x[engine == i]

            out_y.append(np.array((max_engine_time - j, 1), ndmin=2))
            #마스킹하기 -99로
            xtemp = np.zeros((1, max_time, d))
            xtemp += mask_value #앞에 마스크 씌우기
#             xtemp = np.full((1, max_time, d), mask_value)
            
            xtemp[:, max_time-min(j, 99)-1:max_time, :] = engine_x[max(0, j-max_time+1):j+1, :]
            this_x.append(xtemp)
            
        this_x = np.concatenate(this_x)
        out_x.append(this_x)
        
    out_x = np.concatenate(out_x)
    out_y = np.concatenate(out_y)
    return out_x, out_y


# In[12]:


# # Configurable observation look-back period for each engine/day
max_time = 100
mask_value = -99

train_x, train_y = build_data(engine=train[:, 0], time=train[:, 1], x=train[:, 2:], max_time=max_time, is_test=False, mask_value=mask_value)
test_x,_ = build_data(engine=test[:, 0], time=test[:, 1], x=test[:, 2:], max_time=max_time, is_test=True, mask_value=mask_value)



# In[13]:


# train_orig.groupby('unit_number')['time'].describe()


# In[14]:


# always observed in our case
test_y = test_y_orig.copy()
test_y['E'] = 1


# In[56]:


print('train_x =', train_x.shape, 'train_y =', train_y.shape, 'test_x =', test_x.shape, 'test_y =', test_y.shape)


# In[16]:

tte_mean_train = np.nanmean(train_y[:,0])
mean_u = np.nanmean(train_y[:,1])

# Initialization value for alpha-bias 
init_alpha = -1.0/np.log(1.0-1.0/(tte_mean_train+1.0) )
init_alpha = init_alpha/mean_u
print('tte_mean_train', tte_mean_train, 'init_alpha: ',init_alpha,'mean uncensored train: ',mean_u)


# In[17]:


import keras.backend as K
K.set_epsilon(1e-10)
print('epsilon', K.epsilon())

history = History()
weightwatcher = WeightWatcher()
nanterminator = callbacks.TerminateOnNaN()
# reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', 
#                                         factor=0.5, 
#                                         patience=50, 
#                                         verbose=0, 
#                                         mode='auto', 
#                                         epsilon=0.0001, 
#                                         cooldown=0, 
#                                         min_lr=1e-8)

n_features = train_x.shape[-1]

# Start building our model
model = Sequential()

# Mask parts of the lookback period that are all zeros (i.e., unobserved) so they don't skew the model
model.add(Masking(mask_value=mask_value, input_shape=(None, n_features)))

# model.add(BatchNormalization())

# LSTM is just a common type of RNN. You could also try anything else (e.g., GRU).
# model.add(GRU(20, activation='tanh', recurrent_dropout=0.25))
model.add(GRU(20, activation='tanh', recurrent_dropout=0.25))

# model.add(Dense(20))

# We need 2 neurons to output Alpha and Beta parameters for our Weibull distribution
# model.add(TimeDistributed(Dense(2)))
model.add(Dense(2))

# Apply the custom activation function mentioned above
# model.add(Activation(activate))

model.add(Lambda(wtte.output_lambda, 
                 arguments={"init_alpha":init_alpha, 
                            "max_beta_value":100.0, 
                            "alpha_kernel_scalefactor":0.5
                           },
                ))

# Use the discrete log-likelihood for Weibull survival data as our loss function
loss = wtte.loss(kind='discrete',reduce_loss=False).loss_function

model.compile(loss=loss, optimizer=adam(lr=.01, clipvalue=0.5))
# model.compile(loss=loss, optimizer=RMSprop(lr=0.01))


# In[18]:


model.summary()


# In[97]:


#model.fit(train_x, train_y,
#          epochs=1,
#          batch_size=100, 
#          verbose=1,
#          validation_data=(test_x, test_y),
#          callbacks=[nanterminator,history,weightwatcher])


# In[39]:


model.fit(train_x, train_y,
              epochs=10,
              batch_size=100, 
              verbose=1,
              validation_data=(test_x, test_y),
              callbacks=[nanterminator,history,weightwatcher])


# In[72]:


# Fit!
# model.fit(train_x, train_y, epochs=200, batch_size=train_x.shape[0]//10, verbose=1, validation_data=(test_x, test_y),
#          callbacks=[history,weightwatcher])


# In[40]:


plt.plot(history.history['loss'],    label='training')
plt.plot(history.history['val_loss'],label='validation')
plt.title('loss')
plt.legend()

weightwatcher.plot()


# In[42]:


# Make some predictions and put them alongside the real TTE and event indicator values
test_predict = model.predict(test_x)
test_predict = np.resize(test_predict, (100, 2))
test_result = np.concatenate((test_y, test_predict), axis=1)


# In[43]:


test_results_df = pd.DataFrame(test_result, columns=['T', 'E', 'alpha', 'beta'])
test_results_df['unit_number'] = np.arange(1, test_results_df.shape[0]+1)

# test_results_df = pd.concat([test_x_orig, test_results_df], axis=1)
# test_results_df = test_results_df.merge(test_x_orig, on=['unit_number'], how='right')


# In[44]:


test_results_df


# In[45]:


def weibull_pdf(alpha, beta, t):
    return (beta/alpha) * (t/alpha)**(beta-1)*np.exp(- (t/alpha)**beta)


# In[46]:


def weibull_median(alpha, beta):
    return alpha*(-np.log(.5))**(1/beta)


# In[47]:


def weibull_mean(alpha, beta):
    return alpha * math.gamma(1 + 1/beta)


# In[48]:


def weibull_mode(alpha, beta):
    assert np.all(beta > 1)
    return alpha * ((beta-1)/beta)**(1/beta)


# In[49]:


test_results_df['T'].describe()


# In[50]:


get_ipython().run_line_magic('matplotlib', 'inline')
import math
import matplotlib.pyplot as plt
t=np.arange(0,300)
alpha = test_results_df['alpha'].mean()
beta = test_results_df['beta'].mean()

plt.plot(t, weibull_pdf(alpha,beta, t))
mu = weibull_mean(alpha, beta)
median =weibull_median(alpha, beta)
mode = weibull_mode(alpha, beta)
plt.axvline(mu, ls='--', label='mean')
plt.axvline(median, ls='--', color='red', label='median')
plt.axvline(mode, ls='--', color='green', label='mode')
n, bins, patches = plt.hist(test_results_df['T'], 20, normed=1, facecolor='grey', alpha=0.75, label='T')
plt.legend()

plt.gcf().set_size_inches(12.5, 8)
plt.title('Average Weibull distribution over test set')
print('alpha', alpha, 'beta', beta)


# In[51]:


import seaborn as sns
palette=sns.color_palette("RdBu_r", 50)


# In[52]:


sns.palplot(palette)


# In[53]:


train_orig.describe()


# In[54]:


feature_columns = [x for x in test_x_orig.columns if x not in {'unit_number', 'time'}]

mins=train_orig[feature_columns].min()
maxs=train_orig[feature_columns].max()

for unit_no, grp in test_x_orig.groupby('unit_number'):
    df=grp.set_index('time')
    df = df[feature_columns]
    df=(df - mins)/ (maxs - mins)
    df.plot(figsize=(12.5,8))
    plt.title(unit_no)
    plt.show()


# In[ ]:





# In[34]:


def plot_weibull_predictions(results_df):

    fig, axarr = plt.subplots(3, figsize=(20,30))

    t=np.arange(0,400)

    palette = sns.color_palette("RdBu_r", results_df.shape[0] + 1)
    color_dict = dict(enumerate(palette))

    for i, row in enumerate(results_df.iterrows()):
        alpha=row[1]['alpha']
        beta = row[1]['beta']
        T = row[1]['T']
        label = 'a={} b={}'.format(alpha, beta)

        color = color_dict[i]
        ax= axarr[0]
        mode = weibull_mode(alpha, beta)
        y_max = weibull_pdf(alpha, beta, mode)    

        ax.plot(t, weibull_pdf(alpha, beta, t), color=color, label=label)
        ax.scatter(T, weibull_pdf(alpha,beta, T), color=color, s=100)
        ax.vlines(mode, ymin=0, ymax=y_max, colors=color, linestyles='--')

        ax.set_title('Weibull distributions')
    
    ax=axarr[1]
    
    median_predictions = weibull_median(results_df['alpha'], results_df['beta'])
    mean_predictions = results_df[['alpha', 'beta']].apply(lambda row: weibull_mean(row[0], row[1]), axis=1)
    mode_predictions = weibull_mode(results_df['alpha'], results_df['beta'])
#     x = results_df['time']
    
#     ax.scatter(x, results_df['T'], label='survival_time', color='black')

#     ax.scatter(results_df['T'], median_predictions, label='median_prediction')
#     ax.scatter(results_df['T'], mean_predictions, label='mean_prediction')
    ax.scatter(results_df['T'], mode_predictions, label='m_prediction')
    ax.set_title('MAP prediction Vs. true')
    

    ax.legend()
    
    ax=axarr[2]
    sns.distplot(results_df['T'] - mode_predictions, ax=ax)
    ax.set_title('Error')

#     ax.plot(x, results_df['alpha'], label='alpha')
#     ax.legend()
    
#     ax = axarr[3]
#     ax.plot(x, results_df['beta'], label='beta')
#     ax.legend()
    
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     fig.suptitle(title)
    plt.show()


# In[35]:


plot_weibull_predictions(results_df=test_results_df)


# In[36]:


test_results_df['predicted_mu'] = test_results_df[['alpha', 'beta']].apply(lambda row: weibull_mean(row[0], row[1]), axis=1)
test_results_df['predicted_median'] = test_results_df[['alpha', 'beta']].apply(lambda row: weibull_median(row[0], row[1]), axis=1)
test_results_df['predicted_mode'] = test_results_df[['alpha', 'beta']].apply(lambda row: weibull_mode(row[0], row[1]), axis=1)


# In[37]:


import seaborn as sns
sns.jointplot(data=test_results_df, y='T', x='predicted_mode',kind="reg")


# In[101]:


sns.jointplot(data=test_results_df, y='T', x='predicted_median',kind="kde" )


# In[102]:


test_results_df['error'] = test_results_df['T']-test_results_df['predicted_median']


# In[103]:


test_results_df['error'].describe()


# In[104]:


sns.distplot(test_results_df['error'], bins=20)


# # Training evaluation

# In[68]:


test_y.shape


# In[67]:


train_y.shape


# In[72]:


train_predict=model.predict(train_x)
# train_predict = np.resize(train_predict, (20631, 2))
# train_result = np.concatenate((train_y, train_predict), axis=1)


# In[73]:


train_predict.shape


# In[105]:


train_results_df = pd.DataFrame(train_result, columns=['T', 'E', 'alpha', 'beta'])


# In[109]:


train_results_df[['unit_number', 'time']] = train_orig[['unit_number', 'time']]


# In[186]:


train_results_df['unit_number'].nunique()


# In[65]:


train_results_df.shape


# In[247]:


train_results_df.groupby('unit_number')['beta'].describe()


# In[ ]:


for unit_number, grp in train_results_df.groupby('unit_number'):
    plot_weibull_predictions(grp, unit_number)


# In[ ]:





# In[193]:


train_results_df['predicted_mu'] = train_results_df[['alpha', 'beta']].apply(lambda row: weibull_mean(row[0], row[1]), axis=1)
train_results_df['predicted_median'] = train_results_df[['alpha', 'beta']].apply(lambda row: weibull_median(row[0], row[1]), axis=1)


# In[194]:


import seaborn as sns
sns.jointplot(data=train_results_df, y='T', x='predicted_median',kind="reg")


# In[195]:


sns.jointplot(data=train_results_df, y='T', x='predicted_median',kind="kde" )


# In[197]:


train_results_df['error'] = train_results_df['T']-train_results_df['predicted_median']


# In[198]:


train_results_df['error'].describe()


# In[199]:


sns.distplot(train_results_df['error'], bins=20)


# In[ ]:




