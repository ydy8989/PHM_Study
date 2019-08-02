from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Masking
from keras.optimizers import RMSprop
from keras import backend as k
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

i = 2
a= '0{}_M02_DC_train.csv'.format(i)
b= 'train_ttf\\0{}_M02_DC_train.csv'.format(i)
c = 'train_faults\\0{}_M02_train_fault_data.csv'.format(i)
aa = pd.read_csv(a)
bb = pd.read_csv(b)
cc = pd.read_csv(c)
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


aa1, bb1, cc1 = cutoff(aa, cc, bb, 'FlowCool Pressure Dropped Below Limit')

#%%
#Fixtureshutterposition ONE-HOT-ENCODING
def fixtureshutter_outlier(aa):
    aa.FIXTURESHUTTERPOSITIONixtureshutterposition
    return result

#%%
'''
module : lot, stage parsing
'''

lotlst = []
stglst = []
for idx, i in enumerate(cc1.time):
    print('Fault number :', idx+1)
    if len(aa[aa.time==i])==0:
        print(i)
        before_faultInd = max(aa[aa.time<i].index)
        print('Lot:',aa.iloc[before_faultInd]['Lot'],'stage', aa.iloc[before_faultInd]['stage'])
        lotlst.append(aa.iloc[before_faultInd]['Lot'])
        stglst.append(aa.iloc[before_faultInd]['stage'])
    else:
        print(i)
        print('lot error number:',aa[aa.time==i]['Lot'])#55)
        print('stage error number:',aa[aa.time==i]['stage'])#1
        lotlst.append(aa[aa.time==i]['Lot'].iloc[0])
        stglst.append(aa[aa.time==i]['stage'].iloc[0])
    print('======================================================')

print('length of lotlst:',len(lotlst))
print('length of stglst:',len(stglst))

#%%
outliers=[]
dataset= [10,12,12,13,12,11,14,13,15,10,10,10,100,12,14,13, 12,10, 10,11,12,15,12,13,12,11,14,13,15,10,15,12,10,14,13,15,10]
dataset = aa.FIXTURESHUTTERPOSITION
def mad_based_outlier(points, thresh=2):
#    if len(points.shape) == 1:
#        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh 
len(a = mad_based_outlier(dataset))
len(dataset[a])
dataset[a].value_counts()
len(dataset)

#%%
#aaa.loc[(4164,44),:].set_index('time').ix[:,2:].plot(figsize = (12,8))
aaa = aa.set_index(['Lot'])
for num, grp_lot in aa.groupby('Lot'):
    for num2, grp_stg in grp_lot.groupby('stage'):
        print(grp_stg.iloc[0][['Lot','stage']])
        grp_stg.set_index('time').ix[:,6:].plot(figsize = (12,8))
        plt.show()
#        print(grp_stg) # 하나하나가 미니df

#에러를 보유했던 랏과 스테이지 plot
for (a,b) in zip(lotlst, stglst):
    print('lot :',a,'stage',b)
    test_time = aa[aa.Lot==a].time.index
    aa.loc[test_time].ix[:,6:].plot(figsize = (12,8))
    plt.show()
    #
#    bb.loc[test_time].ix[:,1:].plot()
    
aa.ix[473500:473800,6:].plot(figsize = (13,10))

aa[(450000<aa.time) & (aa.time<500000)]

def split_norm_df(aaa):
    #이상치 제거한다.
        #분산으로 제거하면 좋음.
    #쪼갠다
          
    #정규화 때린다
    #다시 붙인다 'time'순으로
    #붙인걸 리턴한다.
    return result

#%%
len(cc1)

#
#(8627086-8697340)/4
#395385
#aa[386350:387000].ix[:,:5]
#


aaa = aa.set_index(['Lot','stage'])
aa.set_index(['Lot']).head()

for num, grp in aaa.groupby('Lot'):
    print(grp.head())
    break

aa[aa.Lot == 54].stage.value_counts()
#grp.set_index('time').ix[:,2:].plot(subplots = True, grid = True, figsize = (12, 40))
aaa.loc[(4164,44),:].set_index('time').ix[:,2:].plot(figsize = (12,8))
grp.loc[(201,68),:].set_index('time').ix[:,2:].plot(figsize = (12,8))
len(grp.loc[(201,48),:].set_index('time'))

grp.loc[(312,63),:].set_index('time').ix[:,2:].plot(figsize = (12,8))
grp.loc[(312,5),:].set_index('time').ix[:,2:].plot(figsize = (12,8))

grp.head()
grp.tail()


#%%
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


"""
    Load and parse engine data files into:
       - an (engine/day, observed history, sensor readings) x tensor, where observed history is 100 days, zero-padded
         for days that don't have a full 100 days of observed history (e.g., first observed day for an engine)
       - an (engine/day, 2) tensor containing time-to-event and 1 (since all engines failed)
    There are probably MUCH better ways of doing this, but I don't use Numpy that much, and the data parsing isn't the
    point of this demo anyway.
"""
def load_file(name):
    with open(name, 'r') as file:
        return np.loadtxt(file, delimiter=',')

np.set_printoptions(suppress=True, threshold=10000)

train = load_file('train.csv')
test_x = load_file('test_x.csv')
test_y = load_file('test_y.csv')

# Combine the X values to normalize them, then split them back out
all_x = np.concatenate((train[:, 2:26], test_x[:, 2:26]))
all_x = normalize(all_x, axis=0)

train[:, 2:26] = all_x[0:train.shape[0], :]
test_x[:, 2:26] = all_x[train.shape[0]:, :]

# Make engine numbers and days zero-indexed, for everybody's sanity
train[:, 0:2] -= 1
test_x[:, 0:2] -= 1

# Configurable observation look-back period for each engine/day
max_time = 100

def build_data(engine, time, x, max_time, is_test):
    # y[0] will be days remaining, y[1] will be event indicator, always 1 for this data
    out_y = np.empty((0, 2), dtype=np.float32)

    # A full history of sensor readings to date for each x
    out_x = np.empty((0, max_time, 24), dtype=np.float32)

    for i in range(100):
        print("Engine: " + str(i))
        # When did the engine fail? (Last day + 1 for train data, irrelevant for test.)
        max_engine_time = int(np.max(time[engine == i])) + 1

        if is_test:
            start = max_engine_time - 1
        else:
            start = 0

        this_x = np.empty((0, max_time, 24), dtype=np.float32)

        for j in range(start, max_engine_time):
            engine_x = x[engine == i]

            out_y = np.append(out_y, np.array((max_engine_time - j, 1), ndmin=2), axis=0)

            xtemp = np.zeros((1, max_time, 24))
            xtemp[:, max_time-min(j, 99)-1:max_time, :] = engine_x[max(0, j-max_time+1):j+1, :]
            this_x = np.concatenate((this_x, xtemp))

        out_x = np.concatenate((out_x, this_x))

    return out_x, out_y

train_x, train_y = build_data(train[:, 0], train[:, 1], train[:, 2:26], max_time, False)
test_x = build_data(test_x[:, 0], test_x[:, 1], test_x[:, 2:26], max_time, True)[0]

train_u = np.zeros((100, 1), dtype=np.float32)
train_u += 1
test_y = np.append(np.reshape(test_y, (100, 1)), train_u, axis=1)

"""
    Here's the rest of the meat of the demo... actually fitting and training the model.
    We'll also make some test predictions so we can evaluate model performance.
"""

# Start building our model
model = Sequential()

# Mask parts of the lookback period that are all zeros (i.e., unobserved) so they don't skew the model
model.add(Masking(mask_value=0., input_shape=(max_time, 24)))

# LSTM is just a common type of RNN. You could also try anything else (e.g., GRU).
model.add(LSTM(20, input_dim=24))

# We need 2 neurons to output Alpha and Beta parameters for our Weibull distribution
model.add(Dense(2))

# Apply the custom activation function mentioned above
model.add(Activation(activate))

# Use the discrete log-likelihood for Weibull survival data as our loss function
model.compile(loss=weibull_loglik_discrete, optimizer=RMSprop(lr=.001))

# Fit!
model.fit(train_x, train_y, nb_epoch=50, batch_size=2000, verbose=2, validation_data=(test_x, test_y))

# Make some predictions and put them alongside the real TTE and event indicator values
test_predict = model.predict(test_x)
test_predict = np.resize(test_predict, (100, 2))
test_result = np.concatenate((test_y, test_predict), axis=1)

# TTE, Event Indicator, Alpha, Beta
print(test_result)

#%%
import pandas as pd
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

