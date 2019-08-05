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

i1 = 1
i2 = 1
a= '0{}_M0{}_DC_train.csv'.format(i1,i2)
b= 'train_ttf/0{}_M0{}_DC_train.csv'.format(i1,i2)
c = 'train_faults/0{}_M0{}_train_fault_data.csv'.format(i1,i2)

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
#def fixtureshutter_outlier(aa):
#    aa.FIXTURESHUTTERPOSITIONixtureshutterposition
#    return result

#%%
#aa.ix[:,7:].plot(subplots = True, grid = True, figsize = (12,34))

def std_based_outlier(df):
    for i in range(7, len(df.iloc[1])): 
        df.iloc[:,i] = df.iloc[:,i].replace(0, np.NaN) #null
        df = df[~(np.abs(df.iloc[:,i] - df.iloc[:,i].mean()) > (3*df.iloc[:,i].std()))].fillna(0)
        print('{}th-colum processing.......'.format(i))       
    return df
outlier_aa = std_based_outlier(aa1)
#outlier_aa.ix[321800:321900,-4:-1].plot(figsize = (12,6), subplots = True)
#outlier_aa.plot(subplots = True, grid = True, figsize = (12,34))



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
##%%
#outliers=[]
#dataset= [10,12,12,13,12,11,14,13,15,10,10,10,100,12,14,13, 12,10, 10,11,12,15,12,13,12,11,14,13,15,10,15,12,10,14,13,15,10]
#dataset = aa.FIXTURESHUTTERPOSITION
#def mad_based_outlier(points, thresh=2):
##    if len(points.shape) == 1:
##        points = points[:,None]
#    median = np.median(points, axis=0)
#    diff = np.sum((points - median)**2)
#    diff = np.sqrt(diff)
#    med_abs_deviation = np.median(diff)
#    modified_z_score = 0.6745 * diff / med_abs_deviation
#    return modified_z_score > thresh 
#len(a = mad_based_outlier(dataset))
#len(dataset[a])
#dataset[a].value_counts()
#len(dataset)

#%%
#aaa.loc[(4164,44),:].set_index('time').ix[:,2:].plot(figsize = (12,8))
#every lots and stages's 
for num, grp_lot in aa.groupby('Lot'):
    for num2, grp_stg in grp_lot.groupby('stage'):
        print(grp_stg.iloc[0][['Lot','stage']])
        grp_stg.set_index('time').ix[:,6:].plot(figsize = (12,8))
        plt.show()
#        print(grp_stg) # 하나하나가 미니df

#%%
#에러를 보유했던 랏과 스테이지 plot
for (a,b) in zip(lotlst, stglst):
    print('lot :',a,'stage',b)
    test_time = aa[aa.Lot==a].time.index
    aa.loc[test_time].ix[:,7:].plot(figsize = (12,8))
    plt.show()
    
#    bb.loc[test_time].ix[:,1:].plot()
   #%% 
#
#aa.ix[473500:473800,6:].plot(figsize = (13,10))
#aa[(450000<aa.time) & (aa.time<500000)]
#cc
#grp_stg[sen_col]
obj_col = ['time',  'Lot', 'runnum', 'recipe', 'recipe_step']
sen_col = list(aa.columns[7:]).append('runnum')
def split_norm_df(aa):
    #이상치 제거한다.
        #분산으로 제거하면 좋음.
        #pass
        #this process is inevitable proess...        
    #쪼갠다
    aa = aa.fillna(method='ffill')
#    aa_copy = aa.copy()
#    aa_copy[sen_col]=0
    df = pd.DataFrame()
    for num, grp_lot in aa.groupby('Lot'): 
        for num2, grp_stg in grp_lot.groupby('stage'):
            test_index = grp_stg.index
            norm_arr = normalize(grp_stg[sen_col], axis=0)
#            aa_copy.loc[test_index][sen_col] = norm_arr
            df_tmp = pd.DataFrame(norm_arr, index=test_index, columns=sen_col)
            df = df.append(df_tmp)
            df = df.sort_index()
            print('=====================================')
            print('lot num :',grp_stg.iloc[0]['Lot'],'stage num :',grp_stg.iloc[0]['stage'])
            print('=====================================')
    org_df = aa.drop(['Tool','stage'], axis = 1)[obj_col]
    result = pd.concat([org_df,df],axis = 1)
#    
#            #axis =0 and 1 comparison
#                #plt.plot(normalize(grp_stg.ix[:199251,7:14],axis = 0)[:,:])
#            grp_stg.set_index('time').ix[:,6:].plot(figsize = (12,8))
#            plt.show()
#            aa4 = aa.copy()
#            aa4.ix[:50,7:9]
#            aa3 = normalize(aa.ix[50:100,7:9], axis = 0)
#            aa4.ix[50:100,7:9] = aa3
#            aa4.ix[:55,7:9]
#            
#         grp_stg.tail()
    #정규화 때린다
    #다시 붙인다 'time'순으로
    #붙인걸 리턴한다.
    return result

# 저장 파일 있으면, 불러오고 없으면 만들기
norm_save_file = False

if norm_save_file:    
    print('Save file loading.......')
    new_aa = pd.read_csv('norm_sensor_data2.csv')
    new_aa = new_aa.drop(['Unnamed: 0'], axis = 1)
else:
    print('There is no save file!')
    print('Dataframe normalizeing Start!!.....')
    new_aa = split_norm_df(outlier_aa)
    new_aa.to_csv("norm_sensor_data2.csv", mode='w')

new_aa.head()
# =============================================================================
# #플로팅 
# new_aa.info()
# new_aa.ix[:,5:].plot(figsize = (12,34), subplots = True, grid = True)
# aa.ix[:,-4:-1].plot(figsize = (12,8), subplots = True, grid = True)
# # new_aa.ix[:,-4:-1].plot(figsize = (12,8), subplots = True, grid = True)
# =============================================================================
 #%%
# =============================================================================
##%%
#
#aaa = aa.set_index(['Lot','stage'])
#aa.set_index(['Lot']).head()
#
#for num, grp in aaa.groupby('Lot'):
#    print(grp.head())
#    break
#
#aa[aa.Lot == 54].stage.value_counts()
##grp.set_index('time').ix[:,2:].plot(subplots = True, grid = True, figsize = (12, 40))
#aaa.loc[(4164,44),:].set_index('time').ix[:,2:].plot(figsize = (12,8))
#grp.loc[(201,68),:].set_index('time').ix[:,2:].plot(figsize = (12,8))
#len(grp.loc[(201,48),:].set_index('time'))
#
#grp.loc[(312,63),:].set_index('time').ix[:,2:].plot(figsize = (12,8))
#grp.loc[(312,5),:].set_index('time').ix[:,2:].plot(figsize = (12,8))
#
#grp.head()
#grp.tail()

# 현재상태 : stage 컬럼은 삭제되었고, lot별로 남은 rul만 매칭 시키면 됨.

bb1 = bb1[['time','TTF_FlowCool Pressure Dropped Below Limit']]
bb1

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
#디폴트 세팅
np.set_printoptions(suppress=True, threshold=10000)
#%%
new_aa[new_aa.Lot==589].plot(subplots=True, figsize = (12,34))
#bb1.plot(figsize = )

#%%
train = pd.read_csv('C:/Users/hbee/Desktop/proj1/data/predictive_maintenance/train.csv', header = None)
test_x = pd.read_csv('C:/Users/hbee/Desktop/proj1/data/predictive_maintenance/test_x.csv', header = None)
test_y = pd.read_csv('C:/Users/hbee/Desktop/proj1/data/predictive_maintenance/test_y.csv', header = None)#여기서 테스트 


# Combine the X values to normalize them, then split them back out
# normalize 한 다음에, 다시 합치는 과정
train = train.as_matrix()
test_x = test_x.as_matrix()

all_x = np.concatenate((train[:, 2:26], test_x[:, 2:26]))
all_x = normalize(all_x, axis=0)

train[:, 2:26] = all_x[0:train.shape[0], :]
test_x[:, 2:26] = all_x[train.shape[0]:, :] 

# Make engine numbers and days zero-indexed, for everybody's sanity
train[:, 0:2] -= 1
test_x[:, 0:2] -= 1

# Configurable observation look-back period for each engine/day
max_time = 100
#%%
#new_aa test 랏 종류를 하나씩 받아오는
lot_list = list(new_aa["Lot"].value_counts().head().index)
sort_lot_list = sorted(lot_list)
len(lot_list)
new_aa = new_aa.drop(['runnum'], axis = 1)

new_aa.time = (new_aa.time - min(new_aa.time))/4
#lot별로 다시 시작하는 time 컬럼 만들어야함;;;
aaaa = new_aa.set_index('time')
bbbb = bb1.set_index('time')
max(aaaa.index)
max(bbbb.index)
pd.concat([bbbb,aaaa],axis = 1)
pd.concat([new_aa,bb1],axis = 1)
bb1
new_aa.tail()


#만드는 김에 lot 순서도 다시 재정렬.....


new_aa.info()
new_aaM = new_aa.as_matrix()
new_aaM.shape
#%%
sampleRate = 5 
TimeStepSize = 1 # every 600 second
n_skipSample = 3 # skip a lot of samples

a= '0{}_M0{}_DC_train.csv'.format(i1,i2)
b= 'train_ttf/0{}_M0{}_DC_train.csv'.format(i1,i2)
c = 'train_faults/0{}_M0{}_train_fault_data.csv'.format(i1,i2)

aa = pd.read_csv(a)
bb = pd.read_csv(b)
cc = pd.read_csv(c)

#노필요 컬럼 드랍시킴
aa = aa.drop(['Tool'], axis = 1)
#aa = aa.drop(['Lot'], axis = 1)
#그냥 인덱스 재설정? 정도가되겠다
aa.index = range(0,len(aa))
bb.index = range(0,len(bb))

#그냥 앞으로 당긴거, fault time 기준으로.
aa1, bb1, cc1 = cutoff(aa, cc, \
                    bb, 'FlowCool Pressure Dropped Below Limit')    

aa1 = aa1.fillna(method = 'ffill')
aa1['recipe'] = aa1['recipe'] + 200
label = bb1['TTF_FlowCool Pressure Dropped Below Limit']
#label.plot()


def series_to_supervised(data, y, n_in=100, n_skip = 30, dropnan=True):
#    data = df_select
#    y = y_select
#    n_in = TimeStepSize
#    n_skip = n_skipSample
#    dropnan = True
#    
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
df_select, y_select = aa1.ix[::sampleRate], label.ix[::sampleRate]
df, y = series_to_supervised(df_select, y_select, TimeStepSize, n_skipSample, True)
#list(df["Lot"].value_counts().index)


#%%
max_time = 100
mask_value = -99

#df.info()
import tqdm
from tqdm import tqdm

def build_data(engine, time, x, max_time, is_test, mask_value):
#    df2[:, 2], df2[:, 0], df2[:, 4:], max_time, False, mask_value
#    engine=df2[:, 2]
#    time=df2[:, 0]
#    x=df2[:, 4:]
#    max_time=max_time
#    is_test=False
#    mask_value=mask_value

    # y[0] will be days remaining, y[1] will be event indicator, always 1 for this data
#    out_y = []
    
    #피쳐 갯수
    d = x.shape[1]

    # A full history of sensor readings to date for each x
    out_x = []
    
    #우리꺼는 stage 기준이면, 조금 더 다르게 해야함. 100개가 아니라 134개? 일걸
#    n_engines=len(lot_list)
    for i in lot_list:
#    for i in tqdm(range(n_engines)):
#        i = 589
        # When did the engine fail? (Last day + 1 for train data, irrelevant for test.)
        #이게 문제같은데; 
        #왜냐면 우리 데이터는 4초마다 한번씩 인덱스가 찍히고, turbofan 데이터는 1초에 한번 찍히는데, 아래 for문은 range로 돌아가니깐
        max_engine_time = len(time[engine == i]) # 해당 엔진넘버에서의 맥스 시간, 그러니깐 고장 직전 시간을 봄
        #테스트 true면, 한칸짜리로 걍 연습하는거임...
        if is_test:
            start = max_engine_time - 1
        else:
            start = 0

        this_x = []
        
        for j in range(start, max_engine_time):
            if j%100==0:
                print("Lot num: " + str(i), '({}/{})'.format(j,max_engine_time))
            engine_x = x[engine == i]
#            out_y.append(np.array((max_engine_time - j, 1), ndmin=2))
            #마스킹하기 -99로
            xtemp = np.zeros((1, max_time, d))
            xtemp += mask_value #앞에 마스크 씌우기
#            xtemp.shape
#             xtemp = np.full((1, max_time, d), mask_value)
            
            xtemp[:, max_time-min(j, 99)-1:max_time, :] = engine_x[max(0, j-max_time+1):j+1, :]
            this_x.append(xtemp)
            
        print('========= Lot {} success ========='.format(i))            
        print(j, i)
        this_x = np.concatenate(this_x)

        out_x.append(this_x)
                
    out_x = np.concatenate(out_x)
    
#    out_y = np.concatenate(out_y)
    return out_x#, out_y

lot_list = list(df["Lot"].value_counts().index)
df2 = df.as_matrix()
train_x = build_data(df2[:, 2], df2[:, 0], df2[:, 4:], max_time, False, mask_value)
#train_x.shape
#df.shape
#y.shape
#%%

def build_data(engine, time, x, max_time, is_test):
    engine = new_aaM[:, 1]
    time = new_aaM[:, 0]
    x = new_aaM[:, 2:]
#    
    is_test = False
    # y[0] will be days remaining, y[1] will be event indicator, always 1 for this data
#    out_y = np.empty((0, 2), dtype=np.float32)

    # A full history of sensor readings to date for each x
    out_x = []#np.empty((0, max_time, 19), dtype=np.float32)

    for i in lot_list:
        
        # When did the engine fail? (Last day + 1 for train data, irrelevant for test.)
        i = 589
        max_engine_time = int(np.max(time[engine == i])) + 1

        if is_test:
            start = max_engine_time - 1
        else:
            start = 0

        this_x = []#np.empty((0, max_time, 19), dtype=np.float32)

        for j in range(start, max_engine_time):#start 시간부터 287초 까지(id = 2 일때 마지막 시간 287임)
            j = 0
            if j%100==0:
                print("Lot num: " + str(i), '({}/{})'.format(j,max_engine_time))
            engine_x = x[engine == i] # 우리껄로 치면 lot=589일때 센서값들 전체

#            out_y = np.append(out_y, np.array((max_engine_time - j, 1), ndmin=2), axis=0)
            
            xtemp = np.zeros((1, max_time, 19))
            xtemp.append(engine_x[max(0, j-max_time+1):j+1, :])
            
            xtemp[:, max_time-min(j, 99)-1:max_time, :] = engine_x[max(0, j-max_time+1):j+1, :]
            this_x.append(xtemp)
            
        this_x = np.concatenate(this_x)
        out_x.append(this_x)
    
    out_x = np.concatenate(out_x)
    out_x.shape
#    out_y = np.concatenate(out_y)
#
#        out_x = np.concatenate((out_x, this_x))

    return out_x#, out_y

train_x = build_data(new_aaM[:, 1], new_aaM[:, 0], new_aaM[:, 2:], max_time, False)
#train_y = ?????

#%%

def build_data(engine, time, x, max_time, is_test):
#    engine = train[:, 0]
#    time = train[:, 1]
#    x = train[:, 2:26]
#    is_test = False
    # y[0] will be days remaining, y[1] will be event indicator, always 1 for this data
    out_y = np.empty((0, 2), dtype=np.float32)

    # A full history of sensor readings to date for each x
    out_x = np.empty((0, max_time, 24), dtype=np.float32)

    for i in range(100):
        print("Engine: " + str(i))
        # When did the engine fail? (Last day + 1 for train data, irrelevant for test.)
#        i = 1
        max_engine_time = int(np.max(time[engine == i])) + 1

        if is_test:
            start = max_engine_time - 1
        else:
            start = 0

        this_x = np.empty((0, max_time, 24), dtype=np.float32)

        for j in range(start, max_engine_time):#start 시간부터 287초 까지(id = 2 일때 마지막 시간 287임)
#            j = 0
            engine_x = x[engine == i] # 우리껄로 치면 lot=589일때 센서값들 전체

            out_y = np.append(out_y, np.array((max_engine_time - j, 1), ndmin=2), axis=0)

            xtemp = np.zeros((1, max_time, 24))
            xtemp[:, max_time-min(j, 99)-1:max_time, :] = engine_x[max(0, j-max_time+1):j+1, :]
            this_x = np.concatenate((this_x, xtemp))

        out_x = np.concatenate((out_x, this_x))

    return out_x, out_y





train_x, train_y = build_data(train[:, 0], train[:, 1], train[:, 2:26], max_time, False)
train_x.shape
train_y.shape


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
