
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
line=pd.read_csv('./Alibaba/gd_train_data.txt',header=None,names=['User_city','Line_name','Terminal_id','Card_id','Create_city','Deal_time','Card_type'])


# In[3]:

line.head()


# In[8]:

line=line.drop(['Terminal_id','Card_id'],axis=1)
line.head()


# In[4]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.scatter(line.Line_name,line.Deal_time)


# In[20]:


line.Card_type.value_counts().plot(kind='bar')


# In[50]:

import datetime
dealtime=line.Deal_time.value_counts()
datedata=pd.DataFrame(dealtime)
datedata.index
wd=[]
for i in datedata.index:
      s=str(i)
      wd.append(datetime.datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8])).weekday())
datedata['wd']=pd.Series(wd,index=datedata.index)
datedata


# In[71]:

import re
m = re.search("(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})", "2015/11/09")
int(m.group('year'))*10000+int(m.group('month'))*100+int(m.group('day'))


# In[7]:

card=line.Card_id.value_counts()
len(card)    


# In[10]:

terminal=line.Terminal_id.value_counts()
len(terminal)


# In[27]:

line.Line_name.value_counts()


# In[24]:

line.Card_type.value_counts()


# In[25]:

line.User_city.value_counts()


# In[26]:

line.Create_city.value_counts()


# In[32]:

bus=pd.read_csv('./Alibaba/gd_line_desc.txt',header=None,names=['Line_name','Start','End', 'Stop_cnt','Line_type'])
bus=bus.drop(['Start','End'],axis=1)
bus


# In[8]:

weather=pd.read_csv('./Alibaba/gd_weather_report.txt',header=None,names=['Date_time','Weather','Temperature', 'Wind_direction_force'])
weather.head()


# In[12]:

weather.Weather.value_counts()


# In[88]:

weathermap={
'多云/多云'    :   0,
    '晴/晴':      1,
'雷阵雨/雷阵雨'    :  2,
'晴/多云'      :  3,
'多云/晴'       :  4,
'小雨/小雨'    :    5,
'多云/阴'      :   6,
'阴/阴'      :    7,
'小雨/多云'    :    8,
'阴/多云'      : 9,
'雷阵雨/多云'    :10,
'小雨/阴'   :11,
'阴/小雨'   :12,
'中到大雨/雷阵雨'  :13,
'小雨/小到中雨'  :14,
'多云/大雨' :15,
'中雨/中到大雨'   :16,
'霾/霾'   :17,
'大雨/中到大雨'  :18,
'雷阵雨/阵雨' :19,
'小到中雨/阴' :20,
'大雨/中雨'  :21,
'多云/雷阵雨' :22,
'多云/小雨' :23,
'晴/雷阵雨'  :24,
'中到大雨/中雨'   :25,
'晴/小雨'  :26,
'大雨/大到暴雨'   :27
}
weather['Weater_value']=weather['Weather'].map(weathermap)
weather


# In[89]:

weather.Wind_direction_force.value_counts()


# In[93]:

wind_direction_map={
'无持续风向≤3级/无持续风向≤3级' :0,
'北风3-4级/无持续风向≤3级':1,
'无持续风向≤3级/北风3-4级':2,
'北风3-4级/北风3-4级':3,
'无持续风向≤3级/北风4-5级':4,
'北风4-5级/北风4-5级':5,
'北风4-5级/北风3-4级':6 ,
'东风4-5级/东南风3-4级':7,              
'无持续风向微风转3-4级/北风微风转3-4级':8,
'东北风3-4级/东风4-5级':9,
'东北风3-4级/东北风3-4级':10,
'北风4-5级/无持续风向≤3级':11,    
'东北风3-4级/无持续风向≤3级':12,       
}
weather['wind_direction_value']=weather['Wind_direction_force'].map(wind_direction_map)
weather.head()


# In[115]:

low=[]
high=[]
year=[]
month=[]
day=[]
date_value=[]
for i in weather.Temperature:
    test=re.search('(\w*)℃/(\w*)℃',i)
    high.append(int(test.group(1)))
    low.append(int(test.group(2)))
weather['low_temp']=pd.Series(low,index=weather.index)
weather['high_temp']=pd.Series(high,index=weather.index)
for i in weather.Date_time:
    test=re.search('(\w*)/(\w*)/(\w*)',i)
    year.append(int(test.group(1)))
    month.append(int(test.group(2)))
    day.append(int(test.group(3)))
    date_value.append(int(test.group(1))*10000+int(test.group(2))*100+int(test.group(3)))
weather['year']=pd.Series(year,index=weather.index)
weather['month']=pd.Series(month,index=weather.index)
weather['day']=pd.Series(day,index=weather.index)
weather['date_value']=pd.Series(date_value,index=weather.index)
weather.head()


# In[118]:

line_1=line[line.Line_name==281]
line_1.head()


# In[122]:

import datetime
dealtime=line_1.Deal_time.value_counts()
datedata=pd.DataFrame(dealtime)
datedata.index
wd=[]
dv=[]
hour=[]
for i in datedata.index:
      s=str(i)
      dv.append(int(s[0:8]))
      hour.append(int(s[8:10]))
      wd.append(datetime.datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8])).weekday())
datedata['wd']=pd.Series(wd,index=datedata.index)
datedata['date_value']=pd.Series(dv,index=datedata.index)
datedata['hour']=pd.Series(hour,index=datedata.index)
datedata.head()


# In[148]:

weather=weather.drop(['year','month','day'],axis=1)
weather.head()


# In[149]:

test=pd.merge(datedata,weather,how='inner',on='date_value')
test.head()


# In[152]:

test=test.drop(['date_value'],axis=1)
test.head()


# In[154]:

target=test[0].values
target


# In[155]:

test=test.drop([0],axis=1)
test.head()


# In[157]:

dataset=test.values
dataset


# In[158]:

from sklearn import svm
clf=svm.SVR()
clf.fit(dataset,target)


# In[159]:

clf.predict([1,7,1,0,23,31])


# In[169]:

clf.set_params(class_weight=True).fit(dataset,target)
clf.score(dataset,target)


# ###Turn the int data to float

# In[234]:

from sklearn import preprocessing
X=dataset.astype(float)
y=target.astype(float)
y


# ###Preprocessing the data

# In[181]:

X_scaled=preprocessing.scale(X)
X_scaled
X_scaled.std(axis=0)


# In[214]:

from sklearn import linear_model
logistic=linear_model.LogisticRegression(C=1e5)
lassolars=linear_model.LassoLars(alpha=2)
bayesianrige=linear_model.BayesianRidge()


# In[224]:

from sklearn import decomposition
pca=decomposition.PCA()
pca.fit(X)


# In[225]:

pca.explained_variance_


# ###K-Fold cross-validation

# In[222]:

from sklearn import cross_validation
k_fold=cross_validation.KFold(len(X_scaled),n_folds=10)
svr=svm.SVR(C=100,kernel='rbf',gamma=1.1,epsilon=0.03)
cross_validation.cross_val_score(svr,X_scaled,target,cv=k_fold)


# ###Grid Search

# In[ ]:

from sklearn.grid_search import GridSearchCV
Crange=np.logspace(-2,10,40)
grid=GridSearchCV(svm.SVR(),param_grid={'C':Crange},cv=k_fold)
grid.fit(X_scaled,y)
grid.best_params_


# In[240]:

get_ipython().magic(u'matplotlib inline')
scores=[g[1] for g in grid.grid_scores_]
plt.semilogx(Crange,scores)

