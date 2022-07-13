# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 19:44:57 2022

@author: lalith kumar
"""

# with outliers.
# standardization not done.

import numpy as np
import pandas as pd
df = pd.read_csv('E:\P-127\POWER PLANT PROJECT\energy_production.csv',sep=';')
df.shape
list(df)
df.head()
df.info()
df.describe()
df.columns
df.cov()
df.corr()
df.isnull().sum()
df.duplicated().sum()

#checking for duplicated 
df.duplicated().sum()
df_new=df.drop_duplicates()
df_new.duplicated().sum()
df_new.shape

X = df.iloc[:,:4]
list(X)
Y = df['energy_production']


# ----------------------MODEL BUILDING---------------------------

from sklearn.model_selection import train_test_split,KFold,cross_val_score,RepeatedKFold
X_train,X_test, Y_train, Y_test  = train_test_split(X,Y, test_size = 0.20, random_state = 100)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.linear_model import Lasso 
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor


# selecting best models
model_selc = [LinearRegression(),
             DecisionTreeRegressor(),
             RandomForestRegressor(n_estimators = 10),
             GradientBoostingRegressor(),
             SVR(kernel='poly'),
             xgb.XGBRegressor(),
             Ridge(alpha=4),
             Lasso(alpha=4),
             KNeighborsRegressor(),
             AdaBoostRegressor()]
             
             
             

kfold = RepeatedKFold(n_splits=5, n_repeats=10, random_state= None)
cv_results = []
cv_results_mean =[]
for ele in model_selc:
    cross_results = cross_val_score(ele,X_train,Y_train, cv=kfold, scoring ='r2')
   
    cv_results.append(cross_results)
   
    cv_results_mean.append(cross_results.mean())
    print("\n MODEL: ",ele,"\nMEAN R2:",cross_results.mean())
    
'''    
#  out put of best models
 MODEL:  LinearRegression() 
 
MEAN R2: 0.929015950812553

 MODEL:  DecisionTreeRegressor() 
MEAN R2: 0.9246089493622504

 MODEL:  RandomForestRegressor(n_estimators=10) 
MEAN R2: 0.9550266618698638

 MODEL:  GradientBoostingRegressor() 
MEAN R2: 0.9480616171779339

 MODEL:  SVR(kernel='poly') 
MEAN R2: 0.6942568255790692

MODEL:  XGBRegressor
MEAN R2: 0.9636109884431717

MODEL:  Ridge(alpha=4) 
MEAN R2: 0.9290482240194154

 MODEL:  Lasso(alpha=4) 
MEAN R2: 0.9254057016985563

 MODEL:  KNeighborsRegressor() 
MEAN R2: 0.9437242192429127

 MODEL:  AdaBoostRegressor() 
MEAN R2: 0.9052916087955134'''

    
# MODEL FITTING.
# 1) LINEAR REGRESSION.

# regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_train)

#for predict the test values
y_prdict=reg.predict(X_test)
y_prdict

#plot on prdtion
plt.scatter(Y_test, y_prdict)
plt.show()

#testing score
test_data_model_score=reg.score(X_test,Y_test)
print ('score of test data',test_data_model_score)
#score of test data  0.9266343475459896

train_data_model_score=reg.score(X_train,Y_train)
print ('score of test data',train_data_model_score)
#score of train data 0.9292095855394127

# mean_absolute_error
from sklearn.metrics import mean_absolute_error
mae1=mean_absolute_error(Y_test, y_prdict)
mae1
#3.6616719535588587

#for mean squared error
from sklearn.metrics import mean_squared_error
mse1=mean_squared_error(Y_test, y_prdict)
mse1
# 21.51960792430798

#for root mean squared error
from sklearn.metrics import mean_squared_error
rmse1=np.sqrt(mean_squared_error(Y_test, y_prdict))
rmse1
# 4.638923142746383

#root mean squared  log error
from sklearn.metrics import mean_squared_log_error
rmsle1=np.sqrt(mean_squared_log_error(Y_test, y_prdict))
rmsle1
#  0.010208909397732644

from sklearn.metrics import r2_score
r2score1= r2_score(y_prdict,Y_test)*100
r2score1
#92.03263005159204

############################################################################################

# 2)GradientBoostingRegressor.

GBR = GradientBoostingRegressor(learning_rate=0.1,n_estimators=100) # lr = 0.1, est = 100
GBR.fit(X_train,Y_train)

#for predict the test values
y_prdict=GBR.predict(X_test)
y_prdict

#plot on prdtion
plt.scatter(Y_test, y_prdict)
plt.show()

#testing score
test_data_model_score=GBR.score(X_test,Y_test)
print ('score of test data',test_data_model_score)
#score of test data 0.9470494411870924

train_data_model_score=GBR.score(X_train,Y_train)
print ('score of test data',train_data_model_score)
#score of train data 0.9534556226471778


# mean_absolute_error
from sklearn.metrics import mean_absolute_error
mae2=mean_absolute_error(Y_test, y_prdict)
mae2
# 2.9746828814593833

#for mean squared error
from sklearn.metrics import mean_squared_error
mse2=mean_squared_error(Y_test, y_prdict)
mse2
#15.529791430989563

#for root mean squared error
from sklearn.metrics import mean_squared_error
rmse2=np.sqrt(mean_squared_error(Y_test, y_prdict))
rmse2
# 3.9407856362646223

#root mean squared  log error
from sklearn.metrics import mean_squared_log_error
rmsle2=np.sqrt(mean_squared_log_error(Y_test, y_prdict))
rmsle2
# 0.008657122911138846

from sklearn.metrics import r2_score
r2score2= r2_score(y_prdict,Y_test)*100
r2score2
#94.39318406130339

###########################################################################################

#3)DECISION TREE REGRESSOR.

DTR = DecisionTreeRegressor(random_state = 0) 
DTR.fit(X_train,Y_train)

#for predict the test values
y_prdict=DTR.predict(X_test)
y_prdict

#plot on prdtion
plt.scatter(Y_test, y_prdict)
plt.show()

#testing score
test_data_model_score=DTR.score(X_test,Y_test)
print ('score of test data',test_data_model_score)
#score of test data  0.9287116793451914

train_data_model_score=DTR.score(X_train,Y_train)
print ('score of test data',train_data_model_score)
#score of train data 1.0

# mean_absolute_error
from sklearn.metrics import mean_absolute_error
mae3=mean_absolute_error(Y_test, y_prdict)
mae3
# 3.128688610240334

#for mean squared error
from sklearn.metrics import mean_squared_error
mse3=mean_squared_error(Y_test, y_prdict)
mse3
#20.91028510971786

#for root mean squared error
from sklearn.metrics import mean_squared_error
rmse3=np.sqrt(mean_squared_error(Y_test, y_prdict))
rmse3
# 4.572776520858838

#root mean squared  log error
from sklearn.metrics import mean_squared_log_error
rmsle3=np.sqrt(mean_squared_log_error(Y_test, y_prdict))
rmsle3
#0.01004607098287805

from sklearn.metrics import r2_score
r2score3= r2_score(y_prdict,Y_test)*100
r2score3
#92.78126074631119

##########################################################################################

#4)randomforest regressor

RFR= RandomForestRegressor(n_estimators =50, random_state = 0)
RFR.fit(X_train,Y_train)

#for predict the test values
y_prdict=RFR.predict(X_test)
y_prdict

#plot on prdtion
plt.scatter(Y_test, y_prdict)
plt.show()

#testing score
test_data_model_score=RFR.score(X_test,Y_test)
print ('score of test data',test_data_model_score)
#score of test data 0.9595280918868389

train_data_model_score=RFR.score(X_train,Y_train)
print ('score of test data',train_data_model_score)
#score of train data 0.9943109287169933


# mean_absolute_error
from sklearn.metrics import mean_absolute_error
mae4=mean_absolute_error(Y_test, y_prdict)
mae4
# 2.4278965517241353

#for mean squared error
from sklearn.metrics import mean_squared_error
mse4=mean_squared_error(Y_test, y_prdict)
mse4
# 11.87121718967606

#for root mean squared error
from sklearn.metrics import mean_squared_error
rmse4=np.sqrt(mean_squared_error(Y_test, y_prdict))
rmse4
#  3.4454632764950577

#root mean squared  log error
from sklearn.metrics import mean_squared_log_error
rmsle4=np.sqrt(mean_squared_log_error(Y_test, y_prdict))
rmsle4
# 0.007572007743226049

from sklearn.metrics import r2_score
r2score4= r2_score(y_prdict,Y_test)*100
r2score4
#95.75835543808579

########################################################################################

#5)SUPPORT VECTOR REGRESSOR


SVR=SVR(kernel='linear')
SVR.fit(X_train,Y_train)

#for predict the test values
y_prdict=SVR.predict(X_test)
y_prdict

#plot on prdtion
plt.scatter(Y_test, y_prdict)
plt.show()

#testing score
test_data_model_score=SVR.score(X_test,Y_test)
print ('score of test data',test_data_model_score)
#score of test data  0.9262407148045863

train_data_model_score=SVR.score(X_train,Y_train)
print ('score of test data',train_data_model_score)
#score of train data 0.9287101721019402


# mean_absolute_error
from sklearn.metrics import mean_absolute_error
mae5=mean_absolute_error(Y_test, y_prdict)
mae5
#3.6536372800780623

#for mean squared error
from sklearn.metrics import mean_squared_error
mse5=mean_squared_error(Y_test, y_prdict)
mse5
# 21.63506825182953

#for root mean squared error
from sklearn.metrics import mean_squared_error
rmse5=np.sqrt(mean_squared_error(Y_test, y_prdict))
rmse5
# 4.6513512286033105

#root mean squared  log error
from sklearn.metrics import mean_squared_log_error
rmsle5=np.sqrt(mean_squared_log_error(Y_test, y_prdict))
rmsle5
#0.010252310164177568

from sklearn.metrics import r2_score
r2score5= r2_score(y_prdict,Y_test)*100
r2score5
#92.32767925926039

####################################################################################

#6)XGBOOST

XGB=xgb.XGBRegressor(n_estimators=100,eta=0.001,gamma=10,learning_rate=0.5)
XGB.fit(X_train,Y_train)

#for predict the test values
y_prdict=XGB.predict(X_test)
y_prdict

#plot on prdtion
plt.scatter(Y_test, y_prdict)
plt.show()

#testing score
test_data_model_score=XGB.score(X_test,Y_test)
print ('score of test data',test_data_model_score)
#score of test data 0.9628323322871019

train_data_model_score=XGB.score(X_train,Y_train)
print ('score of test data',train_data_model_score)
#score of train data 0.9882049595130928

# mean_absolute_error
from sklearn.metrics import mean_absolute_error
mae6=mean_absolute_error(Y_test, y_prdict)
mae6
# 2.3444989185871377

#for mean squared error
from sklearn.metrics import mean_squared_error
mse6=mean_squared_error(Y_test, y_prdict)
mse6
# 10.902017632078016

#for root mean squared error
from sklearn.metrics import mean_squared_error
rmse6=np.sqrt(mean_squared_error(Y_test, y_prdict))
rmse6
#3.301820351272615

#root mean squared  log error
from sklearn.metrics import mean_squared_log_error
rmsle6=np.sqrt(mean_squared_log_error(Y_test, y_prdict))
rmsle6
# 0.007260217224010928

from sklearn.metrics import r2_score
r2score6= r2_score(y_prdict,Y_test)*100
r2score6
#96.16311786444702

#################################################################################

#7).Ridge REGRESSOR

RG= Ridge(alpha=0.01) # lambda
RG.fit(X_train,Y_train)

#for predict the test values
y_prdict=RG.predict(X_test)
y_prdict

#plot on prdtion
plt.scatter(Y_test, y_prdict)
plt.show()

#testing score
test_data_model_score=RG.score(X_test,Y_test)
print ('score of test data',test_data_model_score)
#score of test data 0.9266343473998

train_data_model_score=RG.score(X_train,Y_train)
print ('score of test data',train_data_model_score)
#score of train data  0.9292095855394104

# mean_absolute_error
from sklearn.metrics import mean_absolute_error
mae7=mean_absolute_error(Y_test, y_prdict)
mae7
#  3.6616719828409616

#for mean squared error
from sklearn.metrics import mean_squared_error
mse7=mean_squared_error(Y_test, y_prdict)
mse7
#  21.51960796718831

#for root mean squared error
from sklearn.metrics import mean_squared_error
rmse7=np.sqrt(mean_squared_error(Y_test, y_prdict))
rmse7
#4.638923147368181

#root mean squared  log error
from sklearn.metrics import mean_squared_log_error
rmsle7=np.sqrt(mean_squared_log_error(Y_test, y_prdict))
rmsle7
#0.01020890938352068

from sklearn.metrics import r2_score
r2score7= r2_score(y_prdict,Y_test)*100
r2score7
# 92.03262975451196


################################################################################

#8) LASSO REGRESSOR

LSR = Lasso(alpha=4)
LSR.fit(X_train,Y_train)

#for predict the test values
y_prdict=LSR.predict(X_test)
y_prdict

#plot on prdtion
plt.scatter(Y_test, y_prdict)
plt.show()

#testing score
test_data_model_score=LSR.score(X_test,Y_test)
print ('score of test data',test_data_model_score)
#score of test data 0.9283304025571093

train_data_model_score=LSR.score(X_train,Y_train)
print ('score of test data',train_data_model_score)
#score of train data0.9255951912228451

# mean_absolute_error
from sklearn.metrics import mean_absolute_error
mae8=mean_absolute_error(Y_test, y_prdict)
mae8
#  3.7732162521101245

#for mean squared error
from sklearn.metrics import mean_squared_error
mse8=mean_squared_error(Y_test, y_prdict)
mse8
#   22.730952174365093

#for root mean squared error
from sklearn.metrics import mean_squared_error
rmse8=np.sqrt(mean_squared_error(Y_test, y_prdict))
rmse8
# 4.767698834276877

#root mean squared  log error
from sklearn.metrics import mean_squared_log_error
rmsle8=np.sqrt(mean_squared_log_error(Y_test, y_prdict))
rmsle8
#0.010475907771005332

from sklearn.metrics import r2_score
r2score8= r2_score(y_prdict,Y_test)*100
r2score8
#  90.9757664531752
#---------------------------------------------------------------------------

# 9) K.N.N

knn = KNeighborsRegressor(n_neighbors=5, p=2) # k =5 # p=2 --> Eucledian distance
knn.fit(X_train,Y_train)


#for predict the test values
y_prdict=knn.predict(X_test)
y_prdict

#plot on prdtion
plt.scatter(Y_test, y_prdict)
plt.show()

#testing score
test_data_model_score=knn.score(X_test,Y_test)
print ('score of test data',test_data_model_score)
#score of test data 0.9447051892240363

train_data_model_score=knn.score(X_train,Y_train)
print ('score of test data',train_data_model_score)
#score of train data  0.9652458689836608

# mean_absolute_error
from sklearn.metrics import mean_absolute_error
mae9=mean_absolute_error(Y_test, y_prdict)
mae9
#  2.9416290491118073

#for mean squared error
from sklearn.metrics import mean_squared_error
mse9=mean_squared_error(Y_test, y_prdict)
mse9
# 16.219069937304067

#for root mean squared error
from sklearn.metrics import mean_squared_error
rmse9=np.sqrt(mean_squared_error(Y_test, y_prdict))
rmse9
#4.027290644751639

#root mean squared  log error
from sklearn.metrics import mean_squared_log_error
rmsle9=np.sqrt(mean_squared_log_error(Y_test, y_prdict))
rmsle9
#0.008850567261172221

from sklearn.metrics import r2_score
r2score9= r2_score(y_prdict,Y_test)*100
r2score9
# 94.22259851932193


###########################################################################################

# 10)neural network regression

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# create model
model = Sequential()
model.add(Dense(6, input_dim=4,  activation='relu')) #input layer
model.add(Dense(1, activation='relu')) #output layer


# Compile model
model.compile(loss='msle', optimizer='adam', metrics=['msle'])

# loss --> Regression --> mean square error
# loss --> multi class --> categorical_cross entropy


# Fit the model
history = model.fit(X, Y, validation_split=0.33, epochs=500, batch_size=150)

# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#output:loss: 1.2198e-04 - msle: 1.2198e-04
#msle: 0.01%

# Visualize training history

# list all data in history
history.history.keys()


# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['msle'])
plt.plot(history.history['val_msle'])
plt.title('msle')
plt.ylabel('msle')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
###########################################################################################

# 11) AdaBoostRegressor

AB = AdaBoostRegressor(base_estimator=DTR,n_estimators=100) 
AB.fit(X_train,Y_train)

#for predict the test values
y_prdict=AB.predict(X_test)
y_prdict

#plot on prdtion
plt.scatter(Y_test, y_prdict)
plt.show()

#testing score
test_data_model_score=AB.score(X_test,Y_test)
print ('score of test data',test_data_model_score)
#score of test data  0.9605643876429985

train_data_model_score=AB.score(X_train,Y_train)
print ('score of test data',train_data_model_score)
#score of train data 0.9999349610158265

# mean_absolute_error
from sklearn.metrics import mean_absolute_error
mae10=mean_absolute_error(Y_test, y_prdict)
mae10
#  2.2645454545454538

#for mean squared error
from sklearn.metrics import mean_squared_error
mse10=mean_squared_error(Y_test, y_prdict)
mse10
#11.567250992685468

#for root mean squared error
from sklearn.metrics import mean_squared_error
rmse10=np.sqrt(mean_squared_error(Y_test, y_prdict))
rmse10
# 3.401066155293876

#root mean squared  log error
from sklearn.metrics import mean_squared_log_error
rmsle10=np.sqrt(mean_squared_log_error(Y_test, y_prdict))
rmsle10
# 0.007467573909852638

from sklearn.metrics import r2_score
r2score10= r2_score(y_prdict,Y_test)*100
r2score10
#95.96870584699053

###########################################################################################

#tabulating the rmse values

data={'Model':pd.Series(['LinearRegression','DecisionTree_Regressor','RandomForest_Regressor','Gradientboosting_regressor','supportvector_regressor','XGB_Regressor','Ridge_regressor','Lasso_regressor','KNeighbors_Regressor','AdaBoost_Regressor']),'RMSE':pd.Series([rmse1,rmse2,rmse3,rmse4,rmse5,rmse6,rmse7,rmse8,rmse9,rmse10]),
      'R2SCORES':pd.Series([r2score1,r2score2,r2score3,r2score4,r2score5,r2score6,r2score7,r2score8,r2score9,r2score10]),'MAE':pd.Series([mae1,mae2,mae3,mae4,mae5,mae6,mae7,mae8,mae9,mae10]),'RMSLE':pd.Series([rmsle1,rmsle2,rmsle3,rmsle4,rmsle5,rmsle6,rmsle7,rmsle8,rmsle9,rmsle10]),
      'MSE':pd.Series([mse1,mse2,mse3,mse4,mse5,mse6,mse7,mse8,mse9,mse10])}
data
pd.set_option("display.max.columns",None)
data
RESULTS=pd.DataFrame(data)
RESULTS
RESULTS.sort_values(['RMSE','R2SCORES'])

'''
Out[37]: 
                        Model      RMSE   R2SCORES       MAE     RMSLE  \
5               XGB_Regressor  3.301820  96.163118  2.344499  0.007260   
9          AdaBoost_Regressor  3.408441  95.960924  2.278924  0.007488   
3  Gradientboosting_regressor  3.445463  95.758355  2.427897  0.007572   
1      DecisionTree_Regressor  3.940874  94.392869  2.974830  0.008657   
8        KNeighbors_Regressor  4.027291  94.222599  2.941629  0.008851   
2      RandomForest_Regressor  4.572777  92.781261  3.128689  0.010046   
0            LinearRegression  4.638923  92.032630  3.661672  0.010209   
6             Ridge_regressor  4.638923  92.032630  3.661672  0.010209   
4     supportvector_regressor  4.651351  92.327679  3.653637  0.010252   
7             Lasso_regressor  4.767699  90.975766  3.773216  0.010476   

         MSE  
5  10.902018  
9  11.617473  
3  11.871217  
1  15.530490  
8  16.219070  
2  20.910285  
0  21.519608  
6  21.519608  
4  21.635068  
7  22.730952 
'''
###########################################################################################





