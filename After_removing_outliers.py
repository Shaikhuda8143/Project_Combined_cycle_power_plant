# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:42:34 2022

@author: CG-DTE
"""

import numpy as np
import pandas as pd
df = pd.read_csv('D:/EXCELR INTERN PROJECT/energy_production.csv',sep=';')
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

# heatmap.
import seaborn as sns
sns.heatmap(df.corr(),annot=True,cmap="Blues")

# pair plot.
sns.pairplot(df)

#relation plot.
sns.relplot(x='temperature',y='energy_production',data=df)
sns.relplot(x='exhaust_vacuum',y='energy_production',data=df)
sns.relplot(x='amb_pressure',y='energy_production',data=df)
sns.relplot(x='r_humidity',y='energy_production',data=df)

# histogram
sns.displot(df['temperature'])
sns.displot(df['exhaust_vacuum'])
sns.displot(df['amb_pressure'])
sns.displot(df['r_humidity'])
sns.displot(df['energy_production'])

# histogram.
df['temperature'].hist()
df['exhaust_vacuum'].hist()
df['amb_pressure'].hist()
df['r_humidity'].hist()
df['energy_production'].hist()

# distrubution plot.

sns.distplot(df['temperature'], bins = 10, kde = True)
sns.distplot(df['exhaust_vacuum'], bins = 10, kde = True)
sns.distplot(df['amb_pressure'], bins = 10, kde = True)
sns.distplot(df['r_humidity'], bins = 10, kde = True)
sns.distplot(df['energy_production'], bins = 10, kde = True)

# pair plpot.
import matplotlib.pyplot as plt
sns.pairplot(df,diag_kind='kde')
plt.show


#pip install sweetviz
import sweetviz as sv

my_report = sv.analyze([df,"Energy_production"],target_feat="energy_production")
my_report.show_html("Report.html") # Default arguments will generate to "SWEETVIZ_REPORT.html"

#pip install dtale
import dtale
dtale.show(df,open_browser=True)
dtale.show(df)

#checking for duplicated 
df[df.duplicated()].shape
df[df.duplicated()]

#droping the duplicated
data_new=df.drop_duplicates().reset_index(drop=True)
data_new[df.duplicated()]


import matplotlib.pyplot as plt
#checking for the outliers
#construct box plot 
plt.figure(figsize=(8,8))
data_new.iloc[:,0:].boxplot(vert=0)
plt.show()

def remove_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lower_range=Q1-(1.5 * IQR)
    upper_range=Q3+(1.5 * IQR)
    return lower_range,upper_range
for column in data_new.iloc[:,0:].columns:
    lr,ur=remove_outlier(data_new[column])
    data_new[column]=np.where(data_new[column]>ur,ur,data_new[column])#cap an outlier
    data_new[column]=np.where(data_new[column]<lr,lr,data_new[column])
    
plt.figure(figsize=(12,8))
data_new.iloc[:,0:].boxplot(vert=0)
plt.show()  

data_new.shape
data_new.duplicated().sum()

#  fitting X&Y
X = data_new.iloc[:,:4]
list(X)
Y = data_new['energy_production']

# standardization
from sklearn.preprocessing import StandardScaler
X_scale = StandardScaler().fit_transform(X)
X_scale
type(X_scale)


from sklearn.model_selection import train_test_split,KFold,cross_val_score,RepeatedKFold,GridSearchCV,StratifiedKFold,RandomizedSearchCV
X_train,X_test, Y_train, Y_test  = train_test_split(X_scale,Y, test_size = 0.20, random_state = 100)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from pickle import load,dump
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score

#selecting best models
model_selc = [LinearRegression(),
             DecisionTreeRegressor(),
             RandomForestRegressor(n_estimators = 10),
             GradientBoostingRegressor(),
             SVR(kernel='linear'),
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
    
# MODEL FITTING.
# 1) LINEAR REGRESSION.

#regression

reg = LinearRegression()
reg.fit(X_train,Y_train)

#for predict the test values
y_prdict=reg.predict(X_test)
y_prdict

'''
#plot on prdtion
plt.scatter(Y_test, y_prdict)
plt.show()
'''
plt.figure(figsize=(10,10))
plt.scatter(Y_test, y_prdict, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_prdict), max(Y_test))
p2 = min(min(y_prdict), min(Y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.title('LINEAR REGRESSION',fontsize=20)
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')


#testing score
test_data_model_score=reg.score(X_test,Y_test)
print ('score of test data',test_data_model_score)
#score of test data 0.932718036242918

train_data_model_score=reg.score(X_train,Y_train)
print ('score of train data',train_data_model_score)
#score of train data 0.9273093296060387


#checking mean_absolute_error
mae1=mean_absolute_error(Y_test, y_prdict)
mae1#3.5789233461077354

#for mean squared error
mse1=mean_squared_error(Y_test, y_prdict)
mse1# 20.05725732567963

#for root mean squared error
rmse1=np.sqrt(mean_squared_error(Y_test, y_prdict))
rmse1# 4.478532943462584

#root mean squared  log error
rmsle1=np.sqrt(mean_squared_log_error(Y_test, y_prdict))
rmsle1#   0.009880681515995183

#r2score
r2score1= r2_score(y_prdict,Y_test)*100
r2score1#92.70943833140339

############################################################################################

# 2)GradientBoostingRegressor.

GBR = GradientBoostingRegressor(learning_rate=0.1,n_estimators=100) # lr = 0.1, est = 100
GBR.fit(X_train,Y_train)


#for predict the test values
y_prdict=GBR.predict(X_test)
y_prdict

'''
#plot on prdtion
plt.scatter(Y_test, y_prdict)
plt.show()'''

plt.figure(figsize=(10,10))
plt.scatter(Y_test, y_prdict, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_prdict), max(Y_test))
p2 = min(min(y_prdict), min(Y_test))
plt.plot([p1, p2], [p1, p2],'b-')
plt.title('GRADIENTBOOSTING REGRESSOR',fontsize=20)
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')


#testing score
test_data_model_score=GBR.score(X_test,Y_test)
print ('score of test data',test_data_model_score)
#score of test data 0.9527154573452785

train_data_model_score=GBR.score(X_train,Y_train)
print ('score of train data',train_data_model_score)
#score of train data 0.952306441895972


#checking mean_absolute_error
mae2=mean_absolute_error(Y_test, y_prdict)
mae2# 2.9010275998231503

#for mean squared error
mse2=mean_squared_error(Y_test, y_prdict)
mse2#14.09587631801247

#for root mean squared error
rmse2=np.sqrt(mean_squared_error(Y_test, y_prdict))
rmse2# 3.754447538322046

#root mean squared  log error
rmsle2=np.sqrt(mean_squared_log_error(Y_test, y_prdict))
rmsle2#   0.008269798901142494

#r2score
r2score2= r2_score(y_prdict,Y_test)*100
r2score2#94.94296642386058

###########################################################################################

#3)DECISION TREE REGRESSOR.

DTR = DecisionTreeRegressor(random_state = 100) 
DTR.fit(X_train,Y_train)

#for predict the test values
y_prdict=DTR.predict(X_test)
y_prdict

'''
#plot on prdtion
plt.scatter(Y_test, y_prdict)
plt.show()
'''
plt.figure(figsize=(10,10))
plt.scatter(Y_test, y_prdict, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_prdict), max(Y_test))
p2 = min(min(y_prdict), min(Y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.title('DECISION TREE REGRESSOR',fontsize=20)
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')


#testing score
test_data_model_score=DTR.score(X_test,Y_test)
print ('score of test data',test_data_model_score)
#score of test data 0.9323582064280144

train_data_model_score=DTR.score(X_train,Y_train)
print ('score of train data',train_data_model_score)
#score of train data 1.0


#checking mean_absolute_error
mae3=mean_absolute_error(Y_test, y_prdict)
mae3# 3.131248688352571


#for mean squared error
mse3=mean_squared_error(Y_test, y_prdict)
mse3#20.164525288562434

#for root mean squared error
rmse3=np.sqrt(mean_squared_error(Y_test, y_prdict))
rmse3# 4.490592521730562

#root mean squared  log error
rmsle3=np.sqrt(mean_squared_log_error(Y_test, y_prdict))
rmsle3#    0.009859523705385134

#r2score
r2score3= r2_score(y_prdict,Y_test)*100
r2score3#93.27186155526006

##########################################################################################

#4)randomforest regressor


RFR= RandomForestRegressor(n_estimators =50, random_state = 100)
RFR.fit(X_train,Y_train)

#for predict the test values
y_prdict=RFR.predict(X_test)
y_prdict

'''
#plot on prdtion
plt.scatter(Y_test, y_prdict)
plt.show()'''

plt.figure(figsize=(10,10))
plt.scatter(Y_test, y_prdict, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_prdict), max(Y_test))
p2 = min(min(y_prdict), min(Y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.title('RANDOM FOREST REGRESSOR',fontsize=20)
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')


#testing score
test_data_model_score=RFR.score(X_test,Y_test)
print ('score of test data',test_data_model_score)
#score of test data 0.9644382887341248

train_data_model_score=RFR.score(X_train,Y_train)
print ('score of test data',train_data_model_score)
#score of train data score of test data 0.9942178937370695

#checking mean_absolute_error
mae4=mean_absolute_error(Y_test, y_prdict)
mae4#  2.369634627492126

#for mean squared error
mse4=mean_squared_error(Y_test, y_prdict)
mse4# 10.601212479118546

#for root mean squared error
rmse4=np.sqrt(mean_squared_error(Y_test, y_prdict))
rmse4# 3.255950318895936

#root mean squared  log error
rmsle4=np.sqrt(mean_squared_log_error(Y_test, y_prdict))
rmsle4#     0.007163737075346798
#r2score
r2score4= r2_score(y_prdict,Y_test)*100
r2score4#96.28555854866407

########################################################################################

#5)SUPPORT VECTOR REGRESSOR


SVR=SVR(kernel='linear')
SVR.fit(X_train,Y_train)

#for predict the test values
y_prdict=SVR.predict(X_test)
y_prdict

'''
#plot on prdtion
plt.scatter(Y_test, y_prdict)
plt.show()
'''
plt.figure(figsize=(10,10))
plt.scatter(Y_test, y_prdict, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_prdict), max(Y_test))
p2 = min(min(y_prdict), min(Y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.title('SUPPORT VECTOR REGRESSOR',fontsize=20)
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')


#testing score
test_data_model_score=SVR.score(X_test,Y_test)
print ('score of test data',test_data_model_score)
#score of test data 0.9323981412429594

train_data_model_score=SVR.score(X_train,Y_train)
print ('score of train data',train_data_model_score)
#score of test data 0.9268129221113746


#checking mean_absolute_error
mae5=mean_absolute_error(Y_test, y_prdict)
mae5#3.5637999793784085

#for mean squared error
mse5=mean_squared_error(Y_test, y_prdict)
mse5# 20.152620421122787

#for root mean squared error
rmse5=np.sqrt(mean_squared_error(Y_test, y_prdict))
rmse5# 4.489167007488448

#root mean squared  log error
rmsle5=np.sqrt(mean_squared_log_error(Y_test, y_prdict))
rmsle5#0.009924358914778183

#r2score
r2score5= r2_score(y_prdict,Y_test)*100
r2score5# 92.98018170943332

####################################################################################

#6)XGBOOST


XGB=xgb.XGBRegressor(n_estimators=100,eta=0.001,gamma=10,learning_rate=0.5)
XGB.fit(X_train,Y_train)

#for predict the test values
y_prdict=XGB.predict(X_test)
y_prdict

'''
#plot on prdtion
plt.scatter(Y_test, y_prdict)
plt.show()
'''
plt.figure(figsize=(10,10))
plt.scatter(Y_test, y_prdict, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_prdict), max(Y_test))
p2 = min(min(y_prdict), min(Y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.title('XGBOOST REGRESSOR',fontsize=20)
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')


#testing score
test_data_model_score=XGB.score(X_test,Y_test)
print ('score of test data',test_data_model_score)
#score of test data 0.9665988364354935

train_data_model_score=XGB.score(X_train,Y_train)
print ('score of train data',train_data_model_score)
#score of test data 0.9859995156739711


#checking mean_absolute_error
mae6=mean_absolute_error(Y_test, y_prdict)
mae6# 2.3199225353068846

#for mean squared error
mse6=mean_squared_error(Y_test, y_prdict)
mse6# 9.940419677379243

#for root mean squared error
rmse6=np.sqrt(mean_squared_error(Y_test, y_prdict))
rmse6#3.152843110175202

#root mean squared  log error
rmsle6=np.sqrt(mean_squared_log_error(Y_test, y_prdict))
rmsle6#0.006958237730504955

#r2score
r2score6= r2_score(y_prdict,Y_test)*100
r2score6# 96.5462495332469

#################################################################################

#7).Ridge REGRESSOR

RG= Ridge(alpha=0.01) # lambda
RG.fit(X_train,Y_train)

#for predict the test values
y_prdict=RG.predict(X_test)
y_prdict

'''
#plot on prdtion
plt.scatter(Y_test, y_prdict)
plt.show()
'''
plt.figure(figsize=(10,10))
plt.scatter(Y_test, y_prdict, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_prdict), max(Y_test))
p2 = min(min(y_prdict), min(Y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.title("RIDGE REGRESSOR")
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')


#testing score
test_data_model_score=RG.score(X_test,Y_test)
print ('score of test data',test_data_model_score)
#score of test data 0.9327180350820679

train_data_model_score=RG.score(X_train,Y_train)
print ('score of train data',train_data_model_score)
#score of train data 0.9273093296060364

#checking mean_absolute_error
mae7=mean_absolute_error(Y_test, y_prdict)
mae7#  3.578923405505193

#for mean squared error
mse7=mean_squared_error(Y_test, y_prdict)
mse7#  20.0572576717378

#for root mean squared error
rmse7=np.sqrt(mean_squared_error(Y_test, y_prdict))
rmse7#4.478532982097799

#root mean squared  log error
rmsle7=np.sqrt(mean_squared_log_error(Y_test, y_prdict))
rmsle7#0.009880681572665575

#r2score
r2score7= r2_score(y_prdict,Y_test)*100
r2score7# 92.70943792789367


################################################################################

#8) LASSO REGRESSOR


LSR = Lasso(alpha=4)
LSR.fit(X_train,Y_train)

#for predict the test values
y_prdict=LSR.predict(X_test)
y_prdict

'''
#plot on prdtion
plt.scatter(Y_test, y_prdict)
plt.show()
'''
plt.figure(figsize=(10,10))
plt.scatter(Y_test, y_prdict, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_prdict), max(Y_test))
p2 = min(min(y_prdict), min(Y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.title('LASSO REGRESSOR')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')


#testing score
test_data_model_score=LSR.score(X_test,Y_test)
print ('score of test data',test_data_model_score)
#score of test data0.8558543870456319

train_data_model_score=LSR.score(X_train,Y_train)
print ('score of train data',train_data_model_score)
#score of train data  0.8541986120815863

#checking mean_absolute_error
mae8=mean_absolute_error(Y_test, y_prdict)
mae8#  3.7164156213624193

#for mean squared error
mse8=mean_squared_error(Y_test, y_prdict)
mse8#   21.365243789998974

#for root mean squared error
rmse8=np.sqrt(mean_squared_error(Y_test, y_prdict))
rmse8# 4.622255270968813

#root mean squared  log error
rmsle8=np.sqrt(mean_squared_log_error(Y_test, y_prdict))
rmsle8#0.010173678501454252

#r2score
r2score8= r2_score(y_prdict,Y_test)*100
r2score8#  71.4652753314278
###########################################################################################

# 9) KNN regressor

knn = KNeighborsRegressor(n_neighbors=5, p=2) # k =5 # p=2 --> Eucledian distance
knn.fit(X_train,Y_train)


#for predict the test values
y_prdict=knn.predict(X_test)
y_prdict

'''
#plot on prdtion
plt.scatter(Y_test, y_prdict)
plt.show()'''

plt.figure(figsize=(10,10))
plt.scatter(Y_test, y_prdict, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_prdict), max(Y_test))
p2 = min(min(y_prdict), min(Y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.title('K-NEIGHBORS REGRESSOR')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')

#testing score
test_data_model_score=knn.score(X_test,Y_test)
print ('score of test data',test_data_model_score)
#score of test data  0.9520063526324694

train_data_model_score=knn.score(X_train,Y_train)
print ('score of train data',train_data_model_score)
#score of train data 0.9638290944489041

# mean_absolute_error
mae9=mean_absolute_error(Y_test, y_prdict)
mae9
#  2.8151573976914994

#for mean squared error
mse9=mean_squared_error(Y_test, y_prdict)
mse9
# 14.30726574396642

#for root mean squared error
rmse9=np.sqrt(mean_squared_error(Y_test, y_prdict))
rmse9
#3.7824946455965303

#root mean squared  log error
rmsle9=np.sqrt(mean_squared_log_error(Y_test, y_prdict))
rmsle9
#0.008337608169687899

#r2score
r2score9= r2_score(y_prdict,Y_test)*100
r2score9#94.9918734120857

###########################################################################################

#10) ADA BOOST REGRESSOR

AB = AdaBoostRegressor(base_estimator=DTR,n_estimators=100)
AB.fit(X_train,Y_train)


#for predict the test values
y_prdict=knn.predict(X_test)
y_prdict

'''
#plot on prdtion
plt.scatter(Y_test, y_prdict)
plt.show()'''

plt.figure(figsize=(10,10))
plt.scatter(Y_test, y_prdict, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_prdict), max(Y_test))
p2 = min(min(y_prdict), min(Y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.title('ADABOOST REGRESSOR',fontsize=20)
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')

#testing score
test_data_model_score=AB.score(X_test,Y_test)
print ('score of test data',test_data_model_score)
#score of test data  0.9641090688110406

train_data_model_score=AB.score(X_train,Y_train)
print ('score of train data',train_data_model_score)
#score of train data score of test data 0.999931478181766

# mean_absolute_error
mae10=mean_absolute_error(Y_test, y_prdict)
mae10
#  2.8151573976914994

#for mean squared error
mse10=mean_squared_error(Y_test, y_prdict)
mse10
# 14.30726574396642

#for root mean squared error
rmse10=np.sqrt(mean_squared_error(Y_test, y_prdict))
rmse10
#3.7824946455965303

#root mean squared  log error
rmsle10=np.sqrt(mean_squared_log_error(Y_test, y_prdict))
rmsle10
#0.008337608169687899

#r2score
r2score10= r2_score(y_prdict,Y_test)*100
r2score10#94.9918734120857 

##################################################################################################
# 11)neural network regression

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
#output:loss: 1.2283e-04 - msle: 1.2283e-04
#msle: 0.01%


# Visualize training history

# list all data in history
history.history.keys()


# summarize history for accuracy
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

####################################################################################################

             
#tabulating the values

data={'Model':pd.Series(['LinearRegression','DecisionTree_Regressor','RandomForest_Regressor','Gradientboosting_regressor','supportvector_regressor','XGB_Regressor','Ridge_regressor','Lasso_regressor','KNeighbors_Regressor','AdaBoost_Regressor']),'RMSE':pd.Series([rmse1,rmse2,rmse3,rmse4,rmse5,rmse6,rmse7,rmse8,rmse9,rmse10]),
      'R2SCORES':pd.Series([r2score1,r2score2,r2score3,r2score4,r2score5,r2score6,r2score7,r2score8,r2score9,r2score10]),'MAE':pd.Series([mae1,mae2,mae3,mae4,mae5,mae6,mae7,mae8,mae9,mae10]),'RMSLE':pd.Series([rmsle1,rmsle2,rmsle3,rmsle4,rmsle5,rmsle6,rmsle7,rmsle8,rmsle9,rmsle10]),
      'MSE':pd.Series([mse1,mse2,mse3,mse4,mse5,mse6,mse7,mse8,mse9,mse10])}
data
pd.set_option("display.max.columns",None)
data
RESULTS=pd.DataFrame(data)
RESULTS
RESULTS.sort_values(['RMSE','R2SCORES','MAE','RMSLE','MSE','R2SCORES'])



##################################################################################################




        
        
        
        
        





