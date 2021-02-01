
import numpy as np
import requests
import pandas as pd
import json

#read data from url
url = "https://api.rootnet.in/covid19-in/stats/daily"
r = requests.get(url)
json_data = json.loads(r.content)

print(json_data)
"""
parse1 = json_data['data'][0]


ls1=[]

for region1 in parse1['regional']:
    ls1.append(region1)

df1=pd.DataFrame(ls1)

df1['date'] = parse1['day']
"""

#fetch and parse json data to dataframe
parse_all =json_data['data']

data_ls = []
gbl = globals()
for index in range(len(parse_all)):
    data=parse_all[index]
    
    ls_all=[]
    
    for region in data['regional']:
        ls_all.append(region)
    gbl['df_'+str(index)]=pd.DataFrame(ls_all)
    #print(gbl['df_'+str(index)])
    
    gbl['df_'+str(index)]['date'] = data['day']
    
    data_ls.append(gbl['df_'+str(index)])
    
#print(data_ls)

data_df = pd.concat(data_ls, ignore_index=True)

data_df.to_csv("dataindia_15-06-2020.csv", index=False)

#df=data_df.copy()

import pandas as pd
df_csv=pd.read_csv("dataindia_15-06-2020.csv")

#preprocess
df = df_csv

#preprocess imp columns
df['date']=pd.to_datetime(df.date).dt.date
df_dropped = df.drop(columns=['loc', 'confirmedCasesIndian','confirmedCasesForeign','discharged','deaths'])
gr_df = df_dropped.groupby(['date'])['totalConfirmed'].sum().reset_index() 
y= gr_df
gr_df.drop(gr_df.head(10).index, inplace=True)


#renaming accorfing to algo
gr_df = gr_df.astype({"totalConfirmed": float})
gr_df.rename(columns={'totalConfirmed':'y','date':'ds'}, inplace=True)
gr_df.count




import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd 
from fbprophet import Prophet


#apply fbprophet algo by fb
model = Prophet(interval_width=0.95)
model.fit(gr_df)


#after fitting
future = model.make_future_dataframe(periods=15)
future.tail()


#Prophet returns a large DataFrame with many interesting columns, but we subset our output to the columns most relevant to forecasting, which are:

#ds: the datestamp of the forecasted value
#yhat: the forecasted value of our metric (in Statistics, yhat is a notation traditionally used to represent the predicted values of a value y)
#yhat_lower: the lower bound of our forecasts
#yhat_upper: the upper bound of our forecasts

forecast = model.predict(future)
#forecast = model.predict(test)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


fc = forecast[['ds', 'yhat']].rename(columns = {'Date': 'ds', 'Forecast': 'yhat'})


#ploTTing
ig1 = model.plot(forecast)

# xaxis is the time
# y axis gives the confirmed cases 

model.plot_components(forecast)

#testing

new_date = pd.DataFrame(pd.date_range(start='2020-4-20', end='2020-4-28', freq='D'), columns=['ds'])

new_date = pd.DataFrame(pd.to_datetime(new_date['ds']).dt.date)

#Predictions
predictions = model.predict(new_date)


------------->
#testing on different trick after model fitting
#spliiting
y = gr_df
train = y[:int(0.75*(len(y)))]
test = y[int(0.75*(len(y))):]

#plotting the data
#train['y'].plot()
#test['y'].plot()


predictions1 = model.predict(test)

predictions1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()


fc = predictions1[['ds', 'yhat']].rename(columns = {'Date': 'ds', 'Forecast': 'yhat'})




#error calculation
from math import sqrt

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error
plt.style.use('fivethirtyeight') # For plots

#MSE
mse = mean_squared_error(y_true=test['y'],
                   y_pred=predictions1['yhat'])

rmse = sqrt(mse)
print('RMSE: {}, MSE:{}'.format(rmse,mse))
#MAE
mae = mean_absolute_error(y_true=test['y'],
                   y_pred=predictions1['yhat'])

#MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_true=test['y'],
                   y_pred=predictions1['yhat'])




"""### Forecast quality scoring metrics
- __R squared__
- __Mean Absolute Error__
- __Median Absolute Error__
- __Mean Squared Error__
- __Mean Squared Logarithmic Error__
- __Mean Absolute Percentage Error__
"""



from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error

"""__R squared__, coefficient of determination (it can be interpreted as a percentage of variance explained by the model), (-inf, 1] 
- sklearn.metrics.r2_score
"""

r2_score(test.y, predictions1['yhat'])

"""__Mean Absolute Error__, it is an interpretable metric because it has the same unit of measurement as the initial series, [0, +inf)
- sklearn.metrics.mean_absolute_error
"""

mean_absolute_error(test.y, predictions1['yhat'])

"""__Median Absolute Error__, again an interpretable metric, particularly interesting because it is robust to outliers, [0, +inf)
- sklearn.metrics.median_absolute_error
"""

median_absolute_error(test.y, predictions1['yhat'])

"""__Mean Squared Error__, most commonly used, gives higher penalty to big mistakes and vise versa, [0, +inf)
- sklearn.metrics.mean_squared_error
"""

mean_squared_error(test.y, predictions1['yhat'])

"""__Mean Squared Logarithmic Error__, practically the same as MSE but we initially take logarithm of the series, as a result we give attention to small mistakes as well, usually is used when data has exponential trends, [0, +inf)
- sklearn.metrics.mean_squared_log_error
"""

mean_squared_log_error(test.y, predictions1['yhat'])

"""__Mean Absolute Percentage Error__, same as MAE but percentage, — very convenient when you want to explain the quality of the model to your management, [0, +inf), 
- not implemented in sklearn
"""

def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(y_true=test['y'],
                   y_pred=predictions1['yhat'])

"""__Function to evaluate forecast using above metrics:__"""

def evaluate_forecast(y,pred):
    results = pd.DataFrame({'r2_score':r2_score(y, pred),
                           }, index=[0])
    results['mean_absolute_error'] = mean_absolute_error(y, pred)
    results['median_absolute_error'] = median_absolute_error(y, pred)
    results['mse'] = mean_squared_error(y, pred)
    results['msle'] = mean_squared_log_error(y, pred)
    results['mape'] = mean_absolute_percentage_error(y, pred)
    results['rmse'] = np.sqrt(results['mse'])
    return results

evaluate_forecast(test.y, predictions1['yhat'])




------------->
#testing on different trick after model fitting

#testing

new_date = pd.DataFrame(pd.date_range(start='2020-4-20', end='2020-4-28', freq='D'), columns=['ds'])

new_date = pd.DataFrame(pd.to_datetime(new_date['ds']).dt.date)

#Predictions
predictions2 = model.predict(new_date)


predictions2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()




----------->
#LSTM MODELS

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.layers import LSTM
from keras  import callbacks
from keras import optimizers
import pandas as pd 
import tensorflow as tf
import numpy as np
import pandas as pd

df_csv = pd.read_csv("dataindia.csv")

#preprocess
df = df_csv

#preprocess imp columns
df['date']=pd.to_datetime(df.date).dt.date
df_dropped = df.drop(columns=['loc', 'confirmedCasesIndian','confirmedCasesForeign','discharged','deaths'])
gr_df = df_dropped.groupby(['date'])['totalConfirmed'].sum().reset_index() 
gr_df.drop(gr_df.head(3).index, inplace=True)


gr_df.index = gr_df['date']
gr_df = gr_df.drop("date", axis=1)

print(gr_df.head())
gr_df.shape

#PREPROCESS
dataset = gr_df.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset)

#or
#scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
#scaled = min_max_scaler.fit_transform(dataset['totalConfirmed'].values.reshape(-1, 1))
print('Min', np.min(scaled))
print('Max', np.max(scaled))

#splitting
train_size = int(len(scaled) * 0.70)
test_size = len(scaled - train_size)
train, test = scaled[0:train_size, :], scaled[train_size: len(scaled), :]
print('train: {}\ntest: {}'.format(len(train), len(test)))

#CREATE RNN
def create_dataset(dataset, look_back=1):
    print(len(dataset), look_back)
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        print(i)
        print('X {} to {}'.format(i, i+look_back))
        print(a)
        print('Y {}'.format(i + look_back))
        print(dataset[i + look_back, 0])
        dataset[i + look_back, 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

#reshape into X=t and Y=t+1


look_back = 1
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)



#reshape input to be [samples, time steps, features]


X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape)
print(X_test.shape)


#create and fit the LSTM network


batch_size = 1
model = Sequential()
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=batch_size, verbose=2, shuffle=True)


#Make preditions

import math
from sklearn.metrics import mean_squared_error

trainPredict = model.predict(X_train, batch_size=batch_size)
model.reset_states()

testPredict = model.predict(X_test, batch_size=batch_size)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
y_train = scaler.inverse_transform([y_train])
y_train = y_train.reshape(-1,1)

testPredict = scaler.inverse_transform(testPredict)
y_test = scaler.inverse_transform([y_test])
y_test = y_test.reshape(-1,1)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test, testPredict))
print('Test Score: %.2f RMSE' % (testScore))




trainPredictPlot = np.empty_like(scaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(scaled)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(scaled)-1, :] = testPredict
# plot baseline and predictions
plt.figure(figsize=(20,10))
plt.plot(scaler.inverse_transform(scaled))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()





---------------------------------------------------------------------------->
# ARIMA MODELS


import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import statsmodels.api as sm
import matplotlib
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt


#read data
df_csv=pd.read_csv("dataindia_19-04-2020.csv")

#preprocess
df = df_csv

#preprocess imp columns
df['date']=pd.to_datetime(df.date).dt.date
df_dropped = df.drop(columns=['loc', 'confirmedCasesIndian','confirmedCasesForeign','discharged','deaths'])
gr_df = df_dropped.groupby(['date'])['totalConfirmed'].sum().reset_index() 
gr_df.drop(gr_df.head(3).index, inplace=True)


#renaming accorfing to algo
gr_df = gr_df.astype({"totalConfirmed": float})

gr_df = gr_df.set_index('date')


y = gr_df

#plotting input data
y.plot(figsize=(15, 6))
plt.show()



# ARIMA example
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error


#Parameter Selection for the ARIMA Time Series Model


p = d = q = range(0, 9)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))



for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
        
        

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 0, 0),
                                seasonal_order=(0, 8, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])

print(results.summary())

#this used for large dataset
"""
#divide into train and validation set
train = y[:int(0.75*(len(y)))]
valid = y[int(0.75*(len(y))):]

#plotting the data
train['Total_confirmed_cases'].plot()
valid['Total_confirmed_cases'].plot()

start_index = valid.index.min()
end_index = valid.index.max()
pred = results.predict(start=start_index, end=end_index)

or
"""
#prediction
start_index = pd.to_datetime('2020-03-13')
end_index = pd.to_datetime('2020-04-19')

predictions = results.predict(start=start_index, end=end_index)

print(predictions)

#or predict with steps
pred_uc = results.get_forecast(steps=30)
pred_ci = pred_uc.conf_int()

#error
# report performance
mse = mean_squared_error(y[start_index:end_index], predictions)
rmse = sqrt(mse)
print('RMSE: {}, MSE:{}'.format(rmse,mse))

#plotting error
plt.plot(y.Total_confirmed_cases)
plt.plot(predictions, color='red')
plt.title('RMSE: %.4f'% rmse)
plt.show()


#plotting prediction to next 30 date

pred_uc = results.get_forecast(steps=10)
pred_ci = pred_uc.conf_int()

ax = y.plot(label='observed', figsize=(15, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)

plt.ylim(top=30000) #ymax is your value
plt.ylim(bottom=-30000) #ymin is your value

ax.set_xlabel('Date')
ax.set_ylabel('case')

plt.legend()
plt.show()







"""
#fetch data world wide

import json
import numpy as np
import requests
import pandas as pd
#read data from url
url = "https://coronavirus-tracker-api.herokuapp.com/v2/locations"
r = requests.get(url)
json_data = json.loads(r.content)
"""

data=json_data["locations"]

df=pd.DataFrame(data)

