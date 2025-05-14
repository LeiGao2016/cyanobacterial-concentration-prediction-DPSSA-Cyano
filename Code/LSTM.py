
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout
from keras.optimizers import Adam

# from tensorflow.keras.optimizers import Adam
from numpy.random import seed
from utils import *

# GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_visible_devices([gpus[0]], "GPU")

def evaluation_metric1(y_test,y_hat):
    MSE = metrics.mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = metrics.mean_absolute_error(y_test,y_hat)
    R2 = metrics.r2_score(y_test,y_hat)
    MRE = metrics.mean_absolute_percentage_error(y_test,y_hat)
    var=np.var(y_hat)
    NSE=1-(MSE/var)
    print('MSE: %.5f' % MSE)
    print('RMSE: %.5f' % RMSE)
    print('MAE: %.5f' % MAE)
    print('R2: %.5f' % R2)
    print('MRE: %.5f' % MRE)
    print('NSE: %.5f' % NSE)

def data_split(sequence, n_timestamp):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp

        if end_ix > len(sequence):
            break

        seq_x = sequence[i:end_ix, 0:sequence.shape[1]]
        seq_y = sequence[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def create_dataset(dataset, look_back=20):
    dataX, dataY = [], []
    for i in range(look_back, len(dataset)):
        dataX.append(dataset[(i - look_back):i, :])
        dataY.append(dataset[i, -1])
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)

    return TrainX, Train_Y

def lstm(model_type,yuan_X_train):
    if model_type == 1:
        # single-layer LSTM
        yuan_model = Sequential()
        yuan_model.add(LSTM(units=150, activation='tanh',
                    input_shape=(yuan_X_train.shape[1], yuan_X_train.shape[2])))
        yuan_model.add(Dense(units=1))

    if model_type == 2:
        # multi-layer LSTM
        yuan_model = Sequential()
        yuan_model.add(LSTM(units=50, activation='tanh', return_sequences=True,
                    input_shape=(yuan_X_train.shape[1], yuan_X_train.shape[2]))) #70
#        yuan_model.add(LSTM(units=120, activation='tanh', return_sequences=True)) #60
        yuan_model.add(LSTM(units=50, activation='tanh'))  # 60
        yuan_model.add(Dropout(0.1))
        yuan_model.add(Dense(1))

    if model_type == 3:
        # BiLSTM
        yuan_model = Sequential()
        yuan_model.add(GRU(units=50, activation='tanh', return_sequences=True,
                            input_shape=(yuan_X_train.shape[1], yuan_X_train.shape[2])))  # 70
        yuan_model.add(GRU(units=50, activation='tanh'))  # 60
        yuan_model.add(Dropout(0.1))
        yuan_model.add(Dense(1))


    return yuan_model

seed(1)
tf.random.set_seed(1)

n_timestamp = 5 #Timestamp
n_epochs = 300
# ====================================
#      model type：
#            1. single-layer LSTM
#            2. multi-layer LSTM
#            3. bidirectional LSTM
# ====================================
model_type = 2
#得到环境变量
Environment = pd.DataFrame([])
Discharge = pd.read_csv('./Discharge_RECONSTRUCTION.csv', index_col='Date')
Velocity = pd.read_csv('./Velocity_RECONSTRUCTION.csv', index_col='Date')
Temperature = pd.read_csv('./Temperature_RECONSTRUCTION.csv', index_col='Date')
Salinity = pd.read_csv('./Salinity_RECONSTRUCTION.csv', index_col='Date')

Environment.loc[:,'Discharge'] = Discharge.loc[:,'0']+Discharge.loc[:,'1']
Environment.loc[:,'Velocity'] = Velocity.loc[:,'0']+Velocity.loc[:,'1']
Environment.loc[:,'Temperature'] = Temperature.loc[:,'0']+Temperature.loc[:,'1']
Environment.loc[:,'Salinity'] = Salinity.loc[:,'0']+Salinity.loc[:,'1']

yuan_data = pd.read_csv('./LOCK_9_RECONSTRUCTION.csv', index_col='Date')
#yuan_data.index = pd.to_datetime(yuan_data['trade_date'], format='%Y%m%d')
yuan_data = yuan_data.iloc[0:1248, :] #The first 24 years
yuan_data = yuan_data.loc[:, ['0', '1']]
yuan_data.loc[:, ['Discharge', 'Velocity', 'Temperature', 'Salinity']] = Environment.loc[:, ['Discharge', 'Velocity', 'Temperature', 'Salinity']]
yuan_data.loc[:,'ALL'] = (np.array(yuan_data.loc[:,'0'])+np.array(yuan_data.loc[:,'1']))
yuan_data = yuan_data.drop(['0','1'], axis=1)

idx = int(len(yuan_data)-156) #The past three years
yuan_training_set = yuan_data.iloc[0:idx, :]
yuan_test_set = yuan_data.iloc[idx-n_timestamp:, :]

sc = MinMaxScaler(feature_range=(0, 1))
yuan_sc = MinMaxScaler(feature_range=(0, 1))
yuan_training_set_scaled = yuan_sc.fit_transform(yuan_training_set) #The data is centralized in the range of [0,1] using the yuan_training_set for training.
yuan_testing_set_scaled = yuan_sc.fit_transform(yuan_test_set) #The data is centralized in the range of [0,1] using the yuan_test_set for training.

#yuan_X_train, yuan_y_train = data_split(yuan_training_set_scaled, n_timestamp)
yuan_X_train, yuan_y_train = create_dataset(yuan_training_set_scaled, n_timestamp)
yuan_X_train = yuan_X_train.reshape(yuan_X_train.shape[0], yuan_X_train.shape[1], 5)

yuan_X_test, yuan_y_test  = create_dataset(yuan_testing_set_scaled, n_timestamp)
#yuan_X_test, yuan_y_test = data_split(yuan_testing_set_scaled, n_timestamp)
yuna_X_test = yuan_X_test.reshape(yuan_X_test.shape[0], yuan_X_test.shape[1], 5)

yuan_model = lstm(model_type,yuan_X_train)
print(yuan_model.summary())
adam = Adam(learning_rate=0.01)
yuan_model.compile(optimizer=adam, loss='mse')
yuan_history = yuan_model.fit(yuan_X_train, yuan_y_train,
                              batch_size=32,
                              epochs=n_epochs,
                              validation_data=(yuan_X_test, yuan_y_test),
                              validation_freq=1)

plt.figure(figsize=(10, 6))
plt.plot(yuan_history.history['loss'], label='Training Loss')
plt.plot(yuan_history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

prediction = yuan_model.predict(yuan_X_test)
prediction = np.repeat(prediction,5, axis=-1)
prediction = yuan_sc.inverse_transform(prediction)
prediction_list = np.array(prediction[:, -1]).flatten().tolist()
prediction1 = {
    'date': yuan_data.index[idx:],
    'concentration': prediction_list
}
prediction1 = pd.DataFrame(prediction1)
prediction1 = prediction1.set_index(['date'], drop=True)

observation = yuan_y_test.reshape(len(yuan_y_test),1)
observation = np.repeat(observation,5, axis=-1)
observation = yuan_sc.inverse_transform(observation)
observation_list = np.array(observation[:, -1]).flatten().tolist()
observation1 = {
    'date': yuan_data.index[idx:],
    'concentration': observation_list
}
observation1 = pd.DataFrame(observation1)
observation1 = observation1.set_index(['date'], drop=True)
#yuan_predicted_stock_price1.to_csv('C:/Users/86150/Desktop/一些文件/研究生文件/python/Singular_Spectrum_Analysis/results/qian5prediction.csv')

plt.figure(figsize=(10, 6))
plt.plot(observation1['concentration'], label='Observed Concentration')
plt.plot(prediction1['concentration'], label='Predicted Concentration')
plt.title('Cyanobacteria Concentration Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('concentration', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()

yhat = yuan_data.loc[idx+1:, 'ALL']
yhat = yhat.T
yhat1 = prediction1['concentration']
yhat1 = yhat1.T
evaluation_metric1(yhat1, yhat)

prediction2 = yuan_model.predict(yuan_X_train)
prediction2 = np.repeat(prediction2,5, axis=-1)
prediction2 = yuan_sc.inverse_transform(prediction2)
prediction2_list = np.array(prediction2[:, -1]).flatten().tolist()
prediction2 = {
    'date': yuan_data.index[n_timestamp:idx],
    'concentration': prediction2_list
}
prediction2 = pd.DataFrame(prediction2)
prediction2 = prediction2.set_index(['date'], drop=True)

plt.figure(figsize=(10, 4))
plt.plot(yuan_data['ALL'], label='Observed Concentration')
plt.plot(prediction1['concentration'], label='Test set',linestyle='--', linewidth=1.5)
plt.plot(prediction2['concentration'], label='Training Set',linestyle='--', linewidth=1.5)
plt.title('Cyanobacteria Concentration Prediction')
plt.xlabel('Week', fontsize=12, verticalalignment='top')
plt.ylabel('concentration', fontsize=12, horizontalalignment='center')
plt.legend()
plt.show()

prediction3 = prediction2.append(prediction1)
# prediction3.to_csv('./prediction01.csv')