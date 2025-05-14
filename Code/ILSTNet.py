
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential,K,Model,Input
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Conv1D, concatenate, add, GRU
from keras.optimizers import Adam
from keras.layers.core import  Lambda,Activation

# from tensorflow.keras.optimizers import Adam
from numpy.random import seed
from utils import *

# GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_visible_devices([gpus[0]], "GPU")

def create_dataset1(dataset, look_back=20, skip=1):
    dataX, dataX2, dataY = [], [], []  #Using the first 10 current environment + historical data points to predict the concentration at the 11th point.
    for i in range(look_back*skip, len(dataset)):
        dataX.append(dataset[(i-look_back):i, :])
        dataY.append(dataset[i, -1])
        temp=[]
        for j in range(i-look_back*skip, i, skip):
            temp.append(dataset[j, :])
        dataX2.append(temp)
    TrainX = np.array(dataX)
    TrainX2 = np.array(dataX2)
    TrainY = np.array(dataY)
    return TrainX, TrainX2, TrainY

def LSTNet(yuan_X_trainX1, yuan_X_trainX2, yuan_y_train):
    input1 = Input(shape=(yuan_X_trainX1.shape[1], yuan_X_trainX1.shape[2]))
    conv1 = Conv1D(filters=10, kernel_size=5, strides=1, activation='relu')  # for input1 48 6
    # It's a probelm that I can't find any way to use the same Conv1D layer to train the two inputs,
    conv2 = Conv1D(filters=10, kernel_size=5, strides=1, activation='relu')  # for input2 48 6
    conv2.set_weights(conv1.get_weights())  # at least use same weight

    conv1out = conv1(input1)
    lstm1out = GRU(80,activation='tanh', return_sequences = True)(conv1out)#conv1out
#    lstm1out = GRU(150,activation='tanh', return_sequences = True)(lstm1out)#conv1out
    lstm1out = GRU(90,activation='tanh',)(lstm1out)  #g2:170 g3:170
    lstm1out = Dropout(0.15)(lstm1out)

    input2 = Input(shape=(yuan_X_trainX2.shape[1], yuan_X_trainX2.shape[2]))
    conv2out = conv2(input2)
    lstm2out = GRU(80,activation='tanh', return_sequences = True)(conv2out)#conv2out
#    lstm2out = GRU(160,activation='tanh', return_sequences = True)(lstm2out)#conv2out
    lstm2out = GRU(90,activation='tanh')(lstm2out)
    lstm2out = Dropout(0.15)(lstm2out) #0.2

    lstm_out = concatenate([lstm1out, lstm2out])  # concatenate
#    output = Dense(yuan_y_train.shape[1])(lstm_out)
    output = Dense(1)(lstm_out)

    # highway  Using Dense to simulate the AR autoregressive process, adding a linear component to the prediction, while also enabling the output to respond to changes in the scale of the input.
    highway_window = 5
    # Extract the time dimension of the last three windows while retaining all input dimensions.
    z = Lambda(lambda k: k[:, -highway_window:, :])(input1)
    z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
    z = Lambda(lambda k: K.reshape(k, (-1, highway_window * yuan_X_trainX1.shape[2])))(z)
#    z = Dense(yuan_y_train.shape[1])(z)
    z = Dense(1)(z)

#    z = Lambda(lambda k: K.reshape(k, (-1, yuan_y_train.shape[1])))(z) 
    z = Lambda(lambda k: K.reshape(k, (-1, 1)))(z)  

    output = add([output, z])
    output = Activation('sigmoid')(output)
    model = Model(inputs=[input1, input2], outputs=output)

    return model

seed(1)
tf.random.set_seed(1)

n_timestamp = 5 #Timestamp 10
n_epochs = 1500 #g2:271 g3：270
skip=4 #g2:11

#Obtain the environmental variables.
Environment = pd.DataFrame([])
Discharge = pd.read_csv('./Discharge_RECONSTRUCTION.csv', index_col='Date')
Velocity = pd.read_csv('./Velocity_RECONSTRUCTION.csv', index_col='Date')
Temperature = pd.read_csv('./Temperature_RECONSTRUCTION.csv', index_col='Date')
Salinity = pd.read_csv('./Salinity_RECONSTRUCTION.csv', index_col='Date')

Environment.loc[:,'Discharge'] = Discharge.loc[:,'2']
Environment.loc[:,'Velocity'] = Velocity.loc[:,'2']
Environment.loc[:,'Temperature'] = Temperature.loc[:,'2']
Environment.loc[:,'Salinity'] = Salinity.loc[:,'2']
#get PRIOR
yuan_data = pd.read_csv('./LOCK_9_RECONSTRUCTION.csv', index_col='Date')
#yuan_data.index = pd.to_datetime(yuan_data['trade_date'], format='%Y%m%d')
yuan_data = yuan_data.iloc[0:1248, :] #The previous 24 years
yuan_data = yuan_data.loc[:, ['2']]
yuan_data.loc[:, ['Discharge', 'Velocity', 'Temperature', 'Salinity']] = Environment.loc[:, ['Discharge', 'Velocity', 'Temperature', 'Salinity']]
yuan_data.loc[:,'ALL'] = np.array(yuan_data.loc[:,'2'])
yuan_data = yuan_data.drop(['2'], axis=1)

yuan_sc = MinMaxScaler(feature_range=(0, 1))
yuan_data_scaled = yuan_sc.fit_transform(yuan_data)
yuan_X1, yuan_X2, yuan_Y = create_dataset1(yuan_data_scaled, n_timestamp, skip)

idx = int(len(yuan_X1)-156) #yuan_X1
yuan_X_trainX1 = yuan_X1[0:idx, :, :]
yuan_X_trainX2 = yuan_X2[0:idx, :, :]
yuan_y_train = yuan_Y[0:idx]
yuan_X_testX1 = yuan_X1[idx:, :, :]
yuan_X_testX2 = yuan_X2[idx:, :, :]
yuan_y_test = yuan_Y[idx:]

yuan_model = LSTNet(yuan_X_trainX1, yuan_X_trainX2,yuan_y_train)
print(yuan_model.summary())
adam = Adam(learning_rate=0.01)
yuan_model.compile(optimizer=adam,
                   loss='mse')
yuan_history = yuan_model.fit([yuan_X_trainX1, yuan_X_trainX2], yuan_y_train,
                              batch_size=32,
                              epochs=n_epochs,
                              validation_data=([yuan_X_testX1, yuan_X_testX1], yuan_y_test),
                              validation_freq=1)

plt.figure(figsize=(10, 6))
plt.plot(yuan_history.history['loss'], label='Training Loss')
plt.plot(yuan_history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()



prediction = yuan_model.predict([yuan_X_testX1, yuan_X_testX2])
prediction = np.repeat(prediction,5, axis=-1)
prediction = yuan_sc.inverse_transform(prediction)
prediction_list = np.array(prediction[:, -1]).flatten().tolist()
prediction1 = {
    'date': yuan_data.index[idx+n_timestamp*skip:],
    'concentration': prediction_list
}
prediction1 = pd.DataFrame(prediction1)
prediction1 = prediction1.set_index(['date'], drop=True)


observation = yuan_y_test.reshape(len(yuan_y_test),1)
observation = np.repeat(observation,5, axis=-1)
observation = yuan_sc.inverse_transform(observation)
observation_list = np.array(observation[:, -1]).flatten().tolist()
observation1 = {
    'date': yuan_data.index[idx+n_timestamp*skip:],
    'concentration': observation_list
}
observation1 = pd.DataFrame(observation1)
observation1 = observation1.set_index(['date'], drop=True)
#yuan_predicted_stock_price1.to_csv('C:/Users/86150/Desktop/一些文件/研究生文件/python/Singular_Spectrum_Analysis/results/19to27g2prediction.csv')

plt.figure(figsize=(10, 6))
plt.plot(observation1['concentration'], label='Observed Concentration')
plt.plot(prediction1['concentration'], label='Predicted Concentration')
plt.title('Cyanobacteria Concentration Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Concentration', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()

yhat = yuan_data.loc[idx+n_timestamp*skip+1:, 'ALL']
yhat = yhat.T
yhat1 = prediction1['concentration']
yhat1 = yhat1.T
evaluation_metric(yhat1, yhat)


prediction = yuan_model.predict([yuan_X_trainX1, yuan_X_trainX2])
prediction = np.repeat(prediction,5, axis=-1)
prediction = yuan_sc.inverse_transform(prediction)
prediction_list = np.array(prediction[:, -1]).flatten().tolist()
prediction2 = {
    'date': yuan_data.index[n_timestamp*skip:idx+n_timestamp*skip],
    'concentration': prediction_list
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
# prediction3.to_csv('./prediction2.csv')