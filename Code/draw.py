import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from sklearn import metrics


def evaluation_metric(y_test,y_hat):
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
    return R2, MAE

## Reconstructed concentration, ensuring it is positive
yuan_data = pd.read_csv('./LOCK_9_RECONSTRUCTION.csv', index_col='Date')
yuan_data = yuan_data.iloc[5:1248, :]#Only 1246 are taken, neither 1 nor 1248 is included.
# yuan_data = yuan_data.reset_index(drop=True)
reconstruction = yuan_data['qianflag']+yuan_data['9']
real = yuan_data['ALL']

prediction = pd.read_csv('./results/Cyano_SSA.csv', index_col='date')

proposed_model = prediction.loc[:,'proposed']
SSA_ILSTNet = prediction.loc[:,'SSA_ILSTNet']
SSA_LSTM = prediction.loc[:,'SSA_LSTM']
ILSTNet = prediction.loc[:,'ILSTNet']
LSTM = prediction.loc[:,'LSTM']
GRU = prediction.loc[:,'GRU']
RNN = prediction.loc[:,'RNN']
CNNLSTM = prediction.loc[:,'CNNLSTM']

idx = int(len(prediction)-156)

print('**********proposed model**********')
yhat = np.array(proposed_model[idx:])
yhat1 = np.array(real[idx:])
proposedevaR2,proposedevaMAE = evaluation_metric(yhat1, yhat)

print('**********SSA-ILSTNet**********')
yhat = np.array(SSA_ILSTNet[idx:])
yhat1 = np.array(real[idx:])
SSA_ILSTNetevaR2,SSA_ILSTNetevaMAE = evaluation_metric(yhat1, yhat)

print('**********SSA-LSTM**********')
yhat = np.array(SSA_LSTM[idx:])
yhat1 = np.array(real[idx:])
SSA_LSTMevaR2,SSA_LSTMevaMAE = evaluation_metric(yhat1, yhat)

print('**********ILSTNet**********')
yhat = np.array(ILSTNet[idx:])
yhat1 = np.array(real[idx:])
ILSTNetevaR2,ILSTNetevaMAE = evaluation_metric(yhat1, yhat)

print('**********LSTM**********')
yhat = np.array(LSTM[idx:])
yhat1 = np.array(real[idx:])
LSTMevaR2,LSTMevaMAE = evaluation_metric(yhat1, yhat)

print('**********GRU**********')
yhat = np.array(GRU[idx:])
yhat1 = np.array(real[idx:])
GRUevaR2,GRUevaMAE = evaluation_metric(yhat1, yhat)

print('**********RNN**********')
yhat = np.array(RNN[idx:])
yhat1 = np.array(real[idx:])
RNNevaR2,RNNevaMAE = evaluation_metric(yhat1, yhat)

print('**********CNNLSTM**********')
yhat = np.array(CNNLSTM[idx:])
yhat1 = np.array(real[idx:])
CNNLSTMevaR2,CNNLSTMevaMAE = evaluation_metric(yhat1, yhat)

#font setting
legend_font = {
    'family': 'Times new roman',  
    'style': 'normal',
    'size': 12,  
    'weight': "normal",  
}

prediction_train = proposed_model.iloc[0:idx]
prediction_test = proposed_model.iloc[idx:]
proposed_model = prediction_train.append(prediction_test)

#Separate result display
idx = int(len(prediction)-156) #yuan_X1
prediction_train = proposed_model.iloc[0:idx]
prediction_test = proposed_model.iloc[idx:]

#Comparison between prediction and original data
plt.figure(figsize=(11, 4))
plt.plot(real,label='Observation',linewidth=2)
plt.plot(reconstruction, label='Reconstruction',linestyle='-.')
plt.title('Observation vs Reconstruction', fontsize=12, fontname="Times New Roman")
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Concentration (Cells number/ml)', fontsize=12, horizontalalignment='center',fontname="Times New Roman")
plt.legend(prop=legend_font)
plt.show()

#Comparison between prediction and original data
plt.figure(figsize=(11, 4))
plt.plot(real,label='Observation',linestyle='-.', color='#8F615D')
plt.plot(prediction_train, label='Training test',linewidth=1.5, color='#DEB340')
plt.plot(prediction_test, label='Testing test',linewidth=1.5, color='#DC8C81') #, color='red'
plt.title('(a) LOCK9', fontsize=12, fontname="Times New Roman")
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Concentration (Cells number/ml)', fontsize=12, horizontalalignment='center',fontname="Times New Roman")
plt.legend(prop=legend_font)
plt.show()

#Ablation study
#Week = np.arange(1253-156+1, 1248+1, 1)
plt.figure(figsize=(11, 4))
plt.plot(real, label='Observation', linewidth=1.5)
plt.plot(SSA_ILSTNet, label='ILSTNet-SSA', linewidth=1, linestyle='dashed')
plt.plot(SSA_LSTM, label='LSTM-SSA', linewidth=1, linestyle='dashed')
plt.plot(ILSTNet, label='ILSTNet', linewidth=1, linestyle='dashed')
plt.plot(LSTM, label='LSTM', linewidth=1, linestyle='dashed')
plt.plot(proposed_model, label='DPSSA-Cyano', linewidth=1, color='red')
plt.xlabel('Time (Week)', fontsize=12, fontname="Times New Roman", verticalalignment='top')
plt.ylabel('Concentration (Cells number/ML)', fontsize=12, fontname="Times New Roman", horizontalalignment='center')
plt.title('(a) LOCK9', fontsize=12, fontname="Times New Roman")  #, fontname="Times New Roman"
plt.legend(prop=legend_font)

#The final prediction results of the ablation study are magnified.
idx1 = int(len(prediction)-52) #yuan_X1
plt.figure(figsize=(11, 4))
plt.plot(real.loc[idx1:], label='Observation', linewidth=1.5)
plt.plot(SSA_ILSTNet.loc[idx1:], label='ILSTNet-SSA', linewidth=1, linestyle='dashed')
plt.plot(SSA_LSTM.loc[idx1:], label='LSTM-SSA', linewidth=1, linestyle='dashed')
plt.plot(ILSTNet.loc[idx1:], label='ILSTNet', linewidth=1, linestyle='dashed')
plt.plot(LSTM.loc[idx1:], label='LSTM', linewidth=1, linestyle='dashed')
plt.plot(proposed_model.loc[idx1:], label='DPSSA-Cyano',linewidth=1, color='red')
plt.xlabel('Time (Week)', fontsize=12, fontname="Times New Roman", verticalalignment='top')
plt.ylabel('Concentration (Cells number/ML)', fontsize=12, fontname="Times New Roman", horizontalalignment='center')
plt.title('2017', fontsize=12, fontname="Times New Roman")  #, fontname="Times New Roman"
plt.legend(prop=legend_font)

#Comapration
plt.figure(figsize=(11, 4))
plt.plot(real, label='Observation', linewidth=1.5)
plt.plot(GRU, label='GRU', linewidth=1, linestyle='dashed')
plt.plot(RNN, label='RNN', linewidth=1, linestyle='dashed')
plt.plot(CNNLSTM, label='CNN-LSTM', linewidth=1, linestyle='dashed')
plt.plot(proposed_model, label='DPSSA-Cyano', linewidth=1, color='red')
plt.xlabel('Time (Week)', fontsize=12, fontname="Times New Roman", verticalalignment='top')
plt.ylabel('Concentration (Cells number/ML)', fontsize=12, fontname="Times New Roman", horizontalalignment='center')
plt.title('(a) LOCK9', fontsize=12, fontname="Times New Roman")  #, fontname="Times New Roman"
plt.legend(prop=legend_font)
plt.show()

#The final prediction results of the comparative experiment are magnified.
idx1 = int(len(prediction)-52) #yuan_X1
plt.figure(figsize=(11, 4))
plt.plot(real.loc[idx1:], label='Observation', linewidth=1.5)
plt.plot(GRU.loc[idx1:], label='GRU', linewidth=1, linestyle='dashed')
plt.plot(RNN.loc[idx1:], label='RNN', linewidth=1, linestyle='dashed')
plt.plot(CNNLSTM.loc[idx1:], label='CNNLSTM', linewidth=1, linestyle='dashed')
plt.plot(proposed_model.loc[idx1:], label='DPSSA-Cyano',linewidth=1, color='red')
plt.xlabel('Time (Week)', fontsize=12, fontname="Times New Roman", verticalalignment='top')
plt.ylabel('Concentration (Cells number/ML)', fontsize=12, fontname="Times New Roman", horizontalalignment='center')
plt.title('2017', fontsize=12, fontname="Times New Roman")  #, fontname="Times New Roman"
plt.legend(prop=legend_font)


#The final prediction results are magnified.
idx2=int(len(prediction)-104)
plt.figure(figsize=(12, 4))
plt.plot(real.loc[idx2:],label='Observation', color='#1f77b4')
plt.plot(prediction_test.loc[idx2:], label='Testing set',linestyle='--',linewidth=1.5, color='#d62728') #, color='red'
plt.title('(a) LOCK9', fontsize=12, fontname="Times New Roman")
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Concentration (Cells number/ml)', fontsize=12, horizontalalignment='center',fontname="Times New Roman")
plt.legend(prop=legend_font)
x_major_locator=MultipleLocator(4)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.set_ylim(bottom=0)
plt.xlim(1144,1248)
plt.show()