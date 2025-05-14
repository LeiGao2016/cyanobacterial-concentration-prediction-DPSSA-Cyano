import pandas as pd

summary_model = pd.DataFrame([])

#SSA-LSTM-LSTNet
prediction1 =pd.read_csv('./prediction01.csv', index_col='date')
prediction2 = pd.read_csv('./prediction2.csv', index_col='date')
prediction3 = pd.read_csv('./prediction3.csv', index_col='date')
prediction4 = pd.read_csv('./prediction45.csv', index_col='date')
prediction5 = pd.read_csv('./prediction67.csv', index_col='date')
prediction6 = pd.read_csv('./prediction8.csv', index_col='date')
prediction7 = pd.read_csv('./prediction9.csv', index_col='date')
prediction = (prediction1+prediction2+prediction3+prediction4+prediction5+prediction6+prediction7).fillna(prediction1)
prediction = prediction['concentration'].apply(lambda x: max(0, x))
summary_model['proposed'] = prediction
#SSA-ILSTNet
prediction1 =pd.read_csv('./ablation_experiments/prediction_LSTNet_01.csv', index_col='date')
prediction2 = pd.read_csv('./prediction2.csv', index_col='date')
prediction3 = pd.read_csv('./prediction3.csv', index_col='date')
prediction4 = pd.read_csv('./prediction45.csv', index_col='date')
prediction5 = pd.read_csv('./prediction67.csv', index_col='date')
prediction6 = pd.read_csv('./prediction8.csv', index_col='date')
prediction7 = pd.read_csv('./prediction9.csv', index_col='date')
prediction = (prediction1+prediction2+prediction3+prediction4+prediction5+prediction6+prediction7).fillna(prediction1)
prediction = prediction['concentration'].apply(lambda x: max(0, x))
summary_model['SSA_ILSTNet'] = prediction

#SSA-LSTM
prediction1 =pd.read_csv('./prediction01.csv', index_col='date')
prediction2 = pd.read_csv('./ablation_experiments/prediction_LSTM_2.csv', index_col='date')
prediction3 = pd.read_csv('./ablation_experiments/prediction_LSTM_3.csv', index_col='date')
prediction4 = pd.read_csv('./ablation_experiments/prediction_LSTM_45.csv', index_col='date')
prediction5 = pd.read_csv('./ablation_experiments/prediction_LSTM_67.csv', index_col='date')
prediction6 = pd.read_csv('./ablation_experiments/prediction_LSTM_8.csv', index_col='date')
prediction7 = pd.read_csv('./ablation_experiments/prediction_LSTM_9.csv', index_col='date')
prediction = (prediction1+prediction2+prediction3+prediction4+prediction5+prediction6+prediction7).fillna(prediction1)
prediction = prediction['concentration'].apply(lambda x: max(0, x))
summary_model['SSA_LSTM'] = prediction

#LSTM
prediction1 =pd.read_csv('./ablation_experiments/prediction_LSTM_total.csv', index_col='date')
prediction = (prediction1)
prediction = prediction['concentration'].apply(lambda x: max(0, x))
summary_model['LSTM'] = prediction

#ILSTNet
prediction1 =pd.read_csv('./ablation_experiments/prediction_LSTNet_total.csv', index_col='date')
prediction = (prediction1)
prediction = prediction['concentration'].apply(lambda x: max(0, x))
summary_model['ILSTNet'] = prediction

#GRU
prediction1 =pd.read_csv('./compared_experiments/prediction_GRU_total.csv', index_col='date')
prediction = (prediction1)
prediction = prediction['concentration'].apply(lambda x: max(0, x))
summary_model['GRU'] = prediction

#BiLSTM
prediction1 =pd.read_csv('./compared_experiments/prediction_RNN_total.csv', index_col='date')
prediction = (prediction1)
prediction = prediction['concentration'].apply(lambda x: max(0, x))
summary_model['RNN'] = prediction

#CNNLSTM
prediction1 =pd.read_csv('./compared_experiments/prediction_CNNLSTM_total.csv', index_col='date')
prediction = (prediction1)
prediction = prediction['concentration'].apply(lambda x: max(0, x))
summary_model['CNNLSTM'] = prediction

summary_model.to_csv('./results/Cyano_SSA.csv')