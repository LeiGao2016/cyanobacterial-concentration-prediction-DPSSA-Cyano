from matplotlib import pyplot as plt
from mySSA import mySSA
import pandas as pd
import numpy as np
from matplotlib.pylab import rcParams

# ====================================
#      The decomposition procedure for the overall concentration.
# ====================================

ts = pd.read_csv('LOCK_9.csv', parse_dates=True, index_col='Date')
Discharge = ts.loc[:,'Discharge']
Velocity = ts.loc[:,'Velocity']
Temperature = ts.loc[:,'Temperature']
Salinity = ts.loc[:,'Salinity']

ts = ts.drop('Discharge', axis=1)
ts = ts.drop('Velocity', axis=1)
ts = ts.drop('Temperature', axis=1)
ts = ts.drop('Salinity', axis=1)
ts = ts.drop('Prior', axis=1)
# ts = ts.drop('ALL', axis=1)
ts = ts.iloc[0:1253, :]
ssa = mySSA(ts)
K = 10 
suspected_seasonality = 1  #12
ssa.embed(embedding_dimension=K, suspected_frequency=suspected_seasonality, verbose=True)
ssa.decompose(verbose=True)
# First enable display of graphs in the notebook
contribs = pd.DataFrame([])
rcParams['figure.figsize'] = 11, 4
contribs['contribution'] = ssa.view_s_contributions(return_df=1) 
ssa.view_s_contributions(adjust_scale=True) 
flag = 10
qian_flag_sumcontribs = contribs.loc[0:flag,:].sum()

rcParams['figure.figsize'] = 11, 4
ssa.ts.plot(title='Original Time Series') # This is the original series for comparison
flag = 9 
streamsflag = [i for i in range(flag)] #5
reconstructedflag = ssa.view_reconstruction(*[ssa.Xs[i] for i in streamsflag], names=streamsflag, return_df=True)
streamsnoise = [i for i in range(flag, ssa.embedding_dimension, 1)] #5
reconstructednoise = ssa.view_reconstruction(*[ssa.Xs[i] for i in streamsnoise], names=streamsnoise, return_df=True)

streams10 = [i for i in range(flag)] #10
reconstructed10 = ssa.view_reconstruction(*[ssa.Xs[i] for i in streams10],
                                          names=streams10, return_df=True, plot=False)
ts_copy10 = ssa.ts.copy()
ts_copy10['Reconstruction'] = reconstructed10.Reconstruction.values
ts_copy10.plot(title='Original vs. Reconstructed Time Series')
plt.show()

#rcParams['figure.figsize'] = 11, 2
streams0 = 0
reconstructed0 = ssa.view_reconstruction(ssa.Xs[0], names=streams0, return_df=True)
streams1 = 1
reconstructed1 = ssa.view_reconstruction(ssa.Xs[1], names=streams1, return_df=True)
streams2 = 2
reconstructed2 = ssa.view_reconstruction(ssa.Xs[2], names=streams2, return_df=True)
streams3 = 3
reconstructed3 = ssa.view_reconstruction(ssa.Xs[3], names=streams3, return_df=True)
streams4 = 4
reconstructed4 = ssa.view_reconstruction(ssa.Xs[4], names=streams4, return_df=True)
streams5 = 5
reconstructed5 = ssa.view_reconstruction(ssa.Xs[5], names=streams5, return_df=True)
streams6 = 6
reconstructed6 = ssa.view_reconstruction(ssa.Xs[6], names=streams6, return_df=True)
streams7 = 7
reconstructed7 = ssa.view_reconstruction(ssa.Xs[7], names=streams7, return_df=True)
streams8 = 8
reconstructed8 = ssa.view_reconstruction(ssa.Xs[8], names=streams8, return_df=True)
streams9 = 9
reconstructed9 = ssa.view_reconstruction(ssa.Xs[9], names=streams9, return_df=True)



ts_copy15 = ssa.ts.copy()
ts_copy15['qianflag'] = reconstructedflag.Reconstruction.values
ts_copy15['0'] = reconstructed0.Reconstruction.values
ts_copy15['1'] = reconstructed1.Reconstruction.values
ts_copy15['2'] = reconstructed2.Reconstruction.values
ts_copy15['3'] = reconstructed3.Reconstruction.values
ts_copy15['4'] = reconstructed4.Reconstruction.values
ts_copy15['5'] = reconstructed5.Reconstruction.values
ts_copy15['6'] = reconstructed6.Reconstruction.values
ts_copy15['7'] = reconstructed7.Reconstruction.values
ts_copy15['8'] = reconstructed8.Reconstruction.values
ts_copy15['9'] = reconstructed9.Reconstruction.values


# ts_copy15['Discharge'] = Discharge
# ts_copy15['Velocity'] = Velocity
# ts_copy15['Temperature'] = Temperature
# ts_copy15['Salinity'] = Salinity
ts_copy15.to_csv('./LOCK_9_RECONSTRUCTION.csv')

