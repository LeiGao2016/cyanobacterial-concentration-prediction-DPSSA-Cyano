import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_auto_corr(timeSeries, k):
    '''
    Descr:
        输入：时间序列timeSeries，滞后阶数k
        输出：时间序列timeSeries的k阶自相关系数
        l：序列timeSeries的长度
        timeSeries1，timeSeries2:拆分序列1，拆分序列2
        timeSeries_mean:序列timeSeries的均值
        timeSeries_var:序列timeSeries的每一项减去均值的平方的和

    '''
    l = len(timeSeries)
    # Extract the two arrays that need to be calculated.
    timeSeries1 = timeSeries[0:l - k]
    timeSeries2 = timeSeries[k:]
    timeSeries_mean = timeSeries.mean()
    timeSeries_var = np.array([i ** 2 for i in timeSeries - timeSeries_mean]).sum()
    auto_corr = 0
    for i in range(l - k):
        temp = (timeSeries1[i] - timeSeries_mean) * (timeSeries2[i] - timeSeries_mean) / timeSeries_var
        auto_corr = auto_corr + temp
    return auto_corr

#yuan_data = pd.read_csv('LOCK_9.csv', index_col='Date')
yuan_data = pd.read_csv('./Temperature_RECONSTRUCTION.csv', index_col='Date')
from pandas.plotting import autocorrelation_plot

x= yuan_data['2']
plt.figure('x')
plt.plot(x)
y = x
x = np.array(x)
N = len(x)
plt.figure('Autocorrelation')
autocorrelation_plot(x)
plt.show()
print(get_auto_corr(x,5))

