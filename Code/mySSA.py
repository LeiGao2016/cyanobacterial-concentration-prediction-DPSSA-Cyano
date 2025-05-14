import numpy as np
import pandas as pd
from numpy import matrix as m
from pandas import DataFrame as df
from scipy import linalg

try:
    import seaborn
except:
    pass
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 11, 4


class mySSA(object):
    '''Singular Spectrum Analysis object'''

    def __init__(self, time_series):

        self.ts = pd.DataFrame(time_series)
        self.ts_name = self.ts.columns.tolist()[0] 
        if self.ts_name == 0:
            self.ts_name = 'ts'
        self.ts_v = self.ts.values
        self.ts_N = self.ts.shape[0]
#        self.freq = self.ts.index.inferred_freq
        self.freq = 12

    @staticmethod
    def _printer(name, *args):
        '''Helper function to print messages neatly'''
        print('-' * 40)
        print(name + ':')
        for msg in args:
            print(msg)

    @staticmethod
    def _dot(x, y):
        '''Alternative formulation of dot product to allow missing values in arrays/matrices'''
        pass

    @staticmethod
    def get_contributions(X=None, s=None, plot=True): ## X is the trajectory matrix, s is the singular value
        '''Calculate the relative contribution of each of the singular values'''
        lambdas = np.power(s, 2) 
        frob_norm = np.linalg.norm(X) 
        ret = df(lambdas / (frob_norm ** 2), columns=['Contribution']) ## The sum of the squares of all singular values of matrix A should be equal to the square of the Frobenius norm (or 2-norm), which can be used to determine the contribution.
        ret['Contribution'] = ret.Contribution.round(4)
        if plot:
            ax = ret[ret.Contribution != 0].plot.bar(legend=False)
            ax.set_xlabel("Lambda_i")
            ax.set_title('Non-zero contributions of Lambda_i')
            vals = ax.get_yticks()
            ax.set_yticklabels(['{:3.2f}%'.format(x * 100) for x in vals])
            return ax
        return ret[ret.Contribution > 0]

    @staticmethod
    def diagonal_averaging(hankel_matrix):
        '''Performs anti-diagonal averaging from given hankel matrix
        Returns: Pandas DataFrame object containing the reconstructed series'''
        mat = m(hankel_matrix)
        L, K = mat.shape
        L_star, K_star = min(L, K), max(L, K)
        new = np.zeros((L, K))
        if L > K:
            mat = mat.T
        ret = []

        # Diagonal Averaging
        for k in range(1 - K_star, L_star):
            mask = np.eye(K_star, k=k, dtype='bool')[::-1][:L_star, :]
            mask_n = sum(sum(mask))
            ma = np.ma.masked_array(mat.A, mask=1 - mask)
            ret += [ma.sum() / mask_n]

        return df(ret).rename(columns={0: 'Reconstruction'})

    def view_time_series(self):
        '''Plot the time series'''
        self.ts.plot(title='Original Time Series')

    def embed(self, embedding_dimension=None, suspected_frequency=None, verbose=False, return_df=False):
        '''Embed the time series with embedding_dimension window size.
        Optional: suspected_frequency changes embedding_dimension such that it is divisible by suspected frequency'''
        if not embedding_dimension:
            self.embedding_dimension = self.ts_N // 2
        else:
            self.embedding_dimension = embedding_dimension
        if suspected_frequency:
            self.suspected_frequency = suspected_frequency
            self.embedding_dimension = (self.embedding_dimension // self.suspected_frequency) * self.suspected_frequency   #It is better for the window length to be an integer multiple of the period.

        self.K = self.ts_N - self.embedding_dimension + 1
        self.X = m(linalg.hankel(self.ts, np.zeros(self.embedding_dimension))).T[:, :self.K]     #Construct a Hankel matrix. When i+j is a constant in the trajectory matrix, Yij is the same, and this matrix is of the Hankel matrix type.
        self.X_df = df(self.X) #import DataFrame as df
        self.X_complete = self.X_df.dropna(axis=1) # 1, or 'columns' : Drop columns which contain missing value
        self.X_com = m(self.X_complete.values)
        self.X_missing = self.X_df.drop(self.X_complete.columns, axis=1)#drop in columns 含有缺失值的列单独拿出来
        self.X_miss = m(self.X_missing.values)
        self.trajectory_dimentions = self.X_df.shape
        self.complete_dimensions = self.X_complete.shape
        self.missing_dimensions = self.X_missing.shape
        self.no_missing = self.missing_dimensions[1] == 0

        if verbose:
            msg1 = 'Embedding dimension\t:  {}\nTrajectory dimensions\t: {}'
            msg2 = 'Complete dimension\t: {}\nMissing dimension     \t: {}'
            msg1 = msg1.format(self.embedding_dimension, self.trajectory_dimentions)
            msg2 = msg2.format(self.complete_dimensions, self.missing_dimensions)
            self._printer('EMBEDDING SUMMARY', msg1, msg2)

        if return_df:
            return self.X_df

    def decompose(self, verbose=False):
        '''Perform the Singular Value Decomposition and identify the rank of the embedding subspace
        Characteristic of projection: the proportion of variance captured in the subspace'''
        X = self.X_com
        self.S = X * X.T
        self.U, self.s, self.V = linalg.svd(self.S) #svd:Singular value decomposition, where U is the matrix of eigenvectors, s is the vector of eigenvalues arranged in descending order, and A = UsV.
        self.U, self.s, self.V = m(self.U), np.sqrt(self.s), m(self.V)
        self.d = np.linalg.matrix_rank(X)
        Vs, Xs, Ys, Zs = {}, {}, {}, {}
        for i in range(self.d):
            Zs[i] = self.s[i] * self.V[:, i] 
            Vs[i] = X.T * (self.U[:, i] / self.s[i])
            Ys[i] = self.s[i] * self.U[:, i]
            Xs[i] = Ys[i] * (m(Vs[i]).T)   #Obtain orthonormal vectors.
        self.Vs, self.Xs = Vs, Xs
        self.s_contributions = self.get_contributions(X, self.s, False)#Calculate the correlation, where X is the trajectory matrix and s is the singular value.
        self.r = len(self.s_contributions[self.s_contributions > 0])
        self.r_characteristic = round((self.s[:self.r] ** 2).sum() / (self.s ** 2).sum(), 4) 
        self.orthonormal_base = {i: self.U[:, i] for i in range(self.r)} #orthonormal

        if verbose:
            msg1 = 'Rank of trajectory\t\t: {}\nDimension of projection space\t: {}'
            msg1 = msg1.format(self.d, self.r)
            msg2 = 'Characteristic of projection\t: {}'.format(self.r_characteristic) #Projection properties
            self._printer('DECOMPOSITION SUMMARY', msg1, msg2)

    def view_s_contributions(self, adjust_scale=False, cumulative=False, return_df=False):
        '''View the contribution to variance of each singular value and its corresponding signal'''
        contribs = self.s_contributions.copy()
        contribs = contribs[contribs.Contribution != 0]
        if cumulative:
            contribs['Contribution'] = contribs.Contribution.cumsum() #cumsum
        if adjust_scale:
            contribs = (1 / contribs).max() * 1.1 - (1 / contribs)
        ax = contribs.plot.bar(legend=False)
        ax.set_xlabel("Singular_i")
        ax.set_title('Non-zero{} contribution of Singular_i {}'. \
                     format(' cumulative' if cumulative else '', '(scaled)' if adjust_scale else ''))
        if adjust_scale:
            ax.axes.get_yaxis().set_visible(False) 
        vals = ax.get_yticks() 
        ax.set_yticklabels(['{:3.0f}%'.format(x * 100) for x in vals])
        if return_df:
            return contribs

    @classmethod
    def view_reconstruction(cls, *hankel, names=None, return_df=False, plot=True, symmetric_plots=False):
        '''Visualise the reconstruction of the hankel matrix/matrices passed to *hankel'''
        hankel_mat = None
        for han in hankel:
            if isinstance(hankel_mat, m):
                hankel_mat = hankel_mat + han
            else:
                hankel_mat = han.copy()
        hankel_full = cls.diagonal_averaging(hankel_mat)
        title = 'Reconstruction of signal'
        if names or names == 0:
            title += ' associated with singular value{}: {}'
            title = title.format('' if len(str(names)) == 1 else 's', names)
        if plot:
            ax = hankel_full.plot(legend=False, title=title)
            if symmetric_plots:
                velocity = hankel_full.abs().max()[0]
                ax.set_ylim(bottom=-velocity, top=velocity)
        if return_df:
            return hankel_full

    def _forecast_prep(self, singular_values=None):
        self.X_com_hat = np.zeros(self.complete_dimensions)
        self.verticality_coefficient = 0
        self.forecast_orthonormal_base = {}
        if singular_values:  #singular_values 
            try:
                for i in singular_values:
                    self.forecast_orthonormal_base[i] = self.orthonormal_base[i]
            except:
                if singular_values == 0:
                    self.forecast_orthonormal_base[0] = self.orthonormal_base[0]
                else:
                    raise ('Please pass in a list/array of singular value indices to use for forecast')
        else:
            self.forecast_orthonormal_base = self.orthonormal_base
        self.R = np.zeros(self.forecast_orthonormal_base[0].shape)[:-1]
        for Pi in self.forecast_orthonormal_base.values():
            self.X_com_hat += Pi * Pi.T * self.X_com
            pi = np.ravel(Pi)[-1] #Take the last element of vector Pi.
            self.verticality_coefficient += pi ** 2
            self.R += pi * Pi[:-1]
        self.R = m(self.R / (1 - self.verticality_coefficient))
        self.X_com_tilde = self.diagonal_averaging(self.X_com_hat)

    def forecast_recurrent(self, steps_ahead=12, singular_values=None, plot=False, return_df=False, **plotargs):
        '''Forecast from last point of original time series up to steps_ahead using recurrent methodology
        This method also fills any missing data from the original time series.'''
        try:
            self.X_com_hat
        except(AttributeError):
            self._forecast_prep(singular_values)
        self.ts_forecast = np.array(self.ts_v[0])
        for i in range(1, self.ts_N + steps_ahead):
            try:
                if pd.isna(self.ts_v[i]):
                    x = self.R.T * m(self.ts_forecast[max(0, i - self.R.shape[0]): i]).T
                    self.ts_forecast = np.append(self.ts_forecast, x[0])
                else:
                    self.ts_forecast = np.append(self.ts_forecast, self.ts_v[i])
            except(IndexError):
                x = self.R.T * m(self.ts_forecast[i - self.R.shape[0]: i]).T
                if x[0] < 10:
                    x[0] = 10
                self.ts_forecast = np.append(self.ts_forecast, x[0])
        self.forecast_N = i + 1
        forecast_df = df(self.ts_forecast, columns=['Forecast'])
        forecast_df['Original'] = np.append(self.ts_v, [np.nan] * steps_ahead)
        if plot:
            forecast_df.plot(title='Forecasted vs. original time series', **plotargs)
        if return_df:
            return forecast_df


if __name__ == '__main__':    
    from matplotlib import pyplot as plt
    import pandas as pd
    import numpy as np
    from matplotlib.pylab import rcParams

    # ts = pd.read_csv('LOCK_9.csv‘, parse_dates=True, index_col='Month')
    ts = pd.read_csv('LOCK_9.csv', parse_dates=True, index_col='Date')
    ts = ts.drop('Discharge', axis=1)
    ts = ts.drop('Velocity', axis=1)
    ts = ts.drop('Temperature', axis=1)
    ts = ts.drop('Salinity', axis=1)
    ts = ts.drop('Prior', axis=1)
    ssa = mySSA(ts)
    K = 52 * 1  
    suspected_seasonality = 12  # 12
    ssa.embed(embedding_dimension=K, suspected_frequency=suspected_seasonality, verbose=True)
    ssa.decompose(verbose=True)
    # First enable display of graphs in the notebook

    rcParams['figure.figsize'] = 11, 4
    ssa.view_s_contributions()  
    ssa.view_s_contributions(adjust_scale=True)   # Check the contribution of the non-zero proportion
    ssa.view_s_contributions(eexp=True)  # Exponentiate the singular values.
    rcParams['figure.figsize'] = 11, 2
    for i in range(5):  # Only consider the decomposition of the first five elements of Xs, 5.
        ssa.view_reconstruction(ssa.Xs[i], names=i, symmetric_plots=i != 0)  # There are parameters in the SSA class that can be directly called.
    rcParams['figure.figsize'] = 11, 4
    ssa.ts.plot(title='Original Time Series')  # This is the original series for comparison
    flag = 15  # The first flag singular values are considered useful components: 33 for 2 years, 17 for 1 year.

    streams5 = [i for i in range(3)]  # 5
    reconstructed5 = ssa.view_reconstruction(*[ssa.Xs[i] for i in streams5], names=streams5, return_df=True)
    streams47 = [i for i in range(3, 8, 1)]  # 5
    reconstructed47 = ssa.view_reconstruction(*[ssa.Xs[i] for i in streams47], names=streams47, return_df=True)
    streams715 = [i for i in range(8, flag, 1)]  # 5
    reconstructed715 = ssa.view_reconstruction(*[ssa.Xs[i] for i in streams715], names=streams715, return_df=True)
    streams015 = [i for i in range(flag)]  # 5
    reconstructed015 = ssa.view_reconstruction(*[ssa.Xs[i] for i in streams015], names=streams015, return_df=True)
    streamsnoise = [i for i in range(flag, ssa.embedding_dimension, 1)]  # 5
    reconstructednoise = ssa.view_reconstruction(*[ssa.Xs[i] for i in streamsnoise], names=streamsnoise, return_df=True)

    ts_copy5 = ssa.ts.copy()
    ts_copy5['Reconstruction'] = reconstructed5.Reconstruction.values
    ts_copy5.plot(title='Original vs. Reconstructed Time Series')
    streams10 = [i for i in range(flag)]  # 10
    reconstructed10 = ssa.view_reconstruction(*[ssa.Xs[i] for i in streams10],
                                              names=streams10, return_df=True, plot=False)
    ts_copy10 = ssa.ts.copy()
    ts_copy10['Reconstruction'] = reconstructed10.Reconstruction.values
   
    ts_copy10.plot(title='Original vs. Forcasted')
    ssa.forecast_recurrent(steps_ahead=48, singular_values=streams10, plot=True)
    rcParams['figure.figsize'] = 11, 8
    ssa.forecast_recurrent(steps_ahead=ssa.ts.shape[0], singular_values=streams10, plot=True)
    rcParams['figure.figsize'] = 11, 4

    ts_copy15 = ssa.ts.copy()
    ts_copy15['qian3'] = reconstructed5.Reconstruction.values
    ts_copy15['3to7'] = reconstructed47.Reconstruction.values
    ts_copy15['8to15'] = reconstructed715.Reconstruction.values
    ts_copy15['qian15'] = reconstructed015.Reconstruction.values
    ts_copy15.to_csv('./RECONSTRUCTION.csv')

