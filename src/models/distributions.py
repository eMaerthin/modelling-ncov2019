import numpy as np
from collections import namedtuple


class distribution(object):
    def fit(self):
        raise NotImplementedError
    def rvs(self):
        raise NotImplementedError


class LogNorm(distribution):
    def fit(self, sample):
        lsample = np.log(sample)
        self.lmean = lsample.mean()
        self.lstd = lsample.std()

    def rvs(self):
        return np.exp(np.random.normal(self.lmean, self.lstd))


class Gamma(distribution):
    def fit(self, sample):
        sample_mean = np.mean(sample)
        sample_var = np.var(sample)
        self.alpha = sample_mean**2 / sample_var
        self.beta = sample_var / sample_mean

    def rvs(self):
        return np.random.gamma(self.alpha, self.beta)


SimpleStats = namedtuple('DescribeResults', 'nobs minmax mean variance')

def simple_stats(x):
    """Replacement for scipy.stats.describe"""
    x    = np.array(x)
    N    = len(x)
    mean = np.mean(x)
    var  = np.var(x)
    # skew = np.sum((x-mean)**3)/(np.sqrt(var)**1.5) * np.sqrt(N*(N-1))/(N-2)
    # kurt = np.sum((x-mean)**4)/var**2
    return SimpleStats(N, (np.min(x),np.max(x)), mean, var)
