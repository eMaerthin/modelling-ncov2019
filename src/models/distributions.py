import numpy as np


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

