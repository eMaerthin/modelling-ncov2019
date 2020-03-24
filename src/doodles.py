import numpy as np
import ast
import scipy.stats
import seaborn as sns

import mocos_helper as MH 
import matplotlib.pyplot as plt

incubation = np.load("/home/matteo/Projects/corona/modelling-ncov2019/test/models/assets/incubation_period_distribution.npy")
t1 = np.load("/home/matteo/Projects/corona/modelling-ncov2019/test/models/assets/t1_distribution.npy")
t1_t2 = np.load("/home/matteo/Projects/corona/modelling-ncov2019/test/models/assets/t1_t2_distribution.npy")
onset_death = np.load("/home/matteo/Projects/corona/modelling-ncov2019/test/models/assets/onset_death_distribution.npy")

# incubation period, modelled as lognormal
# t1, t1_2 -> gamma
# death: lognormal


sample = incubation
l_sample = np.log(sample)
lmean = l_sample.mean()
lstd = l_sample.std()
X = np.exp(np.random.normal(lmean, lstd))

sample = t1
alpha, beta = fit_gamma_parameters(sample)
X = np.random.gamma(alpha, beta, 1000)

plt.hist(sample, density=True)
plt.hist(X, density=True, alpha=.4)
plt.show()

def par0(sample):
    lsample = np.log(sample)
    loc = 0
    scale = sample.mean()
    shape = lsample.std()
    return shape, loc, scale

def compareLognorms(sample, estim_par_foo):
    params = estim_par_foo(sample)
    logN = scipy.stats.lognorm(*params)
    x = np.linspace(min(sample), max(sample), 100)
    plt.plot(x, logN.pdf(x))
    sns.distplot(sample, rug=True, hist=False, rug_kws={"color": "g"},
    kde_kws={"color": "k", "lw": 3})
    plt.show()

compareLognorms(incubation, par0)
compareLognorms(incubation, par1)


def fit_gamma_parameters(sample):
    sample_mean = np.mean(sample)
    sample_var = np.var(sample)
    alpha = sample_mean**2 / sample_var
    beta = sample_var / sample_mean
    return alpha, beta

def compareGammas(sample, estim_par_foo):
    alpha, beta = estim_par_foo(sample)
    gamma = scipy.stats.gamma(alpha, 0, beta)
    gamma_ss = scipy.stats.gamma(*scipy.stats.gamma.fit(sample, floc=0))
    x = np.linspace(min(sample), max(sample), 100)
    plt.plot(x, gamma.pdf(x))
    plt.plot(x, gamma_ss.pdf(x))
    sns.distplot(sample, rug=True, hist=False, rug_kws={"color": "g"},
    kde_kws={"color": "k", "lw": 3})
    plt.show()

compareGammas(t1, fit_gamma_parameters)

plt.hist(t1)
plt.show()