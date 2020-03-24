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

def par0(sample):
    lsample = np.log(sample)
    loc = 0
    scale = np.exp(lsample.mean())
    shape = lsample.std()
    return shape, loc, scale

def par1(sample):
    loc = min(sample) - np.std(sample)/100
    lsample = np.log(sample - loc)
    scale = np.exp(lsample.mean())
    shape = lsample.std()
    return shape, loc, scale

def compare(sample, estim_par_foo):
    params = estim_par_foo(sample)
    logN = scipy.stats.lognorm(*params)
    x = np.linspace(min(sample), max(sample), 100)
    plt.plot(x, logN.pdf(x))
    sns.distplot(sample, rug=True, hist=False, rug_kws={"color": "g"},
    kde_kws={"color": "k", "lw": 3})
    plt.show()



compare(sample, par0)
compare(sample, par1)

