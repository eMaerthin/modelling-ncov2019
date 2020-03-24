from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

f = Path("/home/matteo/Projects/corona/modelling-ncov2019/data/vroclav") 
p1 = f/"population_experiment0.csv"
x = read_pop_exp_csv(p1)
w = dict(zip(x['idx'], x['household_index']))
147430 in w
w[147430]


p2 = f/'households_experiment0.csv'
y = pd.read_csv(p2, converters={'idx':ast.literal_eval},
                index_col='household_index')
y['capacity'] = y.idx.apply(lambda x: len(x))
y.to_dict().keys()
y.to_dict()['idx'][1011]
y.to_dict()['capacity'][1011]

z[147340]
df_individuals = pd.read_csv(p1)
df_individuals.index = df_individuals.idx
df_individuals['household_index'].to_dict()


w = y.to_dict()['idx']

y.index == np.arange(0,len(y))

x = read_households_csv(p2)
cap = {k: len(v) for k,v in x.items()}


y.to_dict()['capacity'] == cap
ycap = y.to_dict()['capacity']
all(cap[l] == ycap[l] for l in cap)

_household_capacities[147340]


def get_household2inhabitants(household, inhabitants):
    household2inhabitants = defaultdict(list)
    for h, i in zip(household, inhabitants):
        household2inhabitants[h].append(i)
    return household2inhabitants

get_household2inhabitants(x['household_index'], x['idx'])

path = p2

it_csv = iter_csv(path)
cols = next(it_csv)
d = list(it_csv)
inhabitants = [ast.literal_eval(i) for hi, i in d]
household   = [int(hi) for hi,_ in d]
capacities  = [len(i) for i in inhabitants]
