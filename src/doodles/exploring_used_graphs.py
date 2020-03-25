from collections import Counter
import getpass
import networkx as nx
import pandas as pd
from pathlib import Path
from itertools import combinations

from src.read_csv import read_pop_exp_csv, read_households_csv


proj = {'matteo': "/home/matteo/Projects/corona/modelling-ncov2019",
        'cov': "/home/cov/git/modelling-ncov2019"}[getpass.getuser()]
proj = Path(proj)
params_path = proj/"test/models/assets/params_experiment0.json"
df_individuals_path = proj/"data/vroclav/population_experiment0.csv"
df_households_path = proj/"data/vroclav/households_experiment0.csv"

I = pd.read_csv(df_individuals_path)
H = read_households_csv(df_households_path)

Counter(len(i_l) for i_l in H.values())

list(combinations([1], 2))

G = nx.Graph()
for i_l in H.values():
    if len(i_l) > 1:
        for a_b in combinations(i_l, 2):
            G.add_edge(*a_b)
    else:
        G.add_node(i_l[0])

print(nx.number_connected_components(G))
# 277667
G.cc = list(nx.connected_components(G))
Counter(len(cc) for cc in G.cc)
# Counter({1: 94629, 2: 80612, 3: 54403, 4: 32402, 5: 8472, 6: 4766, 7: 2383}) 