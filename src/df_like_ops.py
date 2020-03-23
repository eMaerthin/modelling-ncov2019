from .read_csv import read_households_csv
from collections import defaultdict

def get_household2inhabitants(household, inhabitants):
    household2inhabitants = defaultdict(list)
    for h, i in zip(household, inhabitants):
        household2inhabitants[h].append(i)
    return household2inhabitants
