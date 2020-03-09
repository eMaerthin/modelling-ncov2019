import collections

import numpy as np
from .enums import (FearFunctions, InfectionStatus, ImportIntensityFunctions)
from .constants import *

fear_functions = {
    FearFunctions.FearDisabled: (lambda _: 1),
    FearFunctions.Fear1: (lambda detected, deaths, weight_detected, weight_deaths, scale:
                          1.5 - np.exp((detected * weight_detected + deaths * weight_deaths) / scale)
                          / (1 + np.exp((detected * weight_detected + deaths * weight_deaths) / scale))),
}

active_states = [
    InfectionStatus.Contraction,
    InfectionStatus.Infectious,
    InfectionStatus.StayHome,
    #InfectionStatus.SeeDoctor,
]

termination_states = [
    InfectionStatus.Hospital,
    InfectionStatus.Death,
    #InfectionStatus.Recovered
]

Event = collections.namedtuple('Event', [TIME, PERSON_INDEX, TYPE, INITIATED_BY,
                                         INITIATED_THROUGH, ISSUED_TIME, EPIDEMIC_STATUS])

import_intensity_functions = {
    ImportIntensityFunctions.Exponential: (lambda x, rate, multiplier: multiplier*pow(rate, x)),
    ImportIntensityFunctions.NoImport: (lambda _: 0)
}
