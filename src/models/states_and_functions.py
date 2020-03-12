import collections
from math import exp

import numpy as np
from .enums import (FearFunctions, InfectionStatus, ImportIntensityFunctions)
from .constants import *

fear_functions = {
    FearFunctions.FearDisabled: (lambda *args, **kwargs: 1),
    FearFunctions.Fear1: (lambda detected, deaths, weight_detected, weight_deaths, scale, limit_value:
                          1.0 + limit_value - limit_value / 0.5
                          * np.exp((detected * weight_detected + deaths * weight_deaths) / scale)
                          / (1 + np.exp((detected * weight_detected + deaths * weight_deaths) / scale))),
    FearFunctions.FearTanh: (lambda detected, deaths, weight_detected, weight_deaths, scale, limit_value:
                             -np.tanh(((detected * weight_detected + deaths * weight_deaths) - scale) / (scale / 3))
                             * ((1 - limit_value) / 2) + (1 - (1 - limit_value) / 2))
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
    InfectionStatus.Recovered
]

Event = collections.namedtuple('Event', [TIME, PERSON_INDEX, TYPE, INITIATED_BY,
                                         INITIATED_THROUGH, ISSUED_TIME, EPIDEMIC_STATUS])

import_intensity_functions = {
    ImportIntensityFunctions.Exponential: (lambda x, rate, multiplier: multiplier*exp(rate * x)),
    ImportIntensityFunctions.Polynomial: (lambda x, rate, multiplier: multiplier*pow(rate, x)),
    ImportIntensityFunctions.NoImport: (lambda _: 0)
}
