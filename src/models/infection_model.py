"""
This is mostly based on references/infection_alg.pdf
"""

import collections
import enum
import json
import os
import sys
import typing

import numpy as np
import pandas as pd
from schema import (Schema, And, Use, Or, Optional)
import config

# Todo: below should go to separate file


class SymptomStatus(enum.Enum):  # to be added to nodes?
    No = 'no'
    #Low = 1  # Maybe Staying Home, maybe visit Doctor
    Mild = 'mild'  # Maybe Staying Home, maybe visit Doctor, maybe go to Hospital
    Serious = 'serious'  # Need to go to Hospital immediately!


class EpidemicStatus(enum.Enum):
    NotDetected = 'not_detected'
    Detected = 'detected'

class InfectionStatus(enum.Enum):
    Healthy = 'healthy'
    Contraction = 'contraction'
    Infectious = 'infectious'
    StayingHome = 'staying_home'
    SeeDoctor = 'see_doctor'  # probable, dynamic variable - visiting doctor can cause spreading the disease into vulnerable people (already sick)
    Hospital = 'hospital'  # probable
    DetectionTime = 'detection_time'  # dependent on the epidemics state
    QuarantineAtHome = 'quarantine_at_home'  # dependent on the epidemics state
    QuarantineAtHospital = 'quarantine_at_hospital'  # dependent on the epidemics state
    # SuspiciousOfCovid19 = 4
    Recovered = 'recovered'  # probable
    Death = 'death' # probable, alternative to Recovered


# https://stackoverflow.com/questions/24481852/serialising-an-enum-member-to-json
# quarantine times -

CONTRACTION_TIME = 'contraction_time'
PERSON_INDEX = 'person_index'
INFECTION_STATUS = 'infection_status'
INITIAL_CONDITIONS = 'initial_conditions'
EPIDEMIC_STATUS = 'epidemic_status'
DISEASE_PROGRESSION = 'disease_progression'
NOT_DETECTED = 'not_detected'
DETECTED = 'detected'
DISTRIBUTION = 'distribution'
STOP_SIMULATION_THRESHOLD = 'stop_simulation_threshold'
LAMBDA = 'lambda'
FILEPATH = 'filepath'
SYMPTOM_STATUS = 'symptom_status'
T0 = 't0'
T1 = 't1'
T2 = 't2'
PUBLIC_ENUMS = {
    'InfectionStatus': InfectionStatus,
    'EpidemicStatus': EpidemicStatus,
    'SymptomStatus': SymptomStatus
}

class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj) in PUBLIC_ENUMS.values():
            return {"__enum__": str(obj)}
        return json.JSONEncoder.default(self, obj)


def as_enum(d):
    if "__enum__" in d:
        name, member = d["__enum__"].split(".")
        return getattr(PUBLIC_ENUMS[name], member)
    else:
        return d

distribution_schema = Schema({DISTRIBUTION: str,
                              Optional(LAMBDA): And(Or(Use(float), Use(int)), lambda n: n > 0),
                              Optional(FILEPATH): (lambda x: os.path.exists(x.replace('$ROOT_DIR', config.ROOT_DIR)))})

disease_times_distributions_schema = Schema({T0: distribution_schema,
                                             T1: distribution_schema,
                                             T2: distribution_schema})

master_schemas = {
    INITIAL_CONDITIONS: Schema([{PERSON_INDEX: int, CONTRACTION_TIME: Or(float, int),
                                 INFECTION_STATUS: Or(InfectionStatus.Healthy, InfectionStatus.Contraction,
                                                      InfectionStatus.Infectious, InfectionStatus.StayingHome,
                                                      InfectionStatus.SeeDoctor, InfectionStatus.Hospital,
                                                      InfectionStatus.Recovered, InfectionStatus.Death),
                                 SYMPTOM_STATUS: Or(SymptomStatus.No, SymptomStatus.Mild, SymptomStatus.Serious)
                                 }]),
    STOP_SIMULATION_THRESHOLD: Schema(And(Use(int), lambda n: n > 0)),
    DISEASE_PROGRESSION: Schema({NOT_DETECTED: disease_times_distributions_schema,
                                 DETECTED: disease_times_distributions_schema}),
    EPIDEMIC_STATUS: Schema(Or(EpidemicStatus.NotDetected, EpidemicStatus.Detected))
}

active_states = [InfectionStatus.Contraction, InfectionStatus.Infectious,
                 InfectionStatus.StayingHome, InfectionStatus.SeeDoctor,
                 InfectionStatus.Hospital]
default_initial_conditions = ([{PERSON_INDEX: 0, CONTRACTION_TIME: 0, INFECTION_STATUS: InfectionStatus.Contraction}])
default_stop_simulation_threshold = 10000
default_epidemic_status = EpidemicStatus.NotDetected
default_distribution = {DISTRIBUTION: 'poisson'}
default_disease_times_distributions = {T0: default_distribution,
                                       T1: default_distribution,
                                       T2: default_distribution}
default_disease_progression = {NOT_DETECTED: default_disease_times_distributions,
                               DETECTED: default_disease_times_distributions}
defaults = {
    INITIAL_CONDITIONS: default_initial_conditions,
    EPIDEMIC_STATUS: default_epidemic_status,
    DISEASE_PROGRESSION: default_disease_progression,
    STOP_SIMULATION_THRESHOLD: default_stop_simulation_threshold
}
stop_simulation_threshold_schema = Schema(And(Use(int), lambda x: x > 0))


class EventQueue:
    pass

def _convert_infection_status(x):
    for status in InfectionStatus:
        if x == status.value:
            return status
    raise ValueError(f'invalid status provided: {x}')


class InfectionModel:
    def __init__(self, params_path: str, df_individuals_path: str) -> None:
        with open(params_path, 'r') as params_file:
            params = json.loads(params_file.read(),
                                object_hook=as_enum)  # TODO: check whether this should be moved to different place
        self._params = dict()
        for key, schema in master_schemas.items():
            self._params[key] = schema.validate(params.get(key, defaults[key]))

        self._global_time = 0.0
        self._df_individuals = pd.read_csv(df_individuals_path,
                                           converters={INFECTION_STATUS: _convert_infection_status})  # TODO: Consider networkx graph instead of pandas df
        self._df_individuals[CONTRACTION_TIME] = np.nan
        for initial_condition in self._params[INITIAL_CONDITIONS]:
            for key in [INFECTION_STATUS, CONTRACTION_TIME]:
                self._df_individuals.loc[initial_condition[PERSON_INDEX], key] = initial_condition[key]

    @property
    def global_time(self):
        return self._global_time

    @property
    def df_individuals(self):
        return self._df_individuals

    @property
    def epidemic_status(self):
        return self._params[EPIDEMIC_STATUS]

    @property
    def stop_simulation_threshold(self):
        return self._params[STOP_SIMULATION_THRESHOLD]

    @property
    def disease_progression(self):
        if self.epidemic_status == EpidemicStatus.NotDetected:
            return self._params[DISEASE_PROGRESSION][NOT_DETECTED]
        return self._params[DISEASE_PROGRESSION][DETECTED]

    def active_people(self):
        return self._df_individuals[INFECTION_STATUS].value_counts().filter(items=active_states, axis=0).sum()

    def generate_disease_progression(self):
        pass

    def run_simulation(self):
        number_active_people = self.active_people()
        # TODO
        pass