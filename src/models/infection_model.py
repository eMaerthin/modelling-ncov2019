"""
This is mostly based on references/infection_alg.pdf
"""

import collections
import enum
import json
import os
import typing

import numpy as np
import pandas as pd
from schema import (Schema, And, Use, Or)

# Todo: below should go to separate file


class InfectionStatus(enum.Enum):
    Healthy = -1
    Contraction = 0
    Infectious = 1
    StayingHome = 2
    RequiringHospital = 3
    Terminated = 999


# https://stackoverflow.com/questions/24481852/serialising-an-enum-member-to-json


CONTRACTION_TIME = 'contraction_time'
PERSON_INDEX = 'person_index'
INFECTION_STATUS = 'infection_status'
INITIAL_CONDITIONS = 'initial_conditions'
PUBLIC_ENUMS = {
    'InfectionStatus': InfectionStatus,
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


initial_conditions_schema = Schema([{PERSON_INDEX: int, CONTRACTION_TIME: Or(float, int),
                                     INFECTION_STATUS: Or(InfectionStatus.Healthy, InfectionStatus.Contraction,
                                                          InfectionStatus.Infectious, InfectionStatus.StayingHome,
                                                          InfectionStatus.RequiringHospital, InfectionStatus.Terminated)
                                     }])
active_states = [InfectionStatus.Contraction, InfectionStatus.Infectious,
                 InfectionStatus.StayingHome, InfectionStatus.RequiringHospital]
default_initial_conditions = ([{PERSON_INDEX: 0, CONTRACTION_TIME: 0, INFECTION_STATUS: InfectionStatus.Contraction}])
default_stop_simulation_threshold = 10000
stop_simulation_threshold_schema = Schema(And(Use(int), lambda x: x > 0))


class EventQueue:
    pass


class InfectionModel:
    def __init__(self, params_path: str, df_individuals_path: str) -> None:
        with open(params_path, 'r') as params_file:
            params = json.loads(params_file.read(),
                                object_hook=as_enum)  # TODO: check whether this should be moved to different place

        self._initial_conditions = initial_conditions_schema.validate(
            params.get(INITIAL_CONDITIONS, default_initial_conditions)
        )
        self._global_time = 0.0
        self._stop_simulation_threshold = params.get('stop_simulation_threshold', default_stop_simulation_threshold)
        self._df_individuals = pd.read_csv(df_individuals_path)  # TODO: Consider networkx graph instead of pandas df
        self._df_individuals[CONTRACTION_TIME] = np.nan
        for initial_condition in self._initial_conditions:
            for key in [INFECTION_STATUS, CONTRACTION_TIME]:
                self._df_individuals.loc[initial_condition[PERSON_INDEX], key] = initial_condition[key]

    @property
    def global_time(self):
        return self._global_time

    @property
    def df_individuals(self):
        return self._df_individuals

    @property
    def stop_simulation_threshold(self):
        return self._stop_simulation_threshold

    def active_people(self):
        return self._df_individuals[INFECTION_STATUS].value_counts().filter(items=active_states, axis=0).sum()

    def run_simulation(self):
        number_active_people = self.active_people()
        # TODO
        pass