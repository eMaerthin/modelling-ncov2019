"""
This is mostly based on references/infection_alg.pdf
"""

import collections
import enum
import json
import os

import numpy as np
import pandas as pd
from schema import (Schema, And, Use)

# Todo: below should go to separate file


class DiseaseState(enum.Enum):
    Healthy = -1
    Contraction = 0
    Infectious = 1
    StayingHome = 2
    RequiringHospital = 3
    Terminated = 999

# https://stackoverflow.com/questions/24481852/serialising-an-enum-member-to-json

PUBLIC_ENUMS = {
    'DiseaseState': DiseaseState,
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

CONTRACTION_TIME = 'contraction_time'
PERSON_INDEX = 'person_index'
DISEASE_STATE = 'disease_state'
INITIAL_CONDITIONS = 'initial_conditions'

initial_conditions_schema = Schema([{PERSON_INDEX: int, CONTRACTION_TIME: float,
                                     DISEASE_STATE: [DiseaseState.Healthy, DiseaseState.Contraction,
                                                     DiseaseState.Infectious, DiseaseState.StayingHome,
                                                     DiseaseState.RequiringHospital, DiseaseState.Terminated]
                                     }])
active_states = [DiseaseState.Contraction, DiseaseState.Infectious,
                 DiseaseState.StayingHome, DiseaseState.RequiringHospital]
default_initial_conditions = ([{PERSON_INDEX: 0, CONTRACTION_TIME: 0, DISEASE_STATE: DiseaseState.Contraction}])
default_stop_simulation_threshold = 10000
stop_simulation_threshold_schema = Schema(And(Use(int), lambda x: x > 0))


class InfectionModel:
    def __init__(self, params_path, df_individuals_path):
        params = json.loads(params_path)  # TODO: check whether this should be moved to different place
        self._initial_conditions = initial_conditions_schema.validate(
            params.get(INITIAL_CONDITIONS, default_initial_conditions)
        )
        self._global_time = 0.0
        self._stop_simulation_threshold = params.get('stop_simulation_threshold', default_stop_simulation_threshold)
        self._df_individuals = pd.read_csv(df_individuals_path, headers=True)
        self._df_individuals[DISEASE_STATE] = DiseaseState.Healthy
        self._df_individuals[CONTRACTION_TIME] = np.nan
        for initial_condition in self._initial_conditions:
            self._df_individuals.loc[initial_condition.person_index,
                                     DISEASE_STATE] = initial_condition.disease_state
            self._df_individuals.loc[initial_condition.person_index,
                                     CONTRACTION_TIME] = initial_condition.contraction_time

    def run_simulation(self):
        number_active_people = self._df_individuals[DISEASE_STATE].value_counts().filter(items=active_states, axis=0).sum()
        # TODO
        pass