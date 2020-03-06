"""
This is mostly based on references/infection_alg.pdf
"""

import collections
import enum
import heapq
import json
import logging
import os
import sys
import typing

import numpy as np
import pandas as pd
import scipy.stats
from schema import (Schema, And, Use, Or, Optional)
import config
import fire

# Todo: Part of below should go to separate files

CONTRACTION_TIME = 'contraction_time'
PERSON_INDEX = 'person_index'
INFECTION_STATUS = 'infection_status'
INITIAL_CONDITIONS = 'initial_conditions'
EPIDEMIC_STATUS = 'epidemic_status'
DISEASE_PROGRESSION = 'disease_progression'
NOT_DETECTED = 'not_detected'
DETECTED = 'detected'

DISTRIBUTION = 'distribution'
LAMBDA = 'lambda'
FILEPATH = 'filepath'

EXPECTED_CASE_SEVERITY = 'expected_case_severity'
APPROXIMATE_DISTRIBUTION = 'approximate_distribution'
CASE_SEVERITY_DISTRIBUTION = 'case_severity_distribution'

START_TIME = 'start_time'
STOP_SIMULATION_THRESHOLD = 'stop_simulation_threshold'

ID = 'id'
HOUSEHOLD_ID = 'household_id'
SOURCE = 'source_id'
TARGET = 'target_id'
KERNEL = 'kernel'
INHABITANTS = 'inhabitants'

TMINUS1 = 'tminus1'
T0 = 't0' # contraction->infectious ACTIVE
T1 = 't1' # infectious->stay_home ACTIVE
T2 = 't2' # stay_home->hospital SEVERE/CRITICAL TODO it WAS before : stay_home->see_doctor MILD only
#T3 = 't3' # TODO let's check if we need this: # stay_home/see_doctor->hospital SEVERE/CRITICAL

ASYMPTOMATIC = 'asymptomatic'
MILD = 'mild'
SEVERE = 'severe'
CRITICAL = 'critical'

LOGNORMAL = 'lognormal'
EXPONENTIAL = 'exponential'
POISSON = 'poisson'
FROM_FILE = 'from_file'

DETECTION = 'detection'
QUARANTINE_AT_HOME = 'quarantine_at_home'
QUARANTINE_AT_HOSPITAL = 'quarantine_at_hospital'

SELECTION_ALGORITHM = 'selection_algorithm'
CARDINALITIES = 'cardinalities'

EMPLOYMENT_STATUS = 'employment_status'
PROFESSION_INDEX = 'profession_index'
SOCIAL = 'social'
P_TRANSPORT = 'p_transport'

class EnumWithPublicValue2MemberMap(enum.Enum):
    @classmethod
    def map(cls):
        return cls._value2member_map_


class DiseaseProgressionEvents(EnumWithPublicValue2MemberMap):
    tminus1 = TMINUS1
    t0 = T0
    t1 = T1
    t2 = T2
    #t3 = T3


class ExpectedCaseSeverity(EnumWithPublicValue2MemberMap):  # to be added to nodes?
    Asymptomatic = ASYMPTOMATIC  # 0.6%
    Mild = MILD  # 80.9% Maybe Staying Home, maybe visit Doctor
    Severe = SEVERE  # 13.8%
    Critical = CRITICAL  # 4.7% Need to go to Hospital!


class EpidemicStatus(EnumWithPublicValue2MemberMap):
    NotDetected = NOT_DETECTED
    Detected = DETECTED


class InfectionStatus(EnumWithPublicValue2MemberMap):
    Healthy = 'healthy'
    Contraction = 'contraction'
    Infectious = 'infectious'
    StayHome = 'stay_home'
    #SeeDoctor = 'see_doctor'  # probable, dynamic variable - visiting doctor can cause spreading the disease into vulnerable people (already sick)
    Hospital = 'hospital'  # probable
    Recovered = 'recovered'  # probable
    Death = 'death'  # probable, alternative to Recovered


class StateDependentOnTheEpidemicsState(EnumWithPublicValue2MemberMap):
    Detection = DETECTION  # dependent on the epidemics state
    QuarantineAtHome = QUARANTINE_AT_HOME  # dependent on the epidemics state
    QuarantineAtHospital = QUARANTINE_AT_HOSPITAL  # dependent on the epidemics state


class SelectionAlgorithms(EnumWithPublicValue2MemberMap):
    """
    This enum is intended for storing all initial condition selection algorithms,
    to seed initially infectious agents in the population
    RandomSelection assignes infection status randomly based on provided distribution of sick people
    In the future other algorithms could be added e.g. to limit assigning infection status to people who work or commute
    """
    RandomSelection = 'random_selection'


default_initial_conditions = ([{
    PERSON_INDEX: 0,
    CONTRACTION_TIME: 0,
    INFECTION_STATUS: InfectionStatus.Contraction
}])

default_stop_simulation_threshold = 10000

default_epidemic_status = EpidemicStatus.NotDetected.value

default_distribution = {
    DISTRIBUTION: 'poisson'
}

default_case_severity_distribution = {
    ASYMPTOMATIC: 0.006,
    MILD: 0.809,
    SEVERE: 0.138,
    CRITICAL: 0.047
}

default_disease_times_distributions = {
    T0: default_distribution,
    T1: default_distribution,
    T2: default_distribution
}

default_disease_progression = {
    NOT_DETECTED: default_disease_times_distributions,
    DETECTED: default_disease_times_distributions
}

defaults = {
    INITIAL_CONDITIONS: default_initial_conditions,
    EPIDEMIC_STATUS: default_epidemic_status,
    STOP_SIMULATION_THRESHOLD: default_stop_simulation_threshold,
    DISEASE_PROGRESSION: default_disease_progression,
    CASE_SEVERITY_DISTRIBUTION: default_case_severity_distribution
}

# TODO ensure those distributions are really supported
supported_distributions = [
    LOGNORMAL, EXPONENTIAL, POISSON, FROM_FILE
]

distribution_schema = Schema({
    DISTRIBUTION: Or(*supported_distributions),
    Optional(LAMBDA): And(Or(Use(float), Use(int)), lambda n: n > 0),
    Optional(FILEPATH): (lambda x: os.path.exists(x.replace('$ROOT_DIR', config.ROOT_DIR))),
    Optional(APPROXIMATE_DISTRIBUTION, default=None): Or(None, *supported_distributions)
})

disease_times_distributions_schema = Schema({
    T0: distribution_schema,
    T1: distribution_schema,
    T2: distribution_schema
})

case_severity_distribution_schema = Schema(And({
    ASYMPTOMATIC: And(Use(float), lambda x: x >= 0),
    MILD: And(Use(float), lambda x: x >= 0),
    SEVERE: And(Use(float), lambda x: x >= 0),
    CRITICAL: And(Use(float), lambda x: x >= 0),
}, lambda x: sum(x.values()) == 1))

initial_conditions_schema1 = [{
        PERSON_INDEX: int,
        CONTRACTION_TIME: Or(float, int),
        INFECTION_STATUS: Or(*InfectionStatus.map()),
        Optional(EXPECTED_CASE_SEVERITY): Or(*ExpectedCaseSeverity.map())
    }]

initial_conditions_schema2 = {
        SELECTION_ALGORITHM: Or(*SelectionAlgorithms.map()),
        CARDINALITIES: {Optional(k, default=0): And(Use(int), lambda n: n >= 0) for k in InfectionStatus.map()}
    }

infection_model_schemas = {
    INITIAL_CONDITIONS: Schema(Or(initial_conditions_schema1, initial_conditions_schema2)),
    STOP_SIMULATION_THRESHOLD: Schema(And(Use(int), lambda n: n > 0)),
    DISEASE_PROGRESSION: Schema({
        NOT_DETECTED: disease_times_distributions_schema,
        DETECTED: disease_times_distributions_schema
    }),
    EPIDEMIC_STATUS: Schema(Or(*EpidemicStatus.map())),
    CASE_SEVERITY_DISTRIBUTION: Schema(case_severity_distribution_schema)
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

TIME = 'time'
TYPE = 'type'
INITIATED_BY = 'initiated_by'
INITIATED_THROUGH = 'initiated_through'
ISSUED_TIME = 'issued_time'
Event = collections.namedtuple('Event', [TIME, PERSON_INDEX, TYPE, INITIATED_BY,
                                         INITIATED_THROUGH, ISSUED_TIME, EPIDEMIC_STATUS])


def _convert_enum(enum_class, x):
    for status in enum_class:
        if x == status.value:
            return status
    raise ValueError(f'invalid status provided: {x}')


def _convert_infection_status(x):
    return _convert_enum(InfectionStatus, x)


def _convert_expected_case_severity(x):
    return _convert_enum(ExpectedCaseSeverity, x)


class InfectionModel:

    def __init__(self, params_path: str, df_individuals_path: str) -> None:
        with open(params_path, 'r') as params_file:
            params = json.loads(
                params_file.read()
            )  # TODO: check whether this should be moved to different place
        self._params = dict()
        for key, schema in infection_model_schemas.items():
            self._params[key] = schema.validate(params.get(key, defaults[key]))

        self._global_time = Schema(And(Or(int, float),
                                       lambda x: x >= 0)).validate(
            params.get(START_TIME, 0.0)
        )
        self._epidemic_status = Schema(Or(*EpidemicStatus.map())).validate(
            params.get(EPIDEMIC_STATUS, NOT_DETECTED)
        )

        self._df_individuals, self._df_progression_times, self._df_potential_contractions, self._df_households = \
            self.set_up_data_frames(df_individuals_path)

        self.event_schema = Schema({
            TIME: And(Or(float,
                         int), lambda x: x > 0),
            PERSON_INDEX: And(int, lambda x: x in self._df_individuals.id.values),
            TYPE: Or(*DiseaseProgressionEvents.map(),
                     *StateDependentOnTheEpidemicsState.map()),
            INITIATED_BY: Or(None,
                             And(int, lambda x: x in self._df_individuals.id.values)),
            ISSUED_TIME: Or(None,
                            And(Or(float, int), lambda x: x >= 0)),
            EPIDEMIC_STATUS: Or(*EpidemicStatus.map()),
        })
        self.event_queue = []
        self.parse_initial_conditions()

    def set_up_data_frames(self, df_individuals_path: str) -> typing.Tuple[pd.DataFrame, pd.DataFrame,
                                                                           pd.DataFrame, pd.DataFrame]:
        df_individuals = pd.read_csv(
            df_individuals_path,
            converters={
                INFECTION_STATUS: _convert_infection_status,
                EXPECTED_CASE_SEVERITY: _convert_expected_case_severity
            })  # TODO: Consider networkx graph instead of pandas df

        if EXPECTED_CASE_SEVERITY not in df_individuals:
            df_individuals[EXPECTED_CASE_SEVERITY] = df_individuals.apply(
                lambda x: self.draw_expected_case_severity(x),
                axis=1
            )

        df_progression_times = pd.DataFrame(columns=[
            ID,
            *DiseaseProgressionEvents.map(),
            *StateDependentOnTheEpidemicsState.map()
        ])

        df_potential_contractions = pd.DataFrame(columns=[SOURCE, TARGET, CONTRACTION_TIME, KERNEL])
        df_households = pd.DataFrame(df_individuals.groupby(HOUSEHOLD_ID)[ID].count(),
                                     index=df_individuals[HOUSEHOLD_ID].drop_duplicates().values
                                     ).rename(columns={ID: INHABITANTS})
        return df_individuals, df_progression_times, df_potential_contractions, df_households

    def parse_initial_conditions(self):
        initial_conditions = self._params[INITIAL_CONDITIONS]
        if isinstance(initial_conditions, list): # schema v1
            for initial_condition in initial_conditions:
                for key, value in initial_condition.items():
                    if key == PERSON_INDEX:
                        continue
                    if key == CONTRACTION_TIME:
                        self.apply_contraction_event(initial_condition[PERSON_INDEX], value)
                        continue
                    if key == INFECTION_STATUS:
                        value = _convert_infection_status(value)
                    elif key == EXPECTED_CASE_SEVERITY:
                        value = _convert_expected_case_severity(value)
                    self._df_individuals.loc[initial_condition[PERSON_INDEX], key] = value
        elif isinstance(initial_conditions, dict): #schema v2
            if initial_conditions[SELECTION_ALGORITHM] == SelectionAlgorithms.RandomSelection.value:
                # initially all indices can be drawn
                choice_set = np.arange(len(self._df_individuals))
                for infection_status, cardinality in initial_conditions[CARDINALITIES].items():
                    if cardinality > 0:
                        selected_rows = np.random.choice(choice_set, cardinality, replace=False)
                        for row in selected_rows:
                            self._df_individuals.loc[row, INFECTION_STATUS] = _convert_infection_status(infection_status)
                        # now only previously unselected indices can be drawn in next steps
                        choice_set = np.array(list(set(choice_set) - set(selected_rows)))
        else:
            raise ValueError('invalid schema')

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
    def case_severity_distribution(self):
        return self._params[CASE_SEVERITY_DISTRIBUTION]

    @property
    def disease_progression(self):
        return self._params[DISEASE_PROGRESSION][self.epidemic_status]

    def active_people(self):
        return self._df_individuals[INFECTION_STATUS].value_counts().filter(
            items=active_states,
            axis=0
        ).sum()

    def draw_expected_case_severity(self, features):
        """ Returns case severity based on individual features
        Currently we do not have any data to support thesis that case severity depends on
        age/comorbid condition/other factors so we use case severity distribution
        - see http://weekly.chinacdc.cn/en/article/id/e53946e2-c6c4-41e9-9a9b-fea8db1a8f51
        """
        case_severity_dict = self.case_severity_distribution
        distribution_hist = np.array(list(case_severity_dict.values()))
        dis = scipy.stats.rv_discrete(values=(
            np.arange(len(distribution_hist)),
            distribution_hist
        ))
        selected_key = list(case_severity_dict.keys())[dis.rvs()]
        return _convert_expected_case_severity(selected_key)

    @staticmethod
    def generate_random_sample(**kwargs) -> float:
        distribution = kwargs.get('distribution', 'poisson')
        if distribution == 'from_file':
            filepath = kwargs.get('filepath', None).replace('$ROOT_DIR', config.ROOT_DIR)
            Schema(lambda x: os.path.exists(x)).validate(filepath)
            array = np.load(filepath)
            approximate_distribution = kwargs.get('approximate_distribution', None)
            if approximate_distribution == 'lognormal':
                shape, loc, scale = scipy.stats.lognorm.fit(array, loc=0)
                return scipy.stats.lognorm.rvs(shape, loc=loc, scale=scale)
            if approximate_distribution:
                raise ValueError(f'Approximating to this distribution {approximate_distribution}'
                                 f'is not yet supported but we can quickly add it')

        if distribution == 'lognormal':
            mean = kwargs.get('mean', 0.0)
            sigma = kwargs.get('sigma', 1.0)
            return np.random.lognormal(mean=mean, sigma=sigma)

        lambda_ = kwargs.get('lambda', 1.0)
        if distribution == 'exponential':
            return np.random.exponential(scale=1/lambda_)
        if distribution == 'poisson':
            return np.random.poisson(lam=lambda_)
        raise ValueError(f'Sampling from distribution {distribution} is not yet supported but we can quickly add it')

    def generate_disease_progression(self, person_id, features, contraction_time: float) -> None:
        """Returns list of disease progression events
        "future" disease_progression should be recalculated when the disease will be recognised at the state level
        t0 - time when individual becomes infectious (Mild symptoms)
        t1 - time when individual stay home due to Mild/Serious? symptoms
        t2 - time when individual goes to hospital due to Serious symptoms
        """
        t0 = contraction_time + self.generate_random_sample(**self.disease_progression[T0])
        self.append_event(Event(t0, person_id, T0, person_id, DISEASE_PROGRESSION, contraction_time, self.epidemic_status))
        t1 = t0 + self.generate_random_sample(**self.disease_progression[T1])
        self.append_event(Event(t1, person_id, T1, person_id, DISEASE_PROGRESSION, t0, self.epidemic_status))
        t2 = None
        # t3 = None
        if self._df_individuals.loc[person_id, EXPECTED_CASE_SEVERITY] in [
            ExpectedCaseSeverity.Severe,
            ExpectedCaseSeverity.Critical
        ]:
            t2 = t1 + self.generate_random_sample(**self.disease_progression[T2])
            self.append_event(Event(t2, person_id, T2, person_id, DISEASE_PROGRESSION, t1, self.epidemic_status))

            ''' TODO let's investigate if this is useful. For now, let's assume t2 is going to hospital
            t3 = t2 + self.generate_random_sample(**self.disease_progression[T3])
            self.append_event(Event(t3, person_id, T3, person_id, t2, self.epidemic_status))
            '''
        self._df_progression_times = self._df_progression_times.append({
            ID: person_id,
            TMINUS1: contraction_time,
            T0: t0,
            T1: t1,
            T2: t2,
            #T3: t3
        }, ignore_index=True)
        ''' TODO let's investigate how to add this:
        quarantine_at_home = None
        quarantine_at_hospital = None
        detection = None
        (...)
        ,
            QUARANTINE_AT_HOME: quarantine_at_home,  # this will be not None when the epidemics is already discovered
            QUARANTINE_AT_HOSPITAL: quarantine_at_hospital,  # TODO how to find this
            DETECTION: detection,  # TODO how to find this
        }, ignore_index=True)
        '''

    def run_simulation(self):
        while self.pop_and_apply_event():
            number_active_people = self.active_people()
            if number_active_people >= self.stop_simulation_threshold:
                logging.info(f"The outbreak reached a high number {self.stop_simulation_threshold}")
                break
        print(self._df_progression_times.head())

    def append_event(self, event: Event) -> None:
        heapq.heappush(self.event_queue, event)

    def apply_contraction_event(self, person_id, contraction_time):
        self._df_individuals.loc[person_id, INFECTION_STATUS] = InfectionStatus.Contraction
        self._global_time = contraction_time
        self.generate_disease_progression(person_id,
                                          self._df_individuals.loc[person_id],
                                          contraction_time)

    def add_potential_contractions_from_transport_kernel(self):
        pass

    def add_potential_contractions_from_household_kernel(self):
        pass

    def add_potential_contractions_from_employment_kernel(self):
        pass

    def add_potential_contractions_from_sporadic_kernel(self):
        pass

    def add_potential_contractions_from_friendship_kernel(self):
        pass

    # 'Event', [TIME, PERSON_INDEX, TYPE, INITIATED_BY, INITIATED_THROUGH, ISSUED_TIME, EPIDEMIC_STATUS])
    def pop_and_apply_event(self) -> bool:
        try:
            event = heapq.heappop(self.event_queue)
            type_ = getattr(event, TYPE)
            time = getattr(event, TIME)
            self._global_time = time

            id = getattr(event, PERSON_INDEX)
            initiated_by = getattr(event, INITIATED_BY)
            initiated_through = getattr(event, INITIATED_THROUGH)

            # TODO the remaining two attributes will be useful when we will take into account
            #  change of the social network state to EPIDEMIC mode
            issued_time = getattr(event, ISSUED_TIME)
            epidemic_status = getattr(event, EPIDEMIC_STATUS)

            if type_ == TMINUS1:
                # check if this action is still valid first
                initiated_inf_status = self._df_individuals.loc[initiated_by, INFECTION_STATUS]
                if initiated_inf_status in active_states:
                    if initiated_through != 'household':
                        if initiated_inf_status != InfectionStatus.StayHome:
                            self.apply_contraction_event(id, time)
                    else:
                        self.apply_contraction_event(id, time)
            elif type_ == T0:
                if self._df_individuals.loc[id, INFECTION_STATUS] == InfectionStatus.Contraction:
                    self._df_individuals.loc[id, INFECTION_STATUS] = InfectionStatus.Infectious
                else:
                    logger.error('state machine failure')
                if self._df_individuals.loc[id, P_TRANSPORT] > 0:
                    self.add_potential_contractions_from_transport_kernel(id)
                if self._df_individuals.loc[id, EMPLOYMENT_STATUS] > 0:
                    self.add_potential_contractions_from_employment_kernel(id)
                if self._df_households.loc[self._df_individuals.loc[id, HOUSEHOLD_ID], INHABITANTS] > 1:
                    self.add_potential_contractions_from_household_kernel(id)
                self.add_potential_contractions_from_friendship_kernel(id)
                self.add_potential_contractions_from_sporadic_kernel(id)
            elif type_ == T1:
                if self._df_individuals.loc[id, INFECTION_STATUS] == InfectionStatus.Infectious:
                    self._df_individuals.loc[id, INFECTION_STATUS] = InfectionStatus.StayHome
            elif type_ == T2:
                if self._df_individuals.loc[id, INFECTION_STATUS] == InfectionStatus.StayHome:
                    self._df_individuals.loc[id, INFECTION_STATUS] = InfectionStatus.Hospital

            # TODO: add more important logic
            return True
        except IndexError:
            return False

    def recalculate_event_queue(self):
        # TODO -> this will be important with state actions implemented like quarantine at home
        pass


logger = logging.getLogger(__name__)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    pd.set_option('display.max_columns', None)
    fire.Fire(InfectionModel)