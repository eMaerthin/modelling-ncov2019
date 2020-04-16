"""
This is mostly based on references/infection_alg.pdf
"""
import ast
from functools import (lru_cache, partial)
import json
import logging
import mocos_helper
#import random
import time
from collections import defaultdict
import pickle
import psutil
from shutil import copyfile
from math import log

from git import Repo
import pandas as pd
import scipy.optimize
import scipy.stats

from src.models.schemas import *
from src.models.defaults import *
from src.models.states_and_functions import *
from src.visualization.visualize import Visualize


import click

from dotenv import find_dotenv, load_dotenv

from queue import (PriorityQueue)
q = PriorityQueue()


class InfectionModel:
    def __init__(self, params_path: str, df_individuals_path: str, df_households_path: str = '') -> None:
        self.params_path = params_path
        self.df_individuals_path = df_individuals_path
        self.df_households_path = df_households_path
        logger.info('Loading params...')
        self._params = dict()
        with open(params_path, 'r') as params_file:
            params = json.loads(
                params_file.read()
            )  # TODO: check whether this should be moved to different place
        logger.info('Parsing params...')
        for key, schema in infection_model_schemas.items():
            self._params[key] = schema.validate(params.get(key, defaults[key]))
        default_household_input_path = os.path.join(self._params[OUTPUT_ROOT_DIR], self._params[EXPERIMENT_ID],
                                                    'input_df_households.csv')  # TODO: ensure households are valid!
        if df_households_path == '':
            self.df_households_path = default_household_input_path
        self._global_time = None
        self._max_time = None
        self._vis = None
        self._max_time_offset = 0.0
        self._expected_case_severity = None
        self._df_individuals = None
        self._df_households = None
        #self._individuals_gender = None
        self._individuals_age = None
        self._individuals_household_id = None
        self._individuals_indices = None
        self._households_capacities = None
        self._households_inhabitants = None
        self._init_for_stats = None

        self._affected_people = 0
        self._active_people = 0
        self._quarantined_people = 0
        self._detected_people = 0
        self._immune_people = 0
        self._deaths = 0
        self._icu_needed = 0

        self._disable_friendship_kernel = False
        self._set_up_data_frames()
        self._infection_status = None
        self._detection_status = None
        self._quarantine_status = None
        self._expected_case_severity = None
        if self._params[REUSE_EXPECTED_CASE_SEVERITIES]:
            self._expected_case_severity = self.draw_expected_case_severity()
        self._infections_dict = None
        self._progression_times_dict = None

        t0_f, t0_args, t0_kwargs = self.setup_random_distribution(T0)
        self.rv_t0 = lambda: t0_f(*t0_args, **t0_kwargs)

        t1_f, t1_args, t1_kwargs = self.setup_random_distribution(T1)
        self.rv_t1 = lambda: t1_f(*t1_args, **t1_kwargs)

        t2_f, t2_args, t2_kwargs = self.setup_random_distribution(T2)
        self.rv_t2 = lambda: t2_f(*t2_args, **t2_kwargs)

        tdeath_f, tdeath_args, tdeath_kwargs = self.setup_random_distribution(TDEATH)
        self.rv_tdeath = lambda: tdeath_f(*tdeath_args, **tdeath_kwargs)

        # TODO: This should be refactored
        self.fear_fun = dict()
        self.fear_weights_detected = dict()
        self.fear_weights_deaths = dict()
        self.fear_scale = dict()
        self.fear_loc = dict()
        self.fear_limit_value = dict()

        self.serial_intervals = []

        self.band_time = None
        self._last_affected = None
        self._per_day_increases = {}

        self._disable_constant_age_kernel = False
        self._constant_age_helper_age_dict = {}
        self._constant_age_individuals = defaultdict(list)
        self._setup_constant_age_kernel()

    def _setup_constant_age_kernel(self):
        if self._params[CONSTANT_AGE_SETUP] is None:
            self._disable_constant_age_kernel = True
            return

        if isinstance(self._params[CONSTANT_AGE_SETUP][AGE], int):
            self._constant_age_helper_age_dict[self._params[CONSTANT_AGE_SETUP][AGE]] = 0
        else:
            if self._params[CONSTANT_AGE_SETUP][INTER_AGE_CONTACTS]:
                # so all ages specified can be mixed
                for age in self._params[CONSTANT_AGE_SETUP][AGE]:
                    self._constant_age_helper_age_dict[age] = 0
            else:
                for i, age in enumerate(self._params[CONSTANT_AGE_SETUP][AGE]):
                    self._constant_age_helper_age_dict[age] = i
        for age, individual_list_key in self._constant_age_helper_age_dict.items():
            self._constant_age_individuals[individual_list_key].extend([
                k for k, v in self._individuals_age_dct.items() if v==age
            ])

    def get_detection_status_(self, person_id):
        return self._detection_status.get(person_id, default_detection_status)

    def get_quarantine_status_(self, person_id):
        return self._quarantine_status.get(person_id, default_quarantine_status)

    def get_infection_status(self, person_id):
        return self._infection_status.get(person_id, InfectionStatus.Healthy.value)

    @staticmethod
    def parse_random_seed(random_seed):
        mocos_helper.seed(random_seed)

    def _set_up_data_frames(self) -> None:
        """
        The purpose of this method is to set up two dataframes.
        One is self._df_individuals that stores features for the population
        Second is self._df_households that stores list of people idx per household
        building df_households is time consuming, therefore we try to reuse previously computed df_households
        :return:
        """
        logger.info('Set up data frames: Reading population csv...')
        self._df_individuals = pd.read_csv(self.df_individuals_path)
        self._df_individuals.index = self._df_individuals.idx
        self._individuals_age = self._df_individuals[AGE].values
        self._individuals_age_dct = self._df_individuals[AGE].to_dict()
        self._individuals_gender_dct = self._df_individuals[GENDER].to_dict()
        self._individuals_household_id = self._df_individuals[HOUSEHOLD_ID].to_dict()
        self._individuals_indices = self._df_individuals.index.values
        if SOCIAL_COMPETENCE in self._df_individuals.columns:
            logger.info('Set up data frames: Social competence and loading social activity sampler...')
            self._social_activity_scores = self._df_individuals[SOCIAL_COMPETENCE].to_dict()

            self._social_activity_sampler = mocos_helper.AgeDependentFriendSampler(
                self._individuals_indices,
                self._individuals_age,
                self._df_individuals[GENDER].values,
                self._df_individuals[SOCIAL_COMPETENCE].values
            )
            self._disable_friendship_kernel = False
        else:
            logger.info('Social competence missing - Disable friendship kernel...')
            self._disable_friendship_kernel = True
        logger.info('Set up data frames: Building households df...')

        if os.path.exists(self.df_households_path):
            self._df_households = pd.read_csv(self.df_households_path, index_col=HOUSEHOLD_ID,
                                              converters={ID: ast.literal_eval})
        else:
            self._df_households = pd.DataFrame({ID: self._df_individuals.groupby(HOUSEHOLD_ID)[ID].apply(list)})
            os.makedirs(os.path.dirname(self.df_households_path), exist_ok=True)
            self._df_households.to_csv(self.df_households_path)
        self._df_households[CAPACITY] = self._df_households[ID].apply(lambda x: len(x))
        d = self._df_households.to_dict()
        self._households_inhabitants = d[ID] #self._df_households[ID]
        self._households_capacities = d[CAPACITY] #self._df_households[CAPACITY]
        if not self._params[LOG_OUTPUTS]:
            self._df_households = None
            self._df_individuals = None

    @staticmethod
    def append_event(event: Event) -> None:
        q.put(event)

    def _fill_queue_based_on_auxiliary_functions(self) -> None:
        # TODO: THIS IS NOT WORKING WHEN CAP = INF, let's fix it
        #  (one possible way to fix it: generate say first N events and a followup "filling EVENT"
        #  on time T(N) of N-th event - at T(N) generate N more events and enqueue next portion.
        #  Alternatively add just one event of type AUXILIARY_FUNCTION/IMPORT_INTENSITY
        #  that will then draw time of next event of that type
        """
        The purpose of this method is to mark some people of the population as sick according to provided function.
        Possible functions: see possible values of ImportIntensityFunctions enum
        Outcome of the function can be adjusted by overriding default parameters:
        multiplier, rate, cap, infectious_probability.
        :return:
        """
        def _generate_event_times(func, rate, multiplier, cap, root_buffer=100, root_guess=0) -> list:
            """
            Here a naive way of generating event times is proposed.
            The idea is to generate N events
            :param func: currently two functions are supported: exponential a*exp(r*t) and polynomial a*r^t
            :param rate: a parameter that is making the slope more steep
            :param multiplier: a parameter that scales the time down
            :param cap: the maximum amount of cases generated and added to queue list
            :param root_buffer: one-directional range to find roots in
            :param root_guess: guess on first solution (i=1)
            :return:
            """
            root_min = root_guess - root_buffer
            root_max = root_guess + root_buffer
            time_events_ = []

            def bisect_fun(x, integer):
                return func(x, rate=rate, multiplier=multiplier) - integer

            for i in range(1, 1 + cap):
                bisect_fun = partial(bisect_fun, integer=i)
                root = scipy.optimize.bisect(bisect_fun, root_min, root_max)
                time_events_.append(root)
                root_min = root
                root_max = root + root_buffer
            return time_events_

        import_intensity = self._params[IMPORT_INTENSITY]
        f_choice = ImportIntensityFunctions(import_intensity[FUNCTION])
        if f_choice == ImportIntensityFunctions.NoImport:
            return
        func = import_intensity_functions[f_choice]
        multiplier = import_intensity[MULTIPLIER]
        rate = import_intensity[RATE]
        cap = import_intensity[CAP]
        infectious_prob = import_intensity[INFECTIOUS]
        event_times = _generate_event_times(func=func, rate=rate, multiplier=multiplier, cap=cap)
        for event_time in event_times:
            person_id = self._individuals_indices[mocos_helper.randint(0, len(self._individuals_indices))]
            t_state = TMINUS1
            if mocos_helper.rand() < infectious_prob:
                t_state = T0
            self.append_event(Event(event_time, person_id, t_state, None, IMPORT_INTENSITY, self.global_time))

    def _fill_queue_based_on_initial_conditions(self):
        """
        The purpose of this method is to mark some people of the population as sick according to provided
        initial conditions.
        Conditions can be provided using one of two supported schemas.
        schema v1 is list with details per person, while schema v2 is dictionary specifying selection algorithm
        and cardinalities of each group of patients (per symptom).
        :return:
        """
        def _assign_t_state(status):
            if status == CONTRACTION:
                return TMINUS1
            if status == INFECTIOUS:
                return T0
            if status == IMMUNE:
                return TRECOVERY
            raise ValueError(f'invalid initial infection status {status}')

        initial_conditions = self._params[INITIAL_CONDITIONS]
        if isinstance(initial_conditions, list):  # schema v1
            for initial_condition in initial_conditions:
                person_idx = initial_condition[PERSON_INDEX]
                t_state = _assign_t_state(initial_condition[INFECTION_STATUS])
                if EXPECTED_CASE_SEVERITY in initial_condition:
                    self._expected_case_severity[person_idx] = initial_condition[EXPECTED_CASE_SEVERITY]
                self.append_event(Event(initial_condition[CONTRACTION_TIME], person_idx, t_state, None,
                                        INITIAL_CONDITIONS, self.global_time))
        elif isinstance(initial_conditions, dict):  # schema v2
            if initial_conditions[SELECTION_ALGORITHM] == InitialConditionSelectionAlgorithms.RandomSelection.value:
                # initially all indices can be drawn
                #choice_set = self._individuals_indices# self._df_individuals.index.values
                choice_set = list(self._individuals_indices)
                for infection_status, cardinality in initial_conditions[CARDINALITIES].items():
                    if cardinality > 0:
                        if cardinality < 1:
                            c = cardinality
                            cardinality = int(cardinality * len(choice_set))
                            if cardinality == 0:
                                logger.info(f"too small cardinality provided {cardinality} ({c})")
                                continue
                        else:
                            cardinality = int(cardinality)
                        #selected_rows = np.random.choice(choice_set, cardinality, replace=False)
                        # now only previously unselected indices can be drawn in next steps
                        #choice_set = np.array(list(set(choice_set) - set(selected_rows)))
                        choice_set, selected_rows = mocos_helper.randomly_split_list(choice_set, howmuch=cardinality)
                        t_state = _assign_t_state(infection_status)
                        for row in selected_rows:
                            self.append_event(Event(self.global_time, row, t_state, row, INITIAL_CONDITIONS,
                                                    self.global_time))
            else:
                err_msg = f'Unsupported selection algorithm provided {initial_conditions[SELECTION_ALGORITHM]}'
                logger.error(err_msg)
                raise ValueError(err_msg)
        else:
            err_msg = f'invalid schema provided {initial_conditions}'
            logger.error(err_msg)
            raise ValueError(err_msg)

    @property
    def global_time(self):
        return self._global_time

    @property
    def df_individuals(self):
        return self._df_individuals

    @property
    def stop_simulation_threshold(self):
        return self._params[STOP_SIMULATION_THRESHOLD]

    @property
    def case_severity_distribution(self):
        return self._params[CASE_SEVERITY_DISTRIBUTION]

    @property
    def disease_progression(self):
        return self._params[DISEASE_PROGRESSION][DEFAULT]

    @property
    def affected_people(self):
        return self._affected_people

    @property
    def detected_people(self):
        return self._detected_people

    @property
    def quarantined_people(self):
        return self._quarantined_people

    @property
    def active_people(self):
        return self._active_people

    @property
    def deaths(self):
        return self._deaths

    def draw_expected_case_severity(self):
        case_severity_dict = self.case_severity_distribution
        keys = list(case_severity_dict.keys())
        d = {}
        for age_min, age_max, fatality_prob in default_age_induced_fatality_rates:
            cond_lb = self._individuals_age >= age_min
            cond_ub = self._individuals_age < age_max
            cond = np.logical_and(cond_lb, cond_ub)
            if np.count_nonzero(cond) == 0:
                continue
            age_induced_severity_distribution = dict()
            age_induced_severity_distribution[CRITICAL] = fatality_prob/self._params[DEATH_PROBABILITY][CRITICAL]
            for x in case_severity_dict:
                if x != CRITICAL:
                    age_induced_severity_distribution[x] = case_severity_dict[x] / (1 - case_severity_dict[CRITICAL]) * (1 - age_induced_severity_distribution[CRITICAL])
            realizations = mocos_helper.sample_with_replacement_shuffled((age_induced_severity_distribution[x] for x in case_severity_dict), len(self._individuals_indices[cond]))
            values = [keys[r] for r in realizations]
            df = pd.DataFrame(values, index=self._individuals_indices[cond])
            d = {**d, **df.to_dict()[0]}
        return d

    def setup_random_distribution(self, t):
        params = self.disease_progression[t]
        distribution = params.get(DISTRIBUTION, default_distribution[DISTRIBUTION])
        if distribution == FROM_FILE:
            filepath = params.get('filepath', None).replace('$ROOT_DIR', config.ROOT_DIR)
            Schema(lambda x: os.path.exists(x)).validate(filepath)
            array = np.load(filepath)
            approximate_distribution = params.get('approximate_distribution', None)
            if approximate_distribution == LOGNORMAL:
                shape, loc, scale = scipy.stats.lognorm.fit(array, floc=0)
                return mocos_helper.lognormal, [], {'mean': log(scale), 'sigma': shape}

            if approximate_distribution == GAMMA:
                shape, loc, scale = scipy.stats.gamma.fit(array, floc=0)
                return mocos_helper.gamma, [], {'alpha': shape, 'beta': scale}

            if approximate_distribution:
                raise NotImplementedError(f'Approximating to this distribution {approximate_distribution}'
                                          f'is not yet supported but we can quickly add it if needed')

            raise NotImplementedError(f'Currently not supporting empirical distribution'
                                      f' without approximating it')

        if distribution == LOGNORMAL:
            mean = params.get('mean', 0.0)
            sigma = params.get('sigma', 1.0)
            return mocos_helper.lognormal, [], {'mean': mean, 'sigma': sigma}

        if distribution == EXPONENTIAL:
            lambda_ = params.get('lambda', 1.0)
            return mocos_helper.exponential, [], {'scale': 1/lambda_}

        if distribution == POISSON:
            lambda_ = params.get('lambda', 1.0)
            return mocos_helper.poisson, [], {'lam': lambda_}

        raise ValueError(f'Sampling from distribution {distribution} is not yet supported but we can quickly add it')

    def add_potential_contractions_from_transport_kernel(self, person_id):
        pass

    def set_up_internal_fear(self, kernel_id):
        fear_factors = self._params[FEAR_FACTORS]
        fear_factor = fear_factor_schema.validate(fear_factors.get(kernel_id, fear_factors.get(DEFAULT, None)))
        if not fear_factor:
            return fear_functions[FearFunctions.FearDisabled], 0, 0, 0, 0, 0
        f = fear_functions[FearFunctions(fear_factor[FEAR_FUNCTION])]
        limit_value = fear_factor[LIMIT_VALUE]
        scale = fear_factor[SCALE_FACTOR]
        loc = fear_factor[LOC_FACTOR]
        weights_deaths = fear_factor[DEATHS_MULTIPLIER]
        weights_detected = fear_factor[DETECTED_MULTIPLIER]
        return f, weights_detected, weights_deaths, scale, loc, limit_value

    def fear(self, kernel_id) -> float:
        if kernel_id not in self.fear_fun:
            res = self.set_up_internal_fear(kernel_id)
            (self.fear_fun[kernel_id], self.fear_weights_detected[kernel_id],
             self.fear_weights_deaths[kernel_id], self.fear_scale[kernel_id],
             self.fear_loc[kernel_id],  self.fear_limit_value[kernel_id]) = res
        detected = self.detected_people
        deaths = self.deaths
        time = self._global_time
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            if self._max_time_offset != np.inf:
                time -= self._max_time_offset
            else:
                time = -np.inf

        return self.fear_fun[kernel_id](detected, deaths, time, self.fear_weights_detected[kernel_id],
                                        self.fear_weights_deaths[kernel_id], self.fear_loc[kernel_id],
                                        self.fear_scale[kernel_id], self.fear_limit_value[kernel_id])

    def gamma(self, kernel_id):
        return self._params[TRANSMISSION_PROBABILITIES][kernel_id]

    def household_kernel_old_implementation(self, person_id):
        prog_times = self._progression_times_dict[person_id]
        start = prog_times[T0]
        end = prog_times[T2] or prog_times[TRECOVERY]
        total_infection_rate = (end - start) * self.gamma('household')
        infected = mocos_helper.poisson(total_infection_rate)
        if infected == 0:
           return
        household_id = self._individuals_household_id[person_id]
        inhabitants = self._households_inhabitants[household_id]
        possible_choices = [i for i in inhabitants if i != person_id]
        for choice_idx in mocos_helper.sample_idxes_with_replacement_uniform(len(possible_choices), infected):
            person_idx = possible_choices[choice_idx]
            if self.get_infection_status(person_idx) == InfectionStatus.Healthy:
                contraction_time = mocos_helper.uniform(low=start, high=end)
                self.append_event(Event(contraction_time, person_idx, TMINUS1, person_id, HOUSEHOLD, self.global_time))

    def add_potential_contractions_from_household_kernel(self, person_id):
        if self._params[OLD_IMPLEMENTATION_FOR_HOUSEHOLD_KERNEL]:
            self.household_kernel_old_implementation(person_id)
            return
        prog_times = self._progression_times_dict[person_id]
        start = prog_times[T0]
        end = prog_times[T2] or prog_times[TRECOVERY]
        household_id = self._individuals_household_id[person_id]
        inhabitants = self._households_inhabitants[household_id]
        possible_choices = [i for i in inhabitants if i != person_id]

        for person_idx in possible_choices:
            if self.get_infection_status(person_idx) == InfectionStatus.Healthy:
                scale = len(possible_choices) / self.gamma('household')
                contraction_time = start + mocos_helper.exponential(scale=scale)

                if contraction_time >= end:
                    continue

                self.append_event(Event(contraction_time, person_idx, TMINUS1, person_id, HOUSEHOLD, self.global_time))

    def add_potential_contractions_from_constant_kernel(self, person_id):
        """ Constant kernel draws a number of infections based on base gamma and enqueue randomly selected events """
        prog_times = self._progression_times_dict[person_id]
        start = prog_times[T0]
        end = prog_times[T1]
        if end is None:
            end = prog_times[T2]
        total_infection_rate = (end - start) * self.gamma('constant')
        infected = mocos_helper.poisson(total_infection_rate)
        if infected == 0:
            return

        selected_rows = mocos_helper.nonreplace_sample_few(self._individuals_indices,
                                                           infected, person_id)

        for person_idx in selected_rows:
            if self.get_infection_status(person_idx) == InfectionStatus.Healthy:
                contraction_time = mocos_helper.uniform(low=start, high=end)
                self.append_event(Event(contraction_time, person_idx, TMINUS1, person_id, CONSTANT, self.global_time))

    def add_potential_contractions_from_constant_age_kernel(self, person_id):
        if self._disable_constant_age_kernel is True:
            return
        age = self._individuals_age_dct[person_id]
        if age not in self._constant_age_helper_age_dict:
            return
        prog_times = self._progression_times_dict[person_id]
        start = prog_times[T0]
        end = prog_times[T1]
        if end is None:
            end = prog_times[T2]
        total_infection_rate = (end - start) * self.gamma('constant_age')

        infected = mocos_helper.poisson(total_infection_rate)
        if infected == 0:
            return

        selected_rows = mocos_helper.nonreplace_sample_few(
            self._constant_age_individuals[self._constant_age_helper_age_dict[age]],
            infected,
            person_id
        )

        for person_idx in selected_rows:
            if self.get_infection_status(person_idx) == InfectionStatus.Healthy:
                contraction_time = mocos_helper.uniform(low=start, high=end)
                self.append_event(Event(contraction_time, person_idx, TMINUS1, person_id, CONSTANT_AGE, self.global_time))

    def add_potential_contractions_from_friendship_kernel(self, person_id):
        if self._disable_friendship_kernel is True:
            return
        prog_times = self._progression_times_dict[person_id]
        start = prog_times[T0]
        end = prog_times[T1]
        if end is None:
            end = prog_times[T2]
        total_infection_rate = (end - start) * self.gamma('friendship')
        no_infected = mocos_helper.poisson(total_infection_rate * self._social_activity_scores[person_id])
        # Add a constant multiplicand above?

        age = self._individuals_age_dct[person_id]
        gender = self._individuals_gender_dct[person_id]
        for _ in range(no_infected):
            infected_idx = self._social_activity_sampler.gen(age, gender)
            if self.get_infection_status(infected_idx) == InfectionStatus.Healthy:
                contraction_time = mocos_helper.uniform(low=start, high=end)
                self.append_event(Event(contraction_time, infected_idx, TMINUS1, person_id, FRIENDSHIP, self.global_time))


    def handle_t0(self, person_id):
        self._active_people += 1
        if self.get_infection_status(person_id) in [
            InfectionStatus.Healthy,
            InfectionStatus.Contraction
        ]:
            self._infection_status[person_id] = InfectionStatus.Infectious.value
        else:
            raise AssertionError(f'Unexpected state detected: {self.get_infection_status(person_id)}'
                                 f'person_id: {person_id}')

        household_id = self._individuals_household_id[person_id]  # self._df_individuals.loc[person_id, HOUSEHOLD_ID]
        capacity = self._households_capacities[household_id]  # self._df_households.loc[household_id][ID]
        if capacity > 1:
            self.add_potential_contractions_from_household_kernel(person_id)
        self.add_potential_contractions_from_constant_kernel(person_id)
        self.add_potential_contractions_from_friendship_kernel(person_id)
        self.add_potential_contractions_from_constant_age_kernel(person_id)

    def generate_disease_progression(self, person_id, event_time: float,
                                     initial_infection_status: str) -> None:
        """Returns list of disease progression events
        "future" disease_progression should be recalculated when the disease will be recognised at the state level
        t0 - time when individual becomes infectious (Mild symptoms)
        t1 - time when individual stay home/visit doctor due to Mild/Serious? symptoms
        t2 - time when individual goes to hospital due to Serious symptoms
        tdeath - time when individual dies (depending on death probability)
        trecovery - time when individual is recovered (in case the patient will not die from covid19)

        If person is Infected:
          A - tminus1 is known (event time),
          B - t0 is calculated as tminus1 + rv_t0,

        If person is added to population as Infectious:
          A - t0 is known (event time),
          B - tminus 1 is calculated as t0 - rv_t0

        For all infected:
          A - t1 is calculated as t0 + rv_t1

        If person will develop Severe or Critical symptoms:
          A - t2 is calculated as t0 + rv_t2
          B - if t1 is larger than t2, discard t1
          C - calculate trecovery time as t0 + 6 weeks <- these 6 weeks are from WHO report, in python we use uniform[4w,8w]
          D - calculate tdetection as t2

        If person will develop Asymptomatic or Mild symptoms:
          A - calculate trecovery time as t0 + 2 weeks <- these 2 weeks are from WHO report, in python we use uniform[11d,17d]
          B - draw a random number uniform[0,1] and if less than detection_mild_proba, calculate tdetection as t0 + 2

        Draw a random number uniform[0,1] and if less than death_probability[expected_case(person_id)]:
          A - calculate tdeath time as t0 + rv_tdeath,
          B - discard all times that are larger than tdeath

        """
        if initial_infection_status == InfectionStatus.Contraction:
            tminus1 = event_time
            t0 = tminus1 + self.rv_t0()
            self.append_event(Event(t0, person_id, T0, person_id, DISEASE_PROGRESSION, tminus1))
            self._infection_status[person_id] = initial_infection_status
        elif initial_infection_status == InfectionStatus.Infectious:
            t0 = event_time
            # tminus1 does not to be defined, but for completeness let's calculate it
            tminus1 = t0 - self.rv_t0()
        else:
            raise ValueError(f'invalid initial infection status {initial_infection_status}')
        t2 = None
        if self._expected_case_severity[person_id] in [
            ExpectedCaseSeverity.Severe,
            ExpectedCaseSeverity.Critical
        ]:
            t2 = t0 + self.rv_t2()
            self.append_event(Event(t2, person_id, T2, person_id, DISEASE_PROGRESSION, t0))

        t1 = t0 + self.rv_t1()
        if not t2 or t1 < t2:
            self.append_event(Event(t1, person_id, T1, person_id, DISEASE_PROGRESSION, t0))
        else:
            # if t2 < t1 then we reset t1 to avoid misleading in data exported from the simulation
            t1 = None

        tdetection = None
        trecovery = None
        tdeath = None
        if mocos_helper.rand() <= self._params[DEATH_PROBABILITY][self._expected_case_severity[person_id]]:
            tdeath = t0 + self.rv_tdeath()
            self.append_event(Event(tdeath, person_id, TDEATH, person_id, DISEASE_PROGRESSION, t0))
        else:
            if self._expected_case_severity[person_id] in [
                ExpectedCaseSeverity.Mild,
                ExpectedCaseSeverity.Asymptomatic
            ]:
                trecovery = t0 + mocos_helper.uniform(14.0 - 3.0, 14.0 + 3.0)  # TODO: this should not be hardcoded!
            else:
                trecovery = t0 + mocos_helper.uniform(42.0 - 14.0, 42.0 + 14.0)
            self.append_event(Event(trecovery, person_id, TRECOVERY, person_id, DISEASE_PROGRESSION, t0))

        """ Following is for checking whther tdetection should be picked up"""
        calculate_tdetection = self._params[TURN_ON_DETECTION]
        if self._expected_case_severity[person_id] in [
            ExpectedCaseSeverity.Mild,
            ExpectedCaseSeverity.Asymptomatic
        ]:
            if mocos_helper.rand() > self._params[DETECTION_MILD_PROBA]:
                calculate_tdetection = False
        if calculate_tdetection:
            """ If t2 is defined (severe/critical), then use this time; if not; use some offset from t0 """
            tdetection = t2 or t0 + 2  # TODO: this should not be hardcoded
            ev = Event(tdetection, person_id, TDETECTION, person_id, DETECTION, t0)
            self.append_event(ev)

        self._progression_times_dict[person_id] = {ID: person_id, TMINUS1: tminus1, T0: t0, T1: t1, T2: t2,
                                                   TDEATH: tdeath, TRECOVERY: trecovery, TDETECTION: tdetection}

        if initial_infection_status == InfectionStatus.Infectious:
            self.handle_t0(person_id)

    @property
    def df_infections(self):
        return pd.DataFrame.from_dict(self._infections_dict, orient='index')

    @property
    def df_progression_times(self):
        return pd.DataFrame.from_dict(self._progression_times_dict, orient='index')

    def save_progression_times(self, path):
        with open(path, "w") as f:
            f.write('idx,tminus1,t0,t1,t2,tdeath,trecovery,tdetection,quarantine\n')
            for elem in self._progression_times_dict.values():
                str = f'{elem.get(ID, None)},{elem.get(TMINUS1, None)},{elem.get(T0, None)},'\
                      f'{elem.get(T1, None)},{elem.get(T2, None)},{elem.get(TDEATH, None)},'\
                      f'{elem.get(TRECOVERY, None)},{elem.get(TDETECTION, None)},{elem.get(QUARANTINE, None)}\n'
                f.write(str)

    def save_potential_contractions(self, path):
        with open(path, "w") as f:
            f.write('source_id,target_id,contraction_time,kernel\n')
            for elem in self._infections_dict.values():
                if elem.get(CONTRACTION_TIME) <= self._global_time: # skiping events that were not realized yet
                    str = f'{elem.get(SOURCE, None)},{elem.get(TARGET, None)},{elem.get(CONTRACTION_TIME, None)},'\
                          f'{elem.get(KERNEL, None)}\n'
                    f.write(str)

    def prevalance_at(self, time):
        return len([1 for elem in self._infections_dict.values() if elem.get(CONTRACTION_TIME, np.inf) <= time])

    def mean_day_increase_until(self, time):
        mean_increase = 0.0
        i = 0
        for k, v in self._per_day_increases.items():
            if k <= time:
                mean_increase = (mean_increase * i + v) / (i + 1)
        return mean_increase

    def detected_cases(self, df_r1):
        cond1 = ~df_r1.tdetection.isna()
        cond2a = ~df_r1.trecovery.isna()
        cond2b = df_r1.tdetection > df_r1.trecovery
        cond2 = ~np.logical_and(cond2a, cond2b)
        if len(df_r1[~df_r1.tdeath.isna()]) > 0:
            cond3a = ~df_r1.tdeath.isna()
            cond3b = df_r1.tdetection > df_r1.tdeath
            cond3 = ~np.logical_and(cond3a, cond3b)
            cond23 = np.logical_and(cond2, cond3)
        else:
            cond23 = cond2
        cond = np.logical_and(cond1, cond23)
        df = df_r1[cond]
        detected_cases = df.sort_values(by='tdetection').tdetection
        return detected_cases

    @staticmethod
    def store_parameter(simulation_output_dir, parameter, filename):
        save_path = os.path.join(simulation_output_dir, filename)
        with open(save_path, 'wb') as f:
            pickle.dump(parameter, f)

    def _save_population_parameters(self, simulation_output_dir):
        run_id = f'{int(time.monotonic() * 1e9)}_{self._params[RANDOM_SEED]}'
        if self._params[SAVE_EXPECTED_SEVERITY]:
            self.store_parameter(simulation_output_dir, self._expected_case_severity, 'expected_case_severity.pkl')
        self.store_parameter(simulation_output_dir, self._infection_status, 'infection_status.pkl')
        self.store_parameter(simulation_output_dir, self._detection_status, 'detection_status.pkl')
        self.store_parameter(simulation_output_dir, self._quarantine_status, 'quarantine_status.pkl')

    def _save_dir(self, prefix=''):
        underscore_if_prefix = '_' if len(prefix) > 0 else ''
        json_name = os.path.splitext(os.path.basename(self.params_path))[0]
        run_id = f'{prefix}{underscore_if_prefix}{json_name}_{int(time.monotonic() * 1e9)}_{self._params[RANDOM_SEED]}'
        simulation_output_dir = os.path.join(self._params[OUTPUT_ROOT_DIR],
                                             self._params[EXPERIMENT_ID],
                                             run_id)
        os.makedirs(simulation_output_dir)
        return simulation_output_dir

    def save_serial_interval(self, simulation_output_dir):
        if len(self.serial_intervals) == 0:
            return np.nan
        np_intervals = np.array(self.serial_intervals)
        serial_interval_median = np.median(np_intervals)
        description = scipy.stats.describe(np_intervals)
        serial_interval_str = f'serial interval: measured from {self._params[SERIAL_INTERVAL][MIN_TIME]}'\
                              f' to {self._params[SERIAL_INTERVAL][MAX_TIME]};'\
                              f' median={serial_interval_median}, stats describe: {description}'
        logger.info(serial_interval_str)
        np.save(os.path.join(simulation_output_dir, 'serial_intervals.npy'), np_intervals)
        output_log_file = os.path.join(simulation_output_dir, 'serial_interval_stats.txt')
        with open(output_log_file, "w") as out:
            out.write(serial_interval_str)
        return serial_interval_median

    def log_outputs(self, simulation_output_dir):
        self._save_population_parameters(simulation_output_dir)
        copyfile(self.params_path, os.path.join(simulation_output_dir,
                                                f'input_{os.path.basename(self.params_path)}'))

        if self._params[SAVE_INPUT_DATA]:
            copyfile(self.df_individuals_path, os.path.join(simulation_output_dir,
                                                            f'input_{os.path.basename(self.df_individuals_path)}'))
            household_input_path = os.path.join(self._params[OUTPUT_ROOT_DIR], self._params[EXPERIMENT_ID],
                                                'input_df_households.csv')
            if not os.path.exists(household_input_path):
                self._df_households.to_csv(household_input_path)
        repo = Repo(config.ROOT_DIR)
        git_active_branch_log = os.path.join(simulation_output_dir, 'git_active_branch_log.txt')
        with open(git_active_branch_log, 'w') as f:
            f.write(f'Active branch name {repo.active_branch.name}\n')
            f.write(str(repo.active_branch.log()))

        git_status = os.path.join(simulation_output_dir, 'git_status.txt')
        with open(git_status, 'w') as f:
            f.write(repo.git.status())

        serial_interval = self.save_serial_interval(simulation_output_dir)
        if self._params[ENABLE_VISUALIZATION]:
            self._vis.visualize_simulation(simulation_output_dir, serial_interval, self.fear,
                                           self.active_people, self._max_time_offset, self.detected_cases,
                                           self.df_progression_times,
                                           self.df_infections
                                           )

    def update_max_time_offset(self):
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            if self._max_time_offset == np.inf:
                if self._params[NUMBER_OF_DETECTED_AT_ZERO_TIME] <= self._detected_people:
                    self._max_time_offset = self._global_time
                    self._init_for_stats = self._active_people

    def quick_return_condition(self, initiated_through):
        """ Checks if event of type 'initiated_through' should be abandoned given current situation """
        if initiated_through == HOUSEHOLD:
            return False

        r = mocos_helper.rand()
        if initiated_through == CONSTANT and len(self._params[R_OUT_SCHEDULE]) > 0:
            t = self._global_time - self._max_time_offset
            for s in self._params[R_OUT_SCHEDULE]:
                if s[MIN_TIME] <= t <= s[MAX_TIME]:
                    if r > s[OVERRIDE_R_FRACTION]:
                        return True
                    else:
                        return False

        if r > self.fear(initiated_through):
            return True
        return False

    def add_new_infection(self, person_id, infection_status,
                          initiated_by, initiated_through):
        self._detection_status[person_id] = DetectionStatus.NotDetected.value

        self._infections_dict[len(self._infections_dict)] = {
            SOURCE: initiated_by,
            TARGET: person_id,
            CONTRACTION_TIME: self.global_time,
            KERNEL: initiated_through
        }
        if self.global_time >= self._params[SERIAL_INTERVAL][MIN_TIME]:
            if self.global_time < self._params[SERIAL_INTERVAL][MAX_TIME]:
                if initiated_by is not None:
                    serial_interval = self.global_time - self._progression_times_dict[initiated_by][TMINUS1]
                    self.serial_intervals.append(serial_interval)

        self._affected_people += 1

        self.generate_disease_progression(person_id,
                                          self.global_time,
                                          infection_status)

    # 'Event', [TIME, PERSON_INDEX, TYPE, INITIATED_BY, INITIATED_THROUGH, ISSUED_TIME])
    def process_event(self, event) -> bool:
        type_ = getattr(event, TYPE)
        time = getattr(event, TIME)
        if int(time / self._params[LOG_TIME_FREQ]) != int(self._global_time / self._params[LOG_TIME_FREQ]):
            memory_use = ps.memory_info().rss / 1024 / 1024
            fearC = self.fear(CONSTANT)
            fearH = self.fear(HOUSEHOLD)
            per_day_increase = 0
            if self._last_affected:
                per_day_increase = (self.affected_people - self._last_affected)/self._last_affected*100
            self._last_affected = self.affected_people
            self._per_day_increases[int(self._global_time)] = per_day_increase
            logger.info(f'Time: {time:.2f}'
                         f'\tAffected: {self.affected_people}'
                         f'\tDetected: {self.detected_people}'
                         f'\tQuarantined: {self.quarantined_people}'
                         f'\tPer-day-increase: {per_day_increase:.2f} %'
                         f'\tActive: {self.active_people}'
                         f'\tDeaths: {self.deaths}'
                         f'\tFearC: {fearC}'
                         f'\tFearH: {fearH}'
                         f'\tPhysical memory use: {memory_use:.2f} MB')
        self._global_time = time
        if self._global_time > self._max_time + self._max_time_offset:
            return False
        person_id = getattr(event, PERSON_INDEX)
        initiated_by = getattr(event, INITIATED_BY)
        initiated_through = getattr(event, INITIATED_THROUGH)

        # TODO the remaining attribute will be useful when we will take into account for backtracing
        # issued_time = getattr(event, ISSUED_TIME)
        if initiated_by is None and initiated_through != DISEASE_PROGRESSION:
            if self.get_infection_status(person_id) == InfectionStatus.Healthy:
                if type_ == TMINUS1:
                    self.add_new_infection(person_id, InfectionStatus.Contraction.value,
                                           initiated_by, initiated_through)
                elif type_ == T0:
                    self.add_new_infection(person_id, InfectionStatus.Infectious.value,
                                           initiated_by, initiated_through)
        elif type_ == TMINUS1:
            # check if this action is still valid first
            initiated_inf_status = self._infection_status[initiated_by]
            if initiated_inf_status in active_states:
                if self.quick_return_condition(initiated_through):
                    return True

                current_status = self.get_infection_status(person_id)
                if current_status == InfectionStatus.Healthy:
                    new_infection = False
                    # TODO below is a spaghetti code that should be sorted out! SORRY!
                    if initiated_through != HOUSEHOLD:
                        if initiated_inf_status != InfectionStatus.StayHome:
                            new_infection = True
                        if self.get_quarantine_status_(initiated_by) == QuarantineStatus.Quarantine:
                            new_infection = False
                        if self.get_quarantine_status_(person_id) == QuarantineStatus.Quarantine:
                            new_infection = False
                    else:  # HOUSEHOLD kernel:
                        new_infection = True
                    if new_infection:
                        self.add_new_infection(person_id, InfectionStatus.Contraction.value,
                                               initiated_by, initiated_through)
        elif type_ == T0:
            if self.get_infection_status(person_id) == InfectionStatus.Contraction:
                self.handle_t0(person_id)
        elif type_ == T1:
            if self.get_infection_status(person_id) == InfectionStatus.Infectious:
                self._infection_status[person_id] = InfectionStatus.StayHome.value
        elif type_ == T2:
            if self.get_infection_status(person_id) in [
                InfectionStatus.StayHome,
                InfectionStatus.Infectious
            ]:
                self._infection_status[person_id] = InfectionStatus.Hospital.value
                if self._expected_case_severity[person_id] == ExpectedCaseSeverity.Critical:
                    self._icu_needed += 1
        elif type_ == TDEATH:
            if self.get_infection_status(person_id) not in [
                InfectionStatus.Death,
                InfectionStatus.Recovered
            ]:
                self._deaths += 1
                if self._expected_case_severity[person_id] == ExpectedCaseSeverity.Critical:
                    if self._progression_times_dict[person_id][T2] < self.global_time:
                        self._icu_needed -= 1
                self._active_people -= 1
                self._infection_status[person_id] = InfectionStatus.Death.value

        elif type_ == TRECOVERY: # TRECOVERY is exclusive with regards to TDEATH (when this comment was added)
            if self.get_infection_status(person_id) not in [
                InfectionStatus.Recovered,
                InfectionStatus.Death
            ]:
                if initiated_through != INITIAL_CONDITIONS:
                    self._active_people -= 1
                    if self._expected_case_severity[person_id] == ExpectedCaseSeverity.Critical:
                        if self._progression_times_dict[person_id][T2] < self.global_time:
                            self._icu_needed -= 1
                self._infection_status[person_id] = InfectionStatus.Recovered
                self._immune_people += 1
        elif type_ == TDETECTION:
            if self.get_infection_status(person_id) not in [
                InfectionStatus.Recovered,
                InfectionStatus.Healthy
            ]:
                if self.get_detection_status_(person_id) == DetectionStatus.NotDetected:
                    self._detection_status[person_id] = DetectionStatus.Detected.value
                    self._detected_people += 1
                    self.update_max_time_offset()
                    household_id = self._individuals_household_id[person_id]
                    for inhabitant in self._households_inhabitants[household_id]:
                        if self.get_quarantine_status_(inhabitant) == QuarantineStatus.NoQuarantine:
                            if self.get_infection_status(inhabitant) != InfectionStatus.Death:
                                self._quarantine_status[inhabitant] = QuarantineStatus.Quarantine.value
                                self._quarantined_people += 1
                                if inhabitant not in self._progression_times_dict:
                                    self._progression_times_dict[inhabitant] = {}
                                self._progression_times_dict[inhabitant][QUARANTINE] = self.global_time
                                if self.get_infection_status(inhabitant) in [InfectionStatus.Infectious,
                                                                             InfectionStatus.StayHome]:
                                    # TODO: this has to be implemented better, just a temporary solution:
                                    if self._progression_times_dict[inhabitant].get(TDETECTION, None) is None:
                                        new_detection_time = self.global_time + 2.0
                                        self._progression_times_dict[inhabitant][TDETECTION] = new_detection_time
                                        ev = Event(new_detection_time, inhabitant, TDETECTION,
                                                                person_id, 'quarantine_followed_detection',
                                                                self.global_time)
                                        self.append_event(ev)
        else:
            raise ValueError(f'unexpected status of event: {event}')

        return True

    def run_simulation(self):
        def _inner_loop(iter):
            threshold_type = self._params[STOP_SIMULATION_THRESHOLD_TYPE]
            value_to_be_checked = None
            start = time.time()
            times_mean = 0.0
            i = 0
            while not q.empty():
                event_start = time.time()
                if threshold_type == PREVALENCE:
                    value_to_be_checked = self.affected_people
                elif threshold_type == DETECTIONS:
                    value_to_be_checked = self.detected_people
                if value_to_be_checked is None:
                    logging.error(f"we have an error here")
                if value_to_be_checked >= self.stop_simulation_threshold:
                    logging.info(
                        f"The outbreak reached a high number {self.stop_simulation_threshold} ({threshold_type})")
                    break
                event = q.get()
                if not self.process_event(event):
                    logging.info(f"Processing event {event} returned False")
                    q.task_done()
                    break
                q.task_done()
                event_end = time.time()
                elapsed = event_end - event_start
                times_mean = ( times_mean * i + elapsed ) / (i + 1)
                i += 1

            end = time.time()
            print(f'Sim runtime {end - start}, event proc. avg time: {times_mean}')
            # cleaning up priority queue:
            while not q.empty():
                q.get_nowait()
                q.task_done()
            simulation_output_dir = self._save_dir()
            self.save_progression_times(os.path.join(simulation_output_dir, 'output_df_progression_times.csv'))
            self.save_potential_contractions(os.path.join(simulation_output_dir, 'output_df_potential_contractions.csv'))
            if self._params[LOG_OUTPUTS]:
                logger.info('Log outputs')
                self.log_outputs(simulation_output_dir)

            if self._icu_needed >= self._params[ICU_AVAILABILITY]:
                return True
            if value_to_be_checked >= self.stop_simulation_threshold:
                return True
            return False

        seeds = None
        if isinstance(self._params[RANDOM_SEED], str):
            seeds = eval(self._params[RANDOM_SEED]) # TODO: warning, this is unsafe! not use in production
        elif isinstance(self._params[RANDOM_SEED], int):
            seeds = [self._params[RANDOM_SEED]]
        runs = 0
        output_log = 'Last_processed_time;Total_#Affected;Total_#Detected;Total_#Deceased;Total_#Quarantined;'\
                     'c;c_norm;Init_#people;Band_hit_time;Subcritical;runs;fear;detection_rate;'\
                     'incidents_per_last_day;over_icu;hospitalized;zero_time_offset;total_#immune'
        if self._params[ENABLE_ADDITIONAL_LOGS]:
            output_log += ';Prevalence_30days;Prevalence_60days;Prevalence_90days;Prevalence_120days;'\
                          'Prevalence_150days;Prevalence_180days;Prevalence_360days;'\
                          'increase_10;increase_20;increase_30;increase_40;increase_50;increase_100;increase_150'
        output_log += '\n'
        for i, seed in enumerate(seeds):
            runs += 1
            self.parse_random_seed(seed)
            self.setup_simulation()
            logger.info('Filling queue based on initial conditions...')
            self._fill_queue_based_on_initial_conditions()

            logger.info('Filling queue based on auxiliary functions...')
            self._fill_queue_based_on_auxiliary_functions()
            logger.info('Initialization step is done!')
            outbreak = _inner_loop(i + 1)
            last_processed_time = self._global_time

            c = self._params[TRANSMISSION_PROBABILITIES][CONSTANT]
            c_norm = c * self._params[AVERAGE_INFECTIVITY_TIME_CONSTANT_KERNEL]
            subcritical = self._active_people < self._init_for_stats / 2 # at 200 days

            bandtime = self.band_time
            #if bandtime:
            #    return 0
            fear_ = self.fear(CONSTANT)
            detection_rate = self._params[DETECTION_MILD_PROBA]
            affected = self.affected_people
            detected = self.detected_people
            deceased = self.deaths
            quarantined = self.quarantined_people
            incidents_per_last_day = self.prevalance_at(self._global_time) - self.prevalance_at(self._global_time - 1)
            hospitalized = self._icu_needed
            zero_time_offset = self._max_time_offset
            immune = self._immune_people
            output_add = f'{last_processed_time };{affected};{detected};{deceased};{quarantined};{c};{c_norm};'\
                         f'{self._init_for_stats};{bandtime};{subcritical};{runs};{fear_};{detection_rate};'\
                         f'{incidents_per_last_day};{outbreak};{hospitalized};{zero_time_offset};{immune}'

            if self._params[ENABLE_ADDITIONAL_LOGS]:
                prev30 = self.prevalance_at(30)
                prev60 = self.prevalance_at(60)
                prev90 = self.prevalance_at(90)
                prev120 = self.prevalance_at(120)
                prev150 = self.prevalance_at(150)
                prev180 = self.prevalance_at(180)
                prev360 = self.prevalance_at(360)
                mean_increase_at_10 = self.mean_day_increase_until(10)
                mean_increase_at_20 = self.mean_day_increase_until(20)
                mean_increase_at_30 = self.mean_day_increase_until(30)
                mean_increase_at_40 = self.mean_day_increase_until(40)
                mean_increase_at_50 = self.mean_day_increase_until(50)
                mean_increase_at_100 = self.mean_day_increase_until(100)
                mean_increase_at_150 = self.mean_day_increase_until(150)
                output_add += f'{prev30};{prev60};{prev90};{prev120};{prev150};{prev180};{prev360};'\
                              f'{mean_increase_at_10};{mean_increase_at_20};{mean_increase_at_30};'\
                              f'{mean_increase_at_40};{mean_increase_at_50};{mean_increase_at_100};'\
                              f'{mean_increase_at_150}'
            output_add += '\n'
            logger.info(output_add)
            output_log = f'{output_log}{output_add}'
        logger.info(output_log)
        simulation_output_dir = self._save_dir('aggregated_results')
        output_log_file = os.path.join(simulation_output_dir, 'results.txt')
        if self._params[ENABLE_VISUALIZATION]:
            self._vis.visualize_scenario(simulation_output_dir)
        with open(output_log_file, "w") as out:
            out.write(output_log)


    def setup_simulation(self):
        self._init_for_stats = 0 # TODO support different import methods
        if isinstance(self._params[INITIAL_CONDITIONS], dict):
            cardinalities = self._params[INITIAL_CONDITIONS][CARDINALITIES]
            self._init_for_stats = cardinalities.get(CONTRACTION, 0) + cardinalities.get(INFECTIOUS, 0)

        # TODO  and think how to better group them, ie namedtuple state_stats?
        self._affected_people = 0
        self._active_people = 0
        self._detected_people = 0
        self._quarantined_people = 0
        self._immune_people = 0
        self._deaths = 0
        self._icu_needed = 0
        self._max_time_offset = 0
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            self._max_time_offset = np.inf
        self._fear_factor = {}
        self._infection_status = {}
        self._infections_dict = {}
        self._progression_times_dict = {}
        self._per_day_increases = {}

        self._global_time = self._params[START_TIME]
        self._max_time = self._params[MAX_TIME]

        if not self._params[REUSE_EXPECTED_CASE_SEVERITIES]:
            self._expected_case_severity = self.draw_expected_case_severity()

        self._last_affected = None
        self.band_time = None
        self._quarantine_status = {}
        self._detection_status = {}
        if self._params[ENABLE_VISUALIZATION]:
            self._vis = Visualize(self._params, self.df_individuals,
                                  self._expected_case_severity, logger)


logger = logging.getLogger(__name__)

@click.command()
@click.option('--params-path', type=click.Path(exists=True))
@click.option('--df-individuals-path', type=click.Path(exists=True))
@click.option('--df-households-path', type=click.Path())
@click.argument('run-simulation') #ignored
def runner(params_path, df_individuals_path, run_simulation, df_households_path=''):
    im = InfectionModel(params_path=params_path,
                        df_individuals_path=df_individuals_path,
                        df_households_path=df_households_path or '')
    im.run_simulation()


# TODO: think about separate thread/process to generate random numbers, facilitate sampling
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    pid = os.getpid()
    ps = psutil.Process(pid)
    pd.set_option('display.max_columns', None)
    #fire.Fire(InfectionModel)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    runner()
