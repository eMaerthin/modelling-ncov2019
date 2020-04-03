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
import pickle
import psutil
from shutil import copyfile
from math import log

from git import Repo
from matplotlib import pyplot as plt
import pandas as pd
import scipy.optimize
import scipy.stats

from src.models.schemas import *
from src.models.defaults import *
from src.models.states_and_functions import *


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
        self._max_time_offset = 0.0
        self._expected_case_severity = None
        self._df_individuals = None
        self._df_households = None
        #self._individuals_gender = {}
        self._individuals_age = None
        self._individuals_household_id = {}
        self._individuals_indices = None
        self._all_runs_detected = []
        self._all_runs_prevalence = []
        self._all_runs_severe = []
        self._households_capacities = {}
        self._households_inhabitants = {}
        self._init_for_stats = 0  # TODO support different import methods
        if isinstance(self._params[INITIAL_CONDITIONS], dict):
            cardinalities = self._params[INITIAL_CONDITIONS][CARDINALITIES]
            self._init_for_stats = cardinalities.get(CONTRACTION, 0) + cardinalities.get(INFECTIOUS, 0)

        self._affected_people = 0
        self._active_people = 0
        self._quarantined_people = 0
        self._detected_people = 0
        self._deaths = 0
        self._icu_needed = 0

        self._set_up_data_frames()
        self._infection_status = {}
        self._detection_status = {}
        self._quarantine_status = {}
        self._expected_case_severity = self.draw_expected_case_severity()
        self._infections_dict = {}
        self._progression_times_dict = {}

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
        self.experimental_ub = None
        self.experimental_lb = None

        self.band_time = None
        self._last_affected = None
        self._per_day_increases = {}

    @property
    def xlim(self):
        return (self._params.get(PLOT_XLIM_LEFT, default_plot_xlim_left),
                self._params.get(PLOT_XLIM_RIGHT, default_plot_xlim_right))

    @property
    def ylim(self):
        return (self._params.get(PLOT_YLIM_BOTTOM, default_plot_ylim_bottom),
                self._params.get(PLOT_YLIM_TOP, default_plot_ylim_top))

    @property
    def xlim_cut(self):
        left = self._params.get(PLOT_XLIM_CUT_LEFT, None)
        right = self._params.get(PLOT_XLIM_CUT_RIGHT, None)
        if left is None:
            left = self.xlim[0]
        if right is None:
            right = self.xlim[1]
        return (left, right)

    @property
    def ylim_cut(self):
        bottom = self._params.get(PLOT_YLIM_CUT_BOTTOM, None)
        top = self._params.get(PLOT_YLIM_CUT_TOP, None)
        if bottom is None:
            bottom = self.ylim[0]
        if top is None:
            top = self.ylim[1]
        return (bottom, top)


    def get_detection_status_(self, person_id):
        return self._detection_status.get(person_id, default_detection_status)

    def get_quarantine_status_(self, person_id):
        return self._quarantine_status.get(person_id, default_quarantine_status)

    def get_infection_status(self, person_id):
        return self._infection_status.get(person_id, InfectionStatus.Healthy.value)

    @staticmethod
    def parse_random_seed(random_seed):
        #np.random.seed(random_seed)
        #random.seed(random_seed)
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
        self._social_activity_scores = self._df_individuals.social_competence.to_dict()

        self._social_activity_sampler = mocos_helper.AgeDependentFriendSampler(
            self._individuals_indices,
            self._individuals_age,
            self._df_individuals[GENDER].values,
            self._df_individuals.social_competence.values
            )

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
                        #selected_rows = np.random.choice(choice_set, cardinality, replace=False)
                        # now only previously unselected indices can be drawn in next steps
                        #choice_set = np.array(list(set(choice_set) - set(selected_rows)))
                        choice_set, selected_rows = mocos_helper.randomly_split_list(choice_set, howmuch=cardinality)
                        t_state = _assign_t_state(infection_status)
                        for row in selected_rows:
                            self.append_event(Event(self.global_time, row, t_state, None, INITIAL_CONDITIONS,
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
            #distribution_hist = np.array([age_induced_severity_distribution[x] for x in case_severity_dict])
            #dis = scipy.stats.rv_discrete(values=(
            #    np.arange(len(age_induced_severity_distribution)),
            #    distribution_hist
            #))
            #realizations = dis.rvs(size=len(self._individuals_indices[cond]))
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
                #return scipy.stats.lognorm.rvs, [shape], {'loc':loc, 'scale':scale}

            if approximate_distribution == GAMMA:
                shape, loc, scale = scipy.stats.gamma.fit(array, floc=0)
                return mocos_helper.gamma, [], {'alpha': shape, 'beta': scale}
                #return scipy.stats.gamma.rvs, [shape], {'loc':loc, 'scale':scale}

            if approximate_distribution:
                raise NotImplementedError(f'Approximating to this distribution {approximate_distribution}'
                                          f'is not yet supported but we can quickly add it if needed')

            raise NotImplementedError(f'Currently not supporting empirical distribution'
                                      f' without approximating it')

        if distribution == LOGNORMAL:
            mean = params.get('mean', 0.0)
            sigma = params.get('sigma', 1.0)
            return mocos_helper.lognormal, [], {'mean':mean, 'sigma':sigma}

        if distribution == EXPONENTIAL:
            lambda_ = params.get('lambda', 1.0)
            return mocos_helper.exponential, [], {'scale':1/lambda_}

        if distribution == POISSON:
            lambda_ = params.get('lambda', 1.0)
            return mocos_helper.poisson, [], {'lam':lambda_}

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
        return self._params[TRANSMISSION_PROBABILITIES][kernel_id] * self.fear(kernel_id)

    def add_potential_contractions_from_household_kernel(self, person_id):
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

    def add_potential_contractions_from_constant_kernel(self, person_id):
        prog_times = self._progression_times_dict[person_id]
        start = prog_times[T0]
        end = prog_times[T1]
        if end is None:
            end = prog_times[T2]
        total_infection_rate = (end - start) * self.gamma('constant')
        infected = mocos_helper.poisson(total_infection_rate)
        if infected == 0:
            return
        #possible_choices = self._individuals_indices # self._df_individuals.index.values
        #possible_choices = possible_choices[possible_choices != person_id]
        #r = range(possible_choices.shape[0])
        selected_rows = mocos_helper.nonreplace_sample_few(self._individuals_indices, infected, person_id)
        #selected_rows = possible_choices[selected_rows_ids]
        for person_idx in selected_rows:
            if self.get_infection_status(person_idx) == InfectionStatus.Healthy:
                contraction_time = mocos_helper.uniform(low=start, high=end)
                self.append_event(Event(contraction_time, person_idx, TMINUS1, person_id, CONSTANT, self.global_time))

    def add_potential_contractions_from_friendship_kernel(self, person_id):
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

    def generate_disease_progression(self, person_id, event_time: float,
                                     initial_infection_status: str) -> None:
        """Returns list of disease progression events
        "future" disease_progression should be recalculated when the disease will be recognised at the state level
        t0 - time when individual becomes infectious (Mild symptoms)
        t1 - time when individual stay home/visit doctor due to Mild/Serious? symptoms
        t2 - time when individual goes to hospital due to Serious symptoms
        tdeath - time when individual dies (depending on death probability)
        trecovery - time when individual is recovered (in case the patient will not die from covid19)
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
        return pd.DataFrame.from_dict(self._infections_dict, orient='index') #, ignore_index=True) #orient='index', columns=columns)

    @property
    def df_progression_times(self):
        return pd.DataFrame.from_dict(self._progression_times_dict, orient='index') #, ignore_index=True)

    def doubling_time(self, simulation_output_dir):
        def doubling(x, y, window=100):
            x1 = x[:-window]
            x2 = x[window:]
            y1 = y[:-window]
            y2 = y[window:]
            a = (x2 - x1) * np.log(2)
            b = np.log(y2/y1)
            c = a / b
            return c # (x2 - x1) * np.log(2) / np.log(y2 / y1)

        def plot_doubling(x, ax, label, window=100):
            if len(x) > window:
                xval = x[:-window]
                yval = doubling(x.values, np.arange(1, 1 + len(x)))
                ax.plot(xval[yval<28], yval[yval<28], label=label)
                return True
            return False

        fig, ax = plt.subplots(nrows=1, ncols=1)
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections
        vals = df_r2.contraction_time.sort_values()
        plot_doubling(vals, ax, label='Trend line for prevalence')
        cond1 = df_r2.contraction_time[df_r2.kernel == 'import_intensity'].sort_values()
        cond2 = df_r2.contraction_time[df_r2.kernel == 'constant'].sort_values()
        cond3 = df_r2.contraction_time[df_r2.kernel == 'household'].sort_values()
        plot_doubling(cond1, ax, label='Trend line for # imported cases')
        plot_doubling(cond2, ax, label='Trend line for Infected through constant kernel')
        plot_doubling(cond3, ax, label='Trend line for Infected through household kernel')
        hospitalized_cases = df_r1[~df_r1.t2.isna()].sort_values(by='t2').t2
        ho_cases = hospitalized_cases[hospitalized_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        plot_doubling(ho_cases, ax, label='Trend line for # hospitalized cases')
        plot_doubling(d_cases, ax, label='Trend line for # deceased cases')
        ax.legend(loc='lower right') #legend, loc='upper left')
        ax.set_title(f'Doubling times for simulation of covid19 dynamics\n {self._params[EXPERIMENT_ID]}')
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'doubling_times.png'))
        plt.close(fig)

    def lancet_draw_death_age_cohorts(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections
        df_in = self.df_individuals
        lims = default_age_cohorts_with_descriptions

        fig, ax = plt.subplots(nrows=1, ncols=1)
        for limm, limM, descr in lims:
            cond1 = df_in.age >= limm
            cond2 = df_in.age < limM
            cond = np.logical_and(cond1, cond2)
            filtered = df_r1.loc[df_r1.index.isin(df_in[cond].index)]
            death_cases = filtered[~filtered.tdeath.isna()].sort_values(by='tdeath').tdeath
            d_cases = death_cases[death_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
            d_times = np.arange(1, 1 + len(d_cases))
            ax.plot(np.append(d_cases, df_r2.contraction_time.max(axis=0)),
                    np.append(d_times, len(d_cases)), label=descr)

        ax.legend()
        experiment_id = self._params[EXPERIMENT_ID]
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'lancet_supplementary_deceased_cases_age_analysis.png'))
        plt.close(fig)

    def draw_death_age_cohorts(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections
        df_in = self.df_individuals
        lims = default_age_cohorts_with_descriptions

        fig, ax = plt.subplots(nrows=1, ncols=1)
        for limm, limM, descr in lims:
            cond1 = df_in.age >= limm
            cond2 = df_in.age < limM
            cond = np.logical_and(cond1, cond2)
            filtered = df_r1.loc[df_r1.index.isin(df_in[cond].index)]
            death_cases = filtered[~filtered.tdeath.isna()].sort_values(by='tdeath').tdeath
            d_cases = death_cases[death_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
            d_times = np.arange(1, 1 + len(d_cases))
            ax.plot(np.append(d_cases, df_r2.contraction_time.max(axis=0)),
                    np.append(d_times, len(d_cases)), label=descr)

        ax.legend()
        experiment_id = self._params[EXPERIMENT_ID]
        ax.set_title(f'cumulative deceased cases per age group \n {experiment_id}')
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'deceased_cases_age_analysis.png'))
        plt.close(fig)

    def lancet_store_bins(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections

        fig, ax0 = plt.subplots()
        r2_max_time = df_r2.contraction_time.max()
        if self.active_people < 10:
            if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
                if self._max_time_offset != np.inf:
                    r2_max_time -= self._max_time_offset
            ax0.plot([r2_max_time], [0], 'ro', markersize=5, label='Last reported infection time')

        bins = np.arange(np.minimum(730, int(1 + r2_max_time)))
        cond3 = df_r2.contraction_time.sort_values()
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            if self._max_time_offset != np.inf:
                cond3 -= self._max_time_offset
        legend = []
        arr = []
        if len(cond3) > 0:
            arr.append(cond3)
            legend.append('Infections')
            ax0.hist(arr, bins, histtype='bar', stacked=False, label=legend)
        arr = []
        legend = []
        hospitalized_cases = df_r1[~df_r1.t2.isna()].sort_values(by='t2').t2
        ho_cases = hospitalized_cases[hospitalized_cases <= r2_max_time].sort_values()
        if len(ho_cases) > 0:
            arr.append(ho_cases)
            legend.append('Hospitalized')
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            if self._max_time_offset != np.inf:
                ho_cases -= self._max_time_offset

        ax0.hist(arr, bins, histtype='bar', stacked=False, label=legend)
        ax0.legend()
        ax0.set_ylabel('Incidents')
        ax0.set_xlabel('Time in days')
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'lancet_bins.png'))
        plt.close(fig)

    def store_bins_pl(self, simulation_output_dir):
        #font = {'family': 'arial',
        #        'weight': 'regular',
        #        'size': 16}

        #matplotlib.rc('font', **font)
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections
        fig, (ax2, ax1, ax0) = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
        r2_max_time = df_r2.contraction_time.max()
        detected = df_r1.dropna(subset=['tdetection']).sort_values(by='tdetection').tdetection
        xloc = [3, 8, 13, 18, 23, 28]
        dates = ['13/03/20', '18/03/20', '23/03/20', '28/03/20', '2/03/20', '7/04/20']
        bins = np.arange(np.minimum(730, int(1 + r2_max_time)))
        cond2 = df_r2.contraction_time[df_r2.kernel == 'constant'].sort_values()
        cond3 = df_r2.contraction_time[df_r2.kernel == 'household'].sort_values()
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            if self._max_time_offset != np.inf:
                cond2 -= self._max_time_offset - 3
                cond3 -= self._max_time_offset - 3
        legend = []
        arr = []
        if len(cond2) > 0:
            arr.append(cond2)
            legend.append('Zarażenia poza domem')
        if len(cond3) > 0:
            arr.append(cond3)
            legend.append('Zarażenia w gosp. domowym')
        values, _, _ = ax0.hist(arr, bins, histtype='bar', stacked=True, label=legend, color=['blue', 'grey'])
        # ax0.plot([3]*2, [0, np.amax(values)], 'k-', label='Ogłoszenie zamknięcia granic Polski 13/03/2020')
        ax0.legend()
        ax0.set_ylabel('Zainfekowani')
        ax0.set_xlabel('Data')
        ax0.set_xticks(xloc)
        ax0.set_xticklabels(dates)
        ax0.set_xlim([0, 30])

        arr = []
        legend = []
        hospitalized_cases = df_r1[~df_r1.t2.isna()].sort_values(by='t2').t2
        ho_cases = hospitalized_cases[hospitalized_cases <= r2_max_time].sort_values()
        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= r2_max_time].sort_values()
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            if self._max_time_offset != np.inf:
                ho_cases -= self._max_time_offset - 3
                d_cases -= self._max_time_offset - 3
        if len(d_cases) > 0:
            arr.append(d_cases)
            legend.append('Przypadki śmiertelne')
        if len(ho_cases) > 0:
            arr.append(ho_cases)
            legend.append('Hospitalizowani')
        values, _, _ = ax1.hist(arr, bins, histtype='bar', stacked=True, label=legend, color=['red', 'orange'])
        # ax1.plot([3]*2, [0, np.amax(values)], 'k-', label='Ogłoszenie zamknięcia granic Polski 13/03/2020')
        ax1.set_xlim([0, 30])

        ax1.set_ylabel('Przypadki poważne')
        ax1.set_xlabel('Data')
        ax1.legend()

        ax1.set_xticks(xloc)
        ax1.set_xticklabels(dates)

        detected_cases = self.detected_cases(df_r1)
        det_cases = detected_cases[detected_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            if self._max_time_offset != np.inf:
                det_cases -= self._max_time_offset - 3
        values, _, _ = ax2.hist(det_cases, bins, histtype='bar', stacked=True, label='Zdiagnozowani', color='green')
        # ax2.plot([3]*2, [0, np.amax(values)], 'k-', label='Ogłoszenie zamknięcia granic Polski 13/03/2020')
        ax2.set_xlim([0, 30])
        ax2.set_ylabel('Zdiagnozowani')
        ax2.set_xlabel('Data')

        ax2.set_xticks(xloc)
        ax2.set_xticklabels(dates)
        # ax2.legend()

        fig.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(simulation_output_dir, 'bins_report_pl.png'), dpi=300)
        plt.close(fig)

    def store_bins(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections
        if self._params[TURN_ON_DETECTION]:
            fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1)
        else:
            fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)
        r2_max_time = df_r2.contraction_time.max()
        if self.active_people < 10:
            if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
                if self._max_time_offset != np.inf:
                    r2_max_time -= self._max_time_offset
            ax0.plot([r2_max_time], [0], 'ro', markersize=5, label='Last reported infection time')

        bins = np.arange(np.minimum(730, int(1 + r2_max_time)))
        cond1 = df_r2.contraction_time[df_r2.kernel == 'import_intensity'].sort_values()
        cond2 = df_r2.contraction_time[df_r2.kernel == 'constant'].sort_values()
        cond3 = df_r2.contraction_time[df_r2.kernel == 'household'].sort_values()
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            if self._max_time_offset != np.inf:
                cond1 -= self._max_time_offset
                cond2 -= self._max_time_offset
                cond3 -= self._max_time_offset
        legend = []
        arr = []
        if len(cond1) > 0:
            arr.append(cond1)
            legend.append('Imported')
        if len(cond2) > 0:
            arr.append(cond2)
            legend.append('Constant kernel')
        if len(cond3) > 0:
            arr.append(cond3)
            legend.append('Household')
        ax0.hist(arr, bins, histtype='bar', stacked=True, label=legend)
        ax0.legend()
        arr = []
        legend = []
        hospitalized_cases = df_r1[~df_r1.t2.isna()].sort_values(by='t2').t2
        ho_cases = hospitalized_cases[hospitalized_cases <= r2_max_time].sort_values()
        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= r2_max_time].sort_values()
        recovery_cases = df_r1[~df_r1.trecovery.isna()].sort_values(by='trecovery').trecovery
        r_cases = recovery_cases[recovery_cases <= r2_max_time].sort_values()
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            if self._max_time_offset != np.inf:
                ho_cases -= self._max_time_offset
                d_cases -= self._max_time_offset
                r_cases -= self._max_time_offset
        if len(d_cases) > 0:
            arr.append(d_cases)
            legend.append('Deceased')
        if len(ho_cases) > 0:
            arr.append(ho_cases)
            legend.append('Hospitalized')
        if len(r_cases) > 0:
            arr.append(r_cases)
            legend.append('Recovered')
        ax1.hist(arr, bins, histtype='bar', stacked=True, label=legend)
        ax1.legend()
        ax0.set_title(f'Daily stacked summaries of simulated covid19\n {self._params[EXPERIMENT_ID]}')
        ax0.set_ylabel('Infections')
        ax0.set_xlabel('Time in days')
        ax1.set_ylabel('Outcome')
        ax1.set_xlabel('Time in days')
        if self._params[TURN_ON_DETECTION]:
            detected_cases = self.detected_cases(df_r1)
            det_cases = detected_cases[detected_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
            if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
                if self._max_time_offset != np.inf:
                    det_cases -= self._max_time_offset
            ax2.hist(det_cases, bins, histtype='bar', stacked=True, label='Daily officially detected cases')
            ax2.set_ylabel('Detections')
            ax2.set_xlabel('Time in days')
            ax2.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'bins.png'))
        plt.close(fig)

    def plot_values(self, values, label, ax, yvalues=None, type='plot', reduce_offset=True, dots=False):
        if len(values) > 0:
            x = values
            if reduce_offset:
                if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
                    if self._max_time_offset != np.inf:
                        x -= self._max_time_offset
                    else:
                        return np.arange(0), np.arange(0)
            if yvalues is None:
                y = np.arange(1, 1 + len(x))
            else:
                y = yvalues
            if type == 'plot':
                if dots:
                    ax.plot(x, y, 'ok', label=label)
                else:
                    ax.plot(x, y, label=label)
            elif type == 'semilogy':
                ax.semilogy(x, y, label=label)
            if self._params[USE_TODAY_MARK]:
                today = float(self._params[TODAY_OFFSET])
                counter = sum(np.array(x) <= today)
                label_at_today = f'{label} at T={today}: {counter}'
                ax.plot([self._params[TODAY_OFFSET]] * 2, [0, len(values)], 'k-', label=label_at_today)
            return x, y
        return np.arange(0), np.arange(0)

    def prevalance_at(self, time):
        df_r2 = self.df_infections
        return len(df_r2[df_r2.contraction_time<=time])

    def mean_day_increase_until(self, time):
        mean_increase = 0.0
        i = 0
        for k, v in self._per_day_increases.items():
            if k <= time:
                mean_increase = (mean_increase * i + v) / (i + 1)
        return mean_increase

    def store_graphs(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections

        fig, ax = plt.subplots(nrows=1, ncols=1)
        vals = df_r2.contraction_time.sort_values()
        self.plot_values(vals, 'Prevalence', ax)
        #cond1 = df_r2.contraction_time[df_r2.kernel == 'import_intensity'].sort_values()
        #cond2 = df_r2.contraction_time[df_r2.kernel == 'constant'].sort_values()
        #cond3 = df_r2.contraction_time[df_r2.kernel == 'household'].sort_values()
        #self.plot_values(cond1, 'Imported', ax)
        #self.plot_values(cond2, 'Inf. through constant kernel', ax)
        #self.plot_values(cond3, 'Inf. through household', ax)
        hospitalized_cases = df_r1[~df_r1.t2.isna()].sort_values(by='t2').t2
        ho_cases = hospitalized_cases[hospitalized_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        detected_cases = self.detected_cases(df_r1)
        det_cases = detected_cases[detected_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        self.plot_values(d_cases, 'Deceased', ax)
        self.plot_values(ho_cases, 'Hospitalized', ax)
        self.plot_values(det_cases, 'Detected', ax)
        self.add_observed_curve(ax)

        if QUARANTINE in df_r1.columns:
            quarantined_cases = df_r1[~df_r1.quarantine.isna()].sort_values(by='quarantine').quarantine
            q_cases = quarantined_cases[quarantined_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
            self.plot_values(q_cases, 'Quarantined', ax)

        ax.legend()
        ax.set_title(f'simulation of covid19 dynamics\n {self._params[EXPERIMENT_ID]}')
        if self._params[FEAR_FACTORS].get(CONSTANT, self._params[FEAR_FACTORS][DEFAULT])[FEAR_FUNCTION] != FearFunctions.FearDisabled:
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

            ax2.set_ylabel('fear')  # we already handled the x-label with ax1
            detected = np.arange(0, 1 + len(det_cases)) #.cumsum().values
            yvals = []
            kernel_id = CONSTANT
            x = [0] + list(det_cases)

            for t_, de_ in zip(x, detected):

                yvals.append(self.fear_fun[kernel_id](de_, 0, t_, self.fear_weights_detected[kernel_id],
                                                      self.fear_weights_deaths[kernel_id],
                                                      self.fear_loc[kernel_id],
                                                      self.fear_scale[kernel_id],
                                                      self.fear_limit_value[kernel_id]))
            #if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            #    if self._max_time_offset != np.inf:
            #        x = [elem - self._max_time_offset for elem in x]
            ax2.plot(x, yvals, 'k--')
            ax2.tick_params(axis='y')
            ax2.set_ylim(bottom=0, top=1)
            ax2.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'summary.png'))
        plt.close(fig)

    def detected_cases(self, df_r1):
        df_r1 = self.df_progression_times
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

    def store_detections(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections
        detected_cases = self.detected_cases(df_r1)
        det_cases = detected_cases[detected_cases <= df_r2.contraction_time.max(axis=0)].sort_values()

        self._all_runs_detected.append(det_cases)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_title(f'detected cases in time\n {self._params[EXPERIMENT_ID]}')

        self.plot_values(det_cases, 'Detected', ax)
        self.add_observed_curve(ax)
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'summary_detections.png'))
        plt.close(fig)

    def lancet_store_graphs(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections

        fig, ax = plt.subplots(nrows=1, ncols=1)
        vals = df_r2.contraction_time.sort_values()
        self.plot_values(vals, 'Prevalence', ax)
        self._all_runs_prevalence.append(vals)
        hospitalized_cases = df_r1[~df_r1.t2.isna()].sort_values(by='t2').t2

        ho_cases = hospitalized_cases[hospitalized_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        self._all_runs_severe.append(ho_cases)
        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        detected_cases = self.detected_cases(df_r1)
        det_cases = detected_cases[detected_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        self.plot_values(d_cases, 'Deceased', ax)
        self.plot_values(ho_cases, 'Hospitalized', ax)
        self.plot_values(det_cases, 'Detected', ax)

        if QUARANTINE in df_r1.columns:
            quarantined_cases = df_r1[~df_r1.quarantine.isna()].sort_values(by='quarantine').quarantine
            q_cases = quarantined_cases[quarantined_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
            self.plot_values(q_cases, 'Quarantined', ax)

        ax.legend()
        #ax.set_title(f'simulation of covid19 dynamics\n {self._params[EXPERIMENT_ID]}')
        if self._params[FEAR_FACTORS].get(CONSTANT, self._params[FEAR_FACTORS][DEFAULT])[FEAR_FUNCTION] != FearFunctions.FearDisabled:
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

            ax2.set_ylabel('fear')  # we already handled the x-label with ax1
            detected = np.arange(0, 1 + len(det_cases)) #.cumsum().values
            yvals = []
            kernel_id = CONSTANT
            x = [0] + list(det_cases)

            for t_, de_ in zip(x, detected):
                yvals.append(self.fear_fun[kernel_id](de_, 0, t_, self.fear_weights_detected[kernel_id],
                                                      self.fear_weights_deaths[kernel_id],
                                                      self.fear_loc[kernel_id],
                                                      self.fear_scale[kernel_id],
                                                      self.fear_limit_value[kernel_id]))
            #if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            #    if self._max_time_offset != np.inf:
            #        x = [elem - self._max_time_offset for elem in x]
            ax2.plot(x, yvals, 'k--')
            ax2.tick_params(axis='y')
            ax2.set_ylim(bottom=0, top=1)
            ax2.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'lancet_summary.png'))
        plt.close(fig)

    def store_semilogy(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections
        vals = df_r2.contraction_time.sort_values()
        if self.experimental_ub is None:
            self.experimental_ub = vals
        else:
            self.experimental_ub = np.minimum(vals, self.experimental_ub)
        if self.experimental_lb is None:
            self.experimental_lb = vals
        else:
            self.experimental_lb = np.maximum(vals, self.experimental_lb)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        self.plot_values(vals, 'Prevalence', ax, type='semilogy')
        #cond1 = df_r2.contraction_time[df_r2.kernel == 'import_intensity'].sort_values()
        #cond2 = df_r2.contraction_time[df_r2.kernel == 'constant'].sort_values()
        #cond3 = df_r2.contraction_time[df_r2.kernel == 'household'].sort_values()

        #self.plot_values(cond1, 'Imported', ax, type='semilogy')
        #self.plot_values(cond2, 'Inf. through constant kernel', ax, type='semilogy')
        #self.plot_values(cond3, 'Inf. through household', ax, type='semilogy')

        hospitalized_cases = df_r1[~df_r1.t2.isna()].sort_values(by='t2').t2
        ho_cases = hospitalized_cases[hospitalized_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        detected_cases = self.detected_cases(df_r1)
        det_cases = detected_cases[detected_cases <= df_r2.contraction_time.max(axis=0)].sort_values()

        self.plot_values(d_cases, 'Deceased', ax, type='semilogy')
        self.plot_values(ho_cases, 'Hospitalized', ax, type='semilogy')
        self.plot_values(det_cases, 'Detected', ax, type='semilogy')
        self.add_observed_curve(ax)

        if QUARANTINE in df_r1.columns:
            quarantined_cases = df_r1[~df_r1.quarantine.isna()].sort_values(by='quarantine').quarantine
            q_cases = quarantined_cases[quarantined_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
            self.plot_values(q_cases, 'Quarantined', ax, type='semilogy')

        ax.legend()
        ax.set_title(f'simulation of covid19 dynamics\n {self._params[EXPERIMENT_ID]}')
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'summary_semilogy.png'))
        plt.close(fig)

    def test_bandwidth_plot(self, simulation_output_dir):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        self.plot_values(self.experimental_ub, 'Prevalence UB', ax)
        self.plot_values(self.experimental_lb, 'Prevalence LB', ax)
        ax.legend()
        ax.set_title(f'Test of bandwidth plot (showing min/max across multiple runs)')
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'test_bandwidth_plot_summary_semilogy.png'))
        plt.close(fig)

    def test_lognormal_prevalence(self, simulation_output_dir):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for i, run in enumerate(self._all_runs_prevalence):
            self.plot_values(run, f'Run {i}', ax, reduce_offset=False, type='semilogy')
        self.add_observed_curve(ax)

        #ax.legend()
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_title(f'Sample paths of prevalence')
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'test_lognormal_prevalence.png'))
        plt.close(fig)

    def test_lognormal_detected(self, simulation_output_dir):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for i, run in enumerate(self._all_runs_detected):
            self.plot_values(run, f'Run {i}', ax, reduce_offset=False, type='semilogy')
        self.add_observed_curve(ax)

        #ax.legend()
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_title(f'Sample paths of detected cases')
        #ax.set_title(f'Analiza')
        #xloc = [0, 5, 10, 15, 20]
        #dates = ['12/03/20', '17/03/20', '22/03/20', '27/03/20', '1/04/20', '6/04/20']
        #ax.set_ylabel('Zdiagnozowani (skala logarytmiczna)')
        #ax.set_xlabel('Data')
        #ax.set_xticks(xloc)
        #ax.set_xticklabels(dates, rotation=30)
        fig.tight_layout()

        plt.savefig(os.path.join(simulation_output_dir, 'test_lognormal_detected.png'))
        plt.close(fig)


    def test_lognormal_detected_pl(self, simulation_output_dir):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for i, run in enumerate(self._all_runs_detected):
            self.plot_values(run, f'Run {i}', ax, reduce_offset=False, type='semilogy')
        self.add_observed_curve(ax)

        #ax.legend()
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        xloc = [0, 5, 10, 15, 20]
        dates = ['12/03/20', '17/03/20', '22/03/20', '27/03/20', '1/04/20', '6/04/20']
        ax.set_ylabel('Zdiagnozowani (skala logarytmiczna)')
        ax.set_xlabel('Data')
        ax.set_xticks(xloc)
        ax.set_xticklabels(dates, rotation=30)
        fig.tight_layout()

        plt.savefig(os.path.join(simulation_output_dir, 'test_lognormal_detected_pl.png'))
        plt.close(fig)

    def test_lognormal_severe(self, simulation_output_dir):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for i, run in enumerate(self._all_runs_severe):
            self.plot_values(run, f'Run {i}', ax, reduce_offset=False, type='semilogy')
        self.add_observed_curve(ax)

        #ax.legend()
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_title(f'Sample paths of severe cases')
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'test_lognormal_severe.png'))
        plt.close(fig)

    def test_detected_cases(self, simulation_output_dir):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        x = []
        y = []
        successes = 0
        for i, run in enumerate(self._all_runs_detected):
            x_, y_ = self.plot_values(run.values, f'Run {i}', ax, reduce_offset=False)
            #x_ = x_.values
            #TODO
            if len(x_) > self._params[NUMBER_OF_DETECTED_AT_ZERO_TIME]:
                t0 = x_[self._params[NUMBER_OF_DETECTED_AT_ZERO_TIME]]
                filt_x = x_[x_ <= t0 - 7]
                if len(filt_x) > 0:
                    arg_tminus7 = np.argmax(filt_x)
                    if np.abs(y_[arg_tminus7] - self._params[LAID_CURVE]["-7"]) < 0.1 * self._params[LAID_CURVE]["-7"]:
                        x.extend(list(x_))
                        y.extend(list(y_))
                        successes += 1
        logger.info(f'There are {successes} successes')
        self.add_observed_curve(ax)

        #ax.legend()
        ax.set_xlim(self.xlim_cut)
        ax.set_ylim(self.ylim_cut)
        reduction = (1 - self._params[FEAR_FACTORS][CONSTANT][LIMIT_VALUE]) * 100
        R = 2.34 * self._params[TRANSMISSION_PROBABILITIES][CONSTANT]
        reducted = (100 - reduction) * R / 100
        title = f'Prognoza diagnoz (q={self._params[DETECTION_MILD_PROBA]:.1f}, redukcja R* z {R:.2f} o {reduction:.0f}% do {reducted:.2f})'
        ax.set_title(title)

        #ax.set_title(f'Sample paths of detected cases')
        xloc = [0, 5, 10, 15, 20, 25, 28]
        dates = ['02/04/20', '07/04/20', '12/04/20', '17/04/20', '22/04/20', '27/04/20', '30/04/20']
        ax.set_ylabel('Zdiagnozowani')
        ax.set_xlabel('Data')

        ax.set_xticks(xloc)
        ax.set_xticklabels(dates, rotation=30)
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'detected_cases_pl.png'))
        plt.close(fig)
        if successes > 0:
            xy = np.vstack([x, y])
            z = scipy.stats.gaussian_kde(xy)(xy)
            fig, ax = plt.subplots()
            ax.set_title(title)

            ax.scatter(x, y, c=z, s=1, edgecolor='')
            self.add_observed_curve(ax)
            xloc = [0, -5, -10, -15, -20]
            dates = ['02/04/20', '28/03/20', '23/03/20', '18/03/20', '13/03/20']
            ax.set_ylabel('Zdiagnozowani')
            ax.set_xlabel('Data')
            ax.set_xticks(xloc)
            ax.set_xticklabels(dates, rotation=30)
            ax.set_xlim([-20, 0])
            ax.set_ylim([0, self._params[NUMBER_OF_DETECTED_AT_ZERO_TIME]*1.4])

            fig.tight_layout()
            plt.savefig(os.path.join(simulation_output_dir, 'detected_cases_density_pl.png'))
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.set_title(title)

            ax.scatter(x, y, c=z, s=1, edgecolor='')
            self.add_observed_curve(ax)

            ax.set_xlim(self.xlim_cut)
            ax.set_ylim(self.ylim_cut)
            xloc = [0, 5, 10, 15, 20, 25, 28]
            dates = ['02/04/20', '07/04/20', '12/04/20', '17/04/20', '22/04/20', '27/04/20', '30/04/20']
            ax.set_ylabel('Zdiagnozowani')
            ax.set_xlabel('Data')
            ax.set_xticks(xloc)
            ax.set_xticklabels(dates, rotation=30)
            fig.tight_layout()
            plt.savefig(os.path.join(simulation_output_dir, 'detected_cases_density_pl_cut.png'))
            plt.close(fig)
        return successes

    def test_detected_cases_no_legend(self, simulation_output_dir):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for i, run in enumerate(self._all_runs_detected):
            self.plot_values(run, f'Run {i}', ax, reduce_offset=False)
        self.add_observed_curve(ax)
        #ax.legend()
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        ax.set_title(f'Test of detected cases (detection rate: {self._params[DETECTION_MILD_PROBA]:.2f})')
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'test_detected_cases_no_legend.png'))
        plt.close(fig)

    def add_observed_curve(self, ax):
        if self._params[LAID_CURVE].items():
            laid_curve_x = np.array([float(elem) for elem in self._params[LAID_CURVE].keys()])
            if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
                if self._max_time_offset != np.inf:
                    laid_curve_x = np.array([float(elem) + self._max_time_offset for elem in self._params[LAID_CURVE].keys()])
            laid_curve_y = np.array(list(self._params[LAID_CURVE].values()))
            self.plot_values(laid_curve_x, 'Cases observed in PL', ax, yvalues=laid_curve_y, dots=True)

    def test_detected_cases_no_legend_cut(self, simulation_output_dir):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for i, run in enumerate(self._all_runs_detected):
            self.plot_values(run, f'Run {i}', ax, reduce_offset=False)
        self.add_observed_curve(ax)

        #ax.legend()
        ax.set_title(f'Test of detected cases (detection rate: {self._params[DETECTION_MILD_PROBA]:.2f})')
        ax.set_xlim(self.xlim_cut)
        ax.set_ylim(self.ylim_cut)
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'test_detected_cases_no_legend_cut.png'))
        plt.close(fig)

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

    def log_outputs(self):
        simulation_output_dir = self._save_dir()
        self.df_progression_times.to_csv(os.path.join(simulation_output_dir, 'output_df_progression_times.csv'))
        self.df_infections.to_csv(os.path.join(simulation_output_dir, 'output_df_potential_contractions.csv'))
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

        serial_interval_median = self.save_serial_interval(simulation_output_dir)
        hack = self._params[EXPERIMENT_ID]
        c = self._params[TRANSMISSION_PROBABILITIES][CONSTANT]
        c_norm = c * self._params[AVERAGE_INFECTIVITY_TIME_CONSTANT_KERNEL]
        det = self._params[DETECTION_MILD_PROBA] * 100
        reduced_r = c_norm * self.fear(CONSTANT)

        self._params[EXPERIMENT_ID] = f'{self._params[EXPERIMENT_ID]}\n(median serial interval: {serial_interval_median:.2f} days, R*: {c_norm:.3f}'
        if self._params[TURN_ON_DETECTION]:
            self._params[EXPERIMENT_ID] = f'{self._params[EXPERIMENT_ID]}, Det: {det:.1f}%)'
        else:
            self._params[EXPERIMENT_ID] = f'{self._params[EXPERIMENT_ID]})'
        if self._params[FEAR_FACTORS].get(CONSTANT, self._params[FEAR_FACTORS][DEFAULT])[FEAR_FUNCTION] != FearFunctions.FearDisabled:
            self._params[EXPERIMENT_ID] = f'{self._params[EXPERIMENT_ID]}\n reduction factor: {(1 - self.fear(CONSTANT)):.3f}, reduced R*: {reduced_r:.3f}'
        self.lancet_store_graphs(simulation_output_dir)
        self.lancet_store_bins(simulation_output_dir)
        self.store_bins(simulation_output_dir)
        #self.store_bins_pl(simulation_output_dir)
        self.store_graphs(simulation_output_dir)
        self.store_detections(simulation_output_dir)
        self.store_semilogy(simulation_output_dir)
        self.doubling_time(simulation_output_dir)
        self.lancet_icu_beds(simulation_output_dir)
        self.icu_beds(simulation_output_dir)
        self.lancet_draw_death_age_cohorts(simulation_output_dir)
        self._params[EXPERIMENT_ID] = hack

    def lancet_icu_beds(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections

        fig, ax = plt.subplots(nrows=1, ncols=1)
        cond = [k for k, v in self._expected_case_severity.items() if v == ExpectedCaseSeverity.Critical]
        critical = df_r1.loc[df_r1.index.isin(cond)]
        plus = critical.t2.values
        deceased = critical[~critical.tdeath.isna()]
        survived = critical[critical.tdeath.isna()]
        minus1 = survived.trecovery.values
        minus2 = deceased.tdeath.values
        max_time = df_r2.contraction_time.max(axis=0)
        df_plus = pd.DataFrame({'t': plus, 'd': np.ones_like(plus)})
        df_minus1 = pd.DataFrame({'t': minus1, 'd': -np.ones_like(minus1)})
        df_minus2 = pd.DataFrame({'t': minus2, 'd': -np.ones_like(minus2)})
        df = df_plus.append(df_minus1).append(df_minus2).sort_values(by='t')
        df = df[df.t <= max_time]
        if len(df) == 0:
            return
        cumv = df.d.cumsum().values
        x = df.t.values

        self.plot_values(x, yvalues=cumv, label='ICU required', ax=ax)

        largest_y = cumv.max()
        icu_availability = self._params[ICU_AVAILABILITY]

        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= max_time].sort_values()
        if len(d_cases) > 0:
            self.plot_values(d_cases, 'deceased', ax)
            largest_y = max(largest_y, len(d_cases))
        t = [0, max_time]
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            if self._max_time_offset != np.inf:
                t = [elem - self._max_time_offset for elem in t]
        ax.plot(t, [icu_availability] * 2, label=f'ICU capacity ({icu_availability})')
        cumv_filter_flag = cumv > icu_availability
        if cumv[cumv_filter_flag].any():
            critical_t = df.t.values[cumv_filter_flag].min()
            self.band_time = critical_t

            ax.plot([critical_t] * 2, [0, largest_y], label=f'Critical time {critical_t:.1f}')
        ax.legend()  # 'upper left')
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'lancet_icu_beds_analysis.png'))
        plt.close(fig)

    def icu_beds(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections

        fig, ax = plt.subplots(nrows=1, ncols=1)
        cond = [k for k, v in self._expected_case_severity.items() if v == ExpectedCaseSeverity.Critical]
        critical = df_r1.loc[df_r1.index.isin(cond)]
        plus = critical.t2.values
        deceased = critical[~critical.tdeath.isna()]
        survived = critical[critical.tdeath.isna()]
        minus1 = survived.trecovery.values
        minus2 = deceased.tdeath.values
        max_time = df_r2.contraction_time.max(axis=0)
        df_plus = pd.DataFrame({'t': plus, 'd': np.ones_like(plus)})
        df_minus1 = pd.DataFrame({'t': minus1, 'd': -np.ones_like(minus1)})
        df_minus2 = pd.DataFrame({'t': minus2, 'd': -np.ones_like(minus2)})
        df = df_plus.append(df_minus1).append(df_minus2).sort_values(by='t')
        df = df[df.t <= max_time]
        if len(df) == 0:
            return
        cumv = df.d.cumsum().values
        x = df.t.values
        self.plot_values(x, yvalues=cumv, label='ICU required', ax=ax)

        largest_y = cumv.max()
        icu_availability = self._params[ICU_AVAILABILITY]

        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= max_time].sort_values()
        t = [0, max_time]
        ax.plot(t, [icu_availability] * 2, label='ICU capacity')
        cumv_filter_flag = cumv > icu_availability
        if cumv[cumv_filter_flag].any():
            critical_t = df.t.values[cumv_filter_flag].min()
            self.band_time = critical_t
            ax.plot([critical_t] * 2, [0, largest_y], label=f'Critical time {critical_t:.1f}')
        ax.legend() #'upper left')
        ax.set_title('ICU requirements\n{self._params[EXPERIMENT_ID]}')
        fig.tight_layout()
        plt.savefig(os.path.join(simulation_output_dir, 'icu_beds_analysis.png'))
        plt.close(fig)

    def update_max_time_offset(self):
        if self._params[MOVE_ZERO_TIME_ACCORDING_TO_DETECTED]:
            if self._max_time_offset == np.inf:
                if self._params[NUMBER_OF_DETECTED_AT_ZERO_TIME] <= self._detected_people:
                    self._max_time_offset = self._global_time
                    self._init_for_stats = self._active_people

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
            current_status = self.get_infection_status(person_id)
            if current_status == InfectionStatus.Healthy:
                if initiated_inf_status in active_states:
                    new_infection = False
                    # TODO below is a spaghetti code that shoud be sorted out! SORRY!
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
                    self._icu_needed -= 1
                    self._active_people -= 1
                self._infection_status[person_id] = InfectionStatus.Death.value

        elif type_ == TRECOVERY: # TRECOVERY is exclusive with regards to TDEATH (when this comment was added)
            if self.get_infection_status(person_id) not in [
                InfectionStatus.Recovered,
                InfectionStatus.Death
            ]:
                self._active_people -= 1
                if self._expected_case_severity[person_id] == ExpectedCaseSeverity.Critical:
                    self._icu_needed -= 1
                self._infection_status[person_id] = InfectionStatus.Recovered
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
            start = time.time()
            while not q.empty():
                #if self._icu_needed >= self._params[ICU_AVAILABILITY]:
                #    logging.info('icu')
                #    self.band_time = self._global_time
                #    break
                #print(f'{self._icu_needed} - {self._params[ICU_AVAILABILITY]}')
                if self.affected_people >= self.stop_simulation_threshold:
                    logging.info(f"The outbreak reached a high number {self.stop_simulation_threshold}")
                    break
                event = q.get()
                if not self.process_event(event):
                    logging.info(f"Processing event {event} returned False")
                    q.task_done()
                    break
                q.task_done()
            end = time.time()
            print("Sim runtime", end - start)
            # cleaning up priority queue:
            while not q.empty():
                q.get_nowait()
                q.task_done()
            if self._params[LOG_OUTPUTS]:
                logger.info('Log outputs')
                self.log_outputs()
            else:
                simulation_output_dir = self._save_dir()
                self.store_detections(simulation_output_dir)
            if self._icu_needed >= self._params[ICU_AVAILABILITY]:
                return True
            if self.affected_people >= self.stop_simulation_threshold:
                return True
            return False

        seeds = None
        if isinstance(self._params[RANDOM_SEED], str):
            seeds = eval(self._params[RANDOM_SEED])
        elif isinstance(self._params[RANDOM_SEED], int):
            seeds = [self._params[RANDOM_SEED]]
        runs = 0
        output_log = 'Last_processed_time;Total_#Affected;Total_#Detected;Total_#Deceased;Total_#Quarantined;'\
                     'c;c_norm;Init_#people;Prevalence_30days;Prevalence_60days;Prevalence_90days;'\
                     'Prevalence_120days;Prevalence_150days;Prevalence_180days;Band_hit_time;Subcritical;'\
                     'Prevalence_360days;runs;fear;detection_rate;increase_10;increase_20;increase_30;increase_40;'\
                     'increase_50;increase_100;increase_150;incidents_per_last_day;over_icu;hospitalized;zero_time_offset\n'
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
            c_norm = c*self._params[AVERAGE_INFECTIVITY_TIME_CONSTANT_KERNEL]
            subcritical = self._active_people < self._init_for_stats/2 # at 200 days

            bandtime = self.band_time
            #if bandtime:
            #    return 0
            prev30 = self.prevalance_at(30)
            prev60 = self.prevalance_at(60)
            prev90 = self.prevalance_at(90)
            prev120 = self.prevalance_at(120)
            prev150 = self.prevalance_at(150)
            prev180 = self.prevalance_at(180)
            prev360 = self.prevalance_at(360)
            fear_ = self.fear(CONSTANT)
            detection_rate = self._params[DETECTION_MILD_PROBA]
            affected = self.affected_people
            detected = self.detected_people
            deceased = self.deaths
            quarantined = self.quarantined_people
            mean_increase_at_10 = self.mean_day_increase_until(10)
            mean_increase_at_20 = self.mean_day_increase_until(20)
            mean_increase_at_30 = self.mean_day_increase_until(30)
            mean_increase_at_40 = self.mean_day_increase_until(40)
            mean_increase_at_50 = self.mean_day_increase_until(50)
            mean_increase_at_100 = self.mean_day_increase_until(100)
            mean_increase_at_150 = self.mean_day_increase_until(150)
            incidents_per_last_day = self.prevalance_at(self._global_time) - self.prevalance_at(self._global_time - 1)
            hospitalized = self._icu_needed
            zero_time_offset = self._max_time_offset
            output_add = f'{last_processed_time };{affected};{detected};{deceased};{quarantined};'\
                         f'{c};{c_norm};{self._init_for_stats};{prev30};{prev60};{prev90};{prev120};{prev150};{prev180};'\
                         f'{bandtime};{subcritical};{prev360};{runs};{fear_};{detection_rate};'\
                         f'{mean_increase_at_10};{mean_increase_at_20};{mean_increase_at_30};{mean_increase_at_40};'\
                         f'{mean_increase_at_50};{mean_increase_at_100};{mean_increase_at_150};{incidents_per_last_day};{outbreak};{hospitalized};{zero_time_offset}\n'
            logger.info(output_add)
            output_log = f'{output_log}{output_add}'
        logger.info(output_log)
        simulation_output_dir = self._save_dir('aggregated_results')
        output_log_file = os.path.join(simulation_output_dir, 'results.txt')
        fitting_successes = self.test_detected_cases(simulation_output_dir)
        q_ = self._params[DETECTION_MILD_PROBA]
        rstar_out = 2.34 * self._params[TRANSMISSION_PROBABILITIES][CONSTANT]
        c = self._params[FEAR_FACTORS][CONSTANT][LIMIT_VALUE]
        fitting_successes_str = f'q,rstar,c,successes\n{q_},{rstar_out},{c},{fitting_successes}\n'
        fitting_successes_log_file = os.path.join(simulation_output_dir, 'fitting_successes.txt')
        with open(fitting_successes_log_file, "w") as out_fitting:
            out_fitting.write(fitting_successes_str)
        #self.test_detected_cases_no_legend(simulation_output_dir)
        #self.test_detected_cases_no_legend_cut(simulation_output_dir)
        self.test_lognormal_prevalence(simulation_output_dir)
        self.test_lognormal_detected(simulation_output_dir)
        self.test_lognormal_severe(simulation_output_dir)
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
        self._expected_case_severity = self.draw_expected_case_severity()

        self._last_affected = None
        self.band_time = None
        self._quarantine_status = {}
        self._detection_status = {}

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
