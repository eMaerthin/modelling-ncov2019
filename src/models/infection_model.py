"""
This is mostly based on references/infection_alg.pdf
"""
import ast
from collections import defaultdict
from functools import (lru_cache, partial)
import json
import logging
import random
import time
import psutil

from git import Repo
from matplotlib import pyplot as plt
import pandas as pd
import scipy.optimize
import scipy.stats

from src.models.schemas import *
from src.models.defaults import *
from src.models.states_and_functions import *
import click

from pathlib import Path
from dotenv import find_dotenv, load_dotenv


from queue import (PriorityQueue)
q = PriorityQueue()


class InfectionModel:
    def __init__(self, params_path: str, df_individuals_path: str, df_households_path: str = '') -> None:
        self.params_path = params_path
        self.df_individuals_path = df_individuals_path
        self.df_households_path = df_households_path
        logger.debug('Loading params...')
        self._params = dict()
        with open(params_path, 'r') as params_file:
            params = json.loads(
                params_file.read()
            )  # TODO: check whether this should be moved to different place
        logger.debug('Parsing params...')
        for key, schema in infection_model_schemas.items():
            self._params[key] = schema.validate(params.get(key, defaults[key]))
        default_household_input_path = os.path.join(self._params[OUTPUT_ROOT_DIR], self._params[EXPERIMENT_ID],
                                                    'input_df_households.csv')  # TODO: ensure households are valid!
        if df_households_path == '':
            self.df_households_path = default_household_input_path
        self._global_time = None
        self._max_time = None
        self._expected_case_severity = None
        self._df_individuals = None
        self._df_households = None
        #self._individuals_gender = {}
        self._individuals_age = None
        self._individuals_household_id = {}
        self._individuals_indices = None
        self._households_capacities = {}
        self._households_inhabitants = {}

        self.event_queue = []
        self._affected_people = 0
        self._active_people = 0
        self._deaths = 0

        self._set_up_data_frames()
        self._infection_status = defaultdict(lambda: InfectionStatus.Healthy)
        self._detection_status = defaultdict(lambda: default_detection_status)
        self._expected_case_severity = self.draw_expected_case_severity()
        self._infections_dict = {}
        self._progression_times_dict = {}


    @staticmethod
    def parse_random_seed(random_seed):
        np.random.seed(random_seed)
        random.seed(random_seed)

    def _set_up_data_frames(self) -> None:
        """
        The purpose of this method is to set up two dataframes.
        One is self._df_individuals that stores features for the population
        Second is self._df_households that stores list of people idx per household
        building df_households is time consuming, therefore we try to reuse previously computed df_households
        :return:
        """
        logger.debug('Set up data frames: Reading population csv...')
        self._df_individuals = pd.read_csv(self.df_individuals_path)
        self._df_individuals.index = self._df_individuals.idx
        self._individuals_age = self._df_individuals[AGE].values
        self._individuals_household_id = self._df_individuals[HOUSEHOLD_ID].to_dict()
        self._individuals_indices = self._df_individuals.index.values

        logger.debug('Set up data frames: Building households df...')

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

    def append_event(self, event: Event) -> None:
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
        f_choice = convert_import_intensity_functions(import_intensity[FUNCTION])
        if f_choice == ImportIntensityFunctions.NoImport:
            return
        func = import_intensity_functions[f_choice]
        multiplier = import_intensity[MULTIPLIER]
        rate = import_intensity[RATE]
        cap = import_intensity[CAP]
        infectious_prob = import_intensity[INFECTIOUS]
        event_times = _generate_event_times(func=func, rate=rate, multiplier=multiplier, cap=cap)
        for event_time in event_times:
            person_id = self.df_individuals.index[np.random.randint(len(self.df_individuals))]
            t_state = TMINUS1
            if np.random.rand() < infectious_prob:
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
                    self._expected_case_severity[person_idx] = convert_expected_case_severity(
                        initial_condition[EXPECTED_CASE_SEVERITY]
                    )
                self.append_event(Event(initial_condition[CONTRACTION_TIME], person_idx, t_state, None,
                                        INITIAL_CONDITIONS, self.global_time))
        elif isinstance(initial_conditions, dict):  # schema v2
            if initial_conditions[SELECTION_ALGORITHM] == InitialConditionSelectionAlgorithms.RandomSelection.value:
                # initially all indices can be drawn
                choice_set = self._individuals_indices# self._df_individuals.index.values
                for infection_status, cardinality in initial_conditions[CARDINALITIES].items():
                    if cardinality > 0:
                        selected_rows = np.random.choice(choice_set, cardinality, replace=False)
                        # now only previously unselected indices can be drawn in next steps
                        choice_set = np.array(list(set(choice_set) - set(selected_rows)))
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
    def deaths(self):
        return self._deaths

    def draw_expected_case_severity(self):
        case_severity_dict = self.case_severity_distribution
        keys = [convert_expected_case_severity(x) for x in case_severity_dict]
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
            distribution_hist = np.array([age_induced_severity_distribution[x] for x in case_severity_dict])
            dis = scipy.stats.rv_discrete(values=(
                np.arange(len(age_induced_severity_distribution)),
                distribution_hist
            ))
            realizations = dis.rvs(size=len(self._individuals_indices[cond]))
            values = [keys[r] for r in realizations]
            df = pd.DataFrame(values, index=self._individuals_indices[cond])
            d = {**d, **df.to_dict()[0]}

        return d

    @staticmethod
    def generate_random_sample(**kwargs) -> float:
        def cached_random_gen(**kwargs):
            distribution = kwargs.get(DISTRIBUTION, default_distribution[DISTRIBUTION])
            if distribution == FROM_FILE:
                filepath = kwargs.get('filepath', None).replace('$ROOT_DIR', config.ROOT_DIR)
                Schema(lambda x: os.path.exists(x)).validate(filepath)
                array = np.load(filepath)
                approximate_distribution = kwargs.get('approximate_distribution', None)
                if approximate_distribution == LOGNORMAL:
                    shape, loc, scale = scipy.stats.lognorm.fit(array, floc=0)
                    return scipy.stats.lognorm.rvs, [shape], {'loc': loc, 'scale': scale}

                if approximate_distribution == GAMMA:
                    shape, loc, scale = scipy.stats.gamma.fit(array, floc=0)
                    return scipy.stats.gamma.rvs, [shape], {'loc': loc, 'scale': scale}

                if approximate_distribution:
                    raise NotImplementedError(f'Approximating to this distribution {approximate_distribution}'
                                              f'is not yet supported but we can quickly add it if needed')

                raise NotImplementedError(f'Currently not supporting empirical distribution'
                                          f' without approximating it')

            if distribution == LOGNORMAL:
                mean = kwargs.get('mean', 0.0)
                sigma = kwargs.get('sigma', 1.0)
                return np.random.lognormal, [], {'mean': mean, 'sigma': sigma}

            if distribution == EXPONENTIAL:
                lambda_ = kwargs.get('lambda', 1.0)
                return np.random.exponential, [], {'scale': 1/lambda_}

            if distribution == POISSON:
                lambda_ = kwargs.get('lambda', 1.0)
                return np.random.poisson, [], {'lam': lambda_}

            raise ValueError(f'Sampling from distribution {distribution} is not yet supported but we can quickly add it')
        f, args, kwargs = cached_random_gen(**kwargs)
        return f(*args, **kwargs)

    def add_potential_contractions_from_transport_kernel(self, person_id):
        pass

    def fear(self, kernel_id) -> float:
        @lru_cache(maxsize=32)
        def _internal_fear(kernel_id):
            fear_factors = self._params[FEAR_FACTORS]
            fear_factor = fear_factor_schema.validate(fear_factors.get(kernel_id, fear_factors.get(DEFAULT, None)))
            if not fear_factor:
                return 1.0
            f = fear_functions[convert_fear_functions(fear_factor[FEAR_FUNCTION])]
            limit_value = fear_factor[LIMIT_VALUE]
            scale = fear_factor[SCALE_FACTOR]
            weights_deaths = fear_factor[DEATHS_MULTIPLIER]
            weights_detected = fear_factor[DETECTED_MULTIPLIER]
            return f, weights_detected, weights_deaths, scale, limit_value
        f, weights_detected, weights_deaths, scale, limit_value = _internal_fear(kernel_id)
        detected = self.affected_people
        deaths = self.deaths
        return f(detected, deaths, weights_detected, weights_deaths, scale, limit_value)

    def gamma(self, kernel_id):
        return self._params[TRANSMISSION_PROBABILITIES][kernel_id] * self.fear(kernel_id)

    def add_potential_contractions_from_household_kernel(self, person_id):
        prog_times = self._progression_times_dict[person_id]
        start = prog_times[T0]
        end = prog_times[T2]
        if end is None:
            end = start + 14 # TODO: Fix the bug, this should Recovery Time
        total_infection_rate = (end - start) * self.gamma('household')
        household_id = self._individuals_household_id[person_id] #self._df_individuals.loc[person_id, HOUSEHOLD_ID]
        inhabitants = self._households_inhabitants[household_id] #self._df_households.loc[household_id][ID]
        possibly_affected_household_members = list(set(inhabitants) - {person_id})
        infected = np.minimum(np.random.poisson(total_infection_rate, size=1),
                              len(possibly_affected_household_members))[0]
        if infected == 0:
            return
        possible_choices = possibly_affected_household_members
        selected_rows = np.random.choice(possible_choices, infected, replace=False)
        for person_idx in selected_rows:
            if self._infection_status[person_idx] == InfectionStatus.Healthy:
                contraction_time = np.random.uniform(low=start, high=end)
                self.append_event(Event(contraction_time, person_idx, TMINUS1, person_id, HOUSEHOLD, self.global_time))

    def add_potential_contractions_from_employment_kernel(self, person_id):
        pass

    def add_potential_contractions_from_sporadic_kernel(self, person_id):
        pass

    def add_potential_contractions_from_friendship_kernel(self, person_id):
        pass

    def add_potential_contractions_from_constant_kernel(self, person_id):
        prog_times = self._progression_times_dict[person_id]
        start = prog_times[T0]
        end = prog_times[T1]
        if end is None:
            end = prog_times[T2]
        total_infection_rate = (end - start) * self.gamma('constant')
        infected = np.random.poisson(total_infection_rate, size=1)[0]
        if infected == 0:
            return
        possible_choices = self._individuals_indices # self._df_individuals.index.values
        possible_choices = possible_choices[possible_choices != person_id]
        r = range(possible_choices.shape[0])
        selected_rows_ids = random.sample(r, k=infected)
        selected_rows = possible_choices[selected_rows_ids]
        for person_idx in selected_rows:
            if self._infection_status[person_idx] == InfectionStatus.Healthy:
                contraction_time = np.random.uniform(low=start, high=end)
                self.append_event(Event(contraction_time, person_idx, TMINUS1, person_id, CONSTANT, self.global_time))

    def handle_t0(self, person_id):
        if self._infection_status[person_id] == InfectionStatus.Contraction:
            self._infection_status[person_id] = InfectionStatus.Infectious
        elif self._infection_status[person_id] != InfectionStatus.Infectious:
            raise AssertionError(f'Unexpected state detected: {self._infection_status[person_id]}'
                                 f'person_id: {person_id}')
        '''
        # TODO: we do not use those params yet:
        if P_TRANSPORT in self._df_individuals.columns:
            if self._df_individuals.loc[person_id, P_TRANSPORT] > 0:
                self.add_potential_contractions_from_transport_kernel(person_id)
        if EMPLOYMENT_STATUS in self._df_individuals.columns:
            if self._df_individuals.loc[person_id, EMPLOYMENT_STATUS] > 0:
                self.add_potential_contractions_from_employment_kernel(person_id)
        '''
        household_id = self._individuals_household_id[person_id]  # self._df_individuals.loc[person_id, HOUSEHOLD_ID]
        capacity = self._households_capacities[household_id]  # self._df_households.loc[household_id][ID]
        if capacity > 1:
            self.add_potential_contractions_from_household_kernel(person_id)
        self.add_potential_contractions_from_friendship_kernel(person_id)
        self.add_potential_contractions_from_sporadic_kernel(person_id)
        self.add_potential_contractions_from_constant_kernel(person_id)

    def generate_disease_progression(self, person_id, event_time: float,
                                     initial_infection_status: InfectionStatus) -> None:
        """Returns list of disease progression events
        "future" disease_progression should be recalculated when the disease will be recognised at the state level
        t0 - time when individual becomes infectious (Mild symptoms)
        t1 - time when individual stay home/visit doctor due to Mild/Serious? symptoms
        t2 - time when individual goes to hospital due to Serious symptoms
        """
        if initial_infection_status == InfectionStatus.Contraction:
            tminus1 = event_time
            t0 = tminus1 + self.generate_random_sample(**self.disease_progression[T0])
            self.append_event(Event(t0, person_id, T0, person_id, DISEASE_PROGRESSION, tminus1))
        elif initial_infection_status == InfectionStatus.Infectious:
            t0 = event_time
            # tminus1 does not to be defined, but for completeness let's calculate it
            tminus1 = t0 - self.generate_random_sample(**self.disease_progression[T0])
        else:
            raise ValueError(f'invalid initial infection status {initial_infection_status}')
        t2 = None
        # t3 = None
        if self._expected_case_severity[person_id] in [
            ExpectedCaseSeverity.Severe,
            ExpectedCaseSeverity.Critical
        ]:
            t2 = t0 + self.generate_random_sample(**self.disease_progression[T2])
            self.append_event(Event(t2, person_id, T2, person_id, DISEASE_PROGRESSION, t0))

        t1 = t0 + self.generate_random_sample(**self.disease_progression[T1])
        if not t2 or t1 < t2:
            self.append_event(Event(t1, person_id, T1, person_id, DISEASE_PROGRESSION, t0))
        else:
            t1 = None
        #self._df_progression_times =
        tdetection = None
        trecovery = None
        tdeath = None
        if np.random.rand() <= self._params[DEATH_PROBABILITY][self._expected_case_severity[person_id].value]:
            tdeath = t0 + self.generate_random_sample(**self.disease_progression[TDEATH])
            self.append_event(Event(tdeath, person_id, TDEATH, person_id, DISEASE_PROGRESSION, t0))
        else:
            if self._expected_case_severity[person_id] == ExpectedCaseSeverity.Mild:
                trecovery = t0 + 14
            else:
                trecovery = t0 + 42
            self.append_event(Event(trecovery, person_id, TRECOVERY, person_id, DISEASE_PROGRESSION, t0))
        self._progression_times_dict[person_id] = {ID: person_id, TMINUS1: tminus1, T0: t0, T1: t1, T2: t2,
                                                   TDEATH: tdeath, TRECOVERY: trecovery}
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

        def plot_doubling(x, window=100):
            if len(x) > window:
                xval = x[:-window]
                yval = doubling(x.values, np.arange(1, 1 + len(x)))
                plt.plot(xval[yval<28], yval[yval<28])
                return True
            return False

        df_r1 = self.df_progression_times
        df_r2 = self.df_infections
        plt.close()
        vals = df_r2.contraction_time.sort_values()
        legend = []
        if plot_doubling(vals):
            legend.append('Trend line for total # infected')
        cond1 = df_r2.contraction_time[df_r2.kernel == 'import_intensity'].sort_values()
        cond2 = df_r2.contraction_time[df_r2.kernel == 'constant'].sort_values()
        cond3 = df_r2.contraction_time[df_r2.kernel == 'household'].sort_values()
        if plot_doubling(cond1):
            legend.append('Trend line for # imported cases')
        if plot_doubling(cond2):
            legend.append('Trend line for Infected through constant kernel')
        if plot_doubling(cond3):
            legend.append('Trend line for Infected through household kernel')
        hospitalized_cases = df_r1[~df_r1.t2.isna()].sort_values(by='t2').t2
        ho_cases = hospitalized_cases[hospitalized_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        if plot_doubling(ho_cases):
            legend.append('Trend line for # hospitalized cases')
        if plot_doubling(d_cases):
            legend.append('Trend line for # deceased cases')
        plt.legend(legend, loc='upper left')
        plt.title(f'Doubling times for simulation of covid19 dynamics\n {self._params[EXPERIMENT_ID]}')
        plt.savefig(os.path.join(simulation_output_dir, 'doubling_times.png'))

    def draw_death_age_cohorts(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections
        df_in = self.df_individuals
        plt.close()
        lims = default_age_cohorts_with_descriptions
        legend = []
        for limm, limM, descr in lims:
            cond1 = df_in.age >= limm
            cond2 = df_in.age < limM
            cond = np.logical_and(cond1, cond2)
            filtered = df_r1.loc[df_r1.index.isin(df_in[cond].index)]
            death_cases = filtered[~filtered.tdeath.isna()].sort_values(by='tdeath').tdeath
            d_cases = death_cases[death_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
            d_times = np.arange(1, 1 + len(d_cases))
            plt.plot(np.append(d_cases, df_r2.contraction_time.max(axis=0)), np.append(d_times, len(d_cases)))
            legend.append(descr)

        plt.legend(legend, loc='upper left')
        experiment_id = self._params[EXPERIMENT_ID]
        plt.title(f'cumulative deceased cases per age group \n {experiment_id}')
        plt.savefig(os.path.join(simulation_output_dir, 'deceased_cases_age_analysis.png'))

    def store_bins(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections
        plt.close()
        bins = np.arange(np.minimum(730, 1 + df_r2.contraction_time.max()))
        cond1 = df_r2.contraction_time[df_r2.kernel == 'import_intensity'].sort_values()
        cond2 = df_r2.contraction_time[df_r2.kernel == 'constant'].sort_values()
        cond3 = df_r2.contraction_time[df_r2.kernel == 'household'].sort_values()
        cond1.hist(alpha=0.3, histtype='stepfilled', bins=bins)
        cond2.hist(alpha=0.3, histtype='stepfilled', bins=bins)
        cond3.hist(alpha=0.3, histtype='stepfilled', bins=bins)
        hospitalized_cases = df_r1[~df_r1.t2.isna()].sort_values(by='t2').t2
        ho_cases = hospitalized_cases[hospitalized_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        ho_cases.hist(alpha=0.7, histtype='stepfilled', bins=bins)
        d_cases.hist(alpha=0.9, histtype='stepfilled', bins=bins)
        plt.legend(['# Imported cases', 'Infected through constant kernel',
                    'Infected through household kernel', '# hospitalized cases', '# deceased cases'],
                   loc='upper left')
        plt.title(f'1 day binning of simulated covid19\n {self._params[EXPERIMENT_ID]}')
        plt.savefig(os.path.join(simulation_output_dir, 'bins.png'))

    def store_graphs(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections
        vals = df_r2.contraction_time.sort_values()
        plt.plot(vals, np.arange(1, 1 + len(vals)))
        cond1 = df_r2.contraction_time[df_r2.kernel == 'import_intensity'].sort_values()
        cond2 = df_r2.contraction_time[df_r2.kernel == 'constant'].sort_values()
        cond3 = df_r2.contraction_time[df_r2.kernel == 'household'].sort_values()
        plt.plot(cond1, np.arange(1, 1 + len(cond1)))
        plt.plot(cond2, np.arange(1, 1 + len(cond2)))
        plt.plot(cond3, np.arange(1, 1 + len(cond3)))
        hospitalized_cases = df_r1[~df_r1.t2.isna()].sort_values(by='t2').t2
        ho_cases = hospitalized_cases[hospitalized_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        plt.plot(ho_cases, np.arange(1, 1 + len(ho_cases)))
        plt.plot(d_cases, np.arange(1, 1 + len(d_cases)))
        plt.legend(['Total # infected', '# Imported cases', 'Infected through constant kernel',
                    'Infected through household kernel', '# hospitalized cases', '# deceased cases'],
                   loc='upper left')
        plt.title(f'simulation of covid19 dynamics\n {self._params[EXPERIMENT_ID]}')
        plt.savefig(os.path.join(simulation_output_dir, 'summary.png'))

    def store_semilogy(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections
        plt.close()
        vals = df_r2.contraction_time.sort_values()
        plt.semilogy(vals, np.arange(1, 1 + len(vals)))
        cond1 = df_r2.contraction_time[df_r2.kernel == 'import_intensity'].sort_values()
        cond2 = df_r2.contraction_time[df_r2.kernel == 'constant'].sort_values()
        cond3 = df_r2.contraction_time[df_r2.kernel == 'household'].sort_values()
        plt.semilogy(cond1, np.arange(1, 1 + len(cond1)))
        plt.semilogy(cond2, np.arange(1, 1 + len(cond2)))
        plt.semilogy(cond3, np.arange(1, 1 + len(cond3)))
        hospitalized_cases = df_r1[~df_r1.t2.isna()].sort_values(by='t2').t2
        ho_cases = hospitalized_cases[hospitalized_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        plt.semilogy(ho_cases, np.arange(1, 1 + len(ho_cases)))
        plt.semilogy(d_cases, np.arange(1, 1 + len(d_cases)))
        plt.legend(['Total # infected', '# Imported cases', 'Infected through constant kernel',
                    'Infected through household kernel', '# hospitalized cases', '# deceased cases'],
                    loc='upper left')
        plt.title(f'simulation of covid19 dynamics\n {self._params[EXPERIMENT_ID]}')
        plt.savefig(os.path.join(simulation_output_dir, 'summary_semilogy.png'))

    def log_outputs(self):
        run_id = f'{int(time.monotonic() * 1e9)}_{self._params[RANDOM_SEED]}'
        simulation_output_dir = os.path.join(self._params[OUTPUT_ROOT_DIR],
                                             self._params[EXPERIMENT_ID],
                                             run_id)
        os.makedirs(simulation_output_dir)
        self.df_progression_times.to_csv(os.path.join(simulation_output_dir, 'output_df_progression_times.csv'))
        self.df_infections.to_csv(os.path.join(simulation_output_dir, 'output_df_potential_contractions.csv'))
        self._df_individuals[EXPECTED_CASE_SEVERITY] = pd.Series(self._expected_case_severity)
        self._df_individuals[INFECTION_STATUS] = pd.Series(self._infection_status)
        self._df_individuals[DETECTION] = pd.Series(self._detection_status)
        self._df_individuals.to_csv(os.path.join(simulation_output_dir, 'output_df_individuals.csv'))
        self._df_households.to_csv(os.path.join(simulation_output_dir, 'output_df_households.csv'))
        if self._params[SAVE_INPUT_DATA]:
            from shutil import copyfile
            copyfile(self.df_individuals_path, os.path.join(simulation_output_dir,
                                                            f'input_{os.path.basename(self.df_individuals_path)}'))
            copyfile(self.params_path, os.path.join(simulation_output_dir,
                                                    f'input_{os.path.basename(self.params_path)}'))
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

        self.store_graphs(simulation_output_dir)
        self.store_bins(simulation_output_dir)
        self.store_semilogy(simulation_output_dir)
        self.store_event_queue(simulation_output_dir)
        self.store_expected_case_severity(simulation_output_dir)
        self.store_infection_status(simulation_output_dir)
        self.doubling_time(simulation_output_dir)
        self.icu_beds(simulation_output_dir)
        self.draw_death_age_cohorts(simulation_output_dir)

    def store_expected_case_severity(self, simulation_output_dir):
        import pickle
        with open(os.path.join(simulation_output_dir, 'save_expected_case_severity.pkl'), 'wb') as f:
            pickle.dump(self._expected_case_severity, f)

    def store_infection_status(self, simulation_output_dir):
        import pickle
        with open(os.path.join(simulation_output_dir, 'save_infection_status.pkl'), 'wb') as f:
            pickle.dump(self._infection_status, f)

    def icu_beds(self, simulation_output_dir):
        df_r1 = self.df_progression_times
        df_r2 = self.df_infections
        df_in = self.df_individuals
        plt.close()
        cond = df_in[EXPECTED_CASE_SEVERITY] == ExpectedCaseSeverity.Critical
        critical = df_r1.loc[df_r1.index.isin(df_in[cond].index)]
        plus = critical.t2.values
        deceased = critical[~critical.tdeath.isna()]
        survived = critical[critical.tdeath.isna()]
        minus1 = survived.t2.values + FOUR_WEEKS
        minus2 = deceased.tdeath.values
        df = pd.DataFrame({'t': plus, 'd': np.ones_like(plus)}).append(
            pd.DataFrame({'t': minus1, 'd': -np.ones_like(minus1)})).append(
            pd.DataFrame({'t': minus2, 'd': -np.ones_like(minus2)})
        ).sort_values(by='t')
        df = df[df.t <= df_r2.contraction_time.max(axis=0)]
        cumv = df.d.cumsum().values
        plt.plot(df.t.values, cumv)
        death_cases = df_r1[~df_r1.tdeath.isna()].sort_values(by='tdeath').tdeath
        d_cases = death_cases[death_cases <= df_r2.contraction_time.max(axis=0)].sort_values()
        plt.plot(d_cases, np.arange(1, 1 + len(d_cases)))
        icu_availability = self._params[ICU_AVAILABILITY]
        plt.plot(d_cases, np.ones(len(d_cases)) * icu_availability)
        legend = ['ICU beds required', '# deceased cases', 'ICU beds available']
        if sum(cumv > icu_availability) > 0:
            critical_t = df.t.values[cumv > icu_availability].min()
            plt.plot([critical_t] * 2, [0, max(d_cases.max(), cumv.max())])
            legend.append(f'Critical time {critical_t:.1f}')
        plt.legend(legend, loc='upper left')
        plt.title(f'ICU beds needed assuming 4 weeks for recovery \n {self._params[EXPERIMENT_ID]}')
        plt.savefig(os.path.join(simulation_output_dir, 'icu_beds_analysis.png'))

    def store_event_queue(self, simulation_output_dir):
        import pickle
        with open(os.path.join(simulation_output_dir, 'save_state_event_queue.pkl'), 'wb') as f:
            pickle.dump(self.event_queue, f)

    def add_new_infection(self, person_id, infection_status,
                          initiated_by, initiated_through):
        self._infection_status[person_id] = infection_status
        self._detection_status[person_id] = DetectionStatus.NotDetected

        self._infections_dict[len(self._infections_dict)] = {
            SOURCE: initiated_by,
            TARGET: person_id,
            CONTRACTION_TIME: self.global_time,
            KERNEL: initiated_through
        }

        self._affected_people += 1
        self._active_people += 1
        self.generate_disease_progression(person_id,
                                          self.global_time,
                                          infection_status)

    # 'Event', [TIME, PERSON_INDEX, TYPE, INITIATED_BY, INITIATED_THROUGH, ISSUED_TIME])
    def process_event(self, event) -> bool:
        type_ = getattr(event, TYPE)
        time = getattr(event, TIME)
        if int(time / self._params[LOG_TIME_FREQ]) != int(self._global_time / self._params[LOG_TIME_FREQ]):
            memory_use = ps.memory_info().rss / 1024 / 1024
            logger.debug(f'Time: {time:.2f}'
                         f'\tAffected people: {self.affected_people}'
                         f'\tActive: {self._active_people}'
                         f'\tPhysical memory use: {memory_use:.2f} MB')
        self._global_time = time
        if self._global_time > self._max_time:
            return False
        person_id = getattr(event, PERSON_INDEX)
        initiated_by = getattr(event, INITIATED_BY)
        initiated_through = getattr(event, INITIATED_THROUGH)

        # TODO the remaining attribute will be useful when we will take into account for backtracing
        # issued_time = getattr(event, ISSUED_TIME)
        if initiated_by is None and initiated_through != DISEASE_PROGRESSION:
            if type_ == TMINUS1:
                if self._infection_status[person_id] == InfectionStatus.Healthy:
                    self.add_new_infection(person_id, InfectionStatus.Contraction,
                                           initiated_by, initiated_through)
            if type_ == T0:
                if self._infection_status[person_id] in [InfectionStatus.Healthy, InfectionStatus.Contraction]:
                    self.add_new_infection(person_id, InfectionStatus.Infectious,
                                           initiated_by, initiated_through)
            return True
        if type_ == TMINUS1:
            # check if this action is still valid first
            initiated_inf_status = self._infection_status[initiated_by]
            current_status = self._infection_status[person_id]
            if current_status == InfectionStatus.Healthy:
                if initiated_inf_status in active_states:
                    if initiated_through != HOUSEHOLD:
                        if initiated_inf_status != InfectionStatus.StayHome:
                            self.add_new_infection(person_id, InfectionStatus.Contraction,
                                                   initiated_by, initiated_through)
                    else:
                        self.add_new_infection(person_id, InfectionStatus.Contraction,
                                               initiated_by, initiated_through)
        elif type_ == T0:
            self.handle_t0(person_id)
        elif type_ == T1:
            if self._infection_status[person_id] == InfectionStatus.Infectious:
                self._infection_status[person_id] = InfectionStatus.StayHome
        elif type_ == T2:
            if self._infection_status[person_id] in [
                InfectionStatus.StayHome,
                InfectionStatus.Infectious
            ]:
                self._infection_status[person_id] = InfectionStatus.Hospital
        elif type_ == TDEATH:
            if self._infection_status[person_id] != InfectionStatus.Death:
                self._infection_status[person_id] = InfectionStatus.Death
                self._deaths += 1
                self._active_people -= 1
        elif type_ == TRECOVERY:
            if self._infection_status[person_id] != InfectionStatus.Recovered:
                self._active_people -= 1
                self._infection_status[person_id] = InfectionStatus.Recovered

        # TODO: add more important logic

        if self._active_people == 0:
            return False
        return True

    def run_simulation(self):
        def _inner_loop(iter):
            while True:
                if self.affected_people >= self.stop_simulation_threshold:
                    logging.debug(f"The outbreak reached a high number {self.stop_simulation_threshold}")
                    break
                event = q.get()
                if not self.process_event(event):
                    logging.debug(f"Processing event {event} returned False")
                    q.task_done()
                    break
                q.task_done()
            # cleaning up priority queue:
            while not q.empty():
                q.get_nowait()
                q.task_done()
            if self._params[LOG_OUTPUTS]:
                logger.info('Log outputs')
                self.log_outputs()
            if self.affected_people >= self.stop_simulation_threshold:
                return True
            return False

        seeds = None
        if isinstance(self._params[RANDOM_SEED], str):
            seeds = eval(self._params[RANDOM_SEED])
        elif isinstance(self._params[RANDOM_SEED], int):
            seeds = [self._params[RANDOM_SEED]]
        outbreak_proba = 0.0
        mean_time_when_no_outbreak = 0.0
        mean_affected_when_no_outbreak = 0.0
        no_outbreaks = 0
        for i, seed in enumerate(seeds):
            self.parse_random_seed(seed)
            self.setup_simulation()
            logger.debug('Filling queue based on initial conditions...')
            self._fill_queue_based_on_initial_conditions()
            logger.debug('Filling queue based on auxiliary functions...')
            self._fill_queue_based_on_auxiliary_functions()
            logger.debug('Initialization step is done!')
            outbreak = _inner_loop(i + 1)
            outbreak_proba = (i * outbreak_proba + outbreak) / (i + 1)
            if not outbreak:
                mean_time_when_no_outbreak = (mean_time_when_no_outbreak * no_outbreaks + self._global_time) / (
                            no_outbreaks + 1)
                mean_affected_when_no_outbreak = (mean_affected_when_no_outbreak * no_outbreaks + self._affected_people) / ( no_outbreaks + 1)
                no_outbreaks += 1
        c = self._params[TRANSMISSION_PROBABILITIES][CONSTANT]
        c_norm = c/0.376
        init_people = self._params[INITIAL_CONDITIONS][CARDINALITIES].get(CONTRACTION, 0) + self._params[INITIAL_CONDITIONS][CARDINALITIES].get(INFECTIOUS, 0)
        output_log = f'\nMean Time\tMean #Affected\tWins freq.\tc\tc_norm\tInit #people\n{mean_time_when_no_outbreak}\t{mean_affected_when_no_outbreak}\t{outbreak_proba}\t{c}\t{c_norm}\t{init_people}'
        logger.info(output_log)
        run_id = f'{int(time.monotonic() * 1e9)}_{self._params[RANDOM_SEED]}'
        simulation_output_dir = os.path.join(self._params[OUTPUT_ROOT_DIR],
                                             self._params[EXPERIMENT_ID],
                                             run_id)
        os.makedirs(simulation_output_dir)
        output_log_file = os.path.join(simulation_output_dir, 'results.txt')
        with open(output_log_file, "w") as out:
            out.write(output_log)




    def setup_simulation(self):
        self.event_queue = []

        self._affected_people = 0
        self._active_people = 0
        self._deaths = 0
        # TODO add more online stats like: self._active_people = 0
        # TODO  and think how to better group them, ie namedtuple state_stats?

        self._fear_factor = {}
        self._infection_status = defaultdict(lambda: InfectionStatus.Healthy)
        self._infections_dict = {}
        self._progression_times_dict = {}

        self._global_time = self._params[START_TIME]
        self._max_time = self._params[MAX_TIME]
        self._expected_case_severity = self.draw_expected_case_severity()


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
