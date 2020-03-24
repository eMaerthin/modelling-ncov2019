from functools import (partial)
import json
import logging
import random
import sys
import time
import numpy as np
from pathlib import Path
import psutil
from collections import defaultdict

import scipy.optimize
import scipy.stats

from src.read_csv import read_pop_exp_csv, read_households_csv
from src.df_like_ops import get_household2inhabitants

from src.models.schemas import *
from src.models.defaults import *
from src.models.states_and_functions import *

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
            params = json.loads(params_file.read())  # TODO: check whether this should be moved to different place
        
        logger.info('Parsing params...')
        for key, schema in infection_model_schemas.items():
            self._params[key] = schema.validate(params.get(key, defaults[key]))
        default_household_input_path = os.path.join(self._params[OUTPUT_ROOT_DIR], self._params[EXPERIMENT_ID],
                                                    'input_df_households.csv')  # TODO: ensure households are valid!
        if df_households_path == '':
            self.df_households_path = default_household_input_path
        self.global_time = None
        self._max_time = None
        self._expected_case_severity = None

        logger.info('Set up individuals.')
        self._individuals = read_pop_exp_csv(self.df_individuals_path)
        self._individuals_age = np.array(self._individuals[AGE])
        self._individuals_indices = np.array(self._individuals[ID])
        self._individuals_household_id = dict(zip(self._individuals_indices, self._individuals[HOUSEHOLD_ID]))
        
        logger.info('Set up households.')
        if os.path.exists(self.df_households_path):
            self._households_inhabitants = read_households_csv(self.df_households_path)
        else:
            self._households_inhabitants = get_household2inhabitants(self._individuals[HOUSEHOLD_ID], self._individuals[ID])    
        self._households_capacities = {k: len(v) for k,v in self._households_inhabitants.items()}

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

        self.fear_fun = dict()
        self.fear_weights_detected = dict()
        self.fear_weights_deaths = dict()
        self.fear_scale = dict()
        self.fear_limit_value = dict()

        self.serial_intervals = []
        self.experimental_ub = None
        self.experimental_lb = None


    @staticmethod
    def parse_random_seed(random_seed):
        np.random.seed(random_seed)
        random.seed(random_seed)

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
            person_id = self._individuals_indices[np.random.randint(len(self._individuals_indices))]
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
                    self._expected_case_severity[person_idx] = initial_condition[EXPECTED_CASE_SEVERITY]
                self.append_event(Event(initial_condition[CONTRACTION_TIME], person_idx, t_state, None,
                                        INITIAL_CONDITIONS, self.global_time))
        elif isinstance(initial_conditions, dict):  # schema v2
            if initial_conditions[SELECTION_ALGORITHM] == InitialConditionSelectionAlgorithms.RandomSelection.value:
                # initially all indices can be drawn
                choice_set = self._individuals_indices
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
    def stop_simulation_threshold(self):
        return self._params[STOP_SIMULATION_THRESHOLD]

    @property
    def case_severity_distribution(self):
        return self._params[CASE_SEVERITY_DISTRIBUTION]

    @property
    def disease_progression(self):
        return self._params[DISEASE_PROGRESSION][DEFAULT]

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
            distribution_hist = np.array([age_induced_severity_distribution[x] for x in case_severity_dict])
            dis = scipy.stats.rv_discrete(values=(
                np.arange(len(age_induced_severity_distribution)),
                distribution_hist
            ))
            print(np.arange(len(age_induced_severity_distribution)))
            print(distribution_hist)

            realizations = dis.rvs(size=len(self._individuals_indices[cond]))
            # list(MH.sample_with_replacement(self.distribution_hist, len(self._individuals_indices[cond])))
            print(realizations)
            print(len(realizations))
            print()

            for indiv, realization in zip(self._individuals_indices[cond], realizations):
                d[indiv] = keys[realization]
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
                return scipy.stats.lognorm.rvs, [shape], {'loc':loc, 'scale':scale}

            if approximate_distribution == GAMMA:
                shape, loc, scale = scipy.stats.gamma.fit(array, floc=0)
                return scipy.stats.gamma.rvs, [shape], {'loc':loc, 'scale':scale}

            if approximate_distribution:
                raise NotImplementedError(f'Approximating to this distribution {approximate_distribution}'
                                          f'is not yet supported but we can quickly add it if needed')

            raise NotImplementedError(f'Currently not supporting empirical distribution'
                                      f' without approximating it')

        if distribution == LOGNORMAL:
            mean = params.get('mean', 0.0)
            sigma = params.get('sigma', 1.0)
            return np.random.lognormal, [], {'mean':mean, 'sigma':sigma}

        if distribution == EXPONENTIAL:
            lambda_ = params.get('lambda', 1.0)
            return np.random.exponential, [], {'scale':1/lambda_}

        if distribution == POISSON:
            lambda_ = params.get('lambda', 1.0)
            return np.random.poisson, [], {'lam':lambda_}

        raise ValueError(f'Sampling from distribution {distribution} is not yet supported but we can quickly add it')

    def add_potential_contractions_from_transport_kernel(self, person_id):
        pass

    def set_up_internal_fear(self, kernel_id):
        fear_factors = self._params[FEAR_FACTORS]
        fear_factor = fear_factor_schema.validate(fear_factors.get(kernel_id, fear_factors.get(DEFAULT, None)))
        if not fear_factor:
            return 1.0
        f = fear_functions[FearFunctions(fear_factor[FEAR_FUNCTION])]
        limit_value = fear_factor[LIMIT_VALUE]
        scale = fear_factor[SCALE_FACTOR]
        weights_deaths = fear_factor[DEATHS_MULTIPLIER]
        weights_detected = fear_factor[DETECTED_MULTIPLIER]
        return f, weights_detected, weights_deaths, scale, limit_value

    def fear(self, kernel_id) -> float:
        if kernel_id not in self.fear_fun:
            (self.fear_fun[kernel_id], self.fear_weights_detected[kernel_id],
             self.fear_weights_deaths[kernel_id], self.fear_scale[kernel_id],
             self.fear_limit_value[kernel_id]) = self.set_up_internal_fear(kernel_id)
        detected = self.affected_people
        deaths = self.deaths
        return self.fear_fun[kernel_id](detected, deaths, self.fear_weights_detected[kernel_id],
                                        self.fear_weights_deaths[kernel_id], self.fear_scale[kernel_id],
                                        self.fear_limit_value[kernel_id])

    def gamma(self, kernel_id):
        return self._params[TRANSMISSION_PROBABILITIES][kernel_id] * self.fear(kernel_id)

    def add_potential_contractions_from_household_kernel(self, person_id):
        prog_times = self._progression_times_dict[person_id]
        start = prog_times[T0]
        end = prog_times[T2] or prog_times[TRECOVERY] # sometimes T2 is not defined (MILD cases)
        total_infection_rate = (end - start) * self.gamma('household')
        household_id = self._individuals_household_id[person_id]
        inhabitants = self._households_inhabitants[household_id]
        possible_choices = list(set(inhabitants) - {person_id})
        infected = np.random.poisson(total_infection_rate, size=1)[0]
        if infected == 0:
            return
        #selected_rows = set(np.random.choice(possible_choices, infected, replace=True))
        selected_rows = set(random.choices(possible_choices, k=infected))
        for person_idx in selected_rows:
            if self.infection_status[person_idx] == InfectionStatus.Healthy:
                contraction_time = np.random.uniform(low=start, high=end)
                self.append_event(Event(contraction_time, person_idx, TMINUS1, person_id, HOUSEHOLD, self.global_time))

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
        possible_choices = self._individuals_indices
        possible_choices = possible_choices[possible_choices != person_id]
        r = range(possible_choices.shape[0])
        selected_rows_ids = random.sample(r, k=infected)
        selected_rows = possible_choices[selected_rows_ids]
        for person_idx in selected_rows:
            if self.infection_status[person_idx] == InfectionStatus.Healthy:
                contraction_time = np.random.uniform(low=start, high=end)
                self.append_event(Event(contraction_time, person_idx, TMINUS1, person_id, CONSTANT, self.global_time))

    def handle_t0(self, person_id):
        self.active_people += 1
        if self.infection_status[person_id] in [
            InfectionStatus.Healthy,
            InfectionStatus.Contraction
        ]:
            self.infection_status[person_id] = InfectionStatus.Infectious.value
        else:
            raise AssertionError(f'Unexpected state detected: {self.infection_status[person_id]}'
                                 f'person_id: {person_id}')
        household_id = self._individuals_household_id[person_id]
        capacity = self._households_capacities[household_id]
        if capacity > 1:
            self.add_potential_contractions_from_household_kernel(person_id)
        self.add_potential_contractions_from_constant_kernel(person_id)

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
            self.infection_status[person_id] = initial_infection_status
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
        if np.random.rand() <= self._params[DEATH_PROBABILITY][self._expected_case_severity[person_id]]:
            tdeath = t0 + self.rv_tdeath()
            self.append_event(Event(tdeath, person_id, TDEATH, person_id, DISEASE_PROGRESSION, t0))
        else:
            if self._expected_case_severity[person_id] in [
                ExpectedCaseSeverity.Mild,
                ExpectedCaseSeverity.Asymptomatic
            ]:
                trecovery = t0 + 14  # TODO: this should not be hardcoded!
            else:
                trecovery = t0 + 42
            self.append_event(Event(trecovery, person_id, TRECOVERY, person_id, DISEASE_PROGRESSION, t0))

        """ Following is for checking whther tdetection should be picked up"""
        calculate_tdetection = self._params[TURN_ON_DETECTION]
        if self._expected_case_severity[person_id] in [
            ExpectedCaseSeverity.Mild,
            ExpectedCaseSeverity.Asymptomatic
        ]:
            if np.random.uniform() > self._params[DETECTION_MILD_PROBA]:  # TODO: this should not be hardcoded!
                calculate_tdetection = False
        if calculate_tdetection:
            """ If t2 is defined (severe/critical), then use this time; if not; use some offset from t0 """
            tdetection = t2 or t0 + 2  # TODO: this should not be hardcoded
            self.append_event(Event(tdetection, person_id, TDETECTION, person_id, DETECTION, t0))

        self._progression_times_dict[person_id] = {ID: person_id, TMINUS1: tminus1, T0: t0, T1: t1, T2: t2,
                                                   TDEATH: tdeath, TRECOVERY: trecovery, TDETECTION: tdetection}

        if initial_infection_status == InfectionStatus.Infectious:
            self.handle_t0(person_id)


    def add_new_infection(self, person_id, infection_status,
                          initiated_by, initiated_through):
        self.detection_status[person_id] = DetectionStatus.NotDetected.value

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

        self.affected_people += 1
        self.generate_disease_progression(person_id,
                                          self.global_time,
                                          infection_status)

    # 'Event', [TIME, PERSON_INDEX, TYPE, INITIATED_BY, INITIATED_THROUGH, ISSUED_TIME])
    def process_event(self, event) -> bool:
        type_ = getattr(event, TYPE)
        time = getattr(event, TIME)
        if int(time / self._params[LOG_TIME_FREQ]) != int(self.global_time / self._params[LOG_TIME_FREQ]):
            memory_use = ps.memory_info().rss / 1024 / 1024
            logger.info(f'Time: {time:.2f}'
                         f'\tAffected: {self.affected_people}'
                         f'\tDetected: {self.detected_people}'
                         f'\tQuarantined: {self.quarantined_people}'
                         f'\tActive: {self.active_people}'
                         f'\tDeaths: {self.deaths}'
                         f'\tPhysical memory use: {memory_use:.2f} MB')
        self.global_time = time
        if self.global_time > self._max_time:
            return False
        person_id = getattr(event, PERSON_INDEX)
        initiated_by = getattr(event, INITIATED_BY)
        initiated_through = getattr(event, INITIATED_THROUGH)

        # TODO the remaining attribute will be useful when we will take into account for backtracing
        # issued_time = getattr(event, ISSUED_TIME)
        if initiated_by is None and initiated_through != DISEASE_PROGRESSION:
            if self.infection_status[person_id] == InfectionStatus.Healthy:
                if type_ == TMINUS1:
                    self.add_new_infection(person_id, InfectionStatus.Contraction.value,
                                           initiated_by, initiated_through)
                elif type_ == T0:
                    self.add_new_infection(person_id, InfectionStatus.Infectious.value,
                                           initiated_by, initiated_through)
        elif type_ == TMINUS1:
            # check if this action is still valid first
            initiated_inf_status = self.infection_status[initiated_by]
            if self.infection_status[person_id] == InfectionStatus.Healthy:
                if initiated_inf_status in active_states:
                    new_infection = False
                    # TODO below is a spaghetti code that shoud be sorted out! SORRY!
                    if initiated_through != HOUSEHOLD:
                        if initiated_inf_status != InfectionStatus.StayHome:
                            new_infection = True
                        if self.quarantine_status[initiated_by] == QuarantineStatus.Quarantine:
                            new_infection = False
                        if self.quarantine_status[person_id] == QuarantineStatus.Quarantine:
                            new_infection = False
                    else:  # HOUSEHOLD kernel:
                        new_infection = True
                    if new_infection:
                        self.add_new_infection(person_id, InfectionStatus.Contraction.value,
                                               initiated_by, initiated_through)
        elif type_ == T0:
            if self.infection_status[person_id] == InfectionStatus.Contraction:
                self.handle_t0(person_id)
        elif type_ == T1:
            if self.infection_status[person_id] == InfectionStatus.Infectious:
                self.infection_status[person_id] = InfectionStatus.StayHome.value
        elif type_ == T2:
            if self.infection_status[person_id] in [
                InfectionStatus.StayHome,
                InfectionStatus.Infectious
            ]:
                self.infection_status[person_id] = InfectionStatus.Hospital.value
        elif type_ == TDEATH:
            if self.infection_status[person_id] != InfectionStatus.Death:
                self.infection_status[person_id] = InfectionStatus.Death.value
                self.deaths += 1
                self.active_people -= 1
        elif type_ == TRECOVERY: # TRECOVERY is exclusive with regards to TDEATH (when this comment was added)
            if self.infection_status[person_id] != InfectionStatus.Recovered:
                self.active_people -= 1
                self.infection_status[person_id] = InfectionStatus.Recovered
        elif type_ == TDETECTION:
            if self.infection_status[person_id] not in [
                InfectionStatus.Recovered,
                InfectionStatus.Healthy
            ]:
                if self.detection_status[person_id] == DetectionStatus.NotDetected:
                    self.detection_status[person_id] = DetectionStatus.Detected.value
                    self.detected_people += 1
                    household_id = self._individuals_household_id[person_id]
                    for inhabitant in self._households_inhabitants[household_id]:
                        if self.quarantine_status[inhabitant] == QuarantineStatus.NoQuarantine:
                            if self.infection_status[inhabitant] != InfectionStatus.Death:
                                self.quarantine_status[inhabitant] = QuarantineStatus.Quarantine.value
                                self.quarantined_people += 1
                                if inhabitant not in self._progression_times_dict:
                                    self._progression_times_dict[inhabitant] = {}
                                self._progression_times_dict[inhabitant][QUARANTINE] = self.global_time
                                if self.infection_status[inhabitant] != InfectionStatus.Healthy:
                                    # TODO: this has to be implemented better, just a temporary solution:
                                    new_detection_time = self.global_time + 2.0
                                    self._progression_times_dict[inhabitant][TDETECTION] = new_detection_time
                                    self.append_event(Event(new_detection_time, inhabitant, TDETECTION, None,
                                                            'quarantine_followed_detection', self.global_time))
        else:
            raise ValueError(f'unexpected status of event: {event}')

        return True

    def run_simulation(self):
        def _inner_loop(iter):
            while not q.empty():
                if self.affected_people >= self.stop_simulation_threshold:
                    logging.info(f"The outbreak reached a high number {self.stop_simulation_threshold}")
                    break
                event = q.get()
                if not self.process_event(event):
                    logging.info(f"Processing event {event} returned False")
                    q.task_done()
                    break
                q.task_done()
            # cleaning up priority queue:
            while not q.empty():
                q.get_nowait()
                q.task_done()
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

        seeds = [seeds[0]]

        
        for i, seed in enumerate(seeds):
            self.parse_random_seed(seed)
            self.reset_simulation_params()
            logger.info('Filling queue based on initial conditions...')
            self._fill_queue_based_on_initial_conditions()
            logger.info('Filling queue based on auxiliary functions...')
            self._fill_queue_based_on_auxiliary_functions()
            logger.info('Initialization step is done!')
            outbreak = _inner_loop(i + 1)
            outbreak_proba = (i * outbreak_proba + outbreak) / (i + 1)
            if not outbreak:
                mean_time_when_no_outbreak = (mean_time_when_no_outbreak * no_outbreaks + self.global_time) / (
                            no_outbreaks + 1)
                mean_affected_when_no_outbreak = (mean_affected_when_no_outbreak * no_outbreaks + self.affected_people) / ( no_outbreaks + 1)
                no_outbreaks += 1

    def reset_simulation_params(self):
        # TODO  and think how to better group them, ie namedtuple state_stats?
        self.affected_people = 0 
        self.active_people = 0
        self.detected_people = 0
        self.quarantined_people = 0
        self.deaths = 0

        self.detection_status  = defaultdict(lambda:default_detection_status)
        self.quarantine_status = defaultdict(lambda:default_quarantine_status)
        self.infection_status  = defaultdict(lambda: InfectionStatus.Healthy.value)

        self._fear_factor = {}
        self._infections_dict = {}
        self._progression_times_dict = {}

        self.global_time = self._params[START_TIME]
        self._max_time = self._params[MAX_TIME]
        self._expected_case_severity = self.draw_expected_case_severity()


logger = logging.getLogger(__name__)
# TODO: think about separate thread/proc:ess to generate random numbers, facilitate sampling
if __name__ == '__main__':

    import getpass
    proj = {'matteo': "/home/matteo/Projects/corona/modelling-ncov2019",
            'cov': "/home/cov/git/modelling-ncov2019"}[getpass.getuser()]
    proj = Path(proj)
    params_path = proj/"test/models/assets/params_experiment0.json"
    df_individuals_path = proj/"data/vroclav/population_experiment0.csv"
    df_households_path = proj/"data/vroclav/households_experiment0.csv"

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    pid = os.getpid()
    ps = psutil.Process(pid)
    im = InfectionModel(params_path=params_path,
                        df_individuals_path=df_individuals_path,
                        df_households_path=df_households_path or '')
    im.run_simulation()
    # from pprint import pprint
    # pprint(im._params)
