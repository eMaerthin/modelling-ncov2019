"""
This is mostly based on references/infection_alg.pdf
"""
import csv
from functools import lru_cache
import heapq
from io import BytesIO
import json
import logging
import math
import random
from time import time_ns
import typing

import fire
from git import Repo
import pandas as pd
import scipy.optimize
import scipy.stats
from tqdm import tqdm

from src.models.schemas import *
from src.models.defaults import *
from src.models.states_and_functions import *


class InfectionModel:

    def __init__(self, params_path: str, df_individuals_path: str) -> None:
        self.params_path = params_path
        self.df_individuals_path = df_individuals_path
        logger.info('Loading params...')
        with open(params_path, 'r') as params_file:
            params = json.loads(
                params_file.read()
            )  # TODO: check whether this should be moved to different place
        self._params = dict()
        self.event_queue = []
        self._affected_people = 0
        logger.info('Parsing params...')
        for key, schema in infection_model_schemas.items():
            self._params[key] = schema.validate(params.get(key, defaults[key]))

        np.random.seed(self._params[RANDOM_SEED])
        random.seed(self._params[RANDOM_SEED])

        self._global_time = self._params[START_TIME]
        logger.info('Setting up data frames...')
        self._df_individuals, self._df_households = self._set_up_data_frames(df_individuals_path)

        self._infections_dict = {}
        self._progression_times_dict = {}
        self.event_schema = event_schema_fun(self._df_individuals)

        logger.info('Filling queue based on initial conditions...')
        self._fill_queue_based_on_initial_conditions()
        logger.info('Filling queue based on auxiliary functions...')
        self._fill_queue_based_on_auxiliary_functions()
        logger.info('Initialization step is done!')

    def _set_up_data_frames(self, df_individuals_path: str) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info('Set up data frames: Reading population csv...')
        df_individuals = pd.read_csv(
            df_individuals_path,
            converters={
                INFECTION_STATUS: convert_infection_status,
                EXPECTED_CASE_SEVERITY: convert_expected_case_severity
            })  # TODO: Consider networkx graph instead of pandas df
        df_individuals.index = df_individuals.idx
        logger.info('Set up data frames: drawing expected case severity...')
        if EXPECTED_CASE_SEVERITY not in df_individuals:
            df_individuals[EXPECTED_CASE_SEVERITY] = ExpectedCaseSeverity.UnseenNode
            """df_individuals.apply(
                lambda x: self.draw_expected_case_severity(x),
                axis=1
            )"""

        '''df_progression_times = pd.DataFrame(columns=[
            ID,
            *DiseaseProgressionEvents.map(),
            *StateDependentOnTheEpidemicsState.map()
        ])'''
        logger.info('Set up data frames: Building households df...')
        df_households = df_individuals.groupby(HOUSEHOLD_ID)[ID].apply(list)
        return df_individuals, df_households

    def _fill_queue_based_on_auxiliary_functions(self) -> None:
        def _generate_event_times(f, rate, multiplier, cap, root_buffer=100, root_guess=0):
            root_min = root_guess - root_buffer
            root_max = root_guess + root_buffer
            time_events_ = []
            for i in range(1, 1 + cap):
                root = scipy.optimize.bisect(lambda x: f(x, rate=rate, multiplier=multiplier) - i, root_min, root_max)
                time_events_.append(root) #, f(root, rate=rate, multiplier=multiplier)))
                root_min = root
                root_max = root + root_buffer
            return time_events_

        import_intensity = self._params[IMPORT_INTENSITY]
        f_choice = convert_import_intensity_functions(import_intensity[FUNCTION])
        if f_choice == ImportIntensityFunctions.NoImport:
            return
        f = import_intensity_functions[f_choice]
        multiplier = import_intensity[MULTIPLIER]
        rate = import_intensity[RATE]
        cap = import_intensity[CAP]
        infectious_prob = import_intensity[INFECTIOUS]
        event_times = _generate_event_times(f=f, rate=rate, multiplier=multiplier, cap=cap)
        for event_time in event_times:
            person_id = self.df_individuals.index[np.random.randint(len(self.df_individuals))]
            t_state = TMINUS1
            if np.random.rand() < infectious_prob:
                t_state = T0
            self.append_event(Event(event_time, person_id, t_state,
                                    None, IMPORT_INTENSITY, self.global_time,
                                    self.epidemic_status))

    def _fill_queue_based_on_initial_conditions(self):
        def _assign_t_state(status):
            if status == CONTRACTION:
                return TMINUS1
            if status == INFECTIOUS:
                return T0
            raise ValueError(f'invalid initial infection status {status}')

        initial_conditions = self._params[INITIAL_CONDITIONS]
        if isinstance(initial_conditions, list): # schema v1
            for initial_condition in initial_conditions:
                for key, value in initial_condition.items():
                    if key == PERSON_INDEX:
                        continue
                    if key == INFECTION_STATUS:
                        continue
                    if key == CONTRACTION_TIME:
                        t_state = _assign_t_state(initial_condition[INFECTION_STATUS])
                        self.append_event(Event(value, initial_condition[PERSON_INDEX], t_state,
                                                None, INITIAL_CONDITIONS, self.global_time,
                                                self.epidemic_status))
                        continue
                    elif key == EXPECTED_CASE_SEVERITY:
                        value = convert_expected_case_severity(value)
                    self._df_individuals.loc[initial_condition[PERSON_INDEX], key] = value
        elif isinstance(initial_conditions, dict): #schema v2
            if initial_conditions[SELECTION_ALGORITHM] == SelectionAlgorithms.RandomSelection.value:
                # initially all indices can be drawn
                choice_set = self._df_individuals.index.values
                for infection_status, cardinality in initial_conditions[CARDINALITIES].items():
                    if cardinality > 0:
                        selected_rows = np.random.choice(choice_set, cardinality, replace=False)
                        t_state = _assign_t_state(infection_status)
                        for row in selected_rows:
                            self.append_event(Event(self.global_time, row, t_state, None, INITIAL_CONDITIONS,
                                                    self.global_time, self.epidemic_status))
                            self._df_individuals.loc[row, INFECTION_STATUS] = convert_infection_status(infection_status)
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
        return self._params[DISEASE_PROGRESSION].get(self.epidemic_status, self._params[DISEASE_PROGRESSION][DEFAULT])

    def affected_people(self):
        return self._affected_people

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
        selected_key = list(case_severity_dict)[dis.rvs()]
        return convert_expected_case_severity(selected_key)

    @staticmethod
    def generate_random_sample(**kwargs) -> float:
        @lru_cache(maxsize=32)
        def cached_random_gen(**kwargs):
            distribution = kwargs.get('distribution', 'poisson')
            if distribution == FROM_FILE:
                filepath = kwargs.get('filepath', None).replace('$ROOT_DIR', config.ROOT_DIR)
                Schema(lambda x: os.path.exists(x)).validate(filepath)
                array = np.load(filepath)
                approximate_distribution = kwargs.get('approximate_distribution', None)
                if approximate_distribution == LOGNORMAL:
                    shape, loc, scale = scipy.stats.lognorm.fit(array, floc=0)
                    return scipy.stats.lognorm.rvs, [shape], {'loc': loc, 'scale': scale}
                elif approximate_distribution == GAMMA:
                    shape, loc, scale = scipy.stats.gamma.fit(array, floc=0)
                    return scipy.stats.gamma.rvs, [shape], {'loc': loc, 'scale': scale}
                if approximate_distribution:
                    raise ValueError(f'Approximating to this distribution {approximate_distribution}'
                                     f'is not yet supported but we can quickly add it if needed')

            if distribution == LOGNORMAL:
                mean = kwargs.get('mean', 0.0)
                sigma = kwargs.get('sigma', 1.0)
                return np.random.lognormal, [], {'mean': mean, 'sigma': sigma}

            lambda_ = kwargs.get('lambda', 1.0)
            if distribution == 'exponential':
                return np.random.exponential, [], {'scale': 1/lambda_}
            if distribution == 'poisson':
                return np.random.poisson, [], {'lam': lambda_}
            raise ValueError(f'Sampling from distribution {distribution} is not yet supported but we can quickly add it')
        f, args, kwargs = cached_random_gen(**kwargs)
        return f(*args, **kwargs)

    def handle_t0(self, id):
        if self._df_individuals.loc[id, INFECTION_STATUS] == InfectionStatus.Contraction:
            self._df_individuals.loc[id, INFECTION_STATUS] = InfectionStatus.Infectious
        elif self._df_individuals.loc[id, INFECTION_STATUS] != InfectionStatus.Infectious:
            logger.error(f'state machine failure. was {self._df_individuals.loc[id, INFECTION_STATUS]}')
        if self._df_individuals.loc[id, P_TRANSPORT] > 0:
            self.add_potential_contractions_from_transport_kernel(id)
        if self._df_individuals.loc[id, EMPLOYMENT_STATUS] > 0:
            self.add_potential_contractions_from_employment_kernel(id)
        if len(self._df_households.loc[self._df_individuals.loc[id, HOUSEHOLD_ID]]) > 1:
            self.add_potential_contractions_from_household_kernel(id)
        self.add_potential_contractions_from_friendship_kernel(id)
        self.add_potential_contractions_from_sporadic_kernel(id)
        self.add_potential_contractions_from_constant_kernel(id)

    def generate_disease_progression(self, person_id, features, event_time: float,
                                     initial_infection_status: InfectionStatus) -> None:
        """Returns list of disease progression events
        "future" disease_progression should be recalculated when the disease will be recognised at the state level
        t0 - time when individual becomes infectious (Mild symptoms)
        t1 - time when individual stay home/visit doctor due to Mild/Serious? symptoms
        t2 - time when individual goes to hospital due to Serious symptoms
        """
        tminus1 = None
        t0 = None
        if initial_infection_status == InfectionStatus.Contraction:
            tminus1 = event_time
            t0 = tminus1 + self.generate_random_sample(**self.disease_progression[T0])
            self.append_event(Event(t0, person_id, T0, person_id, DISEASE_PROGRESSION,
                                    tminus1, self.epidemic_status))
        elif initial_infection_status == InfectionStatus.Infectious:
            t0 = event_time
        else:
            raise ValueError(f'invalid initial infection status {infection_status}')
        t2 = None
        # t3 = None
        if self._df_individuals.loc[person_id, EXPECTED_CASE_SEVERITY] in [
            ExpectedCaseSeverity.Severe,
            ExpectedCaseSeverity.Critical
        ]:
            t2 = t0 + self.generate_random_sample(**self.disease_progression[T2])
            self.append_event(Event(t2, person_id, T2, person_id, DISEASE_PROGRESSION, t0, self.epidemic_status))

        t1 = t0 + self.generate_random_sample(**self.disease_progression[T1])
        if not t2 or t1 < t2:
            self.append_event(Event(t1, person_id, T1, person_id, DISEASE_PROGRESSION, t0, self.epidemic_status))
        else:
            t1 = None
        #self._df_progression_times =
        tdetection = None
        trecovery = None
        tdeath = None
        if np.random.rand() <= self._params[DEATH_PROBABILITY][self._df_individuals.loc[person_id, EXPECTED_CASE_SEVERITY].value]:
            tdeath = t0 + self.generate_random_sample(**self.disease_progression[TDEATH])
        else:
            #trecovery =
            pass
        self._progression_times_dict[person_id] = {ID: person_id, TMINUS1: tminus1, T0: t0, T1: t1, T2: t2,
                                                   TDEATH: tdeath}
                                                   #TDETECTION: tdetection, TRECOVERY: trecovery, TDEATH: tdeath}
        if initial_infection_status == InfectionStatus.Infectious:
            self.handle_t0(person_id)
        '''.append({
            ID: person_id,
            TMINUS1: tminus1,
            T0: t0,
            T1: t1,
            T2: t2,
        }, ignore_index=True)'''
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

    def log_outputs(self):
        simulation_output_dir = os.path.join(self._params[OUTPUT_ROOT_DIR], self._params[EXPERIMENT_ID], str(time_ns()))
        os.makedirs(simulation_output_dir)
        self.df_progression_times.to_csv(os.path.join(simulation_output_dir, 'output_df_progression_times.csv'))
        self.df_infections.to_csv(os.path.join(simulation_output_dir, 'output_df_potential_contractions.csv'))
        self._df_individuals.to_csv(os.path.join(simulation_output_dir, 'output_df_individuals.csv'))
        self._df_households.to_csv(os.path.join(simulation_output_dir, 'output_df_households.csv'))
        if self._params[SAVE_INPUT_DATA]:
            from shutil import copyfile
            copyfile(self.df_individuals_path, os.path.join(simulation_output_dir,
                                                            f'input_{os.path.basename(self.df_individuals_path)}'))
            copyfile(self.params_path, os.path.join(simulation_output_dir,
                                                    f'input_{os.path.basename(self.params_path)}'))
        repo = Repo(config.ROOT_DIR)
        git_active_branch_log = os.path.join(simulation_output_dir, 'git_active_branch_log.txt')
        with open(git_active_branch_log, 'w') as f:
            f.write(f'Active branch name {repo.active_branch.name}\n')
            f.write(str(repo.active_branch.log()))

        git_status = os.path.join(simulation_output_dir, 'git_status.txt')
        with open(git_status, 'w') as f:
            f.write(repo.git.status())

    def get_current_progress(self):
        return (self._df_individuals[INFECTION_STATUS] != InfectionStatus.Healthy).sum()

    def run_simulation(self):
        with tqdm(total=None) as pbar:
            while self.pop_and_apply_event():
                affected = self.affected_people() #get_current_progress()
                pbar.set_description(f'Time: {self.global_time:.2f} - Affected: {affected}')
                if affected >= self.stop_simulation_threshold:
                    logging.info(f"The outbreak reached a high number {self.stop_simulation_threshold}")
                    break
        self.log_outputs()

    def append_event(self, event: Event) -> None:
        heapq.heappush(self.event_queue, event)

    @property
    def df_infections(self):
        return pd.DataFrame.from_dict(self._infections_dict, orient='index') #, ignore_index=True) #orient='index', columns=columns)

    @property
    def df_progression_times(self):
        return pd.DataFrame.from_dict(self._progression_times_dict, orient='index') #, ignore_index=True)

    def add_new_infection(self, person_id, infection_status,
                          initiated_by, initiated_through):
        self._df_individuals.loc[person_id, INFECTION_STATUS] = infection_status
        # https://stackoverflow.com/questions/41888080/python-efficient-way-to-add-rows-to-dataframe
        self._infections_dict[len(self._infections_dict)] = {
            SOURCE: initiated_by,
            TARGET: person_id,
            CONTRACTION_TIME: self.global_time,
            KERNEL: initiated_through
        }
        if self._df_individuals.loc[person_id, EXPECTED_CASE_SEVERITY] == ExpectedCaseSeverity.UnseenNode:
            self._df_individuals.loc[person_id, EXPECTED_CASE_SEVERITY] = self.draw_expected_case_severity(
                self._df_individuals.loc[person_id]
            )
        self._affected_people += 1
        self.generate_disease_progression(person_id,
                                          self._df_individuals.loc[person_id],
                                          self.global_time,
                                          infection_status)


    def add_potential_contractions_from_transport_kernel(self, id):
        pass

    def fear(self, kernel_id) -> float:
        return 1.0

    def gamma(self, kernel_id):
        return self._params[TRANSMISSION_PROBABILITIES][kernel_id] * self.fear(kernel_id)

    def add_potential_contractions_from_household_kernel(self, id):
        prog_times = self._progression_times_dict[id]
        start = prog_times[T0]
        end = prog_times[T2]
        if not end:
            end = prog_times[T1]
        total_infection_rate = (end - start) * self.gamma('household')
        household_id = self._df_individuals.loc[id, HOUSEHOLD_ID]

        possibly_affected_household_members = list(set(self._df_households.loc[household_id]) - {id})
        #possibly_affected_household_members = inhabitants[inhabitants != id]
        '''cond1 = self._df_individuals[HOUSEHOLD_ID] == household_id
        cond2a = self._df_individuals[INFECTION_STATUS] == termination_states[0]
        cond2b = self._df_individuals[INFECTION_STATUS] == termination_states[1]
        cond2 = ~np.logical_and(cond2a, cond2b)
        cond3 = self._df_individuals[ID] != id
        cond = np.logical_and(np.logical_and(cond1, cond2), cond3)
        possibly_affected_household_members = self._df_individuals[cond]'''
        infected = np.minimum(np.random.poisson(total_infection_rate, size=1),
                              len(possibly_affected_household_members))[0]
        if infected == 0:
            return
        possible_choices = possibly_affected_household_members #.index.values
        selected_rows = np.random.choice(possible_choices, infected, replace=False)
        for row in selected_rows:
            if self._df_individuals.loc[row, INFECTION_STATUS] == InfectionStatus.Healthy:
                contraction_time = np.random.uniform(low=start, high=end)
                person_idx = self._df_individuals.loc[row, ID]
                self.append_event(Event(contraction_time, person_idx, TMINUS1,
                                        id, HOUSEHOLD, self.global_time,
                                        self.epidemic_status))

    def add_potential_contractions_from_employment_kernel(self, id):
        pass

    def add_potential_contractions_from_sporadic_kernel(self, id):
        pass

    def add_potential_contractions_from_friendship_kernel(self, id):
        pass

    def add_potential_contractions_from_constant_kernel(self, id):
        prog_times = self._progression_times_dict[id]
        start = prog_times[T0]
        end = prog_times[T1]
        if not end:
            end = prog_times[T2]
        total_infection_rate = (end - start) * self.gamma('constant')
        infected = np.random.poisson(total_infection_rate, size=1)
        if infected == 0:
            return
        possible_choices = self._df_individuals[self._df_individuals[ID] != id].index.values
        selected_rows = np.random.choice(possible_choices, infected, replace=False)
        for row in selected_rows:
            if self._df_individuals.loc[row, INFECTION_STATUS] == InfectionStatus.Healthy:
                contraction_time = np.random.uniform(low=start, high=end)
                person_idx = self._df_individuals.loc[row, ID]
                self.append_event(Event(contraction_time, person_idx, TMINUS1,
                                        id, CONSTANT, self.global_time,
                                        self.epidemic_status))

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
            if not initiated_by and initiated_through != DISEASE_PROGRESSION:
                if type_ == TMINUS1:
                    if self._df_individuals.loc[id, INFECTION_STATUS] == InfectionStatus.Healthy:
                        self.add_new_infection(id, InfectionStatus.Contraction,
                                               initiated_by, initiated_through)
                if type_ == T0:
                    if self._df_individuals.loc[id, INFECTION_STATUS] in [InfectionStatus.Healthy,
                                                                          InfectionStatus.Contraction]:
                            self.add_new_infection(id, InfectionStatus.Infectious,
                                                   initiated_by, initiated_through)
                return True
            if type_ == TMINUS1:
                # TODO handle initial parameter induced contractions: both externally and by initial conditions
                # check if this action is still valid first

                initiated_inf_status = self._df_individuals.loc[initiated_by, INFECTION_STATUS]
                if initiated_inf_status in active_states:
                    if initiated_through != 'household':
                        if initiated_inf_status != InfectionStatus.StayHome:
                            self.add_new_infection(id, InfectionStatus.Contraction,
                                                   initiated_by, initiated_through)
                    else:
                        self.add_new_infection(id, InfectionStatus.Contraction,
                                               initiated_by, initiated_through)
            elif type_ == T0:
                self.handle_t0(id)
            elif type_ == T1:
                if self._df_individuals.loc[id, INFECTION_STATUS] == InfectionStatus.Infectious:
                    self._df_individuals.loc[id, INFECTION_STATUS] = InfectionStatus.StayHome
            elif type_ == T2:
                if self._df_individuals.loc[id, INFECTION_STATUS] in [
                    InfectionStatus.StayHome,
                    InfectionStatus.Infectious
                ]:
                    self._df_individuals.loc[id, INFECTION_STATUS] = InfectionStatus.Hospital

            # TODO: add more important logic
            return True
        except IndexError as valerr:
            logger.info(f'Oops IndexError error: {valerr}')
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