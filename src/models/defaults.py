from .enums import *
from .states_and_functions import fear_functions

default_fear_function = FearFunctions.FearDisabled.value

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

default_fear_factor = {
    APPLICABLE_TO: DEFAULT,
    FEAR_FUNCTION: default_fear_function,
    SCALE_FACTOR: 10,
    DETECTED_MULTIPLIER: 0,
    DEATHS_MULTIPLIER: 1
}

default_fear_factors = [default_fear_factor]

default_experiment_id = 'experiment'

default_transmission_probabilities = {
        transmission_way: 1.0 for transmission_way in KernelType.map()
}

default_import_intensity = {
    FUNCTION: NO_IMPORT
}

default_start_time = 0.0

default_output_root_dir = 'outputs'

default_save_input_data = True

defaults = {
    INITIAL_CONDITIONS: default_initial_conditions,
    EPIDEMIC_STATUS: default_epidemic_status,
    STOP_SIMULATION_THRESHOLD: default_stop_simulation_threshold,
    DISEASE_PROGRESSION: default_disease_progression,
    CASE_SEVERITY_DISTRIBUTION: default_case_severity_distribution,
    OUTPUT_ROOT_DIR: default_output_root_dir,
    EXPERIMENT_ID: default_experiment_id,
    SAVE_INPUT_DATA: default_save_input_data,
    TRANSMISSION_PROBABILITIES: default_transmission_probabilities,
    FEAR_FACTORS: default_fear_factors,
    IMPORT_INTENSITY: default_import_intensity,
    START_TIME: default_start_time,
}
