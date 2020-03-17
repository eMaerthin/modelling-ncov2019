from .enums import *

default_fear_function = FearFunctions.FearDisabled.value

default_initial_conditions = ([{
    PERSON_INDEX: 0,
    CONTRACTION_TIME: 0,
    INFECTION_STATUS: InfectionStatus.Contraction
}])

default_stop_simulation_threshold = 10000

default_detection_status = DetectionStatus.NotDetected.value

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
    FEAR_FUNCTION: default_fear_function,
    SCALE_FACTOR: 10,
    LIMIT_VALUE: 0.5,
    DETECTED_MULTIPLIER: 0,
    DEATHS_MULTIPLIER: 1
}

default_fear_factors = {DEFAULT: default_fear_factor}

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

default_death_probability = {
    ASYMPTOMATIC: 0.0,
    MILD: 0.0,
    SEVERE: 0.0,
    CRITICAL: 0.49
}

default_random_seed = 42

default_max_time = float('inf')

default_icu_availability = 100
default_hospital_beds_availability = 5000
default_med_personnel_availability = 400

default_log_time_freq = 1.0

defaults = {
    INITIAL_CONDITIONS: default_initial_conditions,
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
    DEATH_PROBABILITY: default_death_probability,
    RANDOM_SEED: default_random_seed,
    MAX_TIME: default_max_time,
    ICU_AVAILABILITY: default_icu_availability,
    HOSPITAL_BEDS_AVAILABILITY: default_hospital_beds_availability,
    MED_PERSONNEL_AVAILABILITY: default_med_personnel_availability,
    LOG_TIME_FREQ: default_log_time_freq
}

default_age_induced_fatality_rates = [(0, 20, 0.002), (20, 40, 0.002), (40, 50, 0.004), (50, 60, 0.013),
                                      (60, 70, 0.036), (70, 80, 0.08), (80, 200, 0.148)]

default_age_cohorts_with_descriptions = [(0, 20, '[0-19]'), (20, 40, '[20-39]'), (40, 50, '[40-49]'),
                                         (50, 60, '[50-59]'), (60, 70, '[60-69]'), (70, 80, '[70-79]'),
                                         (80, 200, '[80+]')]