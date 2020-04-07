
import os

from schema import (Schema, And, Use, Or, Optional)

from .enums import *
from .defaults import default_fear_factor
import config

# TODO ensure those distributions are really supported


distribution_schema = Schema({
    DISTRIBUTION: Or(*SupportedDistributions.map()),
    Optional(LAMBDA): And(Or(Use(float), Use(int)), lambda n: n > 0),
    Optional(FILEPATH): (lambda x: os.path.exists(x.replace('$ROOT_DIR', config.ROOT_DIR))),
    Optional(APPROXIMATE_DISTRIBUTION, default=None): Or(None, *SupportedDistributions.map())
})

disease_times_distributions_schema = Schema({
    T0: distribution_schema,
    T1: distribution_schema,
    T2: distribution_schema,
    TDEATH: distribution_schema
})

case_severity_distribution_schema = Schema(And({
    ASYMPTOMATIC: And(Use(float), lambda x: x >= 0),
    MILD: And(Use(float), lambda x: x >= 0),
    SEVERE: And(Use(float), lambda x: x >= 0),
    CRITICAL: And(Use(float), lambda x: x >= 0),
}, lambda x: sum(x.values()) == 1))

def probability_condition(x):
    return 0.0 <= x <= 1.0

fear_factor_schema = Schema({
    Optional(FEAR_FUNCTION, default=default_fear_factor[FEAR_FUNCTION]): Or(*FearFunctions.map()),
    Optional(SCALE_FACTOR, default=default_fear_factor[SCALE_FACTOR]): And(Or(Use(float), Use(int)), lambda x: x > 0),
    Optional(LOC_FACTOR, default=default_fear_factor[LOC_FACTOR]): Or(Use(float), Use(int)),
    Optional(LIMIT_VALUE, default=default_fear_factor[LIMIT_VALUE]): And(Or(Use(float), Use(int)), probability_condition),
    Optional(DETECTED_MULTIPLIER, default=default_fear_factor[DETECTED_MULTIPLIER]): And(Or(Use(float), Use(int)), lambda x: x >= 0),
    Optional(DEATHS_MULTIPLIER, default=default_fear_factor[DEATHS_MULTIPLIER]): And(Or(Use(float), Use(int)), lambda x: x >= 0),
})

fear_factors_schema = Schema({
    Optional(kernel_id): fear_factor_schema for kernel_id in [*KernelType.map(), DEFAULT]
})

import_intensity_schema = Schema({
    FUNCTION: Or(*ImportIntensityFunctions.map()),
    Optional(MULTIPLIER, default=1): And(Or(int, float), lambda x: x >= 0),
    Optional(RATE, default=1): And(Or(int, float), lambda x: x >= 0),
    Optional(CAP, default=float('inf')): And(Or(int, float), lambda x: x >= 0),
    Optional(INFECTIOUS, default=0.0): And(float, probability_condition),
})

initial_conditions_schema1 = [{
    PERSON_INDEX: int,
    CONTRACTION_TIME: Or(float, int),
    INFECTION_STATUS: Or(CONTRACTION, INFECTIOUS),
    Optional(EXPECTED_CASE_SEVERITY): Or(*ExpectedCaseSeverity.map())
}]

initial_conditions_schema2 = {
    SELECTION_ALGORITHM: Or(*InitialConditionSelectionAlgorithms.map()),
    CARDINALITIES: {Optional(k, default=0): And(Use(int), lambda n: n >= 0) for k in [
        CONTRACTION, INFECTIOUS
    ]}
}

global_time_schema = Schema(And(Or(int, float), lambda x: x >= 0))

death_probability_schema = Schema({
    ASYMPTOMATIC: And(Use(float), probability_condition),
    MILD: And(Use(float), probability_condition),
    SEVERE: And(Use(float), probability_condition),
    CRITICAL: And(Use(float), probability_condition),
})

random_seed_schema = Schema(Or(str, int, None))
availability_schema = Schema(And(int, lambda x: x > 0))
log_time_freq_schema = Schema(Or(And(float, lambda x: x > 0), And(int, lambda x: x > 0), None))

serial_interval_schema = Schema(And({
    MIN_TIME: And(Use(float), lambda x: 0.0 <= x),
    MAX_TIME: And(Use(float), lambda x: 0.0 <= x)
}, lambda x: x[MIN_TIME] <= x[MAX_TIME]))

detection_mild_proba_schema = Schema(And(Or(int, float), probability_condition))

r_out_schedule_schema = Schema([{
    MIN_TIME: And(Use(float), lambda x: 0.0 <= x),
    MAX_TIME: And(Use(float), lambda x: 0.0 <= x),
    OVERRIDE_R_FRACTION: And(Use(float), lambda x: 0.0 <= x <= 1.0)}])

infection_model_schemas = {
    INITIAL_CONDITIONS: Schema(Or(initial_conditions_schema1, initial_conditions_schema2)),
    STOP_SIMULATION_THRESHOLD: Schema(And(Use(int), lambda n: n > 0)),
    DISEASE_PROGRESSION: Schema(And({
        Optional(NOT_DETECTED): disease_times_distributions_schema,
        Optional(DETECTED): disease_times_distributions_schema,
        Optional(DEFAULT): disease_times_distributions_schema
    }, lambda d: len(d) > 0)),
    CASE_SEVERITY_DISTRIBUTION: case_severity_distribution_schema,
    OUTPUT_ROOT_DIR: Schema(str),
    EXPERIMENT_ID: Schema(And(Use(str), lambda str_: str_.isalnum)),
    SAVE_INPUT_DATA: Schema(bool),
    TRANSMISSION_PROBABILITIES: Schema({
        Optional(transmission_way, default=0.0): And(Use(float), lambda f: 0 <= f) for transmission_way in KernelType.map()
    }),
    FEAR_FACTORS: fear_factors_schema,
    IMPORT_INTENSITY: import_intensity_schema,
    START_TIME: global_time_schema,
    DEATH_PROBABILITY: death_probability_schema,
    RANDOM_SEED: random_seed_schema,
    MAX_TIME: global_time_schema,
    ICU_AVAILABILITY: availability_schema,
    HOSPITAL_BEDS_AVAILABILITY: availability_schema,
    MED_PERSONNEL_AVAILABILITY: availability_schema,
    LOG_TIME_FREQ: log_time_freq_schema,
    LOG_OUTPUTS: Schema(bool),
    SERIAL_INTERVAL: serial_interval_schema,
    DETECTION_MILD_PROBA: detection_mild_proba_schema,
    TURN_ON_DETECTION: Schema(bool),
    AVERAGE_INFECTIVITY_TIME_CONSTANT_KERNEL: global_time_schema,
    SAVE_EXPECTED_SEVERITY: Schema(bool),
    MOVE_ZERO_TIME_ACCORDING_TO_DETECTED: Schema(bool),
    NUMBER_OF_DETECTED_AT_ZERO_TIME: Schema(And(Use(int), lambda x: x >= 0)),
    USE_TODAY_MARK: Schema(bool),
    TODAY_OFFSET: Schema(Or(int, float)),
    LAID_CURVE: Schema(dict),
    PLOT_XLIM_CUT_LEFT: Schema(Or(int, float, None)),
    PLOT_XLIM_CUT_RIGHT: Schema(Or(int, float, None)),
    PLOT_XLIM_LEFT: Schema(Or(int, float, None)),
    PLOT_XLIM_RIGHT: Schema(Or(int, float, None)),
    PLOT_YLIM_CUT_BOTTOM: Schema(Or(int, float, None)),
    PLOT_YLIM_CUT_TOP: Schema(Or(int, float, None)),
    PLOT_YLIM_BOTTOM: Schema(Or(int, float, None)),
    PLOT_YLIM_TOP: Schema(Or(int, float, None)),
    ENABLE_VISUALIZATION: Schema(bool),
    R_OUT_SCHEDULE: r_out_schedule_schema,
    ENABLE_ADDITIONAL_LOGS: Schema(bool),
    REUSE_EXPECTED_CASE_SEVERITIES: Schema(bool),
    REUSE_TIME_DISTRIBUTION_REALIZATIONS: Schema(bool),
}
