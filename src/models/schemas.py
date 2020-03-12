
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

fear_factor_schema = Schema({
    Optional(FEAR_FUNCTION, default=default_fear_factor[FEAR_FUNCTION]): Or(*FearFunctions.map()),
    Optional(SCALE_FACTOR, default=default_fear_factor[SCALE_FACTOR]): And(Or(Use(float), Use(int)), lambda x: x > 0),
    Optional(LIMIT_VALUE, default=default_fear_factor[LIMIT_VALUE]): And(Or(Use(float), Use(int)), lambda x: 1.0 >= x >= 0.0),
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
    Optional(INFECTIOUS, default=0.0): And(float, lambda x: 0.0 <= x <= 1.0),
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
    ASYMPTOMATIC: And(Use(float), lambda x: 0.0 <= x <= 1.0),
    MILD: And(Use(float), lambda x: 0.0 <= x <= 1.0),
    SEVERE: And(Use(float), lambda x: 0.0 <= x <= 1.0),
    CRITICAL: And(Use(float), lambda x: 0.0 <= x <= 1.0),
})

random_seed_schema = Schema(Or(int, None))

infection_model_schemas = {
    INITIAL_CONDITIONS: Schema(Or(initial_conditions_schema1, initial_conditions_schema2)),
    EPIDEMIC_STATUS: Schema(Or(*EpidemicStatus.map())),
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
        Optional(transmission_way, default=1.0): And(Use(float), lambda f: 0 <= f <= 1) for transmission_way in KernelType.map()
    }),
    FEAR_FACTORS: fear_factors_schema,
    IMPORT_INTENSITY: import_intensity_schema,
    START_TIME: global_time_schema,
    DEATH_PROBABILITY: death_probability_schema,
    RANDOM_SEED: random_seed_schema
}


def event_schema_fun(df_individuals):
    return Schema({
        TIME: And(Or(float,
                     int), lambda x: x > 0),
        PERSON_INDEX: And(int, lambda x: x in df_individuals.id.values),
        TYPE: Or(*DiseaseProgressionEvents.map(),
                 *StateDependentOnTheEpidemicsState.map()),
        INITIATED_BY: Or(None,
                         And(int, lambda x: x in df_individuals.id.values)),
        ISSUED_TIME: Or(None,
                        And(Or(float, int), lambda x: x >= 0)),
        EPIDEMIC_STATUS: Or(*EpidemicStatus.map()),
    })