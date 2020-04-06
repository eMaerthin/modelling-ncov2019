
# Todo: Part of below should go to separate files

CONTRACTION_TIME = 'contraction_time'
PERSON_INDEX = 'person_index'
INFECTION_STATUS = 'infection_status'
INITIAL_CONDITIONS = 'initial_conditions'
DISEASE_PROGRESSION = 'disease_progression'
NOT_DETECTED = 'not_detected'
DETECTED = 'detected'
DEFAULT = 'default'

LOG_OUTPUTS = 'log_outputs'

CONTRACTION = 'contraction'
INFECTIOUS = 'infectious'
MAX_TIME = 'max_time'
CAPACITY = 'capacity'

ICU_AVAILABILITY = 'icu_availability'
HOSPITAL_BEDS_AVAILABILITY = 'hospital_beds_availability'
MED_PERSONNEL_AVAILABILITY = 'med_personnel_availability'
LOG_TIME_FREQ = 'log_time_freq'
DISTRIBUTION = 'distribution'
LAMBDA = 'lambda'
FILEPATH = 'filepath'

PLOT_XLIM_LEFT = 'plot_xlim_left'
PLOT_XLIM_CUT_LEFT = 'plot_xlim_cut_left'
PLOT_XLIM_RIGHT = 'plot_xlim_right'
PLOT_XLIM_CUT_RIGHT = 'plot_xlim_cut_right'

PLOT_YLIM_BOTTOM = 'plot_ylim_bottom'
PLOT_YLIM_CUT_BOTTOM = 'plot_ylim_cut_bottom'
PLOT_YLIM_TOP = 'plot_ylim_top'
PLOT_YLIM_CUT_TOP = 'plot_ylim_cut_top'

EXPECTED_CASE_SEVERITY = 'expected_case_severity'
APPROXIMATE_DISTRIBUTION = 'approximate_distribution'
CASE_SEVERITY_DISTRIBUTION = 'case_severity_distribution'

START_TIME = 'start_time'
STOP_SIMULATION_THRESHOLD = 'stop_simulation_threshold'
ENABLE_VISUALIZATION = 'enable_visualization'
ID = 'idx'
HOUSEHOLD_ID = 'household_index'
SOURCE = 'source_id'
TARGET = 'target_id'
KERNEL = 'kernel'
INHABITANTS = 'inhabitants'

R_OUT_SCHEDULE = 'r_out_schedule'
OVERRIDE_R_FRACTION = 'override_r_fraction'

TMINUS1 = 'tminus1'
T0 = 't0' # contraction->infectious ACTIVE
T1 = 't1' # infectious->stay_home ACTIVE
T2 = 't2' # stay_home->hospital SEVERE/CRITICAL TODO it WAS before : stay_home->see_doctor MILD only
#T3 = 't3' # TODO let's check if we need this: # stay_home/see_doctor->hospital SEVERE/CRITICAL
TDEATH = 'tdeath'
TRECOVERY = 'trecovery'
TDETECTION = 'tdetection'

ASYMPTOMATIC = 'asymptomatic'
MILD = 'mild'
SEVERE = 'severe'
CRITICAL = 'critical'

LOGNORMAL = 'lognormal'
EXPONENTIAL = 'exponential'
POLYNOMIAL = 'polynomial'
GAMMA = 'gamma'
POISSON = 'poisson'
FROM_FILE = 'from_file'

DETECTION = 'detection'
QUARANTINE = 'quarantine'

AVERAGE_INFECTIVITY_TIME_CONSTANT_KERNEL = 'average_infectivity_time_constant_kernel'
SAVE_EXPECTED_SEVERITY = 'save_expected_severity'

TURN_ON_DETECTION = 'turn_on_detection'
DETECTION_MILD_PROBA = 'detection_mild_proba'

SELECTION_ALGORITHM = 'selection_algorithm'
CARDINALITIES = 'cardinalities'

EMPLOYMENT_STATUS = 'employment_status'
PROFESSION_INDEX = 'profession_index'
AGE = 'age'
GENDER = 'gender'
SOCIAL = 'social'
P_TRANSPORT = 'public_transport_usage'

MOVE_ZERO_TIME_ACCORDING_TO_DETECTED = 'move_zero_time_according_to_detected'
NUMBER_OF_DETECTED_AT_ZERO_TIME = 'number_of_detected_at_zero_time'

USE_TODAY_MARK = 'use_today_mark'
TODAY_OFFSET = 'today_offset'
LAID_CURVE = 'laid_curve'

FEAR_FACTORS = 'fear_factors'
FEAR_FUNCTION = 'fear_function'
APPLICABLE_TO = 'applicable_to'
SCALE_FACTOR = 'scale_factor'
LOC_FACTOR = 'loc_factor'
LIMIT_VALUE = 'limit_value'
DETECTED_MULTIPLIER = 'detected_multiplier'
DEATHS_MULTIPLIER = 'deaths_multiplier'
EXPERIMENT_ID = 'experiment_id'

TRANSMISSION_PROBABILITIES = 'transmission_probabilities'

SPORADIC = 'sporadic'
HOUSEHOLD = 'household'
FRIENDSHIP = 'friendship'
WORKPLACE = 'workplace'
TRANSPORT = 'transport'
CONSTANT = 'constant'

IMPORT_INTENSITY = 'import_intensity'

TIME = 'time'
TYPE = 'type'
INITIATED_BY = 'initiated_by'
INITIATED_THROUGH = 'initiated_through'
ISSUED_TIME = 'issued_time'

FUNCTION = 'function'
MULTIPLIER = 'multiplier'
RATE = 'rate'
CAP = 'cap'
NO_IMPORT = 'no_import'

SERIAL_INTERVAL = 'serial_interval'
MIN_TIME = 'min_time'


OUTPUT_ROOT_DIR = 'output_root_dir'
SAVE_INPUT_DATA = 'save_input_data'

DEATH_PROBABILITY = 'death_probability'
UNSEEN_NODE = 'unseen_node'

RANDOM_SEED = 'random_seed'

FOUR_WEEKS = 28
