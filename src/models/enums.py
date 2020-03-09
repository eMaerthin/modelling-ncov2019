
import enum

from .constants import *


class EnumWithPublicValue2MemberMap(enum.Enum):
    @classmethod
    def map(cls):
        return cls._value2member_map_


class Gender(EnumWithPublicValue2MemberMap):
    male = 0
    female = 1


class EmploymentStatus(EnumWithPublicValue2MemberMap):
    not_employed = 0
    employed = 1


class DiseaseProgressionEvents(EnumWithPublicValue2MemberMap):
    tminus1 = TMINUS1
    t0 = T0
    t1 = T1
    t2 = T2


class ExpectedCaseSeverity(EnumWithPublicValue2MemberMap):  # to be added to nodes?
    Asymptomatic = ASYMPTOMATIC  # 0.6%
    Mild = MILD  # 80.9% Maybe Staying Home, maybe visit Doctor
    Severe = SEVERE  # 13.8%
    Critical = CRITICAL  # 4.7% Need to go to Hospital!
    UnseenNode = UNSEEN_NODE

class EpidemicStatus(EnumWithPublicValue2MemberMap):
    NotDetected = NOT_DETECTED
    Detected = DETECTED


class InfectionStatus(EnumWithPublicValue2MemberMap):
    Healthy = 'healthy'
    Contraction = CONTRACTION
    Infectious = INFECTIOUS
    StayHome = 'stay_home'
    #SeeDoctor = 'see_doctor'  # probable, dynamic variable - visiting doctor can cause spreading the disease into vulnerable people (already sick)
    Hospital = 'hospital'  # probable
    Recovered = 'recovered'  # probable
    Death = 'death'  # probable, alternative to Recovered


class StateDependentOnTheEpidemicsState(EnumWithPublicValue2MemberMap):
    Detection = DETECTION  # dependent on the epidemics state
    QuarantineAtHome = QUARANTINE_AT_HOME  # dependent on the epidemics state
    QuarantineAtHospital = QUARANTINE_AT_HOSPITAL  # dependent on the epidemics state


class SelectionAlgorithms(EnumWithPublicValue2MemberMap):
    """
    This enum is intended for storing all initial condition selection algorithms,
    to seed initially infectious agents in the population
    RandomSelection assignes infection status randomly based on provided distribution of sick people
    In the future other algorithms could be added e.g. to limit assigning infection status to people who work or commute
    """
    RandomSelection = 'random_selection'


class KernelType(EnumWithPublicValue2MemberMap):
    Sporadic = SPORADIC
    Household = HOUSEHOLD
    Friendship = FRIENDSHIP
    Workplace = WORKPLACE
    Transport = TRANSPORT
    Constant = CONSTANT


class ImportIntensityFunctions(EnumWithPublicValue2MemberMap):
    NoImport = NO_IMPORT
    Exponential = EXPONENTIAL
    Polynomial = POLYNOMIAL


class FearFunctions(EnumWithPublicValue2MemberMap):
    """
    Set of allowed fear functions
    """
    FearDisabled = 'fear_disabled'
    Fear1 = 'fear_function1'


class SupportedDistributions(EnumWithPublicValue2MemberMap):
    Lognormal = LOGNORMAL
    Exponential = EXPONENTIAL
    Poisson = POISSON
    Gamma = GAMMA
    FromFile = FROM_FILE


def _convert_enum(enum_class, x):
    for status in enum_class:
        if x == status.value:
            return status
    raise ValueError(f'invalid status provided: {x}')


def convert_infection_status(x):
    return _convert_enum(InfectionStatus, x)


def convert_expected_case_severity(x):
    return _convert_enum(ExpectedCaseSeverity, x)


def convert_import_intensity_functions(x):
    return _convert_enum(ImportIntensityFunctions, x)