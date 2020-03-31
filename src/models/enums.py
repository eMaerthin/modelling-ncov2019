
import enum

from .constants import *


class EnumWithPublicValue2MemberMap(enum.Enum):
    @classmethod
    def map(cls):
        return cls._value2member_map_

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return other.value == self.value
        else:
            return other == self.value

    def __hash__(self):
        return hash(self.value)


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


class DetectionStatus(EnumWithPublicValue2MemberMap):
    NotDetected = NOT_DETECTED
    Detected = DETECTED


class QuarantineStatus(EnumWithPublicValue2MemberMap):
    NoQuarantine = 0
    Quarantine = 1


class InfectionStatus(EnumWithPublicValue2MemberMap):
    Healthy = 'healthy'
    Contraction = CONTRACTION
    Infectious = INFECTIOUS
    StayHome = 'stay_home'
    Hospital = 'hospital'  # probable
    Recovered = 'recovered'  # probable
    Death = 'death'  # probable, alternative to Recovered


class InitialConditionSelectionAlgorithms(EnumWithPublicValue2MemberMap):
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
    """
    Here we list various ways of importing new infected cases into the population according to provided schedule.
    - NoImport means that we have a lockdown already so no further cases will be imported from the outside.
    - Exponential - there will be added N imported cases (for instance due to travels, migration)
                    up to time T that solves the equation N=a*exp(rT)
    - Polynomial - imported cases are drawn in polynomial rate
                    - N imported cases up to time T that solves the equation N=a*pow(r, T)
    """
    NoImport = NO_IMPORT
    Exponential = EXPONENTIAL
    Polynomial = POLYNOMIAL


class FearFunctions(EnumWithPublicValue2MemberMap):
    """
    Set of allowed fear functions
    TODO: create a Fear class with a function spread (or sow or something) and would allow any concrete class implementing Fear to be applied in a simulation.
    """
    FearDisabled = 'fear_disabled'
    FearSigmoid = 'fear_sigmoid'
    FearTanh = 'fear_tanh'
    FearTanhTime = 'fear_tanh_time'


class SupportedDistributions(EnumWithPublicValue2MemberMap):
    Lognormal = LOGNORMAL
    Exponential = EXPONENTIAL
    Poisson = POISSON
    Gamma = GAMMA
    FromFile = FROM_FILE
