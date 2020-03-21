import numpy as np
from enum import Enum

prop_idx = 'idx'
prop_age = 'age'
prop_gender = 'gender'
prop_employment_status = 'employment_status'
prop_social_competence = 'social_competence'
prop_public_transport_usage = 'public_transport_usage'
prop_public_transport_duration = 'public_transport_duration'
prop_household = 'household_index'
prop_profession = 'profession'

columns = [prop_idx, prop_age, prop_gender, prop_household, prop_employment_status, prop_social_competence,
           prop_public_transport_usage, prop_public_transport_duration, prop_profession]
data_types = {prop_idx: np.uint64,
              prop_age: np.int8,
              prop_gender: np.int8,
              prop_employment_status: np.int8,
              prop_social_competence: np.float,
              prop_public_transport_usage: np.int8,
              prop_public_transport_duration: np.float,
              prop_household: np.uint64,
              prop_profession: np.uint64}


class AgeGroup(Enum):
    YOUNG = 0
    MIDDLE = 1
    ELDERLY = 2


class Gender(Enum):
    NOT_SET = -1
    MALE = 0
    FEMALE = 1


GENDERS = [Gender.FEMALE, Gender.MALE]


def gender_from_string(string):
    if string == 'M':
        return Gender.MALE
    elif string == 'F':
        return Gender.FEMALE
    raise ValueError('Unknown gender {}'.format(string))


class EmploymentStatus(Enum):
    NOT_SET = -1
    NOT_EMPLOYED = 0
    EMPLOYED = 1


class EconomicalGroup(Enum):
    PRZEDPRODUKCYJNY = 0
    PRODUKCYJNY_MOBILNY = 1
    PRODUKCYJNY_NIEMOBILNY = 2
    POPRODUKCYJNY = 3


HOUSEHOLD_NOT_ASSIGNED = -1
PROFESSION_NOT_ASSIGNED = -1
SOCIAL_COMPETENCE_NOT_ASSIGNED = -1
AGE_NOT_SET = -1
PUBLIC_TRANSPORT_USAGE_NOT_SET = -1
PUBLIC_TRANSPORT_DURATION_NOT_SET = -1


class BasicNode(dict):

    def __init__(self, idx: int, age: int = AGE_NOT_SET,
                 gender: Gender = Gender.NOT_SET,
                 household: int = HOUSEHOLD_NOT_ASSIGNED) -> None:
        """
        Creates a node representing a person.
        :param age: (optional) age of the node, defaults to AGE_NOT_SET
        :param gender: (optional) gender of the node, defaults to Gender.NOT_SET
        :param household: (optional) household index of the node, defaults to HOUSEHOLD_NOT_ASSIGNED
        :return: None
        """
        super().__init__()
        self[prop_idx] = idx
        self[prop_age] = age
        self[prop_gender] = gender.value
        self[prop_household] = household

    @property
    def idx(self) -> int:
        return self[prop_idx]

    @property
    def age(self) -> int:
        return self[prop_age]

    @age.setter
    def age(self, age: int) -> None:
        self[prop_age] = age

    @property
    def gender(self) -> int:
        return self[prop_gender]

    @gender.setter
    def gender(self, gender: Gender) -> None:
        self[prop_gender] = gender.value

    @property
    def household(self) -> int:
        return self[prop_household]

    @household.setter
    def household(self, household: int) -> None:
        self[prop_household] = household


class Node(BasicNode):

    def __init__(self, age: int = AGE_NOT_SET,
                 gender: Gender = Gender.NOT_SET,
                 employment_status: EmploymentStatus = EmploymentStatus.NOT_SET,
                 social_competence: float = SOCIAL_COMPETENCE_NOT_ASSIGNED,
                 public_transport_usage: float = PUBLIC_TRANSPORT_USAGE_NOT_SET,
                 public_transport_duration: float = PUBLIC_TRANSPORT_DURATION_NOT_SET,
                 household: int = HOUSEHOLD_NOT_ASSIGNED,
                 profession: int = PROFESSION_NOT_ASSIGNED) -> None:
        """
            Creates a node representing a person.
            :param age: (optional) age of the node, defaults to AGE_NOT_SET
            :param gender: (optional) gender of the node, defaults to Gender.NOT_SET
            :param employment_status: (optional) employement status of the node, defaults to EmploymentStatus.NOT_SET
            :param social_competence: (optional) social competence of the node, defaults to SOCIAL_COMPETENCE_NOT_ASSIGNED
            :param public_transport_usage: (optional) public transport usage of the node. in essence binary, but can also be used as
            frequency, defaults to PUBLIC_TRANSPORT_USAGE_NOT_SET (#TODO: to be decided)
            :param public_transport_duration: (optional) mean duration per day spent in public transport (#TODO: to be decided about
            mean vs other aggregate function), defaults to PUBLIC_TRANSPORT_DURATION_NOT_SET
            :param household: (optional) household index of the node, defaults to HOUSEHOLD_NOT_ASSIGNED
            :param profession: (optional) profession index of the node, defaults to PROFESSION_NOT_ASSIGNED
            :return: None
        """
        super().__init__(0, age, gender, household)
        self[prop_employment_status] = employment_status.value
        self[prop_social_competence] = social_competence
        self[prop_public_transport_usage] = public_transport_usage
        self[prop_public_transport_duration] = public_transport_duration
        self[prop_profession] = profession

    @property
    def employment_status(self) -> int:
        return self[prop_employment_status]

    @employment_status.setter
    def employment_status(self, employment_status: EmploymentStatus) -> None:
        self[prop_employment_status] = employment_status.value

    @property
    def social_competence(self) -> float:
        return self[prop_social_competence]

    @social_competence.setter
    def social_competence(self, social_competence: float) -> None:
        self[prop_social_competence] = social_competence

    @property
    def public_transport_usage(self) -> float:
        return self[prop_public_transport_usage]

    @public_transport_usage.setter
    def public_transport_usage(self, public_transport_usage: float) -> None:
        self[prop_public_transport_usage] = public_transport_usage

    @property
    def public_transport_duration(self) -> float:
        return self[prop_public_transport_duration]

    @public_transport_duration.setter
    def public_transport_duration(self, public_transport_duration: float) -> None:
        self[prop_public_transport_duration] = public_transport_duration

    @property
    def profession(self) -> int:
        return self[prop_profession]

    @profession.setter
    def profession(self, profession: int) -> None:
        self[prop_profession] = profession

    @property
    def economical_group(self) -> EconomicalGroup:
        if self.age < 18:
            return EconomicalGroup.PRZEDPRODUKCYJNY
        if self.age < 45:
            return EconomicalGroup.PRODUKCYJNY_MOBILNY
        if self.gender == Gender.FEMALE.value and self.age < 60:
            return EconomicalGroup.PRODUKCYJNY_NIEMOBILNY
        if self.gender == Gender.MALE.value and self.age < 65:
            return EconomicalGroup.PRODUKCYJNY_NIEMOBILNY
        return EconomicalGroup.POPRODUKCYJNY

    @property
    def age_group(self) -> AgeGroup:
        if self.age < 30:
            return AgeGroup.YOUNG
        if self.age < 60:
            return AgeGroup.MIDDLE
        return AgeGroup.ELDERLY
