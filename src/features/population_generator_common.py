import pandas as pd
from typing import List, Dict, Any
from src.features import entities


def _age_gender_population(age_gender_df: pd.DataFrame) -> pd.DataFrame:
    """Polish census data gives age and gender together. For each age (or age range) there is a number of males and
    females provided. This function generates a dataframe of people with age and gender. The length of the dataframe
    equals the total number of people in the Census data. """
    ages = []
    genders = []
    for idx, row in age_gender_df.iterrows():
        ages.extend([row.Age] * row.Total)
        genders.extend([entities.Gender.MALE.value] * row.Males)
        genders.extend([entities.Gender.FEMALE.value] * row.Females)
    return pd.DataFrame(data={entities.prop_age: ages, entities.prop_gender: genders})


def nodes_to_dataframe(nodes: List[entities.Node]) -> pd.DataFrame:
    """A utility function that takes a list of dictionaries (here, specifically the subclass of dictionary - Node),
    converts these dictionaries into lists and creates a dataframe out of them. """
    return pd.DataFrame(data=_list_of_dicts_to_dict_of_lists(nodes))


def _list_of_dicts_to_dict_of_lists(list_of_dicts: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """A utility function that given a list of dictionaries converts them into a dictionary of named
    (by a key) lists. """
    return {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}

