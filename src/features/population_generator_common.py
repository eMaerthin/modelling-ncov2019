import pandas as pd
from typing import List, Dict, Any
from src.features import entities


def _age_gender_population(age_gender_df: pd.DataFrame) -> pd.DataFrame:
    ages = []
    genders = []
    for idx, row in age_gender_df.iterrows():
        ages.extend([row.Age] * row.Total)
        genders.extend([entities.Gender.MALE.value] * row.Males)
        genders.extend([entities.Gender.FEMALE.value] * row.Females)
    return pd.DataFrame(data={entities.prop_age: ages, entities.prop_gender: genders})


def nodes_to_dataframe(nodes: List[entities.Node]) -> pd.DataFrame:
    return pd.DataFrame(data=_list_of_dicts_to_dict_of_lists(nodes))


def _list_of_dicts_to_dict_of_lists(list_of_dicts: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    return {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}

