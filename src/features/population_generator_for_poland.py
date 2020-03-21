from src.data.datasets import *
from pathlib import Path
import pandas as pd
from contextlib import closing
from typing import Tuple, List, Dict, Optional
from src.features.entities import BasicNode, GENDERS
from src.features.population_generator_common import nodes_to_dataframe
from datetime import datetime
import numpy as np
from tqdm import tqdm

project_dir = Path(__file__).resolve().parents[2]
poland_folder = project_dir / 'data' / 'processed' / 'poland'


class PopulationGenerator:
    household_csv_houshold_index_col = 'household_index'
    household_csv_idx_col = 'idx'
    simulation_population_csv = 'population.csv'
    simulation_household_csv = 'household.csv'

    def __init__(self, voivodship: Path) -> None:
        self.voivodship = voivodship
        self.age_gender_df = pd.read_excel(str(poland_folder / age_gender_xlsx.file_name),
                                           sheet_name=self.voivodship.name)
        self.age_gender_df = self.age_gender_df.astype(int)
        self.age_gender_df['female_probability'] = self.age_gender_df.Females / \
                                                   (self.age_gender_df.Females + self.age_gender_df.Males)
        self.children_df = self.age_gender_df[self.age_gender_df['Age'] < 18].reset_index(drop=True)
        self.children_df['total_probability'] = self.children_df['Total'] / self.children_df['Total'].sum()
        self.adults_df = self.age_gender_df[self.age_gender_df['Age'] >= 18].reset_index(drop=True)
        self.adults_df['total_probability'] = self.adults_df['Total'] / self.adults_df['Total'].sum()
        self.households_headcount_ac_df = self._preprocess_household_headcount_ac()
        self.number_of_households = self.households_headcount_ac_df.households.sum()
        print(self.number_of_households)

    def _preprocess_household_headcount_ac(self) -> pd.DataFrame:
        try:
            df2 = pd.read_excel(str(self.voivodship / households_headcount_ac_xlsx.file_name),
                                sheet_name=households_headcount_ac_xlsx.sheet_name)
        except Exception as e:  # XLRDError No sheet named <'processed'>
            print(str(e))
            df = pd.read_excel(str(self.voivodship / households_headcount_ac_xlsx_raw.file_name),
                               sheet_name=households_headcount_ac_xlsx_raw.sheet_name, header=None,
                               skiprows=29, nrows=6, usecols=[1, 3, 4, 5, 6, 7, 8],
                               names=['children', 0, 1, 2, 3, 4, 5], index_col=0)

            df = df.fillna(0)
            df = df.astype(int)

            as_list = df.index.tolist()
            idx = as_list.index('5+')
            as_list[idx] = 5
            df.index = as_list
            df.index.name = 'children'

            df2 = pd.melt(df.reset_index(), id_vars=['children'], var_name='adults', value_name='households')
            df2 = df2.astype(int)
            df2['headcount'] = df2.children + df2.adults
            df2['probability'] = df2['households'] / df2['households'].sum()

            with closing(pd.ExcelWriter(str(self.voivodship / households_headcount_ac_xlsx.file_name),
                                        engine='openpyxl')) as writer:
                df2.to_excel(writer, sheet_name=households_headcount_ac_xlsx.sheet_name, index=False)
        return df2

    def _draw_a_household(self) -> Tuple[int, int]:
        idx = np.random.choice(self.households_headcount_ac_df.index.tolist(),
                               p=self.households_headcount_ac_df['probability'])
        row = self.households_headcount_ac_df.iloc[idx]
        return int(row['children']), int(row['adults'])

    def _draw_from_subpopulation(self, subpopulation: pd.DataFrame, headcount: int, household_idx: int,
                                 current_index: int) -> Tuple[List[BasicNode], int]:
        nodes = []

        for _ in range(headcount):
            idx = np.random.choice(subpopulation.index.tolist(), p=subpopulation['total_probability'])
            row = subpopulation.iloc[idx]
            age = row['Age']
            gender = GENDERS[np.random.choice([0, 1], p=[row.female_probability, 1 - row.female_probability])]
            nodes.append(BasicNode(current_index, age, gender, household_idx))
            current_index += 1

        return nodes, current_index

    def _draw_children(self, children_count: int, household_idx: int, current_index: int) -> Tuple[
        List[BasicNode], int]:
        return self._draw_from_subpopulation(self.children_df, children_count, household_idx, current_index)

    def _draw_adults(self, adults_count: int, household_idx: int, current_index: int) -> Tuple[List[BasicNode], int]:
        return self._draw_from_subpopulation(self.adults_df, adults_count, household_idx, current_index)

    def _prepare_simulation_folder(self, simulations_folder):
        simulations_folder = simulations_folder / self.voivodship.name
        simulations_folder.mkdir()
        return simulations_folder

    def run(self, household_index: int, population_index: int, simulation_folder: Optional[Path] = None) \
            -> Tuple[int, int]:

        simulation_folder = self._prepare_simulation_folder(simulation_folder)

        nodes: List[BasicNode] = []
        households: Dict[int, List[int]] = {}
        current_index = population_index
        for idx in tqdm(range(self.number_of_households)):
            current_household_idx = idx + household_index
            children_count, adults_count = self._draw_a_household()
            children, current_index = self._draw_children(children_count, current_household_idx, current_index)
            adults, current_index = self._draw_adults(adults_count, current_household_idx, current_index)
            nodes.extend(children)
            nodes.extend(adults)
            households[current_household_idx] = [child.idx for child in children] + [adult.idx for adult in adults]

            if idx % 1000 == 0:
                hdf = pd.DataFrame(data={self.household_csv_houshold_index_col: list(households.keys()),
                                         self.household_csv_idx_col: list(households.values())})
                pdf = nodes_to_dataframe(nodes)
                if idx == 1000:
                    pdf.to_csv(str(simulation_folder / self.simulation_population_csv), index=False)
                    hdf.to_csv(str(simulation_folder / self.simulation_household_csv), index=False)
                else:
                    pdf.to_csv(str(simulation_folder / self.simulation_population_csv), mode='a', header=False,
                               index=False)
                    hdf.to_csv(str(simulation_folder / self.simulation_household_csv), mode='a', header=False,
                               index=False)

                nodes = []
                households = {}

        hdf = pd.DataFrame(data={self.household_csv_houshold_index_col: list(households.keys()),
                                 self.household_csv_idx_col: list(households.values())})
        pdf = nodes_to_dataframe(nodes)
        pdf.to_csv(str(simulation_folder / self.simulation_population_csv), mode='a', header=False, index=False)
        hdf.to_csv(str(simulation_folder / self.simulation_household_csv), mode='a', header=False, index=False)
        return self.number_of_households + household_index, current_index


def prepare_simulations_folder(simulations_folder: Path = None):
    if simulations_folder is None:
        simulations_folder = project_dir / 'data' / 'simulations' / datetime.now().strftime('%Y%m%d_%H%M')
    if not simulations_folder.is_dir():
        simulations_folder.mkdir()
    return simulations_folder


if __name__ == '__main__':
    voivodships = [x for x in poland_folder.iterdir() if x.is_dir() and len(x.name) == 1]
    # change to subprocesses
    next_household_index = 0
    next_person_index = 0
    simulations_folder = prepare_simulations_folder()
    for i, item in enumerate(voivodships):
        next_household_index, next_person_index = PopulationGenerator(item).run(next_household_index, next_person_index,
                                                                                simulations_folder)
