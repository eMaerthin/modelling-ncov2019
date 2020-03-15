from unittest import TestCase
from src.models import infection_model
from collections import defaultdict
import numpy as np
import random
import os
import pandas as pd

class TestInfectionModel(TestCase):

    def __setup_model(self, params_filename='dummy_params.json', individuals_filename='dummy_individuals.csv'):
        params_path = os.path.join('test', 'models', 'assets', params_filename)
        individuals_path = os.path.join('test', 'models', 'assets', individuals_filename)
        self.model = infection_model.InfectionModel(params_path=params_path, df_individuals_path=individuals_path)

    def test_active_people(self):
        self.__setup_model()
        assert 0 == self.model.active_people, f'Expected 0, actual: {self.model.active_people()}'

        self.model.pop_and_apply_event()
        assert 1 == self.model.active_people(), f'Expected at least 1, actual: {self.model.active_people()}'

    def test_individuals_schema1(self):
        self.__setup_model()

        df_individuals = self.model.df_individuals
        assert 10 == len(df_individuals)
        self.model.pop_and_apply_event()

        assert infection_model.InfectionStatus.Contraction == df_individuals.loc[0, infection_model.INFECTION_STATUS]
        assert infection_model.ExpectedCaseSeverity.Severe == df_individuals.loc[0, infection_model.EXPECTED_CASE_SEVERITY]
        for i in range(1, 10):
            assert infection_model.InfectionStatus.Healthy == df_individuals.loc[i, infection_model.INFECTION_STATUS]
        # pd.set_option('display.max_columns', None)
        # print(df_individuals.head())

    def test_individuals_schema2(self):
        self.__setup_model(params_filename='dummy_alternative_specification_of_initial_conditions.json')
        df_individuals = self.model.df_individuals
        health_state_distribution = df_individuals[infection_model.INFECTION_STATUS].value_counts()
        assert 5 == health_state_distribution[infection_model.InfectionStatus.Infectious]
        assert 5 == health_state_distribution[infection_model.InfectionStatus.Contraction]

    def test_auxilliary(self):
        self.__setup_model()
        assert 0 < len([event for event in self.model.event_queue
                        if event.initiated_through == infection_model.IMPORT_INTENSITY])

    def test_stop_threshold(self):
        self.__setup_model()

        assert 1000 == self.model.stop_simulation_threshold

    def test_disease_progression(self):
        self.__setup_model()

        assert 'from_file' == self.model.disease_progression[infection_model.T0][infection_model.DISTRIBUTION]
