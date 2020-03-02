from unittest import TestCase
from src.models import infection_model
import numpy as np
import random
import os
import pandas as pd


class TestInfectionModel(TestCase):

    @classmethod
    def setUpClass(cls):
        dummy_params_path = os.path.join('test', 'models', 'assets', 'dummy_params.json')
        dummy_individuals_path = os.path.join('test', 'models', 'assets', 'dummy_individuals.csv')
        cls.model = infection_model.InfectionModel(params_path=dummy_params_path,
                                                   df_individuals_path=dummy_individuals_path)

    def test_active_people(self):
        assert 1 == self.model.active_people()

    def test_individuals(self):
        df_individuals = self.model.df_individuals
        assert 10 == len(df_individuals)

        assert infection_model.InfectionStatus.Contraction == df_individuals.loc[0, infection_model.INFECTION_STATUS]
        assert infection_model.ExpectedCaseSeverity.Severe == df_individuals.loc[0, infection_model.EXPECTED_CASE_SEVERITY]
        for i in range(1, 10):
            assert infection_model.InfectionStatus.Healthy == df_individuals.loc[i, infection_model.INFECTION_STATUS]
        pd.set_option('display.max_columns', None)
        print(df_individuals.head())

    def test_stop_threshold(self):
        assert 1000 == self.model.stop_simulation_threshold

    def test_epidemic_status(self):
        assert infection_model.EpidemicStatus.NotDetected.value == self.model.epidemic_status

    def test_disease_progression(self):
        assert 'exponential' == self.model.disease_progression[infection_model.T0][infection_model.DISTRIBUTION]
