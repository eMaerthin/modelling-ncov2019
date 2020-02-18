from unittest import TestCase
from src.models import infection_model
import numpy as np
import random
import os


class TestInfectionModel(TestCase):
    def test_init_with_dummy_params(self):
        dummy_params_path = os.path.join('test', 'models', 'assets', 'dummy_params.json')
        dummy_individuals_path = os.path.join('test', 'models', 'assets', 'dummy_individuals.csv')
        model = infection_model.InfectionModel(params_path=dummy_params_path,
                                               df_individuals_path=dummy_individuals_path)
        df_individuals = model.df_individuals
        assert 9 == len(df_individuals)
        assert 1 == model.active_people()
