import networkx as nx
from src.features import entities
from unittest import TestCase


class TestNodes(TestCase):
    def setUp(self):
        self.G = nx.Graph()

    def test_basic_node_with_idx(self):
        idx = 1
        node = entities.BasicNode(idx)

        self.assertEqual(idx, node.idx)
        self.assertEqual(idx, node[entities.prop_idx])

    def test_basic_node_with_gender(self):
        idx = 1
        gender = entities.Gender.FEMALE
        node = entities.BasicNode(idx, gender=gender)

        self.assertEqual(gender.value, node.gender)
        self.assertEqual(gender.value, node[entities.prop_gender])

    def test_basic_node_update_gender(self):
        idx = 1
        gender = entities.Gender.MALE
        node = entities.BasicNode(idx)

        self.assertEqual(entities.Gender.NOT_SET.value, node.gender)

        node.gender = gender
        self.assertEqual(gender.value, node.gender)

    def test_basic_node_all_attributes_as_dict(self):
        idx = 1
        gender = entities.Gender.MALE
        age = 32
        household_index = 15072
        node = entities.BasicNode(idx, age=age, gender=gender, household=household_index)

        self.assertEqual(idx, node[entities.prop_idx])
        self.assertEqual(age, node[entities.prop_age])
        self.assertEqual(gender.value, node[entities.prop_gender])
        self.assertEqual(household_index, node[entities.prop_household])

    def test_basic_node_all_attributes_as_properties(self):
        idx = 1
        gender = entities.Gender.MALE
        age = 32
        household_index = 15072
        node = entities.BasicNode(idx, age=age, gender=gender, household=household_index)

        self.assertEqual(idx, node.idx)
        self.assertEqual(age, node.age)
        self.assertEqual(gender.value, node.gender)
        self.assertEqual(household_index, node.household)

    def test_basic_node_all_attributes_update(self):
        # given
        idx = 1
        gender = entities.Gender.MALE
        age = 32
        household_index = 15072
        node = entities.BasicNode(idx, age=age, gender=gender, household=household_index)

        # when
        new_age = 17
        new_gender = entities.Gender.FEMALE
        new_household = 15234

        node.age = new_age
        node.gender = new_gender
        node.household = new_household

        # then
        self.assertEqual(idx, node.idx)
        self.assertEqual(new_age, node.age)
        self.assertEqual(new_gender.value, node.gender)
        self.assertEqual(new_household, node.household)


