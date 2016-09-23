__author__ = 'mangirish_wagle'

from test_data import TestData
from global_vectors import GlobalVectors


class TestUtil:

    def __init__(self):
        return

    def classify_data_item(self, data_item, decision_tree):

        test_data = TestData()

        return test_data

    def classify_data_set(self, data_set, decision_tree):

        for data_item in data_set:
            test_data = self.classify_data_item(data_item, decision_tree)
            GlobalVectors.test_data_vector.append(test_data)


