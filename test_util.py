__author__ = 'mangirish_wagle'

from random import randint

from test_data import TestData
from global_vectors import GlobalVectors


class TestUtil:

    def __init__(self):
        return

    def classify_data_item(self, data_item, decision_tree):

        test_data = TestData()

        prev_node = None
        current_node = decision_tree

        split_feature_value = data_item[current_node.split_feature_index]

        while len(current_node.children) > 0 and split_feature_value in current_node.children:
            prev_node =current_node
            current_node = current_node.children[split_feature_value]

        pos_count = current_node.pos_neg[0]
        neg_count = current_node.pos_neg[1]

        classified = None
        if pos_count == neg_count:
            rand_num = randint(0,1000) % 2
            classified = GlobalVectors.POSITIVE_VALUE #if rand_num == 1 else GlobalVectors.NEGATIVE_VALUE
        else:
            classified = GlobalVectors.POSITIVE_VALUE if pos_count > neg_count else GlobalVectors.NEGATIVE_VALUE

        test_data.predicted_class = classified
        test_data.is_error = True if data_item[0] != classified else False
        test_data.is_false_negative = True if data_item[0] == GlobalVectors.POSITIVE_VALUE \
                                              and classified == GlobalVectors.NEGATIVE_VALUE else False
        test_data.is_true_negative = True if data_item[0] == GlobalVectors.NEGATIVE_VALUE \
                                              and classified == GlobalVectors.NEGATIVE_VALUE else False
        test_data.is_false_positive = True if data_item[0] == GlobalVectors.NEGATIVE_VALUE \
                                              and classified == GlobalVectors.POSITIVE_VALUE else False
        test_data.is_true_positive = True if data_item[0] == GlobalVectors.POSITIVE_VALUE \
                                              and classified == GlobalVectors.POSITIVE_VALUE else False

        return test_data

    def classify_data_set(self, data_set, decision_tree):

        GlobalVectors.clear_test_data_vector()

        for data_item in data_set:
            test_data = self.classify_data_item(data_item, decision_tree)
            GlobalVectors.test_data_vector.append(test_data)

        # print(GlobalVectors.test_data_vector)

    def get_accuracy(self):

        correct_classification_count = 0
        for data_item in GlobalVectors.test_data_vector:
            if not data_item.is_error:
                correct_classification_count += 1

        accuracy = (correct_classification_count/float(len(GlobalVectors.test_data_vector))) * 100

        return accuracy

    def get_true_negative_count(self):

        true_negative_count = 0
        for data_item in GlobalVectors.test_data_vector:
            if data_item.is_true_negative:
                true_negative_count += 1

        return true_negative_count

    def get_true_positive_count(self):

        true_positive_count = 0
        for data_item in GlobalVectors.test_data_vector:
            if data_item.is_true_positive:
                true_positive_count += 1

        return true_positive_count

    def get_false_positive_count(self):

        false_positive_count = 0
        for data_item in GlobalVectors.test_data_vector:
            if data_item.is_false_positive:
                false_positive_count += 1

        return false_positive_count

    def get_false_negative_count(self):

        false_negative_count = 0
        for data_item in GlobalVectors.test_data_vector:
            if data_item.is_false_negative:
                false_negative_count += 1

        return false_negative_count



