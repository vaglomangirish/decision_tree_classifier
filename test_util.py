__author__ = 'mangirish_wagle'

from random import randint

from test_data import TestData
from global_vectors import GlobalVectors


class TestUtil:

    def __init__(self):
        return

    def classify_data_item(self, data_item, decision_tree):
        """
        Function that classifies a data item from test data set by traversing the trained decision tree.
        :param data_item:
        :param decision_tree:
        :return:
        """

        test_data = TestData()

        current_node = decision_tree

        split_feature_value = data_item[current_node.split_feature_index]

        # Traverse tree until it finds the leaf node or the split on the feature value is not found.
        while len(current_node.children) > 0 and split_feature_value in current_node.children:
            current_node = current_node.children[split_feature_value]
            split_feature_value = data_item[current_node.split_feature_index]

        classified = current_node.class_label

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
        """
        Wrapper function to classify the test data set and creates the test data vector.
        :param data_set:
        :param decision_tree:
        :return:
        """

        GlobalVectors.clear_test_data_vector()

        for data_item in data_set:
            test_data = self.classify_data_item(data_item, decision_tree)
            GlobalVectors.test_data_vector.append(test_data)

        # print(GlobalVectors.test_data_vector)

    def get_accuracy(self):
        """
        Function that returns accuracy of the constructed decision tree based on the test data set.
        :return:
        """

        correct_classification_count = 0
        for data_item in GlobalVectors.test_data_vector:
            if not data_item.is_error:
                correct_classification_count += 1

        accuracy = (correct_classification_count/float(len(GlobalVectors.test_data_vector))) * 100

        return accuracy

    def get_true_negative_count(self):
        """
        Function that returns true negative count of the constructed decision tree based on the test data set.
        :return:
        """

        true_negative_count = 0
        for data_item in GlobalVectors.test_data_vector:
            if data_item.is_true_negative:
                true_negative_count += 1

        return true_negative_count

    def get_true_positive_count(self):
        """
        Function that returns true positive of the constructed decision tree based on the test data set.
        :return:
        """

        true_positive_count = 0
        for data_item in GlobalVectors.test_data_vector:
            if data_item.is_true_positive:
                true_positive_count += 1

        return true_positive_count

    def get_false_positive_count(self):
        """
        Function that returns false positive of the costructed decision tree based on the test data set.
        :return:
        """

        false_positive_count = 0
        for data_item in GlobalVectors.test_data_vector:
            if data_item.is_false_positive:
                false_positive_count += 1

        return false_positive_count

    def get_false_negative_count(self):
        """
        Function that returns false negative of the costructed decision tree based on the test data set.
        :return:
        """

        false_negative_count = 0
        for data_item in GlobalVectors.test_data_vector:
            if data_item.is_false_negative:
                false_negative_count += 1

        return false_negative_count



