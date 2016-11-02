__author__ = 'mangirish_wagle'

import random

from global_vectors import GlobalVectors
from data_import_handler import DataImportHandler
from learn_tree import LearnTree
from test_util import TestUtil
from test_data import TestData


class Bagging:
    """
    Class that contains the bagging related utility functions.
    """

    tree_bags_list = list()


    def __init__(self):
        return

    def get_bootstrap_sample(self, size):
        """
        Function that returns a bootstrap sample of given size with replacement from training dataset.
        :return:
        """

        bootstrap_feature_vectors = []

        # Sampling random bootstraps.
        for k in xrange(size):
            rand_index = random.randint(0, (len(GlobalVectors.train_feature_vectors) - 1))
            bootstrap_feature_vectors.append(GlobalVectors.train_feature_vectors[rand_index])

        return bootstrap_feature_vectors

    def bagged_learn(self, depth, no_of_bags):
        """
        Returns
        :return:
        """

        for k in xrange(no_of_bags):
            bootstrap_feature_vectors = self.get_bootstrap_sample(len(GlobalVectors.train_feature_vectors))
            tree = LearnTree()
            tree.create_decision_tree(bootstrap_feature_vectors, depth)
            # tree.print_decision_tree(tree.decision_tree)

            self.tree_bags_list.append(tree)

    def get_majority_vote(self, result_dict):
        """
        Function that returns the majority vote for test data point classification using bagging.
        :return:
        """

        majority_label = list(result_dict)[0]
        max_count = 0

        for label in result_dict:
            if result_dict[label] >= max_count:

                if result_dict[label] == max_count:
                    majority_label = list([label, majority_label])[random.randint(0,1)]

                majority_label = label
                max_count = result_dict[label]

        return majority_label

    def classify_data_item(self, data_item, learn_tree_list):
        """
        Function that classifies a data item from test data set by traversing the trained decision tree.
        :param data_item:
        :param decision_tree:
        :return:
        """

        result_dict = dict()

        test_data = TestData()

        # Iterating to all the learnt bag trees to find majority vote.
        for ltree in learn_tree_list:
            current_node = ltree.decision_tree

            split_feature_value = data_item[current_node.split_feature_index]

            # Traverse tree until it finds the leaf node or the split on the feature value is not found.
            while len(current_node.children) > 0 and split_feature_value in current_node.children:
                current_node = current_node.children[split_feature_value]

            if current_node.class_label not in result_dict:
                result_dict[current_node.class_label] = 1
            else:
                result_dict[current_node.class_label] += 1

        # Getting the majority vote here.
        classified = self.get_majority_vote(result_dict)

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

    def classify_data_set(self, data_set):
        """
        Wrapper function to classify the test data set and creates the test data vector.
        :param data_set:
        :param decision_tree:
        :return:
        """

        GlobalVectors.clear_test_data_vector()

        for data_item in data_set:
            test_data = self.classify_data_item(data_item, self.tree_bags_list)
            GlobalVectors.test_data_vector.append(test_data)

        # print(GlobalVectors.test_data_vector)

    def print_confusion_matrix(self):
        """
        Function that prints the confusion matrix.
        :return:
        """
        t_util = TestUtil()
        print("{0:20s} {1:18s} {2:18s}".format("", "Predicted Negative", "Predicted Positive"))
        print("{0:20s} {1:18d} {2:18d}".format("Actual Negative", t_util.get_true_negative_count(),
                                               t_util.get_false_positive_count()))
        print("{0:20s} {1:18d} {2:18d}".format("Actual Positive", t_util.get_false_negative_count(),
                                               t_util.get_true_positive_count()))



def main():

    data_handler = DataImportHandler()
    data_handler.import_mushroom_data("datasets/mushroom/agaricuslepiotatrain1.csv", "train", ",");
    data_handler.import_mushroom_data("datasets/mushroom/agaricuslepiotatest1.csv", "test", ",");

    bagg = Bagging()

    bagg.bagged_learn(3, 10)
    bagg.classify_data_set(GlobalVectors.test_feature_vectors)

    bagg.print_confusion_matrix()



if __name__ == "__main__":
    main()

