__author__ = 'mangirish_wagle'

import math

from global_vectors import GlobalVectors
from data_import_handler import DataImportHandler
from learn_tree import LearnTree
from test_util import TestUtil
from test_data import TestData


class Boosting:
    """
    Class that contains boosting related utils.
    """

    tree_boost_list = list()   # List of boosted trees
    tree_alpha_list = list()   # List of Alpha coefficient for every boosted tree.

    errored_index_list = list()

    def __init__(self):
        self.tree_boost_list = list()
        self.tree_alpha_list = list()
        self.errored_index_list = list()
        return

    def get_coeff_alpha(self, weighted_error):
        """
        Function that returns the weight coefficient alpha, given the weighted error.
        :param weighted_error:
        :return:
        """

        alpha = 0.5 * math.log((1.0 - float(weighted_error)) / float(weighted_error))
        return alpha

    def initialize_weights(self):
        """
        Function that initializes the weights of the data points to 1/N where N is number of data points.
        :return:
        """

        for point in GlobalVectors.train_feature_vectors:
            GlobalVectors.train_weight_vector.append(float(1.0/len(GlobalVectors.train_feature_vectors)))

    def analyze_data_for_errors(self, tree):
        """
        Function that populates the errored index list which contains indices of all data points that are misclassified.
        :param tree:
        :return:
        """

        test_util = TestUtil()

        for index in xrange(len(GlobalVectors.train_feature_vectors)):
            test_data = test_util.classify_data_item(GlobalVectors.train_feature_vectors[index], tree)

            if test_data.is_error:
                self.errored_index_list.append(index)

    def get_global_weighted_error(self):
        """
        Function returns the sum of weights of all the misclassified points in the training set.
        :return:
        """

        # Initializing to very small valeu to avoid 0.
        weighted_error_sum = 0.0000000000001

        if GlobalVectors.train_weight_vector is not None and len(GlobalVectors.train_weight_vector) > 0:
            for error_index in self.errored_index_list:
                weighted_error_sum += GlobalVectors.train_weight_vector[error_index]

        return weighted_error_sum

    def get_total_weight_sum(self):
        """
        Function that returns sum of weights of all the points in dataset.
        :return:
        """

        sum = 0.0

        for weight in GlobalVectors.train_weight_vector:
            sum += weight

        return sum

    def normalize_weights(self):
        """
        Function that normalizes the updated weights in every iteration.
        :return:
        """

        total_weight = self.get_total_weight_sum()

        for index in xrange(len(GlobalVectors.train_weight_vector)):
            GlobalVectors.train_weight_vector[index] /= total_weight

    def update_weights(self, alpha):
        """
        Function that updates weights of data points based on whether they are correctly or incorrectly classified.
        For correct classification the weight is multiplied by e^-alpha.
        For misclassification the weight is multiplied by e^alpha.
        :param alpha:
        :return:
        """

        for index in xrange(len(GlobalVectors.train_weight_vector)):

            if index in self.errored_index_list:
                GlobalVectors.train_weight_vector[index] *= math.exp(alpha)
            else:
                GlobalVectors.train_weight_vector[index] *= math.exp(alpha*(-1.0))

    def boosted_learn(self, depth, no_of_trees):
        """
        Wrapper function for the entire Adaboost learning procedure.
        :param depth:
        :param no_of_trees:
        :return:
        """

        self.initialize_weights()

        for count in xrange(no_of_trees):
            ltree = LearnTree()
            ltree.create_decision_tree(GlobalVectors.train_feature_vectors, depth, True)

            self.analyze_data_for_errors(ltree.decision_tree)
            global_weighted_error = self.get_global_weighted_error()
            alpha = self.get_coeff_alpha(global_weighted_error)

            self.update_weights(alpha)
            self.normalize_weights()

            print("Learning Iteration " + str(count) + " | Alpha " + str(count) + ": " + str(alpha))
            # ltree.print_decision_tree(ltree.decision_tree)

            self.tree_boost_list.append(ltree)
            self.tree_alpha_list.append(alpha)

    def get_boosted_classification(self, tree_classify_list):
        """
        Function that returns the ensemble result of all the iterative classifications in Adaboost.
        :param tree_classify_list:
        :return:
        """

        sum_alpha = 0.0

        multiply_factor = None
        multiplicand = None

        for index in xrange(len(tree_classify_list)):
            multiply_factor = self.tree_alpha_list[index]
            if tree_classify_list[index] == GlobalVectors.NEGATIVE_VALUE:
                multiplicand = -1.0
            else:
                multiplicand = 1.0

            sum_alpha += (multiply_factor) * (multiplicand)

        return GlobalVectors.POSITIVE_VALUE if sum_alpha > 0 else GlobalVectors.NEGATIVE_VALUE

    def classify_data_item(self, data_item, learn_tree_list):
        """
        Function that classifies a data item from test data set by traversing the trained decision tree.
        :param data_item:
        :param decision_tree:
        :return:
        """

        result_dict = dict()

        test_data = TestData()

        tree_classify_list = list()

        # Iterating to all the learnt bag trees to find majority vote.
        for ltree in learn_tree_list:
            current_node = ltree.decision_tree

            split_feature_value = data_item[current_node.split_feature_index]

            # Traverse tree until it finds the leaf node or the split on the feature value is not found.
            while len(current_node.children) > 0 and split_feature_value in current_node.children:
                current_node = current_node.children[split_feature_value]
                split_feature_value = data_item[current_node.split_feature_index]

            tree_classify_list.append(current_node.class_label)

        # Getting the majority vote here.
        classified = self.get_boosted_classification(tree_classify_list)

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
            test_data = self.classify_data_item(data_item, self.tree_boost_list)
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


# Unit Testing.
def main():
    data_handler = DataImportHandler()
    data_handler.import_mushroom_data("datasets/mushroom/agaricuslepiotatrain1.csv", "train", ",");
    data_handler.import_mushroom_data("datasets/mushroom/agaricuslepiotatest1.csv", "test", ",");

    boost = Boosting()
    test_util = TestUtil()

    boost.boosted_learn(5, 10)
    boost.classify_data_set(GlobalVectors.test_feature_vectors)

    print(test_util.get_accuracy())

    boost.print_confusion_matrix()

if __name__ == "__main__":
    main()