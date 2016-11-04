__author__ = 'mangirish_wagle'

import math

from global_vectors import GlobalVectors
from data_import_handler import DataImportHandler
from learn_tree import LearnTree
from test_util import TestUtil


class Boosting:
    """
    Class that contains boosting related utils.
    """

    tree_boost_list = list()
    tree_alpha_list = list()

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

        alpha = (1.0/2.0) * math.log((1.0 - float(weighted_error)) / float(weighted_error))
        return alpha

    def initialize_weights(self):

        for point in GlobalVectors.train_feature_vectors:
            GlobalVectors.train_weight_vector.append(float(1.0/len(GlobalVectors.train_feature_vectors)))

    def analyze_data_for_errors(self, tree):

        test_util = TestUtil()

        for index in xrange(len(GlobalVectors.train_feature_vectors)):
            test_data = test_util.classify_data_item(GlobalVectors.train_feature_vectors[index], tree)

            if test_data.is_error:
                self.errored_index_list.append(index)

    def get_global_weighted_error(self):

        weighted_error_sum = 0.0

        if GlobalVectors.train_weight_vector is not None and len(GlobalVectors.train_weight_vector) > 0:
            for error_index in self.errored_index_list:
                weighted_error_sum += GlobalVectors.train_weight_vector[error_index]

        return weighted_error_sum

    def update_weights(self, alpha):

        for index in xrange(len(GlobalVectors.train_weight_vector)):

            if index in self.errored_index_list:
                GlobalVectors.train_weight_vector[index] *= math.exp(alpha)
            else:
                GlobalVectors.train_weight_vector[index] *= math.exp(alpha*(-1.0))

    def boosted_learn(self, depth, no_of_trees):

        self.initialize_weights()

        for count in xrange(no_of_trees):
            ltree = LearnTree()
            ltree.create_decision_tree(GlobalVectors.train_feature_vectors, depth, True)

            self.analyze_data_for_errors(ltree.decision_tree)
            global_weighted_error = self.get_global_weighted_error()
            alpha = self.get_coeff_alpha(global_weighted_error)

            self.update_weights(alpha)

            print("Alpha " + str(count) + ": " + str(alpha))
            ltree.print_decision_tree(ltree.decision_tree)

            self.tree_boost_list.append(ltree)
            self.tree_alpha_list.append(alpha)

def main():
    data_handler = DataImportHandler()
    data_handler.import_mushroom_data("datasets/mushroom/agaricuslepiotatrain1.csv", "train", ",");
    data_handler.import_mushroom_data("datasets/mushroom/agaricuslepiotatest1.csv", "test", ",");

    boost = Boosting()

    boost.boosted_learn(3,5)

if __name__ == "__main__":
    main()