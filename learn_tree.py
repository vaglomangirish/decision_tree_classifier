__author__ = 'mangirish_wagle'

from tree_node import TreeNode
from global_vectors import GlobalVectors
from data_import_handler import DataImportHandler
from eval_util import EvalUtil
from data_utils import DataUtils
from test_util import TestUtil


class LearnTree:

    decision_tree = TreeNode()

    def __init__(self):
        return

    def test_pure_class(self, data_set):

        is_pure = False

        data_utils = DataUtils()

        positive, negative = data_utils.get_pos_neg_count(data_set)

        if positive == 0 or negative == 0:
            is_pure = True

        return is_pure, positive, negative

    def test_termination_condition(self, data_set, max_depth, feature_index_list):

        termination = True

        is_pure_class, positive, negative = self.test_pure_class(data_set)

        if max_depth > 1 and feature_index_list is not None and len(feature_index_list) > 0 and not is_pure_class:
            termination = False

        return termination, positive, negative

    def create_decision_tree(self, data_set, max_depth):

        init_index_list = []

        # Creating initial list of indices. This will contain indices to all features.
        for index in xrange(1, len(data_set[0])):
            init_index_list.append(index)

        self.decision_tree = self.create_node(data_set, max_depth, init_index_list)

    def create_node(self, data_set, max_depth, feature_index_list):

        # tree_node = None

        is_termination_condition, positive, negative = self.test_termination_condition(data_set,
                                                                                      max_depth, feature_index_list)

        tree_node = TreeNode()
        eval_util = EvalUtil()
        data_util = DataUtils()

        max_info_gain_index = eval_util.get_split_attribute_index(data_set, feature_index_list)

        #print max_info_gain_index
        tree_node.set_split_feature_index(max_info_gain_index)

        split_feature_values = data_util.get_feature_discrete_values(data_set, max_info_gain_index)

        tree_node.set_pos_neg(positive, negative)

        revised_index_list = [x for x in feature_index_list if x != max_info_gain_index]

        if not is_termination_condition:
            for value in split_feature_values:
                data_subset = data_util.get_data_subset(data_set, max_info_gain_index, value)
                if len(data_subset) > 0:
                    child_node = self.create_node(data_subset, max_depth - 1, revised_index_list)
                    tree_node.append_child(value, child_node)
                else:
                    tree_node.append_child(value, None)

        return tree_node

    def print_confusion_matrix(self):
        t_util = TestUtil()
        print("{0:20s} {1:18s} {2:18s}".format("", "Predicted Negative", "Predicted Positive"))
        print("{0:20s} {1:18d} {2:18d}".format("Actual Negative", t_util.get_true_negative_count(), t_util.get_false_positive_count()))
        print("{0:20s} {1:18d} {2:18d}".format("Actual Positive", t_util.get_false_negative_count(), t_util.get_true_positive_count()))


# Testing with main
def main():

    monks_handler = DataImportHandler()
    monks_handler.import_monks_data("3", "train")
    monks_handler.import_monks_data("3", "test")

    l_tree = LearnTree()
    t_util = TestUtil()

    l_tree.create_decision_tree(GlobalVectors.train_feature_vectors, 3)
    # l_tree.is_pure_class(GlobalVectors.feature_vectors)

    t_util.classify_data_set(GlobalVectors.test_feature_vectors, l_tree.decision_tree)

    # print(l_tree.decision_tree)

    # print(GlobalVectors.test_data_vector)

    # print(t_util.get_accuracy())
    l_tree.print_confusion_matrix()


if __name__ == "__main__": main()