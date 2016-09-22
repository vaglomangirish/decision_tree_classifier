__author__ = 'mangirish_wagle'

from tree_node import TreeNode
from global_vectors import GlobalVectors
from monks_data_handler import MonksDataHandler
from eval_util import EvalUtil
from data_utils import DataUtils


class LearnTree:

    decision_tree = TreeNode()

    def __init__(self):
        return

    @classmethod
    def test_pure_class(cls, data_set):

        is_pure = False

        data_utils = DataUtils()

        positive, negative = data_utils.get_pos_neg_count(data_set)

        if positive == 0 or negative == 0:
            is_pure = True

        return is_pure, positive, negative

    @classmethod
    def test_termination_condition(cls, data_set, max_depth, feature_index_list):

        termination = True

        is_pure_class, positive, negative = cls.test_pure_class(data_set)

        if max_depth > 0 and feature_index_list is not None and len(feature_index_list) > 0 and not is_pure_class:
            termination = False

        return termination, positive, negative

    @classmethod
    def create_decision_tree(cls, data_set, max_depth):

        init_index_list = []

        # Creating initial list of indices. This will contain indices to all features.
        for index in xrange(1, len(data_set[0])):
            init_index_list.append(index)

        LearnTree.decision_tree = cls.create_node(GlobalVectors.feature_vectors, max_depth, init_index_list)

    @classmethod
    def create_node(cls, data_set, max_depth, feature_index_list):

        tree_node = None

        is_termination_condition, positive, negative = cls.test_termination_condition(data_set,
                                                                                      max_depth, feature_index_list)

        if not is_termination_condition:
            tree_node = TreeNode()
            eval_util = EvalUtil()
            data_util = DataUtils()

            max_info_gain_index = eval_util.get_split_attribute_index(data_set, feature_index_list)

            tree_node.set_split_feature_index(max_info_gain_index)

            split_feature_values = data_util.get_feature_discrete_values(data_set, max_info_gain_index)

            for value in split_feature_values:
                data_subset = data_util.get_data_subset(data_set, max_info_gain_index, value)
                child_node = cls.create_node(data_subset, max_depth - 1, feature_index_list.remove(max_info_gain_index))
                tree_node.set_pos_neg(positive, negative)
                tree_node.append_child(child_node)

        return tree_node


# Testing with main
def main():

    monks_handler = MonksDataHandler()
    monks_handler.import_monks_data("1", "train")
    l_tree = LearnTree()

    l_tree.create_decision_tree(GlobalVectors.feature_vectors, 2)
    #l_tree.is_pure_class(GlobalVectors.feature_vectors)


if __name__ == "__main__": main()