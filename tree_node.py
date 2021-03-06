__author__ = 'mangirish_wagle'

import copy


class TreeNode:

    # Feature to split on.
    split_feature_index = None

    # Map of feature_split_value to children nodes.
    children = {}

    # Positive and negative vector in the format [positive_count, negative_count]
    # pos_neg = [0, 0]

    class_label = None

    def __init__(self):
        self.children = {}
        return

    def __str__(self, level=0, feature_value=None):
        if feature_value is None:
            feature_value = "NA"
        ret = "\t"*level+repr("[" + str(level) + "]" + "Parent feature value = " + feature_value + " | Split Feature Index: "
                              + str(self.split_feature_index) + "| Class:" + self.class_label)+"\n"
        for child in self.children:
            ret += self.children[child].__str__(level+1, child)
        return ret

    def set_split_feature_index(self, index):
        self.split_feature_index = index

    def append_child(self, split_feature_value, child_node):
        self.children[split_feature_value] = child_node

    def set_class_label(self, label):
        self.class_label = label


