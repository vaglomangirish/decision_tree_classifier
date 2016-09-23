__author__ = 'mangirish_wagle'

import copy

class TreeNode:

    # Feature to split on.
    split_feature_index = None

    # Map of feature_split_value to children nodes.
    children = {}

    # Positive and negative vector in the format [positive_count, negative_count]
    pos_neg = [0, 0]

    def __init__(self):
        self.children = {}
        return

    def set_split_feature_index(self, index):
        self.split_feature_index = index

    def append_child(self, split_feature_value, child_node):
        self.children[split_feature_value] = child_node

    def set_pos_neg(self, positive, negative):
        self.pos_neg = [positive, negative]

