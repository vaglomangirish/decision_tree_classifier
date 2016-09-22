__author__ = 'mangirish_wagle'


class TreeNode:

    # Feature to split on.
    split_feature_index = None

    # Map of feature_split_value to children nodes.
    children = {}

    # Positive and negative vector in the format [positive_count, negative_count]
    pos_neg = [0, 0]

    def __init__(self):
        return

    @classmethod
    def set_split_feature_index(cls, index):
        cls.split_feature_index = index

    @classmethod
    def append_child(cls, split_feature_value, child_node):
        cls.children[split_feature_value] = child_node

    @classmethod
    def set_pos_neg(cls, positive, negative):
        cls.pos_neg = [positive, negative]

