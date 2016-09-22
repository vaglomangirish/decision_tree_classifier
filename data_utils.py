__author__ = 'mangirish_wagle'

from global_vectors import GlobalVectors


class DataUtils:

    def __init__(self):
        return

    def get_data_subset(self, data_set, feature_index, feature_value):
        subset = []

        for data_item in data_set:
            if data_item[feature_index] == feature_value:
                subset.append(data_item)

        return subset

    def get_feature_discrete_values(self, data_set, feature_index):

        feature_values = []

        for data_item in data_set:
            if data_item[feature_index] not in feature_values:
                feature_values.append(data_item[feature_index])

        return feature_values

    def get_pos_neg_count(self, data_set):

        positive_count = 0
        negative_count = 0

        for data_item in data_set:
            if data_item[0] == GlobalVectors.POSITIVE_VALUE:
                positive_count += 1
            elif data_item[0] == GlobalVectors.NEGATIVE_VALUE:
                negative_count += 1

        return positive_count, negative_count
