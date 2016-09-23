__author__ = 'mangirish_wagle'

from global_vectors import GlobalVectors
import random


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

    def get_class_label(self, data_set):

        label_count_dict = {}

        # iterate and count all class labels
        for data_item in data_set:
            if data_item[0] not in label_count_dict:
                label_count_dict[data_item[0]] = 1
            else:
                label_count_dict[data_item[0]] += 1

        is_pure_class = True if len(label_count_dict) == 1 else False

        # return class with highest count. If in case of conflict, return a random class with max count.
        max_labels = []

        max_count = 0
        for label in label_count_dict:
            if label_count_dict[label] > max_count:
                del max_labels
                max_labels = []
                max_labels.append(label)
            elif label_count_dict[label] == max_count:
                max_labels.append(label)

        max_label_to_return = random.choice(max_labels)

        return  max_label_to_return, is_pure_class

    def get_pos_neg_count(self, data_set):

        positive_count = 0
        negative_count = 0

        for data_item in data_set:
            if data_item[0] == GlobalVectors.POSITIVE_VALUE:
                positive_count += 1
            elif data_item[0] == GlobalVectors.NEGATIVE_VALUE:
                negative_count += 1

        return positive_count, negative_count
