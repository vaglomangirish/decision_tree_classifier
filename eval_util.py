__author__ = 'mangirish_wagle'

from data_import_handler import DataImportHandler
from global_vectors import GlobalVectors

import math


class EvalUtil:
    """
    Class that provides utility methods for evaluation of data.
    """

    def __init__(self):
        return

    def get_entropy(self, pos_count, neg_count):
        """
        Function to calculate entropy
        :param pos_count: Positive class count
        :param neg_count: Negative class count
        :return: Entropy calculated
        """

        prob_pos = pos_count/float(pos_count + neg_count)
        prob_neg = neg_count/float(pos_count + neg_count)

        if prob_pos == 0.0 or prob_neg == 0.0:
            return 0.0;

        entropy = ( - prob_pos * math.log(prob_pos, 2) ) - ( prob_neg * math.log(prob_neg, 2) )

        return entropy

    def get_information_gain(self, data_set, feature_index):

        entropy_s = self.get_entropy(self.get_total_positive(data_set), self.get_total_negative(data_set))

        feature_val_count_dict = {}

        for vector in data_set:
            feature_val = vector[feature_index]
            class_val = vector[0]

            # feature_val_count_dict record: {value: [postive count, negative_count]}
            if feature_val is not None and feature_val not in feature_val_count_dict:
                feature_val_count_dict[feature_val] = [0, 0]

            # Update positive/ negative count
            if class_val == GlobalVectors.POSITIVE_VALUE:
                feature_val_count_dict[feature_val][0] += 1
            elif class_val == GlobalVectors.NEGATIVE_VALUE:
                feature_val_count_dict[feature_val][1] += 1

        value_entropy_sum = 0

        for value in feature_val_count_dict:
            total_value_count = feature_val_count_dict[value][0] + feature_val_count_dict[value][1]
            value_entropy_sum += (total_value_count/float(len(data_set))) \
                           * self.get_entropy(feature_val_count_dict[value][0], feature_val_count_dict[value][1])

        info_gain = entropy_s - value_entropy_sum

        return info_gain

    def get_total_positive(self, data_set):

        pos_count = 0

        for vector in data_set:
            if vector[0] == GlobalVectors.POSITIVE_VALUE:
                pos_count += 1

        return pos_count

    def get_total_negative(self, data_set):

        neg_count = 0

        for vector in data_set:
            if vector[0] == GlobalVectors.NEGATIVE_VALUE:
                neg_count += 1

        return neg_count

    def get_split_attribute_index(self, data_set, index_list):

        max_info_gain = None
        max_info_gain_index = None

        for index in index_list:
            info_gain = self.get_information_gain(data_set, index)
            if max_info_gain is None or info_gain > max_info_gain:
                max_info_gain = info_gain
                max_info_gain_index = index

        return max_info_gain_index


# Testing with main
def main():
    monks_handler = DataImportHandler()
    monks_handler.import_monks_data("1", "train")
    evalutil = EvalUtil()

    attr = evalutil.get_split_attribute_index(GlobalVectors.train_feature_vectors, [1,3,6])
    print(attr)

if __name__ == "__main__": main()