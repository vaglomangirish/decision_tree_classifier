__author__ = 'mangirish_wagle'

import random

from global_vectors import GlobalVectors
from data_import_handler import DataImportHandler


class FeatureAnalysis:
    """
    Assuming the data point is binary.

    {
      pos_data = {
                  pos: [1,3,5,7],
                  neg: [2,4,6,8],
                  label: <1/0>
                 }
      neg_data = {
                  pos: [11,13,15,17],
                  neg: [12,14,16,18]
                  label: <1/0>
                 }
    }

    """

    # Dicts for nalysis of data features with value 1 and 0
    pos_data = dict()  # 1
    neg_data = dict()  # 0

    POS_TAG = "pos"
    NEG_TAG = "neg"
    MAJOR_LABEL_TAG = "major_label"

    def __init__(self):

        # Init list of positive and negative indices for feature value 1.
        self.pos_data[self.POS_TAG] = list()
        self.pos_data[self.NEG_TAG] = list()
        self.pos_data[self.MAJOR_LABEL_TAG] = None

        # Init list of positive and negative indices for feature value 0.
        self.neg_data[self.POS_TAG] = list()
        self.neg_data[self.NEG_TAG] = list()
        self.neg_data[self.MAJOR_LABEL_TAG] = None

        return

    def analyze_feature(self, feature_index):

        self.__init__()

        for index in xrange (0, len(GlobalVectors.train_feature_vectors)):

            if GlobalVectors.train_feature_vectors[index][feature_index] == GlobalVectors.POSITIVE_VALUE:
                # Feature value 1

                if GlobalVectors.train_feature_vectors[index][GlobalVectors.label_index]\
                                                                == GlobalVectors.POSITIVE_VALUE:
                    self.pos_data[self.POS_TAG].append(index)
                else:
                    self.pos_data[self.NEG_TAG].append(index)
            else:
                # Feature value 0

                if GlobalVectors.train_feature_vectors[index][GlobalVectors.label_index]\
                                                                == GlobalVectors.POSITIVE_VALUE:
                    self.neg_data[self.POS_TAG].append(index)
                else:
                    self.neg_data[self.NEG_TAG].append(index)

        # Assign labels based on majority
        if len(self.pos_data[self.POS_TAG]) > len(self.pos_data[self.NEG_TAG]):
            self.pos_data[self.MAJOR_LABEL_TAG] = GlobalVectors.POSITIVE_VALUE
        elif len(self.pos_data[self.NEG_TAG]) > len(self.pos_data[self.POS_TAG]):
            self.pos_data[self.MAJOR_LABEL_TAG] = GlobalVectors.NEGATIVE_VALUE
        else:
            self.pos_data[self.MAJOR_LABEL_TAG] = random.choice([GlobalVectors.POSITIVE_VALUE,
                                                                 GlobalVectors.NEGATIVE_VALUE])

        if len(self.neg_data[self.POS_TAG]) > len(self.neg_data[self.NEG_TAG]):
            self.neg_data[self.MAJOR_LABEL_TAG] = GlobalVectors.POSITIVE_VALUE
        elif len(self.neg_data[self.NEG_TAG]) > len(self.neg_data[self.POS_TAG]):
            self.neg_data[self.MAJOR_LABEL_TAG] = GlobalVectors.NEGATIVE_VALUE
        else:
            self.neg_data[self.MAJOR_LABEL_TAG] = random.choice([GlobalVectors.POSITIVE_VALUE,
                                                                 GlobalVectors.NEGATIVE_VALUE])

        return

    def get_feature_weighted_error(self, feature_index):
        """
        Function assumes that the global train_weight_vector is initialized.
        :param feature_index:
        :return:
        """
        self.analyze_feature(feature_index)

        weighted_error_sum = 0.0

        error_list1 = list()
        error_list2 = list()

        if GlobalVectors.train_weight_vector is not None and len(GlobalVectors.train_weight_vector) > 0:

            # checking for feature value = 1
            if self.pos_data[self.MAJOR_LABEL_TAG] == GlobalVectors.POSITIVE_VALUE: # Label = 1
                error_list1 = self.pos_data[self.NEG_TAG]
            else:  # Label = 0
                error_list1 = self.pos_data[self.POS_TAG]

            for index in error_list1:
                weighted_error_sum += GlobalVectors.train_weight_vector[index]

            # checking for feature value = 0
            if self.neg_data[self.MAJOR_LABEL_TAG] == GlobalVectors.POSITIVE_VALUE:  # Label = 1
                error_list2 = self.neg_data[self.NEG_TAG]
            else:  # Label = 0
                error_list2 = self.neg_data[self.POS_TAG]

            for index in error_list2:
                weighted_error_sum += GlobalVectors.train_weight_vector[index]

            return weighted_error_sum


def main():

    data_handler = DataImportHandler()
    data_handler.import_mushroom_data("datasets/mushroom/agaricuslepiotatrain1.csv", "train", ",");
    data_handler.import_mushroom_data("datasets/mushroom/agaricuslepiotatest1.csv", "test", ",");

    featanalyze = FeatureAnalysis()
    featanalyze.get_feature_weighted_error(1)

if __name__ == "__main__":
    main()

