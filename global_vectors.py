__author__ = 'mangirish_wagle'


class GlobalVectors:
    """
    Class that stores global data vectors.
    """

    # Every record will be of format [class_label, f1, f2, f3...]
    # where f1, f2, f3 are features.
    feature_names = []
    feature_vectors = []

    # Test vector would contain test data for each data point
    test_data_vector = []

    POSITIVE_VALUE = "1"
    NEGATIVE_VALUE = "0"

    def __init__(self):
        return

    @staticmethod
    def set_feature_names(name_vector):
        GlobalVectors.feature_names = name_vector
        return

    @staticmethod
    def set_feature_vector(vectors):
        GlobalVectors.feature_vectors = vectors
        return

    @staticmethod
    def append_to_feature_vector(vector):
        GlobalVectors.feature_vectors.append(vector)
        return

    @staticmethod
    def append_to_feature_vector(vector):
        GlobalVectors.feature_vectors.append(vector)
        return

