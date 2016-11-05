__author__ = 'mangirish_wagle'


class GlobalVectors:
    """
    Class that stores global data vectors.
    """

    # Index of label in data point vector.
    label_index = 0;

    # Every record will be of format [class_label, f1, f2, f3...]
    # where f1, f2, f3 are features.
    feature_names = []
    train_feature_vectors = []
    test_feature_vectors = []

    # Weight vector would contain weights of each training data point.
    train_weight_vector = []

    # Error train vector contains indices of misclassified points from train data.
    error_train_vector =[]

    # Success train vector contains indices of correctly classified data points.
    success_train_vector = []

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
        GlobalVectors.train_feature_vectors = vectors
        return

    @staticmethod
    def append_to_feature_vector(vector):
        GlobalVectors.train_feature_vectors.append(vector)
        return

    @staticmethod
    def append_to_train_feature_vector(vector):
        GlobalVectors.train_feature_vectors.append(vector)
        return

    @staticmethod
    def append_to_test_feature_vector(vector):
        GlobalVectors.test_feature_vectors.append(vector)
        return

    @staticmethod
    def clear_train_feature_vector():
        del GlobalVectors.train_feature_vectors
        GlobalVectors.train_feature_vectors = []
        return

    @staticmethod
    def clear_test_feature_vector():
        del GlobalVectors.test_feature_vectors
        GlobalVectors.test_feature_vectors = []
        return

    @staticmethod
    def clear_test_data_vector():
        del GlobalVectors.test_data_vector
        GlobalVectors.test_data_vector = []
        return

