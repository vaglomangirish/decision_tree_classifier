__author__ = 'mangirish_wagle'


class TestData:
    """
    Data structure that stores test data for a data point.
    """

    predicted_class = None
    is_false_positive = None
    is_false_negative = None
    is_true_positive = None
    is_true_negative = None

    def __init__(self):
        return

    def set_predicted_class(self, pred_class):
        self.predicted_class = pred_class

    def set_is_false_positive(self, is_fp):
        self.is_false_positive = is_fp

    def set_is_false_negative(self, is_fn):
        self.is_false_negative = is_fn

    def set_is_true_positive(self, is_tp):
        self.is_true_positive = is_tp

    def set_is_true_negative(self, is_tn):
        self.is_true_negative = is_tn
