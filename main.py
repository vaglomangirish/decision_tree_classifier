__author__ = 'mangirish_wagle'

from pprint import pprint

from data_import_handler import DataImportHandler
from learn_tree import LearnTree
from test_util import TestUtil
from global_vectors import GlobalVectors


class Main:

    monks_data_location = "datasets/monks"
    cars_data_location = "datasets/cars"

    def __init__(self):
        return

    def print_analysis_for_depth(self, train_dataset_file_path, test_dataset_file_path, depth):
        """
        Prints analysis for data_set like monks for given depth of decision tree.
        :param depth: Depth of decision tree.
        :return:
        """
        data_import_handler = DataImportHandler()
        l_tree = LearnTree()
        t_util = TestUtil()

        print("Metrics for Depth- " + str(depth))

        # Import train data
        data_import_handler.import_data(train_dataset_file_path, "train")
        # Import test data
        data_import_handler.import_data(test_dataset_file_path, "test")

        # Learning decision tree for depth 1
        l_tree.create_decision_tree(GlobalVectors.train_feature_vectors, depth)
        # Classifying test data.
        t_util.classify_data_set(GlobalVectors.test_feature_vectors, l_tree.decision_tree)

        print("")
        print("Decision tree:")
        l_tree.print_decision_tree(l_tree.decision_tree)

        print("Percentage Accuracy for depth : " + str(t_util.get_accuracy()))

        print("")
        print("Confusion matrix:")
        l_tree.print_confusion_matrix()

    def print_analysis_for_monks(self):

        print("###########################################")
        print("   MONKS DATA SET   ")
        print("###########################################")

        # Monk 1
        print("Showing analysis for Monks-1 and depth 0")
        self.print_analysis_for_depth(self.monks_data_location +
                                      "/monks-1.train", self.monks_data_location + "/monks-1.test", 0)

        print("")
        print("--------------------------")

        print("Showing analysis for Monks-1 and depth 1")
        self.print_analysis_for_depth(self.monks_data_location +
                                      "/monks-1.train", self.monks_data_location + "/monks-1.test", 1)

        print("")
        print("--------------------------")

        print("Showing analysis for Monks-1 and depth 2")
        self.print_analysis_for_depth(self.monks_data_location +
                                      "/monks-1.train", self.monks_data_location + "/monks-1.test", 2)

        print("")
        print("--------------------------")

        print("Showing analysis for Monks-1 and depth 3")
        self.print_analysis_for_depth(self.monks_data_location +
                                      "/monks-1.train", self.monks_data_location + "/monks-1.test", 3)

        print("")
        print("--------------------------")

        print("Showing analysis for Monks-1 and depth 4")
        self.print_analysis_for_depth(self.monks_data_location +
                                      "/monks-1.train", self.monks_data_location + "/monks-1.test", 4)

        print("###########################")
        # Monk 2
        print("Showing analysis for Monks-2 and depth 0")
        self.print_analysis_for_depth(self.monks_data_location +
                                      "/monks-2.train", self.monks_data_location + "/monks-2.test", 0)

        print("")
        print("--------------------------")

        print("Showing analysis for Monks-2 and depth 1")
        self.print_analysis_for_depth(self.monks_data_location +
                                      "/monks-2.train", self.monks_data_location + "/monks-2.test", 1)

        print("")
        print("--------------------------")

        print("Showing analysis for Monks-2 and depth 2")
        self.print_analysis_for_depth(self.monks_data_location +
                                      "/monks-2.train", self.monks_data_location + "/monks-2.test", 2)

        print("")
        print("--------------------------")

        print("Showing analysis for Monks-2 and depth 3")
        self.print_analysis_for_depth(self.monks_data_location +
                                      "/monks-2.train", self.monks_data_location + "/monks-2.test", 3)

        print("")
        print("--------------------------")

        print("Showing analysis for Monks-2 and depth 4")
        self.print_analysis_for_depth(self.monks_data_location +
                                      "/monks-2.train", self.monks_data_location + "/monks-2.test", 4)

        print("###########################")
        # Monk 3

        print("Showing analysis for Monks-3 and depth 0")
        self.print_analysis_for_depth(self.monks_data_location +
                                      "/monks-3.train", self.monks_data_location + "/monks-3.test", 0)

        print("")
        print("--------------------------")

        print("Showing analysis for Monks-3 and depth 1")
        self.print_analysis_for_depth(self.monks_data_location +
                                      "/monks-3.train", self.monks_data_location + "/monks-3.test", 1)

        print("")
        print("--------------------------")

        print("Showing analysis for Monks-3 and depth 2")
        self.print_analysis_for_depth(self.monks_data_location +
                                      "/monks-3.train", self.monks_data_location + "/monks-3.test", 2)

        print("")
        print("--------------------------")

        print("Showing analysis for Monks-3 and depth 3")
        self.print_analysis_for_depth(self.monks_data_location +
                                      "/monks-3.train", self.monks_data_location + "/monks-3.test", 3)

        print("")
        print("--------------------------")

        print("Showing analysis for Monks-3 and depth 4")
        self.print_analysis_for_depth(self.monks_data_location +
                                      "/monks-3.train", self.monks_data_location + "/monks-3.test", 4)

    def print_analysis_for_cars(self):
        """
        This function prints info for custom dataset (Cars)
        :return:
        """

        print("")
        print("###########################################")
        print("   CUSTOM CARS DATA SET   ")
        print("###########################################")

        print("Showing analysis for cars and depth 0")
        self.print_analysis_for_depth(self.cars_data_location +
                                      "/car_data_1.train", self.cars_data_location + "/car_data_1.test", 0)

        print("--------------------------")

        print("Showing analysis for cars and depth 1")
        self.print_analysis_for_depth(self.cars_data_location +
                                      "/car_data_1.train", self.cars_data_location + "/car_data_1.test", 1)

        print("--------------------------")

        print("Showing analysis for cars and depth 2")
        self.print_analysis_for_depth(self.cars_data_location +
                                      "/car_data_1.train", self.cars_data_location + "/car_data_1.test", 2)

        print("--------------------------")

        print("Showing analysis for cars and depth 3")
        self.print_analysis_for_depth(self.cars_data_location +
                                      "/car_data_1.train", self.cars_data_location + "/car_data_1.test", 3)

        print("--------------------------")

        print("Showing analysis for cars and depth 4")
        self.print_analysis_for_depth(self.cars_data_location +
                                      "/car_data_1.train", self.cars_data_location + "/car_data_1.test", 4)


# Testing with main
def main():
    main_obj = Main()
    main_obj.print_analysis_for_monks()
    main_obj.print_analysis_for_cars()

if __name__ == "__main__":
    main()
