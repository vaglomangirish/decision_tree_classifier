__author__ = 'mangirish_wagle'

from global_vectors import GlobalVectors


class DataImportHandler:

    monks_data_location = "datasets/monks"

    def __init__(self):
       return

    def import_monks_data(self, index, data_type):
        """
        Function to import the monk's data set
        Attribute information:
        1. class: 0, 1   Considering 1-> Positive 0-> Negative
        2. a1:    1, 2, 3
        3. a2:    1, 2, 3
        4. a3:    1, 2
        5. a4:    1, 2, 3
        6. a5:    1, 2, 3, 4
        7. a6:    1, 2
        8. Id:    (A unique symbol for each instance)
        :param index: index of the data set 1-> monks1, 2-> monks2
        :param data_type: train/ test
        :return:
        """

        # The name vector stores the names attributes of a vector in feature_vectors.
        name_vector = ["class", "a1", "a2", "a3", "a4", "a5", "a6"]

        GlobalVectors.feature_names = name_vector

        if data_type == "train":
            GlobalVectors.clear_train_feature_vector()
        elif data_type == "test":
            GlobalVectors.clear_test_feature_vector()

        # Reading the data set file to store data in feature_vector in the specified format.
        with open(self.monks_data_location + "/monks-" + index + "." + data_type) as monks_file_data:
            for line in monks_file_data:
                line.strip()
                vector_arr = line.split(" ")
                vector = vector_arr[1:(len(vector_arr)-1)]
                # print(vector)

                if data_type == "train":
                    GlobalVectors.append_to_train_feature_vector(vector)
                elif data_type == "test":
                    GlobalVectors.append_to_test_feature_vector(vector)

    def import_data(self, data_file_path, data_type):
        """
        Generic function to import training and test data.
        :param data_file_path: path to the data file
        :param data_type: test/ train
        :return:
        """

        if data_type == "train":
            GlobalVectors.clear_train_feature_vector()
        elif data_type == "test":
            GlobalVectors.clear_test_feature_vector()

        # Reading the data set file to store data in feature_vector in the specified format.
        with open(data_file_path) as monks_file_data:
            for line in monks_file_data:
                line = line.strip()
                vector_arr = line.split(" ")
                vector = vector_arr[0:(len(vector_arr)-2)]
                # print(vector)

                if data_type == "train":
                    GlobalVectors.append_to_train_feature_vector(vector)
                elif data_type == "test":
                    GlobalVectors.append_to_test_feature_vector(vector)


# Testing with main
def main():
    monks_handler = DataImportHandler()
    monks_handler.import_data(monks_handler.monks_data_location + "/monks-1.test", "test")
    print(GlobalVectors.feature_names)
    print(GlobalVectors.test_feature_vectors)

if __name__ == "__main__": main()