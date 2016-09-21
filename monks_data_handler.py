__author__ = 'mangirish_wagle'

from global_vectors import GlobalVectors


class MonksDataHandler:

    monks_data_location = "datasets/monks"

    def __init__(self):
       return

    @classmethod
    def import_monks_data(cls, index, data_type):
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

        # Reading the data set file to store data in feature_vector in the specified format.
        with open(cls.monks_data_location + "/monks-" + index + "." + data_type) as monks_file_data:
            for line in monks_file_data:
                vector_arr = line.split(" ")
                vector = vector_arr[1:8]
                # print(vector)
                GlobalVectors.append_to_feature_vector(vector)

# Testing with main
def main():
    monks_handler = MonksDataHandler()
    monks_handler.import_monks_data("1", "test")
    print(GlobalVectors.feature_names)
    print(GlobalVectors.feature_vectors)

if __name__ == "__main__": main()