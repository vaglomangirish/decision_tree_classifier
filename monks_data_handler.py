__author__ = 'mangirish_wagle'

from global_vectors import GlobalVectors

class MonksDataHandler:

    monks_data_location = "datasets/monks"

    def __init__(self):
       return

    @classmethod
    def import_monks_data(cls, index, data_type):

        with open(cls.monks_data_location + "/monks-" + index + "." + data_type) as monks_file_data:
            for line in monks_file_data:
                vector_arr = line.split(" ")
                vector = vector_arr[1:8]
                print(vector)
                GlobalVectors.append_to_feature_vector(vector)


def main():
    monks_handler = MonksDataHandler()
    monks_handler.import_monks_data("1", "test")
    print(GlobalVectors.feature_vectors)

if __name__ == "__main__": main()