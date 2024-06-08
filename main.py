from data.dataLoader import dataLoader
import numpy as np
import pandas as pd



def main():
    x_train, x_test, y_train, y_test = dataLoader(
        file_path = "data/data.csv",
        y_index = 0, 
        x_index_list = [3, 4, 5]
    )


    return None


if __name__ == "__main__":
    main()