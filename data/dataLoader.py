import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def dataLoader(
    file_path, 
    y_index,  # Only one column can be specify as y
    x_index_list, 
    test_size = 0.3,
    random_state = 42
):
    _df = pd.read_csv(
        file_path, encoding = "utf-8", na_values = np.nan
    )
    
    y_data = _df.iloc[:, y_index]
    x_data = _df.iloc[:, x_index_list]

    return train_test_split(  # x_train, x_test, y_train, y_test
        x_data, y_data, 
        test_size = test_size, 
        random_state = random_state
    )



if __name__ == "__main__":
    x_train, x_test, y_train, y_test = dataLoader()
    