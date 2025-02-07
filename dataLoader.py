import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def dataLoader(
    file_path, 
    y,  # Only one column can be specify as y
    x_list, 
    test_ratio = 0.3,
    random_state = 0
):
    _df = pd.read_csv(file_path, encoding = "utf-8", na_values = np.nan)

    #----------------------------------------------------------------------------------------
    assert isinstance(y, str) or isinstance(y, int)
    if isinstance(y, str):
        y_data = _df.loc[:, y]
    else:
        y_data = _df.iloc[:, y]
    print("\nDependency:")
    print(y_data.describe())
    #----------------------------------------------------------------------------------------
    
    #----------------------------------------------------------------------------------------
    # Judge x variables
    assert isinstance(x_list, list)
    assert all([isinstance(i, str) for i in x_list]) or all([isinstance(i, int) for i in x_list])
    if all([isinstance(i, str) for i in x_list]):
        x_data = _df.loc[:, x_list]
    else:
        x_data = _df.iloc[:, x_list]
    print("\nIndependency:")
    print(x_data.describe())
    #----------------------------------------------------------------------------------------
    
    # x_train, x_test, y_train, y_test
    return train_test_split(
        x_data, y_data, 
        test_size = test_ratio, 
        random_state = random_state
    )

    