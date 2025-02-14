import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


def dataLoader(file_path, y, x_list, cat_features, test_ratio, random_state) \
    -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load and preprocess data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.
        y (str or int): The column name or index of the dependent variable.
        x_list (list): A list of column names or indices of the independent variables.
        cat_features (list or None): A list of column names or indices of the categorical features.
        test_ratio (float): The ratio of the test set.
        random_state (int): The random state for the train_test_split.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple of training and testing data.
    """

    _df = pd.read_csv(file_path, encoding = "utf-8", na_values = np.nan)

    #----------------------------------------------------------------------------------------
    assert isinstance(y, str) or isinstance(y, int)  # Check if y is either a string or integer
    if isinstance(y, str):
        y_data = _df.loc[:, y]  # Select column using column name if y is a string
    else:
        y_data = _df.iloc[:, y]  # Select column using integer index if y is an integer
    print("\nDependency:")
    print(y_data.info())
    print(y_data.describe())
    #----------------------------------------------------------------------------------------
    
    #----------------------------------------------------------------------------------------
    # Check if x_list is a list
    assert isinstance(x_list, list)
    # Check if all elements in x_list are either strings or integers
    assert all([isinstance(i, str) for i in x_list]) or all([isinstance(i, int) for i in x_list])
    #----------------------------------------------------------------------------------------
    
    #----------------------------------------------------------------------------------------
    # Handle categorical features if specified
    if cat_features is not None:
        # Verify that categorical features are of object or category dtype
        if all([isinstance(col, str) for col in cat_features]):
            pass
        else:
            cat_features = _df.columns[cat_features].tolist()
        assert all([_df.loc[:,col].dtype == 'object' for col in cat_features]) or \
                all([_df.loc[:,col].dtype == 'category' for col in cat_features]), \
                "Categorical features must be of object/category type"
        
        # Convert categorical features to category dtype
        for col in cat_features:
            _df[col] = _df[col].astype('category')
    #----------------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------------
    # Select columns using column names if x_list contains strings
    if all([isinstance(i, str) for i in x_list]):
        x_data = _df.loc[:, x_list]
    # Select columns using integer indices if x_list contains integers 
    else:
        x_data = _df.iloc[:, x_list]
    print("\nIndependency:")
    print(x_data.info())
    print(x_data.describe())
    #----------------------------------------------------------------------------------------
    
    #----------------------------------------------------------------------------------------
    # Split the data into training and testing sets
    return train_test_split(
        x_data, y_data, 
        test_size = test_ratio, 
        random_state = random_state
    )
    #----------------------------------------------------------------------------------------