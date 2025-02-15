import numpy as np
import pandas as pd
from dataLoader import dataLoader
from myshap import myshap
from myregressors import Regr


def main(file_path, y, x_list, model, results_dir,
         cat_features = None,
         trials = 50,
         test_ratio = 0.3,
         shap_ratio = 0.3,
         cross_valid = 5,
         random_state = 0) -> None:
    ###########################################################################
    """Input validation"""
    assert isinstance(shap_ratio, float) and 0.0 < shap_ratio < 1.0, \
        "`shap_ratio` must be a float between 0 and 1"
    ###########################################################################

    ###########################################################################
    """Data preparing"""
    x_train, x_test, y_train, y_test = dataLoader(
        file_path = file_path,
        y = y,
        x_list = x_list,
        cat_features = cat_features,
        test_ratio = test_ratio,
        random_state = random_state
    )
    ###########################################################################

    ###########################################################################
    """Execute machine learning"""
    optimizer = Regr(
        cv = cross_valid,
        random_state = random_state,
        trials = trials,
        results_dir = results_dir,
    )
    optimal_model = optimizer.fit(x_train, y_train, model, cat_features)
    optimizer.evaluate(optimal_model, x_test, y_test, x_train, y_train)
    ###########################################################################

    ###########################################################################
    """SHAP test"""
    """
    np.random.seed(random_state)
    all_data = pd.concat([x_train, x_test])
    shap_data = all_data.loc[
        np.random.choice(
            all_data.index,
            int(len(all_data) * shap_ratio),
            replace = False
        )]
    myshap(optimal_model, model, shap_data, results_dir)
    """
    ###########################################################################

    return None


if __name__ == "__main__":
    
    for i in [
        # "dt", "rf", "gbdt", "xgb", "lgb", "cat", "ada",
        "svr", "knr", "mlp",
    ]:
        print(f"{i} started")
        main(
            file_path = "data.csv",
            y = "y",
            x_list = list(range(1, 16)),
            model = i,
            results_dir = "results/" + i,
            # cat_features = ["x16", "x17"],
            trials = 50,
            test_ratio = 0.3,
            shap_ratio = 0.3,
            cross_valid = 5,
            random_state = 0,
        )
        print(f"{i} finished")
    
    """
    main(
        file_path = "data2.csv",
        y = "y",
        x_list = list(range(1, 18)),
        model = "cat",
        results_dir = "results/cat",
        cat_features = ["x16", "x17"],
        trials = 10,
        test_ratio = 0.3,
        shap_ratio = 0.3,
        cross_valid = 5,
        random_state = 0,
        )
    """