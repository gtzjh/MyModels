import pandas as pd
from autogluon.tabular import TabularPredictor


def ensemble(
        x_train, x_test, y_train, y_test, 
        y_label, 
        train_time = 60*60*2  # About 2 hours
    ):
    train_data = pd.concat([y_train, x_train], axis = 1)
    test_data = pd.concat([y_test, x_test], axis = 1)

    predictor = TabularPredictor(
        label = y_label,
        eval_metric = "r2",
        verbosity = 0
    )
    predictor.fit(
        train_data = train_data,
        time_limit = train_time,
        presets = "best_quality"
    )
    eva = dict(predictor.evaluate(test_data))  # Test the model
    print({
        "R2": round(eva["r2"], 6),
        "MAE": -round(eva["mean_absolute_error"], 6),
        "RMSE": -round(eva["root_mean_squared_error"], 6)
    })

    return None



if __name__ == "__main__":
    ensemble()
