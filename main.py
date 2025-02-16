import numpy as np
import pandas as pd
from dataLoader import dataLoader
from myshap import myshap
from models import Regr
import pathlib


"""
Machine Learning Pipeline for Model Training and Evaluation
A class that handles data loading, model training, and evaluation with SHAP analysis.
Supports various regression models with hyperparameter optimization and cross-validation.
"""
class MLPipeline:
    def __init__(self, file_path, y, x_list, model, results_dir,
                 cat_features=None, trials=50, test_ratio=0.3,
                 shap_ratio=0.3, cross_valid=5, random_state=0):
        self.file_path = file_path
        self.y = y
        self.x_list = x_list
        self.model = model
        self.results_dir = results_dir
        self.cat_features = cat_features
        self.trials = trials
        self.test_ratio = test_ratio
        self.shap_ratio = shap_ratio
        self.cross_valid = cross_valid
        self.random_state = random_state
        self._validate_inputs()
    
    """Validate input parameters"""
    def _validate_inputs(self):
        assert isinstance(self.file_path, (str, pathlib.Path)), \
            "`file_path` must be string or Path object"
        assert isinstance(self.y, (str, int)), \
            "`y` must be either a string or index within the whole dataset"
        assert (isinstance(self.x_list, list) and len(self.x_list) > 0) \
            and all(isinstance(x, (str, int)) for x in self.x_list), \
            "`x_list` must be non-empty list of strings or integers"
        assert isinstance(self.model, str) and self.model in ["cat", "rf", "dt", "lgb", "gbdt", "xgb", "ada", "svr", "knr", "mlp"], \
            "`model` must be a string and must be one of ['cat', 'rf', 'dt', 'lgb', 'gbdt', 'xgb', 'ada', 'svr', 'knr', 'mlp']"
        assert isinstance(self.results_dir, (str, pathlib.Path)), \
            "`results_dir` must be string or Path object"
        assert isinstance(self.cat_features, list) or self.cat_features is None, \
            "`cat_features` must be a list or None"
        assert isinstance(self.trials, int) and self.trials > 0, \
            "`trials` must be a positive integer"
        assert isinstance(self.test_ratio, float) and 0.0 < self.test_ratio < 1.0, \
            "`test_ratio` must be a float between 0 and 1"
        assert isinstance(self.shap_ratio, float) and 0.0 < self.shap_ratio < 1.0, \
            "`shap_ratio` must be a float between 0 and 1"
        assert isinstance(self.cross_valid, int) and self.cross_valid > 0, \
            "`cv` must be a positive integer"
        assert isinstance(self.random_state, int), \
            "`random_state` must be an integer"
        
    """Prepare training and test data"""
    def load_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = dataLoader(
            file_path=self.file_path,
            y=self.y,
            x_list=self.x_list, 
            cat_features=self.cat_features,
            test_ratio=self.test_ratio,
            random_state=self.random_state
        )

    """Optimize and evaluate the model"""
    def optimize(self):
        try:
            optimizer = Regr(
                cv=self.cross_valid,
                random_state=self.random_state,
                trials=self.trials,
                results_dir=self.results_dir,
            )
            self.optimal_model = optimizer.fit(self.x_train, self.y_train, 
                                             self.model, self.cat_features)
            optimizer.evaluate(self.optimal_model, self.x_test, self.y_test, 
                             self.x_train, self.y_train)
        except Exception as e:
            with open("error.txt", "w") as f:
                f.write(f"Model type: {self.model}\n")
                f.write(f"Time: {pd.Timestamp.now()}\n")
                f.write(f"Error: Model optimization failed: {str(e)}\n")
            raise RuntimeError(f"Model optimization failed: {str(e)}")
        
    """Use SHAP for explanation"""
    def explain(self):
        np.random.seed(self.random_state)
        all_data = pd.concat([self.x_train, self.x_test])
        shuffled_indices = np.random.permutation(all_data.index)
        shap_data = all_data.loc[
            np.random.choice(
                shuffled_indices,
                int(len(all_data) * self.shap_ratio),
                replace=False
            )]
        myshap(self.optimal_model, self.model, shap_data, self.results_dir)
        
    """Execute the full pipeline"""
    def run(self):
        self.load_data()
        self.optimize()
        self.explain()
        return None


if __name__ == "__main__":
    for i in [
        "svr", "knr", "mlp", "ada",
        "dt", "rf", "gbdt", "xgb", "lgb", "cat",
    ]:
        try:
            the_model = MLPipeline(
                file_path = "data.csv",
                y = "y",
                x_list = list(range(1, 16)),
                model = i,
                results_dir = "results/" + i,
                cat_features = None,
                trials = 200,
                test_ratio = 0.3,
                shap_ratio = 0.3,
                cross_valid = 5,
                random_state = 0,
            )
            the_model.run()
        except Exception as e:
            with open("error.txt", "w") as f:
                f.write(f"Model type: {i}\n")
                f.write(f"Time: {pd.Timestamp.now()}\n") 
                f.write(f"Error: {str(e)}\n")
            continue
