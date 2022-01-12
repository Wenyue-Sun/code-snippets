# define a ML function
from time import time

from sklearn.datasets import make_classification
from sklearn.datasets import load_digits

import autosklearn.classification
from autosklearn.experimental.askl2 import AutoSklearn2Classifier

import autosklearn.regression

# none of the auto-ml function is working yet
## ...

def run_ml(n_jobs=1):
    # define dataset
    X, y = load_digits(return_X_y=True)
    # X, y = make_classification(n_samples=5000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
    # define the model

    # model = autosklearn.classification.AutoSklearnClassifier(
    #     time_left_for_this_task=31,
    #     ensemble_size=10,
    #     n_jobs=n_jobs,
    #     delete_tmp_folder_after_terminate=True
    # )
    model = autosklearn.regression.AutoSklearnRegressor(
        n_jobs=n_jobs, 
        ensemble_size=10,
        time_left_for_this_task=31
    )
    # model = AutoSklearn2Classifier(
    #     time_left_for_this_task=31,
    #     ensemble_size=10,
    #     n_jobs=n_jobs,
    #     delete_tmp_folder_after_terminate=True
    # )
    # record current time
    start = time()
    # fit the model
    model.fit(X, y)
    y_pred = model.predict(X)
    print(y_pred[0])
    # record current time
    end = time()
    # report execution time
    result = end - start
    return result
    # return f"{result:.3f}s"


if __name__ == "__main__":
    run_ml(1)