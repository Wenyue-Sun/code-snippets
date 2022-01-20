# define a ML function
from time import time

import sklearn.datasets
import sklearn.metrics

import autosklearn.classification

# autoML is quite picky in terms of dependencies

def run_ml(n_jobs=1):
    # define dataset
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=30,
        n_jobs=n_jobs
    )
    start = time()
    print("Start automl")
    # record current time
    automl.fit(X_train, y_train, dataset_name='breast_cancer')        
    # record current time
    end = time()
    print("Automl fit done")
    # report execution time
    result = end - start
    return result
    # return f"{result:.3f}s"


if __name__ == "__main__":
    run_ml(1)