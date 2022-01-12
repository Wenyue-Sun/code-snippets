# define a ML function
from time import time
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

def run_ml(n_jobs=1):
    # define dataset
    X, y = make_classification(n_samples=5000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
    # define the model
    model = RandomForestClassifier(n_estimators=500, n_jobs=n_jobs)
    # record current time
    start = time()
    # fit the model
    model.fit(X, y)
    # record current time
    end = time()
    # report execution time
    result = end - start
    return result
    # return f"{result:.3f}s"
