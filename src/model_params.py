import random

LR_PARAMS = {
    "max_iter" : random.randint(500, 1000),
    "penalty" : random.choice(["L1", "L2"]),
    "C" : random.uniform(0.1, 1.0),
    "solver" : random.choice(["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]),
    "random_state" : 42,
    "multi_class" : "auto"
}


RANDOM_SEARCH_PARAMS = {
    "n_iter" : 100,
    "scoring" : "accuracy",
    "cv" : 5,
    "verbose" : 1,
    "n_jobs" : -1,
    "random_state" : 42
}