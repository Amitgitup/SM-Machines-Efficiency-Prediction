from scipy.stats import randint, uniform

LR_PARAMS = {
    "max_iter" : randint(500, 1000),
    "penalty" : ["l1", "l2"],
    "C" : uniform(0.1, 0.9),
    "solver" : ["liblinear", "saga"],
    "random_state" : [42]
}


RANDOM_SEARCH_PARAMS = {
    "n_iter" : 100,
    "scoring" : "accuracy",
    "cv" : 5,
    "verbose" : 1,
    "n_jobs" : 1,
    "random_state" : 42,
}