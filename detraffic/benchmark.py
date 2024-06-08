"""benchmark."""

import pathlib
import warnings

warnings.filterwarnings("ignore")

import pickle
from datetime import datetime

from experiment import Experiment

EXPERIMENTS_ROOT_PATH = pathlib.Path("experiments")
RESULTS_ROOT_PATH = pathlib.Path("results")
PLOTS_ROOT_PATH = pathlib.Path("plots")

EXPERIMENT_NAME = "test"
NOW = str(datetime.now())

EXPERIMENT_PATH = EXPERIMENTS_ROOT_PATH / EXPERIMENT_NAME
RESULT_PATH = RESULTS_ROOT_PATH / EXPERIMENT_NAME / NOW
PLOT_PATH = PLOTS_ROOT_PATH / EXPERIMENT_NAME / NOW

PLOT_PATH.mkdir(parents=True, exist_ok=True)

experiment_config = pathlib.Path(EXPERIMENT_PATH.with_suffix(".yaml"))

experiment = Experiment.from_config(experiment_config)
results = experiment.run()
experiment.save(RESULT_PATH.with_suffix(".pkl"))

with open(RESULT_PATH.with_suffix(".pkl"), "rb") as f_obj:
    results = pickle.load(f_obj)

Experiment.plot(PLOT_PATH, results, experiment.hyperparams["iters"])
