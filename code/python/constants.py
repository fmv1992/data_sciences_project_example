"""Assign project-wide variables and functions.

Must contain assertions for all assigned variables that are paths.

"""

import glob
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# pylama: ignore=D103


def flush_project_results(*paths):
        flush_files = set(filter(
            os.path.isfile,
            sum(map(lambda x: (
                glob.glob(x + os.sep + '**' + os.sep + '*', recursive=True)
                + glob.glob(x + os.sep + '*', recursive=True)),
                    paths),
                [])))
        for del_file in flush_files:
            os.remove(del_file)


def get_root_dir_based_on_dotgit(path):
    """Scan the folder iteratively for the '.git' root folder."""
    if os.path.isfile(path):
        _this_file = os.path.abspath(__file__)
        _this_folder = os.path.dirname(_this_file)
    else:
        _this_folder = os.path.abspath(path)
    while '.git' not in os.listdir(_this_folder):
        _this_folder = os.path.dirname(_this_folder)
    return os.path.abspath(_this_folder)


# Set projects constants.
Y_COLUMN = 'occupancy'

# Set data paths constants for the entire project.
ROOT_PATH = get_root_dir_based_on_dotgit(__file__)
assert os.path.exists(ROOT_PATH)
DATA_PATH = os.path.join(ROOT_PATH, 'data')
DATASET_PATH = os.path.join(DATA_PATH, 'data.csv.xz')
TMP_PATH = os.path.join(ROOT_PATH, 'tmp')
OUTPUT_PATH = os.path.join(ROOT_PATH, 'output')
OUTPUT_DATA_PROC_PATH = os.path.join(OUTPUT_PATH, 'processed_data')
OUTPUT_MODEL_PATH = os.path.join(OUTPUT_PATH, 'models')
PERSITENT_GRID_PATH = os.path.join(OUTPUT_PATH,
                                   'persistent_grid_object.pickle')
assert os.path.exists(DATA_PATH)
assert os.path.exists(TMP_PATH)
assert os.path.exists(OUTPUT_PATH)
assert os.path.exists(OUTPUT_DATA_PROC_PATH)
assert os.path.exists(OUTPUT_MODEL_PATH)

# Plotting paths.
DATA_EXPLORATION = os.path.join(OUTPUT_PATH, 'data_exploration')
DE_VIOLIN = os.path.join(DATA_EXPLORATION, 'violinplots')
assert os.path.exists(DATA_EXPLORATION)
assert os.path.exists(DE_VIOLIN)


# Set models to be used in the project.
MODELS = [
    # # XGBoost.
    XGBClassifier(),
    # # Random Forest.
    RandomForestClassifier(),
    # # Decision Tree.
    DecisionTreeClassifier()
    ]

# Set grids respective to models defined earlier.
GRIDS = [
    # XGBoost.
    {
        'colsample_bytree': [1],
        'gamma': [0.0,  1, 10.001],  # had problems with [10]
        'learning_rate': [0.3],
        'max_depth': [2, 5, ],
        'n_estimators': [10, 20, ],
        'nthread': [1],
        'n_jobs': [1],
        'silent': [1],
        'subsample': [1],
    },
    # Random Forest.
    {
        'n_estimators': [2, 4],
        'max_depth': [2, 4, ],
        'min_samples_leaf': [.2],
        'n_jobs': [-1],
        'oob_score': [True],
        'bootstrap': [True],
    },
    # Decision Tree.
    {
        'max_depth': [2, 4],
        'min_samples_leaf': [0.01, 0.1]
    },

]

# Set fitting parameters respective to models defined earlier.
MODEL_FITTING_PARAMETERS = [
    # XGBoost.
    {
    },
    # Random Forest.
    {
    },
    # Decision Tree.
    {
    },
]
