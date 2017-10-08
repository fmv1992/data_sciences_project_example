"""Assign project-wide variables2

Must contain assertions for all the assigned variables (such as paths).

"""
import os
import glob

# pylama: ignore=D103


def flush_project_results(*paths):
        flush_files = set(filter(
            os.path.isfile,
            sum(map(lambda x: (
                glob.glob(x + os.sep + '**' + os.sep + '*') + glob.glob(x + os.sep + '*')),
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


ROOT_PATH = get_root_dir_based_on_dotgit(__file__)
assert os.path.exists(ROOT_PATH)

DATA_PATH = os.path.join(ROOT_PATH, 'data', 'data.csv.xz')
TMP_PATH = os.path.join(ROOT_PATH, 'tmp')
OUTPUT_PATH = os.path.join(ROOT_PATH, 'output')
PERSITENT_GRID_PATH = os.path.join(OUTPUT_PATH,
                                   'persistent_grid_object.pickle')
assert os.path.exists(DATA_PATH)
assert os.path.exists(TMP_PATH)
assert os.path.exists(OUTPUT_PATH)

# There are two approaches for models and data processing.
# First approach: zipping
#   (model,
#   map(data_processing_function, dataset),
#   best_grid)
# Second approach: zipping (model, map(data_processing_function, dataset)).
# Sequence of model objects to be iterated.
MODELS = [
    # XGBoost.
    # Random Forest.
    # Decision Tree.
    ]
# TODO: how to combine models, data sets, grid for models and grid for data
# processing.

# Data processing functions
DATA_PROCESSING_PIPELINE = [
    # XGBoost.
    [
        # Normalization + PCA(2).
        # Normalization + PCA(3).
        # Feature selection + PCA.
    ],
    # Random Forest. (may not contain nulls)
    [
        # Remove nulls + PCA + Normalization.
        # Remove nulls + PCA.
    ],
    # Decision Tree.
    ]
# These pipelines get combined into a single transformer using sklearn's
# FeatureUnion.

DATA_PROCESSING_PIPELINE_KWARGS = [
    # XGBoost.
    [
        # Normalization + PCA(2).
        [{}, {}, ],
        # Normalization + PCA(3).
        [{}, {}, ],
        # Feature selection + PCA.
        [{}, {}, ],
    ],
    # Random Forest. (may not contain nulls)
    [
        # Remove nulls + PCA + Normalization.
        # Remove nulls + PCA.
    ],
    # Decision Tree.
    ]

GRIDS = [
    # XGBoost.
    {},
    # Random Forest. (may not contain nulls)
    {},
    # Decision Tree.
    {},
    ]
