"""Assign project-wide variables.

Must contain assertions for all the assigned variables (such as paths).

"""
import os
import glob

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


ROOT_PATH = get_root_dir_based_on_dotgit(__file__)
assert os.path.exists(ROOT_PATH)

DATA_PATH = os.path.join(ROOT_PATH, 'data', 'data.csv.xz')
TMP_PATH = os.path.join(ROOT_PATH, 'tmp')
OUTPUT_PATH = os.path.join(ROOT_PATH, 'output')
OUTPUT_DATA_PROC_PATH = os.path.join(OUTPUT_PATH, 'processed_data')
PERSITENT_GRID_PATH = os.path.join(OUTPUT_PATH,
                                   'persistent_grid_object.pickle')
assert os.path.exists(DATA_PATH)
assert os.path.exists(TMP_PATH)
assert os.path.exists(OUTPUT_PATH)

# Plotting paths.
DATA_EXPLORATION = os.path.join(OUTPUT_PATH, 'data_exploration')
DE_HIST_DF = os.path.join(DATA_EXPLORATION, 'histogram_of_dataframe')
DE_VIOLIN = os.path.join(DATA_EXPLORATION, 'violinplots')


# TODO: how to combine models, data sets, grid for models and grid for data
# processing.
# There are two approaches for models and data processing.
# First approach: zipping
#   (model,
#   map(data_processing_function, dataset),
#   best_grid)
# Second approach: zipping (model, map(data_processing_function, dataset)).
# Sequence of model objects to be iterated.
#
# On the data set processing
#
# The data set processing is tricky because of the following:
#   - Depending on how you iterate over models/data processing the data proc.
#   part may be only calculated once.
#   - Each combination of model + data proc. requires its own persistent grid
#   object. This may point in the direction of saving the processed data sets.
#   This may be an unacceptable performance impact just to accommodate a
#   feature of persistent grid (which is linked to the data set path).

MODELS = [
    # XGBoost.
    # Random Forest.
    # Decision Tree.
    ]

# Data processing functions.
# In practice there will not be a large amount of (models) x (data processing).
DATA_PROCESSING_PIPELINES = [
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


GRIDS = [
    # XGBoost.
    {},
    # Random Forest. (may not contain nulls)
    {},
    # Decision Tree.
    {},
    ]

# Y column.
Y_COLUMN = 'occupancy'
