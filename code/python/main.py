"""Run all the desired analysis of your project.

Those analysis should be defined elsewhere.

"""
import warnings

import data_utilities as du
from data_utilities import sklearn_utilities as sku

import constants
import data_loading
import data_processing
import data_exploration
import grid_search
import models

# pylama: ignore=D103


def filter_warnings():
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)


def main():
    # Filter warnings that polute the project stdout.
    filter_warnings()
    # Rationale: produce cleaner results.

    # Set the random seed for the entire project.
    du.set_random_seed(0)
    # Rationale: ensure reproducibility of the results.

    # Flush previous runs.
    constants.flush_project_results(constants.TMP_PATH, constants.OUTPUT_PATH)
    # Rationale: provide a clear state for the project to run and enforces
    # reproducibility of the results.

    # Load and save data.
    # data_loading.main()
    dataframe = data_loading.load_data(constants.DATASET_PATH,
                                       constants.TMP_PATH)
    data_loading.save_data(dataframe, constants.TMP_PATH,
                           constants.DATASET_PATH)
    # Rationale: *Loading*: load data in the main module and pass it as a first
    # argument to every other defined function (that relates to the data set)
    # thus saving precious time with data loading. *Saving*: for big data sets
    # saving the dataset as a fast read format (such as HDF5) saves time.

    # Load and combine data processing pipelines.
    data_processing.main(dataframe, nan_strategy='drop')
    # Rationale: prepare data to be fed into the models.
    # Different algorithms make use of different data structures. For instance
    # XGBoost allow for nans. Data transformations usually don't.

    # Perform exploratory data analyses.
    data_exploration.main(dataframe)
    # Rationale: conduct exploratory data analyses.

    # Data split.
    # Removed.
    # Rationale: module 'models' should execute this.

    # Perform grid search.
    # Iteration over processed data sets may occur here since they are model
    # dependent.
    grid_search.main(constants.MODELS, constants.GRIDS)
    best_combination_of_datasets_and_grids = (
        grid_search.dict_of_best_datasets_and_grids(constants.MODELS,
                                                    constants.GRIDS))
    best_datasets = best_combination_of_datasets_and_grids['best_datasets']
    best_grids = best_combination_of_datasets_and_grids['best_grids']
    # Rationale: perform grid search as part of machine learning best
    # practices.

    # Summary of what was executed so far:
    # 1) Setting of the random seed for reproducibility.
    # 2) Flusing of intermediate results for a clean run.
    # 3) Data loading and data saving.
    # 4) Conduction of exploratory data analyses.
    # 5) Grid search of best model hyper parameters.
    # To conclude our project we need the grand finale: model selection and
    # evaluation/comparison.
    models.main(constants.MODELS, best_datasets, best_grids)
    # Rationale: train models and output theirs results to empower the modeller
    # to choose the best of them.


if __name__ == '__main__':
    main()
