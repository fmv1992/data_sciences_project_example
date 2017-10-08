"""Run all the desired analysis of your project.

Those analysis should be defined elsewhere.

"""

import data_utilities as du
from data_utilities import sklearn_utilities as sku

import constants
import data_loading
import data_exploration
import grid_search

# pylama: ignore=D103


def main():
    # Set the random seed for the entire project.
    du.set_random_seed(0)
    # Rationale: ensure reproducibility of the results.

    # Flush previous runs.
    constants.flush_project_results(constants.TMP_PATH, constants.OUTPUT_PATH)
    # Rationale: provide a clear state for the project to run and enforces
    # reproducibility of the results.

    # Load, save and split data.
    data_loading.main()
    dataframe = data_loading.load_data(constants.DATA_PATH)
    data_loading.save_data(dataframe, constants.TMP_PATH)
    x_train, x_test, y_train, y_test = data_loading.train_test_split(dataframe)
    # Rationale: *Loading*: load data in the main module and pass it as a first
    # argument to every other defined function (that relates to the data set)
    # thus saving precious time with data loading. *Saving*: for big data sets
    # saving the dataset as a fast read format (such as HDF5) saves time.

    # Load and combine data processing pipelines.
    # TODO:
    data_processing_pipelines = None

    # Perform exploratory data analyses.
    data_exploration.main(dataframe)
    # Rationale: conduct exploratory data analyses.

    # Perform grid search.
    persistent_grid_object = sku.grid_search.PersistentGrid.load_from_path(
        persistent_grid_path=constants.PERSITENT_GRID_PATH,
        dataset_path=constants.DATA_PATH)
    # Iteration over processed data sets may occur here since they are model
    # dependent.
    grid_search.main(dataframe,
                     constants.MODELS,
                     data_processing_pipelines,
                     constants.GRIDS,
                     persistent_grid_object)
    best_grids = grid_search.get_best_grids(  # noqa
        constants.MODELS,
        data_processing_pipelines,
        persistent_grid_object)
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


if __name__ == '__main__':
    main()
