"""Provide grid search functions."""

import numpy as np

from data_utilities import sklearn_utilities as sku

import constants
import data_loading
import data_processing

# pylama: ignore=D103


def main(models, grids):
    dict_of_best_datasets_and_grids(models, grids)


def dict_of_best_datasets_and_grids(models, grids):
    all_grids_results = get_model_and_df_grid_combinations(models, grids)
    datasets_and_grids = get_best_dataset_and_grid_for_each_model(
        models, all_grids_results.copy())
    best_datasets, best_grids = zip(*datasets_and_grids)
    return dict(best_datasets=best_datasets, best_grids=best_grids)


def get_best_dataset_and_grid_for_each_model(models, computed_grids):
    path_grid_tuple = list()
    for model in map(sku.get_estimator_name, models):
        model_filter = list(filter(lambda z: z['model'] == model,
                                   computed_grids))
        best_entry = sorted(model_filter,
                            key=lambda x: np.mean(x['scores']),
                            reverse=True)[0].copy()
        best_dataset = best_entry.pop('path')
        best_entry.pop('model')
        path_grid_tuple.append((best_dataset, best_entry))
    return path_grid_tuple


def get_model_and_df_grid_combinations(models, grids):
    all_grids_results = list()
    # Iterate over data sets.
    for path in data_processing.PROCESSED_DATASETS_PATH:
        df = data_loading.load_data(path, data_processing.PROCESSED_DATA_PATH)
        pgo = sku.grid_search.PersistentGrid.load_from_path(
            persistent_grid_path=constants.PERSITENT_GRID_PATH,
            dataset_path=path)
        # Iterate over models.
        for grid, model in zip(grids, models):
            best_grid = get_best_grid(df, model, grid, pgo).copy()
            best_grid['model'] = sku.get_estimator_name(model)
            best_grid['path'] = path
            all_grids_results.append(best_grid.copy())
    return all_grids_results


# TODO: adapt to get just one best grid.
def get_best_grid(dataframe, model, grid, pgo):
        computed_grids = sku.persistent_grid_search_cv(
            pgo,
            grid,
            model,
            dataframe[data_loading.get_x_columns(dataframe)],
            y=dataframe[constants.Y_COLUMN],
            cv=10,
            scoring='roc_auc')
        one_best_grid = computed_grids[0]
        return one_best_grid
