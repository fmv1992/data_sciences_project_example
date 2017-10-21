"""Provide grid search functions."""

# pylama: ignore=D103

from data_utilities import sklearn_utilities as sku

import constants
import data_loading


def main(dataframe,
         models,
         grids,
         pgo,
         ):
    # Iterate over combination of model and feature union.
    get_best_grids(dataframe, models, grids, pgo,)


def get_best_grids(dataframe,
         models,
         grids,
         pgo,
         ):
    # Iterate over combination of model and feature union.
    best_grids = list()
    for model, grid in zip(models, grids):
        computed_grids = sku.persistent_grid_search_cv(
            pgo,
            grid,
            model,
            dataframe[data_loading.get_x_columns(dataframe)],
            y=dataframe[constants.Y_COLUMN],
            cv=10,
            scoring='roc_auc')
        one_best_grid = computed_grids[0]
        best_grids.append(one_best_grid)
    return best_grids
