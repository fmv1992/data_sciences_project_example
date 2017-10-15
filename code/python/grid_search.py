"""Provide grid search functions."""

# pylama: ignore=D103

from data_utilities import sklearn_utilities as sku

import constants


def main(dataframe,
         models,
         grids,
         pgo,
         ):
    # Combine data transformation with FeatureUnion.
    feature_unions = []  # TODO
    # TODO: adapt feature_unions or data transformation to conform to
    # persistence of the sku.PersistentGrid object.

    # Zip models and feature unions:
    # (model, (fu1, fu2)).

    # Iterate over combination of model and feature union.
    for model in models:
        for fu in feature_unions:
            # TODO: transform data according to feature union.
            sku.grid_search()


def get_best_grids(models,
                   # data_processing_pipelines,
                   persistent_grid_object):
    pass
