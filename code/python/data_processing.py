"""Provide data processing functions."""

import os

# import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import SelectKBest
# from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.manifold import TSNE

import constants
import data_loading


# pylama: ignore=D103

def _process_nan(dataframe, how='drop'):
    if how == 'drop':
        df = dataframe.dropna()
    else:
        raise NotImplementedError
    return df


def _pca(x, *pca_args, **pca_kwargs):
    pca_obj = PCA(*pca_args, **pca_kwargs)
    pca_cols = ['pca' + str(i) for i in range(len(x.columns))]
    pca_df = pd.DataFrame(
        data=pca_obj.fit_transform(x),
        columns=pca_cols)
    return pca_df


def _normalize(x):
    norm = StandardScaler()
    norm.fit(x)
    return norm.transform(x)


def _tsne(x, y, *tsne_args, **tsne_kwargs):
    tsne_obj = TSNE(*tsne_args, **tsne_kwargs)
    tsne_cols = ['tsne' + str(i) for i in range(len(x.columns))]
    tsne_df = pd.DataFrame(
        data=tsne_obj.fit_transform(x),
        columns=tsne_cols)
    return tsne_df


def norm_pca2(x, y):
    norm_df = _normalize(x)
    pca_df = _pca(norm_df, 2)
    return pca_df


def norm_pca3(x, y):
    norm_df = _normalize(x)
    pca_df = _pca(norm_df, 2)
    return pca_df


def norm_tsne2(x, y):
    norm_df = _normalize(x)
    tsne_df = _tsne(norm_df, 3)
    return tsne_df


def main(dataframe, nan_strategy='drop'):
    if nan_strategy == 'drop':
        df = dataframe.dropna()
        del dataframe
    else:
        raise NotImplementedError
    data_path_basename = os.path.basename(constants.DATA_PATH).split('.')[0]
    x_columns = data_loading.get_x_columns(df)
    for processing_pipeline in constants.DATA_PROCESSING_PIPELINES:
        # See notes:
        # TODO: on how to combine models.
        # On constants.py file.

        # fu = FeatureUnion(
        #     constants.DATA_PROCESSING_PIPELINES[processing_pipeline])
        # new_df = pd.DataFrame(data=fu.fit_transform(X=df[x_columns],
        #                                             y=df[constants.Y_COLUMN]))
        # # Save newly computed dataframe.
        # new_data_name = os.path.join(
        #     constants.OUTPUT_DATA_PROC_PATH,
        #     processing_pipeline + '_' + data_path_basename + '.hdf')
        # data_loading.save_data(
        #     new_df,
        #     constants.OUTPUT_DATA_PROC_PATH,
        #     basename=processing_pipeline)

        pass


if __name__ == '__main__':
    dataframe = data_loading.load_data(constants.DATA_PATH)
    main(dataframe)
