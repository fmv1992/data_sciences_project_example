"""Provide data processing functions."""

import os

# import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import SelectKBest
# from sklearn.preprocessing import FunctionTransformer
from sklearn.manifold import TSNE

import constants
import data_loading

# pylama: ignore=D103

# Define constants (paths) to processed dataframes.
DATA_VANILLA = constants.DATASET_PATH
_DATA_OUTPUT = os.path.join(constants.OUTPUT_PATH, 'data_processing')
DATA_PCA2 = os.path.join(constants.DATA_PATH, 'pca2.csv.xz')
DATA_PCA3 = os.path.join(constants.DATA_PATH, 'pca3.csv.xz')
DATA_TSNE2 = os.path.join(constants.DATA_PATH, 'tsne2.csv.xz')
DATA_TSNE3 = os.path.join(constants.DATA_PATH, 'tsne3.csv.xz')
# Rationale: Create non existent paths to processed dataframes so they can be
# used by PersistentGrid objects to compute grids. This makes it easy to deal
# with various dataframes in a more modular way.


def _process_nan(dataframe, how='drop'):
    if how == 'drop':
        df = dataframe.dropna()
    else:
        raise NotImplementedError
    return df


def _pca(x, *pca_args, **pca_kwargs):
    pca_obj = PCA(*pca_args, **pca_kwargs)
    transformed = pca_obj.fit_transform(x)
    pca_cols = ['pca' + str(i) for i in range(transformed.shape[1])]
    pca_df = pd.DataFrame(
        data=transformed,
        columns=pca_cols)
    return pca_df


def _normalize(x):
    norm = StandardScaler()
    norm.fit(x)
    return pd.DataFrame(data=norm.transform(x), columns=x.columns)


def _tsne(x, y, *tsne_args, **tsne_kwargs):
    tsne_obj = TSNE(*tsne_args, **tsne_kwargs)
    transformed = tsne_obj.fit_transform(x)
    tsne_cols = ['tsne' + str(i) for i in range(transformed.shape[1])]
    tsne_df = pd.DataFrame(
        data=transformed,
        columns=tsne_cols)
    return tsne_df


def norm_pca2(x, y):
    if data_loading.dataframe_already_exists(_DATA_OUTPUT, DATA_PCA2):
        return None
    y = pd.DataFrame(y).reset_index()
    norm_df = _normalize(x)
    pca_df = _pca(norm_df, 2)
    joined_df = pd.concat((norm_df, pca_df, y),
                          axis=1,
                          ignore_index=True,
                          verify_integrity=True)
    data_loading.save_data(joined_df, _DATA_OUTPUT, DATA_PCA2)


def norm_pca3(x, y):
    if data_loading.dataframe_already_exists(_DATA_OUTPUT, DATA_PCA3):
        return None
    y = pd.DataFrame(y).reset_index()
    norm_df = _normalize(x)
    pca_df = _pca(norm_df, 3)
    joined_df = pd.concat((norm_df, pca_df, y),
                          axis=1,
                          ignore_index=True,
                          verify_integrity=True)
    data_loading.save_data(joined_df, _DATA_OUTPUT, DATA_PCA3)


def norm_tsne2(x, y):
    if data_loading.dataframe_already_exists(_DATA_OUTPUT, DATA_TSNE2):
        return None
    y = y.reset_index()
    norm_df = _normalize(x)
    tsne_df = _tsne(norm_df, 2)
    joined_df = pd.concat((norm_df, tsne_df, y),
                          axis=1,
                          ignore_index=True,
                          verify_integrity=True)
    data_loading.save_data(joined_df, _DATA_OUTPUT, DATA_TSNE2)


def norm_tsne3(x, y):
    if data_loading.dataframe_already_exists(_DATA_OUTPUT, DATA_TSNE3):
        return None
    y = y.reset_index()
    norm_df = _normalize(x)
    tsne_df = _tsne(norm_df, 3)
    joined_df = pd.concat((norm_df, tsne_df, y),
                          axis=1,
                          ignore_index=True,
                          verify_integrity=True)
    data_loading.save_data(joined_df, _DATA_OUTPUT, DATA_TSNE3)


def main(dataframe, nan_strategy='drop'):
    if nan_strategy == 'drop':
        df = dataframe.dropna()
        del dataframe
    else:
        raise NotImplementedError
    x = df[data_loading.get_x_columns(df)]
    y = df[constants.Y_COLUMN]
    norm_pca2(x, y)
    norm_pca3(x, y)
    norm_tsne2(x, y)
    norm_tsne3(x, y)


if __name__ == '__main__':
    dataframe = data_loading.load_data(constants.DATASET_PATH)
    main(dataframe)
