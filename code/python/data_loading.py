"""Provide data loading and preparation functions."""
import urllib.request
import zipfile
import lzma
import os
import datetime as dt

import pandas as pd
import numpy as np

from data_utilities import pandas_utilities as pu

import constants

# pylama: ignore=D103


def download_data(zipurl, zippath):
    """Download data and places it in the tmp folder."""
    with urllib.request.urlopen(zipurl) as urlf:
        with open(zippath, 'wb') as zipf:
            zipf.write(urlf.read())


def join_csvs_from_zip(zippath, lzma_outpath):
    """Join csvs from zip file into a single compressed file."""
    zipf = zipfile.ZipFile(zippath)
    csv_list = list(map(
        zipf.read, zipf.namelist()))
    # Parse header for one dataframe.
    csv_header = csv_list[0][:csv_list[0].find(b'\n')]
    # Combine dataframes, prepending one header and stripping all other
    # headers.
    with lzma.open(lzma_outpath, 'wb') as f:
        f.write(
            csv_header +
            b'\n'.join(x[x.find(b'\n'):] for x in csv_list))


def save_data(dataframe, output_path, dataframe_path):
    """Save dataframe in the output path.

    The basename is important because it discriminates saved objects. It is
    recommended to be the input path.
    """
    dest_path = _transform_to_unique_path(output_path, dataframe_path)
    if not os.path.exists(dest_path):
        dataframe.to_hdf(
            dest_path,
            key='x',
            mode='w')


def _transform_to_unique_path(dest_path, df_path):
    basename = str(df_path).replace(os.sep, '_').replace('.', '_').lower()
    full_path = os.path.join(dest_path, basename + '.hdf')
    return full_path


def load_data(dataframe_path, quick_load_path=None):
    if quick_load_path is not None:
        quick_load_data_path = _transform_to_unique_path(quick_load_path,
                                                         dataframe_path)
        if os.path.exists(quick_load_data_path):
            df = pd.read_hdf(quick_load_data_path)
            return df
    # Load csv.
    df = pd.read_csv(dataframe_path, parse_dates=True)
    # Reset wrong index.
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    # Correct data dtypes.
    _correct_data_dtypes(df)
    # Correct column names.
    pu.rename_columns_to_lower(df)
    # Sort by datetimes.
    df = df.sort_values('date')
    # Insert missing values.
    df.loc[:, df.columns.drop('occupancy')] = (
        _insert_missing_values(df.loc[:, df.columns.drop('occupancy')]))
    # Convert days to ordinal numbers.
    df = _convert_days_to_ordinal(df)
    return df


def _correct_data_dtypes(dataframe):
    dataframe.date = pd.to_datetime(dataframe.date, format='%Y-%m-%d %H:%M:%S')


def _insert_missing_values(dataframe, fraction=0.03):
    mask = np.random.choice([True, False],
                            size=dataframe.shape,
                            p=(fraction, 1 - fraction))
    return dataframe.mask(mask, np.nan)


def _convert_days_to_ordinal(dataframe):
    dataframe['date'] = dataframe.date.map(
        lambda x: dt.datetime.date(x).toordinal())
    return dataframe


def dataframe_already_exists(dest_path, df_path):
    df_path = _transform_to_unique_path(dest_path, df_path)
    return os.path.exists(df_path)


def get_x_columns(dataframe):
    x_columns = dataframe.columns.tolist()
    x_columns.remove(constants.Y_COLUMN)
    return x_columns


def train_test_split(dataframe, split_out_of_time=True, train_size=0.6):
    if split_out_of_time:
        all_days = sorted(dataframe.date.unique())
        N_TRAINING_DAYS = 3
        X_COLUMNS = get_x_columns(dataframe)
        train_set, test_set = (all_days[:-N_TRAINING_DAYS],
                               all_days[-N_TRAINING_DAYS:])
        return (
            dataframe.loc[dataframe.date.isin(train_set), X_COLUMNS],  # xtrain
            dataframe.loc[dataframe.date.isin(test_set), X_COLUMNS],   # xtest
            dataframe.loc[dataframe.date.isin(train_set),
                          constants.Y_COLUMN],   # ytrain
            dataframe.loc[dataframe.date.isin(test_set),
                          constants.Y_COLUMN],   # ytest
        )
    else:
        raise NotImplementedError


def main():
    # Setup its own constants.
    DATA_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip'  # noqa
    ZIP_PATH = os.path.join(constants.TMP_PATH, 'zip_data.zip')

    # Download data from website.
    download_data(DATA_URL, ZIP_PATH)

    # Join data into a single csv.
    join_csvs_from_zip(ZIP_PATH, constants.DATASET_PATH)

    # Load data (for checking purposes).
    df = load_data(constants.DATASET_PATH)

    # Save data (for checking purposes).
    save_data(df, constants.TMP_PATH, constants.DATASET_PATH)


