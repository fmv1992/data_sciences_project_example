"""Provide model loading functions."""

import constants
import pickle
import os

from sklearn.metrics import roc_auc_score

from data_utilities import sklearn_utilities as sku

import data_loading
import data_processing

# pylama: ignore=D103


def report_models_results(xtrain, xtest, ytrain, ytest, model):
    predicted_proba_true = model.predict_proba(xtest)[:, 1]
    print(30 * '-', sku.get_estimator_name(model))
    print('ROC AUC:', roc_auc_score(ytest, predicted_proba_true))


def save_trained_model(model, output_path):
    model_path = os.path.join(output_path,
                              sku.get_estimator_name(model) + '.pickle')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def load_trained_model(model, output_path):
    model_path = os.path.join(output_path,
                              sku.get_estimator_name(model))
    if os.path.exists(model_path):
        with open(model_path, 'wb') as f:
            loaded_model = pickle.load(f)
    else:
        loaded_model = None
    return loaded_model


def main(models, dataset_paths, best_grids, model_fitting_parameters):

    trained_model_list = list()
    for model, dset_path, grid, fitting_parameters in zip(
            models, dataset_paths, best_grids, model_fitting_parameters):

        # Load data set.
        dataframe = data_loading.load_data(
            dset_path,
            constants.OUTPUT_DATA_PROC_PATH)
        # Divide into training and test data set.
        x_train, x_test, y_train, y_test = data_loading.train_test_split(
            dataframe)

        # Load and train models.
        trained_model = load_trained_model(model,
                                           constants.OUTPUT_MODEL_PATH)
        if trained_model is None:  # if model is not lodaded train and save.
            trained_model = model
            grid.pop('scores')
            trained_model.set_params(**grid)
            trained_model.fit(x_train, y_train, **fitting_parameters)
            save_trained_model(trained_model, constants.OUTPUT_MODEL_PATH)
        trained_model_list.append(trained_model)

        # Report model results.
        report_models_results(x_train, x_test, y_train, y_test, model)
