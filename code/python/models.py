"""Provide model loading functions."""

import data_loading
import constants
import pickle
import os

from sklearn.metrics import roc_auc_score

from data_utilities import sklearn_utilities as sku

# pylama: ignore=D103


def report_models_results(xtrain, xtest, ytrain, ytest, model):
    predicted_proba_true = model.predict_proba(xtest)[:, 1]
    # TODO: add more analyses to assess best classifier.
    print('ROC AUC:', roc_auc_score(ytest, predicted_proba_true))


def save_trained_model(model):
    model_path = os.path.join(constants.OUTPUT_MODEL_PATH,
                              sku.get_estimator_name(model) + '.pickle')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def load_trained_model(model):
    model_path = os.path.join(constants.OUTPUT_MODEL_PATH,
                              sku.get_estimator_name(model))
    if os.path.exists(model_path):
        with open(model_path, 'wb') as f:
            loaded_model = pickle.load(f)
    else:
        loaded_model = None
    return loaded_model


def main(dataframe, models, best_grids):

    x_train, x_test, y_train, y_test = data_loading.train_test_split(dataframe)

    trained_model_list = list()
    for model, params in zip(models, best_grids):
        trained_model = load_trained_model(model)
        if trained_model is None:
            trained_model = model
            params.pop('scores')
            trained_model.set_params(**params)
            # TODO: allow for fitting parameters.
            trained_model.fit(x_train, y_train)
            save_trained_model(trained_model)
        trained_model_list.append(trained_model)

    # Report.
    for model in trained_model_list:
        report_models_results(x_train, x_test, y_train, y_test, model)
