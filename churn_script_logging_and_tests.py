# churn_script_logging_and_tests.py

"""This file tests churn library"""

import logging
import os
import pathlib

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import churn_library as cls
from constants import CATEGORICAL_VALUES

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def _clear_data():
    pathlib.Path('./images/eda/churn_distribution.png').unlink(missing_ok=True)
    pathlib.Path(
        './images/eda/customer_age_distribution.png').unlink(missing_ok=True)
    pathlib.Path(
        './images/eda/marital_status_distribution.png').unlink(missing_ok=True)
    pathlib.Path(
        './images/eda/deleteme.png').unlink(missing_ok=True)
    pathlib.Path('./images/eda/histogram.png').unlink(missing_ok=True)
    pathlib.Path('./models/rfc_model.pkl').unlink(missing_ok=True)
    pathlib.Path('./models/logistic_model.pkl').unlink(missing_ok=True)


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
        '''

    try:
        dataframe = import_data("./tests/data/bank_data.csv")

        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except (AssertionError, FileNotFoundError):
        logging.error(
            "Testing data import: File doesn't exist or doesn't have a matrix-shape.")


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    dataframe = cls.import_data("./tests/data/bank_data.csv")

    try:
        perform_eda(dataframe)

        assert 'Churn' in dataframe.columns
        logging.info(
            "Testing eda: SUCCESS - Target variable have been added to dataframe.")
    except AssertionError:
        logging.error("Testing eda: Churn not in dataframe.")

    try:
        assert os.path.isfile('./images/eda/churn_distribution.png')
        assert os.path.isfile('./images/eda/customer_age_distribution.png')
        assert os.path.isfile('./images/eda/marital_status_distribution.png')
        assert os.path.isfile(
            './images/eda/total_transaction_distribution.png')
        assert os.path.isfile('./images/eda/histogram.png')

        logging.info(
            "Testing eda: SUCCESS - Plots have been rendered into files.")
    except AssertionError:
        logging.error(
            "Testing eda: 'Churn' not in dataframe or images not generated.")


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    dataframe = cls.import_data("./tests/data/bank_data.csv")
    cls.perform_eda(dataframe)

    try:
        encoded_data = encoder_helper(
            dataframe, CATEGORICAL_VALUES, "sample_label")

        for value in CATEGORICAL_VALUES:
            assert f'{value}_sample_label' in encoded_data.columns

        assert len(encoded_data.columns) == (
            len(dataframe.columns) + len(CATEGORICAL_VALUES))

        logging.info("Testing encoding: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing enconding: Missing categorical values or non-expected categories added.")


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''

    dataframe = cls.import_data("./tests/data/bank_data.csv")
    cls.perform_eda(dataframe)

    x_train, x_test, y_train, y_test = perform_feature_engineering(dataframe)

    try:
        assert len(x_train) > 0
        assert len(x_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0

    except AssertionError:
        logging.error(
            "Testing feature engineering: Train and/or test data not split for training.")

    logging.info("Testing feature engineering: SUCCESS")


def test_train_models(train_models):
    '''
    test train_models
    '''
    dataframe = cls.import_data("./tests/data/bank_data.csv")
    cls.perform_eda(dataframe)
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        dataframe)
    train_models(x_train, x_test, y_train, y_test)

    try:
        rf_model = joblib.load('./models/rfc_model.pkl')
        lr_model = joblib.load('./models/logistic_model.pkl')
        assert isinstance(rf_model, RandomForestClassifier)
        assert isinstance(lr_model, LogisticRegression)

        logging.info(
            "Testing model training: SUCCESS - Models have been generated and saved to files.")
    except FileNotFoundError:
        logging.error(
            "Testing model training: Models have not been saved properly.")

    try:
        assert os.path.isfile('./images/results/feature_importance_plot.png')
        assert os.path.isfile('./images/results/logistic_results.png')
        assert os.path.isfile('./images/results/rf_results.png')
        assert os.path.isfile('./images/results/roc_curve_result.png')
        logging.info(
            "Testing model training: SUCCESS - Model results have been rendered to files.")
    except AssertionError:
        logging.error(
            "Testing model training: Results have not been rendered to files.")

    logging.info("Testing model training: SUCCESS")


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
