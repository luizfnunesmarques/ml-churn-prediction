# churn_library_solution.py

"""
This library provides functions to explore, manipulate and train ml models.
Author: Luiz Marques
Creation date: 04/02/2024
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

from constants import MODEL_FEATURES, CATEGORICAL_VALUES

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def _render_classification_report(test_report, train_report, output_file):
    plt.clf()
    plt.rc('figure', figsize=(20, 20))
    plt.text(0.01,
             0.9,
             str('Test data'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01,
             0.7,
             test_report,
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01,
             0.5,
             str('Train data'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01,
             0.3,
             train_report,
             {'fontsize': 10},
             fontproperties='monospace')
    plt.axis('off')
    plt.savefig(f'./images/results/{output_file}')


def _render_plot(output_file, data_command):
    plt.clf()
    plt.figure(figsize=(20, 10))
    data_command()
    plt.savefig(f'./images/eda/{output_file}')
    plt.close()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandadf s dataframe
    '''
    return pd.read_csv(pth, encoding="utf-8")


def perform_eda(dataframe):
    '''
    perform eda on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe. WARNING: the passing object will be modified.

    output:
            None
    '''
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    _render_plot(
        "churn_distribution.png",
        lambda: dataframe['Churn'].hist())  # pylint: disable=unnecessary-lambda
    _render_plot(
        'customer_age_distribution.png',
        lambda: dataframe['Customer_Age'].hist())  # pylint: disable=unnecessary-lambda
    _render_plot(
        'marital_status_distribution.png',
        lambda: dataframe.Marital_Status.value_counts('normalize').plot(
            kind='bar'))
    _render_plot(
        'total_transaction_distribution.png',
        lambda: sns.histplot(
            dataframe['Total_Trans_Ct'],
            stat='density',
            kde=True))
    _render_plot(
        'histogram.png',
        lambda: sns.heatmap(
            dataframe.corr(),
            annot=False,
            cmap='Dark2_r',
            linewidths=2))


def encoder_helper(dataframe, category_lst, response="Churn"):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the
    notebook

    input:
        dataframe: pandas dataframe.
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could be used
            for naming variables or index y column]

    output:
            dataframe: pandas dataframe with new columns for
    '''
    # preserve original dataframe data and structure
    encoded_data_frame = dataframe.copy()

    for category in category_lst:
        encoded_data_frame[f'{category}_{response}'] = encoded_data_frame.groupby(
            category)['Churn'].transform('mean')

    return encoded_data_frame


def perform_feature_engineering(dataframe, response='Churn'):
    '''
    input:
              dataframe: pandas dataframe
    response: string of response name [optional argument that could be used
    for naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    x_data = pd.DataFrame()
    y_data = dataframe['Churn']

    encoded_data = encoder_helper(dataframe,
                                  CATEGORICAL_VALUES,
                                  response)

    x_data[MODEL_FEATURES] = encoded_data[MODEL_FEATURES]

    return train_test_split(x_data, y_data, test_size=0.3, random_state=42)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
            None
    '''
    rf_test_report = str(classification_report(y_test, y_test_preds_rf))
    rf_train_report = str(classification_report(y_train, y_train_preds_rf))
    _render_classification_report(
        rf_test_report,
        rf_train_report,
        'rf_results.png')

    lr_test_report = str(classification_report(y_test, y_test_preds_lr))
    lr_train_report = str(classification_report(y_train, y_train_preds_lr))
    _render_classification_report(
        lr_test_report,
        lr_train_report,
        'logistic_results.png')


def feature_importance_plot(
        model,
        x_data,
        output_path="feature_importances.png"):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_path: path to store the figure

    output:
             None
    '''
    plt.clf()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 30))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_path)
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    random_forest_classifier = RandomForestClassifier(random_state=42)
    lr_model = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    rf_model = GridSearchCV(
        estimator=random_forest_classifier,
        param_grid=param_grid,
        cv=5)

    rf_model.fit(x_train, y_train)
    lr_model.fit(x_train, y_train)

    plt.figure(figsize=(15, 8))
    plot_roc_curve(rf_model, x_test, y_test, ax=plt.gca(), alpha=0.8)
    plot_roc_curve(lr_model, x_test, y_test, ax=plt.gca(), alpha=0.8)
    plt.savefig('./images/results/roc_curve_result.png')

    y_train_preds_rf = rf_model.best_estimator_.predict(x_train)
    y_test_preds_rf = rf_model.best_estimator_.predict(x_test)
    y_train_preds_lr = lr_model.predict(x_train)
    y_test_preds_lr = lr_model.predict(x_test)

    feature_importance_plot(
        rf_model.best_estimator_,
        x_train,
        './images/results/feature_importance_plot.png')
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    joblib.dump(rf_model.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lr_model, './models/logistic_model.pkl')
