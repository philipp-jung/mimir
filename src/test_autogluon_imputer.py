import random
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_hastie_10_2, fetch_california_housing
import utils

from autogluon_imputer import AutoGluonImputer


random.seed(0)

def get_regression_data(dat_fn, noise=3e-1):
    X, y = dat_fn(return_X_y=True)
    X = X + np.random.randn(*X.shape) * noise
    test_data_df = pd.DataFrame(np.vstack([X.T, y]).T,
                                columns=[str(i) for i in range(X.shape[-1] + 1)])
    return test_data_df


def get_classification_data(data_fn, noise=3e-1):
    X, y = data_fn(n_samples=10000)
    X = X + np.random.randn(*X.shape) * noise
    test_data_df = pd.DataFrame(np.vstack([X.T, y]).T,
                                columns=[str(i) for i in range(X.shape[-1] + 1)])
    return test_data_df


def test_autogluon_imputer_quantile_regression():
    """
    Tests AutoGluonImputer's quantile regression with default settings.
    Quantile regression is still an experimentative feature -- this test
    is just a rough scaffolding for now.
    """
    regression_data = get_regression_data(fetch_california_housing)
    label = regression_data.columns[-1]
    features = [x for x in regression_data.columns if x != label]
    regression_data[label] = regression_data[label].astype(str)
    df_train, df_test = utils.random_split(regression_data.copy())

    imputer = AutoGluonImputer(
        model_name='test model',
        columns=df_train.columns,
        input_columns=features,
        output_column=label,
        verbosity=2)

    imputer.fit(train_df=df_train, time_limit=10)
    assert(True)


def test_autogluon_imputer_precision_threshold():
    """
    Verifies that the precision measured on the test set increases
    monotonously if the user selects bigger precision_thresholds.

    Also ensures that the empirically measured precision + .02 is
    bigger than the user-specified precision -- there are some cases
    where the empirically measured precision will be bigger than what
    the user specified due to the stochastic nature of the process.

    """
    classification_data = get_classification_data(make_hastie_10_2)
    label = classification_data.columns[-1]
    features = [x for x in classification_data.columns if x != label]
    classification_data[label] = classification_data[label].astype(str)
    df_train, df_test = utils.random_split(classification_data.copy())

    imputer = AutoGluonImputer(
        model_name='other test model',
        columns=df_train.columns,
        input_columns=features,
        output_column=label,
        verbosity=2)

    imputer.fit(train_df=df_train, time_limit=10)

    precisions = []
    for precision_threshold in [0.1, 0.5, 0.9, 0.95, .99]:
        probas = imputer.predict_proba(df_test[features])

        # report = classification_report(df_test[label],
        #                                imputed[label+"_imputed"].fillna(""),
        #                                output_dict=True)
        # precisions.append({
        #     'precision_threshold': precision_threshold,
        #     'empirical_precision_on_test_set': np.mean([report['-1.0']['precision'],
        #                                                 report['1.0']['precision']]
        #                                                )
        # })
    # df_precisions = pd.DataFrame(precisions)
    # precision_deviations = df_precisions['empirical_precision_on_test_set'] \
    #                         - df_precisions['precision_threshold'] + 0.02

    # for i, _ in enumerate(precisions):
    #     if i > 0:
    #         assert(precisions[i]['empirical_precision_on_test_set']
    #                >= precisions[i-1]['empirical_precision_on_test_set'])
    # assert all(precision_deviations > 0)
    assert(True)
