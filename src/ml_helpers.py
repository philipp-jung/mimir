import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import f1_score
from typing import List, Dict, Tuple
import hpo

from helpers import Corrections


def set_binary_pre_ensembling(predicted_probas: np.ndarray,
                              all_error_correction_suggestions: List[Tuple],
                              corrections: Corrections,
                              model_name: str):
    """
    Similar to set_binary_cleaning_suggestions, the result of the pre-ensembling of the vicinity-features is used
    to set the probabilities of a cleaning-suggestion of the ensembled cleaning model.
    @param predicted_probas: numpy array where each entry is an array of length 2, representing the 2 classes.
    @param all_error_correction_suggestions: list of tuples containing all error corrections, same length as
    predicted_labels.
    @param corrections: Corrections object that contains all correction suggestions used in the final ensembling.
    @param model_name: Name of the correction model.
    @return: None
    """
    for index, predicted_proba in enumerate(predicted_probas):
        error_cell, correction_suggestion = all_error_correction_suggestions[index]
        corrections_dict = corrections.get(model_name).get(error_cell, {})
        corrections_dict[correction_suggestion] = predicted_proba[1]
        corrections.get(model_name)[error_cell] = corrections_dict


def set_binary_cleaning_suggestions(predicted_labels: List[int],
                                    predicted_probas: List[float],
                                    x_test: List[np.ndarray],
                                    all_error_correction_suggestions: List[Tuple[Tuple[int, int], str]],
                                    corrected_cells: Dict):
    """
    After the Classifier has done its work, take its outputs and set them as the correction in the global state d.
    If there is user input available for a cell, it always takes precedence over the ML output.

    @param predicted_labels: array of binary classes whether or not a cleaning suggestion has been selected by ensembling.
    @param predicted_probas: array of floats of class probability that (class == 1).
    @param x_test: array of feature vectors.
    @param all_error_correction_suggestions: list of tuples (cell, correction) containing all error corrections, same length as
    predicted_labels.
    @param corrected_cells: dictionary {error_cell: predicted_correction} that stores the cleaning results.
    @return: None
    """
    suggestions = {}

    for index, predicted_label in enumerate(predicted_labels):
        if predicted_label:
            error_cell, predicted_correction = all_error_correction_suggestions[index]
            if suggestions.get(error_cell) is None:
                suggestions[error_cell] = []
            suggestions[error_cell].append({'correction': predicted_correction,
                                            'probability': predicted_probas[index],
                                            'features': x_test[index]})
            
    for error_cell in suggestions:
        if len(suggestions[error_cell]) > 1: # multiple suggestions were classified by the correction classifier
            # take the suggestions that have probabilities that are equal to the highest probability
            max_prob = max(suggestions[error_cell], key=lambda x: x['probability'])['probability']
            max_prob_suggestions = [suggestion for suggestion in suggestions[error_cell] if suggestion['probability'] == max_prob]
            if error_cell == (152,11):
                a = 1
            # take the suggestion with the highest
            if len(max_prob_suggestions) == 1:
                corrected_cells[error_cell] = max_prob_suggestions[0]['correction']
            else:
                # further filter and take random suggestion with the highest sum of features
                max_sum_suggestion = max(max_prob_suggestions, key=lambda x: sum(x['features']))
                corrected_cells[error_cell] = max_sum_suggestion['correction']
        else:
            corrected_cells[error_cell] = suggestions[error_cell][0]['correction']


def handle_edge_cases(x_train, x_test, y_train) -> Tuple[bool, List, List]:
    """
    Depending on the dataset and how much data has been labeled by the user, the data used to formulate the ML problem
    can lead to an invalid ML problem.
    To prevent the application from stopping unexpectedly, edge-cases are handled here.

    If this software was engineered more elegantly, this function would not exist. But there is always a balance
    between elegance and pace of development. And we run pretty fast.

    @return: A tuple whose first position is a boolean, indicating if the ML problem should be started. The second
    position is a list of predicted labels. Which is used when the ML problem should not be started.
    """
    if sum(y_train) == 0:  # no correct suggestion was created for any of the error cells.
        return False, [], []  # nothing to do, need more user-input to work.

    elif sum(y_train) == len(y_train):  # no incorrect suggestion was created for any of the error cells.
        return False, np.ones(len(x_test)), np.ones(len(x_test))

    elif len(x_train) > 0 and len(x_test) == 0:  # len(x_test) == 0 because all rows have been labeled.
        return False, [], []  # trivial case - use manually corrected cells to correct all errors.

    elif len(x_train) == 0:
        return False, [], []  # nothing to learn because x_train is empty.

    elif len(x_train) > 0 and len(x_test) > 0:
        return True, [], []
    else:
        raise ValueError("Invalid state.")


def generate_train_test_data(column_errors: Dict[int, List[Tuple[int, int]]],
                             labeled_cells: Dict[Tuple[int, int], List],
                             pair_features: Dict[Tuple[int, int], Dict[str, List]],
                             df_dirty: pd.DataFrame,
                             synth_pair_features: Dict[Tuple[int, int], Dict[str, List]],
                             column: int):
    x_train = []  # train feature vectors
    y_train = []  # train labels
    x_test = []  # test features vectors
    all_error_correction_suggestions = []  # all cleaning suggestions for all errors flattened in a list
    corrected_cells = {}  # take user input as a cleaning result if available

    for error_cell in column_errors[column]:
        correction_suggestions = pair_features.get(error_cell, [])
        if error_cell in labeled_cells and labeled_cells[error_cell][0] == 1:
            # If an error-cell has been labeled by the user, use it to create the training dataset.
            # The second condition is always true if error detection and user labeling work without an error.
            for suggestion in correction_suggestions:
                x_train.append(pair_features[error_cell][suggestion])  # Puts features into x_train
                suggestion_is_correction = (suggestion == labeled_cells[error_cell][1])
                y_train.append(int(suggestion_is_correction))
                corrected_cells[error_cell] = labeled_cells[error_cell][1]  # user input as cleaning result
        else:  # put all cells that contain an error without user-correction in the "test" set.
            for suggestion in correction_suggestions:
                x_test.append(pair_features[error_cell][suggestion])
                all_error_correction_suggestions.append([error_cell, suggestion])

    for synth_cell in synth_pair_features:
        if synth_cell[1] == column:
            correction_suggestions = synth_pair_features.get(synth_cell, [])
            for suggestion in correction_suggestions:
                x_train.append(synth_pair_features[synth_cell][suggestion])
                suggestion_is_correction = (suggestion == df_dirty.iloc[synth_cell])
                y_train.append(int(suggestion_is_correction))

    return x_train, y_train, x_test, corrected_cells, all_error_correction_suggestions


def generate_synth_test_data(synth_pair_features: Dict[Tuple[int, int], Dict[str, List]],
                             df_dirty: pd.DataFrame,
                             column: int) -> Tuple[List, List, List]:
    """
    For one column, extract features x_test, labels y_test, and the corresponding list of error_correction_suggestions.
    """
    x_test = []
    y_test = []
    all_error_correction_suggestions = []  # all cleaning suggestions for all errors flattened in a list

    for synth_cell in synth_pair_features:
        if synth_cell[1] == column:
            correction_suggestions = synth_pair_features.get(synth_cell, [])
            for suggestion in correction_suggestions:
                x_test.append(synth_pair_features[synth_cell][suggestion])
                suggestion_is_correction = (suggestion == df_dirty.iloc[synth_cell])
                y_test.append(int(suggestion_is_correction))
                all_error_correction_suggestions.append([synth_cell, suggestion])

    return x_test, y_test, all_error_correction_suggestions


def test_synth_data(d, pair_features, synth_pair_features, classification_model: str, column: int, column_errors: dict, clean_on: str) -> float:
    """
    Test the difference in distribution between user_data and synth_data to determine if using synth_data in the
    cleaning problem is worthwhile.

    @param d: baran dataset object
    @param classification_model: which sklearn model to use for ensembling
    @param column: column that is being cleaned
    @param column_errors: errors per column
    @param clean_on: The data that is cleaned. Valid values are either "user_data" or "synth_data". If "user_data",
    clean errors in the user data using a model ensembling trained with synth_data. If "synth_data", clean errors
    in the synth_data using a model ensembling trained with user_data.
    @return: f1-score of the ensembling model cleaning erronous values.
    """
    synth_x_test, synth_y_test, synth_error_correction_suggestions = generate_synth_test_data(
        synth_pair_features, d.dataframe, column)

    x_train, y_train, x_test, user_corrected_cells, error_correction_suggestions = generate_train_test_data(
        column_errors,
        d.labeled_cells,
        pair_features,
        d.dataframe,
        {},  # keine synth daten: ich will ja wissen, wie gut die Verteilung ohne synth-daten vorhergesagt wird.
        column)

    # this condition enables unsupervised cleaning.
    if len(synth_x_test) > 0 and len(synth_y_test) > 0 and len(x_train) == 0:
        return 1.0  # no user-data available, but synth-data availble: Use that synth data to clean.

    score = 0.0
    is_valid_problem, _, _ = handle_edge_cases(x_train, synth_x_test, y_train)
    if not is_valid_problem:
        return score

    if clean_on == 'synth_data':
        if classification_model == "ABC" or sum(y_train) <= 2:
            gs_clf = sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
        elif classification_model == "CV":
            gs_clf = hpo.cross_validated_estimator(x_train, y_train)
        else:
            raise ValueError('Unknown model.')

        gs_clf.fit(x_train, y_train)

        if len(synth_y_test) > 0:
            synth_predicted_labels = gs_clf.predict(synth_x_test)
            score = f1_score(synth_y_test, synth_predicted_labels)
        return score

    elif clean_on == 'user_data':
        if classification_model == "ABC" or sum(y_train) <= 2:
            gs_clf = sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
        elif classification_model == "CV":
            gs_clf = hpo.cross_validated_estimator(synth_x_test, synth_y_test)
        else:
            raise ValueError('Unknown model.')

        gs_clf.fit(synth_x_test, synth_y_test)

        if len(y_train) > 0:
            user_predicted_labels = gs_clf.predict(x_train)
            score = f1_score(y_train, user_predicted_labels)
        return score

    else:
        raise ValueError('Can either clean_on "user_data" or "synth_data".')
