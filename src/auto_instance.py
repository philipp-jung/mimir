from typing import Union, Dict
from autogluon_imputer import AutoGluonImputer, TargetColumnException
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def train_cleaning_model(df: pd.DataFrame,
                         dataset_name: str,
                         label: int,
                         time_limit : int = 90,
                         use_cache: bool = True,
                         row_threshold: int = 50,
                         verbosity: int = 0) -> Union[AutoGluonImputer, None]:
    """
    Train an autogluon model for the purpose of cleaning data.
    Optionally, you can set verbosity to control how much output AutoGluon
    produces during training.
    Returns a predictor object.
    @param df: Dataframe to clean the imputer model on.
    @param dataset_name: used for storing the imputer model on the drive & caching.
    @param label: index of the column to be imputed.
    @param time_limit: autogluon time limit training the imputer model.
    @param use_cache: Instead of training models, use cached models for the same dataset_name & label if it exists.
    @param row_threshold: Minimal row count of the dataframe to train a model with.
    @param verbosity: integer between 0 and 2 to control autogluon's verbosity.
    @return: AutoGluonImputer, capable of imputing values in the label column.
    """

    if df.shape[0] < row_threshold or df.shape[0] == 0:  # too few rows to train an imputer model with.
        return None
    lhs = list(df.columns)
    del lhs[label - 1]
    rhs = df.columns[label]
    model_name = f'{dataset_name}-{label}-imputer-model'

    # Splitting here maybe needs to be removed, because it doesn't scale well with
    # dataset size. AutoGluon does smart train-test-splits internally, might be reasonable
    # leveraging them. I currently keep this to have control over the data in one place.
    df_train, df_test = train_test_split(df, test_size=.1)

    # Since I concatenate dataframes earlier, duplicate index entries are possible. Which is why I reset the index.
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    try:
        if not use_cache:
            raise FileNotFoundError
        imputer = AutoGluonImputer.load(output_path='./', model_name=model_name)
    except FileNotFoundError:
        try:
            imputer = AutoGluonImputer(
                model_name=model_name,
                columns=df.columns,
                input_columns=lhs,
                output_column=rhs,
                verbosity=verbosity,
            )
            imputer.fit(train_df=df_train,
                        test_df=df_test,
                        time_limit=time_limit)

        except TargetColumnException:
            if verbosity > 0:
                print(f'Could not train an imputer for rhs {label}: Target class has less than 10 occurrences.')
            return None
        except IndexError:
            if verbosity > 0:
                print(f'Could not traind an imputer for rhs {label}: Something went wrong calibrating class probabilities.')
            return None
        # TODO debug what happens when KeyError is raised.
        except (ValueError, KeyError) as e:
            if verbosity > 0:
                print(e)
                print(f'Could not traind an imputer for rhs {label}: AutoGluon did not train a single model successfully.')
            return None
    imputer.save()
    return imputer


def make_cleaning_prediction(df_dirty: pd.DataFrame,
                             probas: pd.DataFrame,
                             target_col: int) -> pd.Series:
    """
    When cleaning, the task is defined such that the ids of the dirty rows
    are known. This uses the knowledge that it is always incorrect to return
    the class that is known to be dirty.

    This does the following:
    1) Check the most probable class. If the class is not the dirty class,
    return it.
    2) If the returned class is the dirty class, return the second most likely
    class.
    """
    dirty_label_mask = pd.DataFrame()
    for c in probas.columns:
        dirty_label_mask[c] = df_dirty[target_col] == c
    probas[dirty_label_mask] = -1  # dirty labels are never chosen

    i_classes = np.argmax(probas.values, axis=1)
    cleaning_predictions = [probas.columns[i] for i in i_classes]
    return pd.Series(cleaning_predictions)


def imputation_based_corrector(df: pd.DataFrame,
                               df_clean_subset: pd.DataFrame,
                               detected_cells: dict) -> Dict[tuple, str]:
    """
    Trains an imputer model for each column in the dataframe. The data used to create that imputer is a
    subset of the original dataframe, containing only cleaned rows.

    Returns the most probable class for each error tuple.
    """
    df = df.astype(str)
    results = {}

    for i_col, col in enumerate(df.columns):
        imputer = train_cleaning_model(df_clean_subset, label=i_col, time_limit=30)
        if imputer is not None:
            error_rows = [row for (row, _) in list(detected_cells.keys())]
            df_rhs_errors = df.iloc[error_rows, :]
            probas = imputer.predict_proba(df_rhs_errors)

            se_predicted = make_cleaning_prediction(df_rhs_errors,
                                                    probas,
                                                    col)
            n_error = 0
            for cell in detected_cells.keys():
                error_row, error_col = cell
                if error_col == i_col:
                    results[cell] = se_predicted.iloc[n_error]
                    n_error += 1
    return results


def get_clean_table(df: pd.DataFrame, detected_cells: dict, user_input: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the subset of a dataframe that doesn't contain any errors.

    @param df: The typed dataframe to be cleaned, containing errors.
    @param detected_cells: Baran standard way of storing information on cells with errors.
    @param user_input: DataFrame containing the user-inputted rows, with correct types.
    @return: Dataframe that doesn't contain values that contain errors.
    """
    error_rows = [row for row, column in detected_cells.keys()]
    clean_mask = [True if x not in error_rows else False for x in range(df.shape[0])]
    clean_subset = df.iloc[clean_mask, :]
    unioned_data = pd.concat([clean_subset, user_input])
    return unioned_data
