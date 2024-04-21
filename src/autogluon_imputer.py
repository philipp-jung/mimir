"""
AutoGluon Imputer
Imputes missing values in tables based on autogluon-tabular. A convenience wrapper around AutoGluon.
"""
import pickle
import inspect
from autogluon.tabular import TabularPredictor
from pathlib import Path

from typing import List, Any

import numpy as np
import pandas as pd


def random_split(data_frame: pd.DataFrame,
                 split_ratios: List[float] = None,
                 seed: int = 10) -> List[pd.DataFrame]:
    """
    Shuffles and splits a Dataframe into partitions.

    :param data_frame: a pandas DataFrame
    :param split_ratios: percentages of splits
    :return:
    """
    if split_ratios is None:
        split_ratios = [.8, .2]
    sections = np.array([int(r * len(data_frame)) for r in split_ratios]).cumsum()
    return np.split(data_frame.sample(frac=1, random_state=seed), sections)[:len(split_ratios)]


class TargetColumnException(Exception):
    """Raised when a target column cannot be used as label for a supervised learning model"""
    pass


class AutoGluonImputer():

    """
    AutoGluonImputer

    :param model_name: name of the AutoGluonImputer (as tring)
    :param input_columns: list of input column names (as strings)
    :param output_column: output column name (as string)
    :param verbosity: verbosity level from 0 to 2
    :param output_path: path to which the AutoGluonImputer is saved to
    """

    def __init__(self,
                 columns: List[str],
                 model_name: str = 'AutoGluonImputer',
                 input_columns: List[str] = None,
                 output_column: str = None,
                 verbosity: int = 0,
                 output_path: str = ''
                 ) -> None:

        self.model_name = model_name
        self.columns = columns  # accessed by the imputer feature generator
        self.input_columns = input_columns
        self.output_column = output_column
        self.verbosity = verbosity
        self.predictor = None
        self.output_path = Path('./') if output_path == '' else Path(output_path)

    @property
    def datawig_model_path(self) -> Path:
        return self.output_path / Path(f'datawigModels/{self.model_name}.pickle')

    @property
    def ag_model_path(self) -> Path:
        return self.output_path / Path(f'agModels/{self.model_name}')

    def fit(self,
            df: pd.DataFrame,
            time_limit: int = 30,
            presets: str = 'medium_quality') -> Any:
        """
        Trains AutoGluonImputer model for a single column

        :param train_df: The data used to run autogluon on.
        :param time_limit: time limit for AutoGluon in seconds
        """

        if self.input_columns is None:
            self.input_columns = [c for c in df.columns if c is not self.output_column]

        if df[self.output_column].value_counts().max() < 10:
            raise TargetColumnException("Maximum class count below 10, "
                                        "cannot train imputation model")

        self.predictor = TabularPredictor(label=self.output_column,
                                      path=self.ag_model_path,
                                      verbosity=self.verbosity,
                                      problem_type='multiclass').\
        fit(train_data=df,
            time_limit=time_limit,
            presets=[presets],
            calibrate=True,
            verbosity=self.verbosity,
            num_cpus=25)  # funny: num_cpus expects a string, and if you pass a string, AG crashes.

        return self

    def predict_proba(self, data_frame: pd.DataFrame):
        """
        Run the imputer on a data_frame and return class probabilities.
        """
        if not self.predictor:
            raise ValueError("No predictor has been trained. Run .fit() first,"
                             " then continue with .predict().")
        probas = self.predictor.predict_proba(data_frame)
        return probas

    def save(self):
        """
        Saves model to disk. Creates the directory
        `{self.output_path}/datawigModels` to to save the model to, and
        creates it, if it doesn't exist.
        """
        if not self.datawig_model_path.parent.exists():
            self.datawig_model_path.parent.mkdir()

        params = {k: v for k, v in self.__dict__.items() if k != 'module'}
        pickle.dump(params, open(self.datawig_model_path, "wb"))

    @staticmethod
    def load(output_path: str, model_name: str) -> Any:
        """
        Loads model from output path. Expects
        - a folder `{output_path}/datawigModels` to exist and contain
          `{model_name}.pickle`, itself containing a serialized
          AutoGluonImputer.
        - a folder `{output_path}/agModels` to exist and contain a folder
          named `model_name`, itself containing AutoGluon serialized models.

        :param model_name: string name of the AutoGluonImputer model
        :param output_path: path containing agModels/ and datawigModels/ folders
        :return: AutoGluonImputer model

        """
        load_path = output_path / Path(f"datawigModels/{model_name}.pickle")
        with open(load_path, 'rb') as f:
            params = pickle.load(f)
        imputer_signature = inspect.getfullargspec(
            AutoGluonImputer.__init__)[0]

        constructor_args = {p: params[p]
                            for p in imputer_signature if p != 'self'}
        non_constructor_args = {p: params[p] for p in params.keys() if
                                p not in ['self'] + list(constructor_args.keys())}

        # use all relevant fields to instantiate AutoGluonImputer
        imputer = AutoGluonImputer(**constructor_args)
        # then set all other args
        for arg, value in non_constructor_args.items():
            setattr(imputer, arg, value)

        # lastly, load AG Model
        ag_load_path = output_path / Path(f"agModels/{model_name}")
        imputer.predictor = TabularPredictor.load(ag_load_path)
        return imputer
