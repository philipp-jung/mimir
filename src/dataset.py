import os
from typing import Union, Dict, Tuple
import pandas as pd

class Dataset:
    """
    The dataset class.
    """

    def __init__(self,
                 dataset_name: str,
                 error_fraction: Union[None, int] = None,
                 version: Union[None, int] = None,
                 error_class: Union[None, str] = None,
                 n_rows: Union[None, int] = None):
        """
        I currently use four different sources of datasets: the original Baran paper, the RENUVER paper, datasets that
        I assemble from OpenML, and hand-selected datasets from the UCI website. Depending on the source, the datasets
        differ:
        - Datasets from the Baran paper are uniquely identified by their name.
        - Datasets from the RENUVER paper are uniquely identified by their name, error_fraction and version in [1, 5].
        - Datasets that I generate from OpenML are identified by their name and error_fraction.
        - Datasets that I generate from UCI are identified by their name and error_fraction.

        Whenever possible, uses datasets that contain dtype information, stored as .parquet, which is beneficial for
        the imputer feature generator.

        @param dataset_name: Name of the dataset.
        @param error_fraction: Baran datasets don't have this. The % of cells containing errors in RENUVER datasets and
        the Ruska-corrupted datasets. The % of values in a column in OpenML datasets. Value between 0 - 100.
        @param version: generating errors in Jenga is not deterministic. So it makes sense to create a couple of
        versions to avoid outlier corruptions.
        @param error_class: Strategy according to which an error has been created. Only applies to OpenML datasets.
        @param n_rows: If n_rows is specified, the dataset gets subsetted to the first n_rows rows.
        """
        self.repaired_dataframe = None  # to be assigned after cleaning suggestions were applied.
        self.error_fraction = None
        self.version = None
        self.error_class = None

        openml_dataset_ids = ["725", "310", "1046", "823", "137", "42493", "4135", "251", "151", "40922", "40498", "30", "1459", "1481", "184", "375", "32", "41027", "6", "40685", "43572"]
        hpi_dataset_ids = ["cddb"]
        renuver_dataset_ids = ["bridges", "cars", "glass", "restaurant"]
        baran_dataset_ids = ["beers", "flights", "hospital", "tax", "rayyan", "toy", "debug", "synth-debug", "food"]
        uci_dataset_ids = ["adult", "breast-cancer", "letter", "nursery"]

        if dataset_name in baran_dataset_ids:
            self.path = f"../datasets/{dataset_name}/dirty.csv"
            self.clean_path = f"../datasets/{dataset_name}/clean.csv"
            if dataset_name == 'food':
                self.parquet_path = f"../datasets/{dataset_name}/dirty.parquet"
                self.typed_clean_path = f"../datasets/{dataset_name}/clean.parquet"
            else:
                self.parquet_path = self.path  # no parquet file is available.
                self.typed_clean_path = self.clean_path  # no parquet file is available.
            self.name = dataset_name

        elif dataset_name in renuver_dataset_ids:
            self.path = f"../datasets/renuver/{dataset_name}/{dataset_name}_{error_fraction}_{version}.csv"
            self.clean_path = f"../datasets/renuver/{dataset_name}/clean.csv"
            self.parquet_path = self.path  # no parquet file is available.
            self.typed_clean_path = self.clean_path  # no parquet file is available.

            self.name = dataset_name
            self.error_fraction = error_fraction
            self.version = version

        elif dataset_name in hpi_dataset_ids:
            if error_class is None:
                raise ValueError('Please specify the error class with which the openml dataset has been corrupted.')
            self.path = f"../datasets/{dataset_name}/{error_class}_{error_fraction}.csv"
            self.parquet_path = f"../datasets/{dataset_name}/{error_class}_{error_fraction}.parquet"
            self.clean_path = f"../datasets/{dataset_name}/clean.csv"
            self.typed_clean_path = os.path.splitext(self.clean_path)[0] + '.parquet'

            self.name = dataset_name
            self.error_class = error_class
            self.error_fraction = error_fraction

        elif dataset_name in openml_dataset_ids:
            if error_class is None:
                raise ValueError('Please specify the error class with which the openml dataset has been corrupted.')
            self.path = f"../datasets/openml/{dataset_name}/{error_class}_{error_fraction}.csv"
            self.parquet_path = f"../datasets/openml/{dataset_name}/{error_class}_{error_fraction}.parquet"
            self.clean_path = f"../datasets/openml/{dataset_name}/clean.csv"
            self.typed_clean_path = os.path.splitext(self.clean_path)[0] + '.parquet'

            self.name = dataset_name
            self.error_class = error_class
            self.error_fraction = error_fraction

        elif dataset_name in uci_dataset_ids:
            self.path = f"../datasets/{dataset_name}/MCAR/dirty_{error_fraction}.csv"
            self.clean_path = f"../datasets/{dataset_name}/clean.csv"
            self.parquet_path = self.path  # no parquet file is available.
            self.typed_clean_path = self.clean_path  # no parquet file is available.

            self.name = dataset_name
            self.error_fraction = error_fraction

        else:
            raise ValueError(f'Dataset with name {dataset_name} is not known. Please add it to the config '
                             f'in dataset.py.')

        self.dataframe = self.read_csv_dataset(self.path)
        self.clean_dataframe = self.read_csv_dataset(self.clean_path)

        if self.parquet_path.endswith('.parquet'):
            self.typed_dataframe = self.read_parquet_dataset(self.parquet_path)
        else:  # if typed .parquet file is unavailable, fall back to .csv file.
            self.typed_dataframe = self.read_csv_dataset(self.parquet_path)

        if self.typed_clean_path.endswith('.parquet'):
            self.typed_clean_dataframe = self.read_parquet_dataset(self.typed_clean_path)
        else:  # if typed .parquet file is unavailable, fall back to .csv file.
            self.typed_clean_dataframe = self.read_csv_dataset(self.typed_clean_path)

        if n_rows is not None:
            self.dataframe = self.dataframe.iloc[:n_rows, :]
            self.clean_dataframe = self.clean_dataframe.iloc[:n_rows, :]
            self.typed_dataframe = self.typed_dataframe.iloc[:n_rows, :]
            self.typed_clean_dataframe = self.typed_clean_dataframe.iloc[:n_rows, :]


    @staticmethod
    def read_parquet_dataset(dataset_path: Union[str, None]):
        """
        This method reads a dataset from a parquet file path. This is nice for the imputer because
        parquet preserves dtypes.
        """
        if dataset_path is None:
            return None
        dataframe = pd.read_parquet(dataset_path)
        return dataframe

    def read_csv_dataset(self, dataset_path):
        """
        This method reads a dataset from a csv file path.
        """
        dataframe = pd.read_csv(dataset_path, sep=",", header="infer", encoding="utf-8", dtype=str,
                                    keep_default_na=False, low_memory=False)
        return dataframe

    @staticmethod
    def write_csv_dataset(dataset_path, dataframe):
        """
        This method writes a dataset to a csv file path.
        """
        dataframe.to_csv(dataset_path, sep=",", header=True, index=False, encoding="utf-8")

    @staticmethod
    def get_dataframes_difference(df_1: pd.DataFrame, df_2: pd.DataFrame) -> Dict:
        """
        This method compares two dataframes df_1 and df_2. It returns a dictionary whose keys are the coordinates of
        a cell. The corresponding value is the value in df_1 at the cell's position if the values of df_1 and df_2 are
        not the same at the given position.
        """
        if df_1.shape != df_2.shape:
            raise ValueError("Two compared datasets do not have equal sizes.")
        difference_dictionary = {}
        for row in range(df_1.shape[0]):
            for col in range(df_1.shape[1]):
                if df_1.iloc[row, col] != df_2.iloc[row, col]:
                    difference_dictionary[(row, col)] = df_1.iloc[row, col]
        return difference_dictionary

    def create_repaired_dataset(self, correction_dictionary):
        """
        This method takes the dictionary of corrected values and creates the repaired dataset.
        """
        self.repaired_dataframe = self.dataframe.copy()
        for cell in correction_dictionary:
            self.repaired_dataframe.iloc[cell] = correction_dictionary[cell]

    def get_df_from_labeled_tuples(self):
        """
        Turns the labeled tuples into a dataframe.
        """
        return self.clean_dataframe.iloc[list(self.labeled_tuples.keys()), :]

    def _get_actual_errors_dictionary_ground_truth(self) -> Dict[Tuple[int, int], str]:
        """
        Returns a dictionary that resolves every error cell to the ground truth.
        """
        return self.get_dataframes_difference(self.clean_dataframe, self.dataframe)

    def get_errors_dictionary(self) -> Dict[Tuple[int, int], str]:
        """
        This method compares the clean and dirty versions of a dataset. The returned dictionary resolves to the error
        values in the dirty dataframe.
        """
        return self.get_dataframes_difference(self.dataframe, self.clean_dataframe)

    def get_correction_dictionary(self):
        """
        This method compares the repaired and dirty versions of a dataset.
        """
        return self.get_dataframes_difference(self.repaired_dataframe, self.dataframe)

    def get_data_quality(self):
        """
        This method calculates data quality of a dataset.
        """
        return 1.0 - float(len(self._get_actual_errors_dictionary_ground_truth())) / (self.dataframe.shape[0] * self.dataframe.shape[1])

    def get_data_cleaning_evaluation(self, correction_dictionary, sampled_rows_dictionary=False):
        """
        This method evaluates data cleaning process.
        """
        actual_errors = self._get_actual_errors_dictionary_ground_truth()
        if sampled_rows_dictionary:
            actual_errors = {(i, j): actual_errors[(i, j)] for (i, j) in actual_errors if i in sampled_rows_dictionary}
        ed_tp = 0.0
        ec_tp = 0.0
        output_size = 0.0
        for cell in correction_dictionary:
            if (not sampled_rows_dictionary) or (cell[0] in sampled_rows_dictionary):
                output_size += 1
                if cell in actual_errors:
                    ed_tp += 1.0
                    if correction_dictionary[cell] == actual_errors[cell]:
                        ec_tp += 1.0
        ed_p = 0.0 if output_size == 0 else ed_tp / output_size
        ed_r = 0.0 if len(actual_errors) == 0 else ed_tp / len(actual_errors)
        ed_f = 0.0 if (ed_p + ed_r) == 0.0 else (2 * ed_p * ed_r) / (ed_p + ed_r)
        ec_p = 0.0 if output_size == 0 else ec_tp / output_size
        ec_r = 0.0 if len(actual_errors) == 0 else ec_tp / len(actual_errors)
        ec_f = 0.0 if (ec_p + ec_r) == 0.0 else (2 * ec_p * ec_r) / (ec_p + ec_r)
        return [ed_p, ed_r, ed_f, ec_p, ec_r, ec_f]
########################################


########################################
if __name__ == "__main__":
    d = Dataset('toy')
    print(d.get_data_quality())
########################################
