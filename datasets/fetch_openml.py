import argparse
import random
from typing import Tuple, List, Dict
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.datasets import fetch_openml
from helpers import apply_corruption, apply_imputer_corruption, validate_export

random.seed(0)
openml_ids_binary = [725, 310, 1046, 823, 137, 42493, 4135, 251, 151, 40922]
openml_ids_multiclass = [40498,
                         30,
                         1459,
                         1481,
                         184,
                         375,
                         32,
                         41027,
                         6,
                         40685,
                         43572,  # imdb movies
                         ]

fractions = [0.01, 0.05, 0.1, 0.3, 0.5]


def dataset_paths(
    data_id: int, corruption: str, error_fraction: int|float
) -> Tuple[Path, Path]:
    directory = Path(f"openml/{data_id}")
    directory.mkdir(exist_ok=True)
    clean_path = directory / "clean"
    corrupt_path = directory / f"{corruption}_{int(100*error_fraction)}"
    return clean_path, corrupt_path


def fetch_corrupt_imputer_dataset(data_id: int) -> Tuple[List[Dict], List[str]]:
    """
    Corrupt only the target column. Erroneous datasets that are generated this way
    are prone to gain heavily from a missing value imputer.
    """
    res = fetch_openml(data_id=data_id, as_frame=True, parser='auto')

    df = res["frame"]
    se_target = res["target"]
    if data_id == 43572:
        df.columns = [x.replace('(', '').replace(')', '') for x in df.columns]
        df['Actors'].fillna('', inplace=True)
        df['Director'].fillna('', inplace=True)
        df['Revenue_Millions'].fillna(-9999999, inplace=True)
        df['Metascore'].fillna(-9999999, inplace=True)
        df['Metascore'] = df['Metascore'].astype('int64')
        se_target = df['Year']
    if se_target is None:
        raise ValueError('OpenML dataset comes without target, specify one manually.')

    clean_path, _ = dataset_paths(data_id, "", 0)
    df.to_csv(str(clean_path) + '.csv', index=False)
    df.to_parquet(str(clean_path) + '.parquet', index=False)
    metadata = []
    dtypes = [str(x) for x in df.dtypes.values]

    #for corruption_name in ["imputer_simple_mcar", "imputer_simple_mar", "imputer_simple_mnar"]:
    for corruption_name in ["imputer_simple_mcar"]:
        for fraction in fractions:
            se_corrupted = apply_imputer_corruption(corruption_name, df, se_target, fraction)
            df_corrupted = df.copy()
            if data_id == 43572:
                df_corrupted['Year'] = se_corrupted
            else:
                df_corrupted.iloc[:, -1] = se_corrupted

            metadata.append(
                {
                    "dataset_id": data_id,
                    "corruption_name": corruption_name,
                    "fraction": fraction,
                }
            )

            clean_path, corrupt_path = dataset_paths(data_id, corruption_name, fraction)
            df_corrupted.to_parquet(str(corrupt_path) + ".parquet", index=False)
            df_corrupted.to_csv(str(corrupt_path) + ".csv", index=False)
            validate_export(str(corrupt_path), df_corrupted, df, fraction)
    return metadata, dtypes


def fetch_corrupt_dataset(data_id: int) -> Tuple[List[Dict], List[str]]:
    """
    Goal of this exercise is to showcase that the imputer approach can do
    meaningful things with continuous features whereas baran and the likes just
    fail miserably.
    """
    res = fetch_openml(data_id=data_id, as_frame=True, parser='auto')

    df = res["frame"]
    if data_id == 43572:
        df.columns = [x.replace('(', '').replace(')', '') for x in df.columns]
        df['Actors'].fillna('', inplace=True)
        df['Director'].fillna('', inplace=True)
        df['Revenue_Millions'].fillna(-9999999, inplace=True)
        df['Metascore'].fillna(-9999999, inplace=True)
        df['Metascore'] = df['Metascore'].astype('int64')
    clean_path, _ = dataset_paths(data_id, "", 0)
    df.to_csv(str(clean_path) + '.csv', index=False)
    df.to_parquet(str(clean_path) + '.parquet', index=False)
    metadata = []
    dtypes = [str(x) for x in df.dtypes.values]

    for corruption_name in ["imputer_simple_mcar", "imputer_simple_mar", "imputer_simple_mnar"]:
        for fraction in fractions:
            df_corrupted = apply_corruption(corruption_name, df, fraction)
            metadata.append(
                {
                    "dataset_id": data_id,
                    "corruption_name": corruption_name,
                    "fraction": fraction,
                }
            )

            clean_path, corrupt_path = dataset_paths(data_id, corruption_name, fraction)
            df_corrupted.to_parquet(str(corrupt_path) + ".parquet", index=False)
            df_corrupted.to_csv(str(corrupt_path) + ".csv", index=False)
            validate_export(str(corrupt_path), df_corrupted, df, fraction)

    return metadata, dtypes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch and corrupt OpenML datasets')
    parser.add_argument('--imputer', action='store_true',
                        help="insert missing values only into dataset's target column.")

    args = parser.parse_args()

    if args.imputer:
        print("Imputer-approach: Inserting errors into target column only.")
    else:
        print("Inserting errors into all columns.")
    metadatas = []
    all_dtypes = []
    for dataset_id in openml_ids_binary:
    #for dataset_id in [1046]:
        if args.imputer:
            metadata, dtypes = fetch_corrupt_imputer_dataset(dataset_id)
        else:
            metadata, dtypes = fetch_corrupt_dataset(dataset_id)
        all_dtypes.extend(dtypes)
        metadata = [{**x, "dataset_type": "binary classification"} for x in metadata]
        metadatas.extend(metadata)
    for dataset_id in openml_ids_multiclass:
    #for dataset_id in []:
        if args.imputer:
            metadata, dtypes = fetch_corrupt_imputer_dataset(dataset_id)
        else:
            metadata, dtypes = fetch_corrupt_dataset(dataset_id)
        all_dtypes.extend(dtypes)
        metadata = [
            {**x, "dataset_type": "multiclass classification"} for x in metadata
        ]
        metadatas.extend(metadata)

    errors = pd.DataFrame(metadatas)
    if args.imputer:
        errors.to_csv("error_stats_openml_imputer.csv", index=False)
    else:
        errors.to_csv("error_stats_openml.csv", index=False)
    c = Counter(all_dtypes)
    print(f'Created datasets, encountered dtypes {c}.')
