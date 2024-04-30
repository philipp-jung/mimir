import pandas as pd
import sqlite3


def create_tables_for_dataset(name: str, path: str, path_dirty: str, n_rows=None):
    connection = sqlite3.connect('src/database.db')
    df_clean = pd.read_csv(path, dtype=str).fillna('__missing__')
    if n_rows is not None:
        df_clean = df_clean.iloc[:n_rows, :]

    df_clean['Label'] = None
    df_clean.to_sql(f"{name}", connection, if_exists="replace", index=False)

    df_dirty = pd.read_csv(path_dirty, dtype=str).fillna('__missing__')

    if n_rows is not None:
        df_dirty = df_dirty.iloc[:n_rows, :]

    df_dirty['Label'] = None
    df_dirty.to_sql(f"{name}_copy", connection, if_exists="replace", index=False)
    df_dirty.to_sql(f"{name}_dirty", connection, if_exists="replace", index=False)

    print(f'Created tables for dataset {name}')
    connection.close()


def main():
    baran_dataset_ids = ["beers", "flights", "hospital", "rayyan", "tax", "food"]
    renuver_dataset_ids = ["bridges", "cars", "glass", "restaurant"]
    openml_dataset_ids = ["6", "137", "151", "184", "1481", "41027", "43572"]

    for dataset_name in renuver_dataset_ids:
        for version in range(1, 6):
            for error_fraction in range(1, 6):
                agg_name = f'{dataset_name}_{error_fraction}_{version}'
                path = f"../../datasets/renuver/{dataset_name}/clean.csv"
                path_dirty = f"../../datasets/renuver/{dataset_name}/{dataset_name}_{error_fraction}_{version}.csv"
                create_tables_for_dataset(agg_name, path, path_dirty)

    for dataset_name in baran_dataset_ids:
        agg_name = f'{dataset_name}'
        path = f"../../datasets/{dataset_name}/clean.csv"
        path_dirty = f"../../datasets/{dataset_name}/dirty.csv"
        create_tables_for_dataset(agg_name, path, path_dirty)

    for dataset_name in openml_dataset_ids:
        for error_class in ['imputer_simple_mcar', 'simple_mcar']:
            for error_fraction in [1, 5, 10]:
                agg_name = f'{dataset_name}_{error_class}_{error_fraction}'
                path = f"../../datasets/openml/{dataset_name}/clean.csv"
                path_dirty = f"../../datasets/openml/{dataset_name}/{error_class}_{error_fraction}.csv"
                create_tables_for_dataset(agg_name, path, path_dirty, 1000)
    print(f'Wrote tables to src/database.db.')


if __name__ == '__main__':
    main()
