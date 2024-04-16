import pandas as pd

def export_table(name: str, path: str, n_rows=None):
    df_clean = pd.read_csv(path, dtype=str)
    if n_rows is not None:
        df_clean = df_clean.iloc[:n_rows, :]

    df_clean.to_csv(f"to_metanome/{name}.csv", index=False)
    print(f'Created csv for dataset {name}')


def main():
    baran_dataset_ids = ["beers", "flights", "hospital", "rayyan", "toy", "food", "tax"]
    renuver_dataset_ids = ["bridges", "cars", "glass", "restaurant"]
    openml_dataset_ids = ["6", "137", "151", "184", "1481", "41027", "43572"]

    for dataset_name in renuver_dataset_ids:
        agg_name = f'{dataset_name}'
        path = f"../../datasets/renuver/{dataset_name}/clean.csv"
        export_table(agg_name, path)

    for dataset_name in baran_dataset_ids:
        agg_name = f'{dataset_name}'
        path = f"../../datasets/{dataset_name}/clean.csv"
        export_table(agg_name, path)

    for dataset_name in openml_dataset_ids:
        agg_name = f'{dataset_name}'
        path = f"../../datasets/openml/{dataset_name}/clean.csv"
        export_table(agg_name, path, 1000)


if __name__ == '__main__':
    main()
