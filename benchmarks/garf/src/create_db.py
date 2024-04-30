import sqlite3
import pandas as pd
from typing import Dict, Optional, List


def create_tables_for_dataset(connection: sqlite3.Connection, name: str, path: str, path_dirty: str):
    df_clean = pd.read_csv(path, dtype=str).fillna('__missing__')

    df_clean['Label'] = None
    df_clean.to_sql(f"{name}", connection, if_exists="replace", index=False)

    df_dirty = pd.read_csv(path_dirty, dtype=str).fillna('__missing__')

    df_dirty['Label'] = None
    df_dirty.to_sql(f"{name}_copy", connection, if_exists="replace", index=False)
    df_dirty.to_sql(f"{name}_dirty", connection, if_exists="replace", index=False)


def create_database(datasets: List[Dict], database_path: Optional[str] = None):
    if database_path is None:
        connection = sqlite3.connect("database.db")
    else:
        connection = sqlite3.connect(database_path)

    for d in datasets:
        print(f"Creating tables for {d['name']} from clean file at {d['path']}, dirty file at {d['path_dirty']}.")
        create_tables_for_dataset(connection, d['name'], d['path'], d['path_dirty'])

    connection.close()
    print("Database successfully created.")


if __name__ == '__main__':
    #datasets = {"Hosp_rules": "./data/Hosp_clean.csv", "Food": "./data/Food_clean.csv"}
    datasets = [{"name": "hospital",
                 "path": "./data/hospital/clean.csv",
                 "path_dirty": "./data/hospital/dirty.csv"
                 },
                ]

    create_database(datasets)
