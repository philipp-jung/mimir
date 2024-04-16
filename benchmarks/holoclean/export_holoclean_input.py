import os
import pandas as pd
from pathlib import Path


def convert_json_to_constraints(json_content):
    """
    Metanome's output has a funky format. We massage it a bit to get the DC format that
    HoloClean expects. Thanks to Bernardo Breve for providing
    this function!
    """
    output=[]
    operationsArr = ['UNEQUAL', 'LESS_EQUAL', 'GREATER_EQUAL', 'EQUAL', 'LESS', 'GREATER']
    operationSign = ['IQ', 'LTE', 'GTE', 'EQ', 'LT', 'GT']

    dataset_name = json_content[0]['predicates'][0]['column1']['tableIdentifier'].split('.')[0]
    for dc in json_content:
        output_string = "t1&t2"
        for p in dc['predicates']:
            index = operationsArr.index(p['op'])
            output_string += '&' + operationSign[index] + '('
            output_string += "t1." + p['column1']['columnIdentifier']
            output_string += ",t2." + p['column2']['columnIdentifier'] + ')'

        output.insert(len(output), output_string)

    return output, dataset_name


def apply_conversion_to_directory(input_path, output_path):
    # List all files
    json_files = [file for file in os.listdir(input_path)]

    for json_file in json_files:
        json_file_path = os.path.join(input_path, json_file)

        # Read JSON content from the file
        with open(json_file_path, 'rt') as f:
            content = f.read().split('\n')
            content = [f'{line},' for line in content[:-1]]
            content[0] = '[' + content[0]
            content[-1] = content[-1] + ']'
            formatted = '\n'.join(content)
            json_content = eval(formatted)
        
        algorithm = 'hydra'

        # Convert JSON to constraints
        constraints, dataset_name = convert_json_to_constraints(json_content)

        # Write the converted constraints to a new file
        output_file_path = os.path.join(output_path, f'{algorithm}_{dataset_name}.txt')

        with open(output_file_path, 'wt') as output_file:
            for x in constraints:
                output_file.write(x+'\n')
        print(f'Converted DCs of dataset {dataset_name} to {output_file_path}.')


class ExportDataset:
    """
    Helper to make exporting datasets easier.
    """
    def __init__(self, export_path: str):
        self.export_path = Path(export_path)

    def export_table(self, name: str, path: str, n_rows=None):
        df = pd.read_csv(path, dtype=str)
        if n_rows is not None:
            df = df.iloc[:n_rows, :]
        df.to_csv(self.export_path/f"{name}.csv", index=False)

    def export_clean_table(self, name: str, path: str, n_rows=None):
        df_clean = pd.read_csv(path, dtype=str)
        if n_rows is not None:
            df_clean = df_clean.iloc[:n_rows, :]
        melted_df = pd.melt(df_clean.reset_index(),
                            id_vars=['index'],
                            var_name='column_name',
                            value_name='cell_value')

        # Rename the columns if needed
        melted_df.columns = ['tid', 'attribute', 'correct_val']
        melted_df.to_csv(self.export_path/f"{name}.csv", index=False)


def export_datasets(export_path: str):
    """
    Export datasets the datasets's clean and dirty version to export_path
    as .csv files.
    """
    e = ExportDataset(export_path)
    baran_dataset_ids = ["beers", "flights", "hospital", "rayyan", "tax", "food"]
    renuver_dataset_ids = ["bridges", "cars", "glass", "restaurant"]
    openml_dataset_ids = ["6", "137", "151", "184", "1481", "43572"]

    for dataset_name in renuver_dataset_ids:
        clean_path = f"../../datasets/renuver/{dataset_name}/clean.csv"
        e.export_clean_table(dataset_name, clean_path)
        for version in range(1, 6):
            for error_fraction in range(1, 6):
                agg_name = f'{dataset_name}_{error_fraction}_{version}'
                path_dirty = f"../../datasets/renuver/{dataset_name}/{dataset_name}_{error_fraction}_{version}.csv"
                e.export_table(agg_name, path_dirty)
    print('Exported RENUVER datasets to ' + export_path)

    for dataset_name in baran_dataset_ids:
        path = f"../../datasets/{dataset_name}/clean.csv"
        e.export_clean_table(dataset_name, path)

        agg_name = f'{dataset_name}_dirty'
        path_dirty = f"../../datasets/{dataset_name}/dirty.csv"
        e.export_table(agg_name, path_dirty)

    print('Exported Baran datasets to ' + export_path)

    for dataset_name in openml_dataset_ids:
        path = f"../../datasets/openml/{dataset_name}/clean.csv"
        e.export_clean_table(dataset_name, path, 1000)
        for error_class in ['imputer_simple_mcar', 'simple_mcar']:
            for error_fraction in [1, 5, 10]:
                agg_name = f'{dataset_name}_{error_class}_{error_fraction}'
                path_dirty = f"../../datasets/openml/{dataset_name}/{error_class}_{error_fraction}.csv"
                e.export_table(agg_name, path_dirty, 1000)
    print('Exported OpenML datasets to ' + export_path)

if __name__ == '__main__':
    export_datasets('holoclean_input/datasets/')

    dc_input_path = 'metanome_output/'
    dc_output_path = 'holoclean_input/dcs/'

    apply_conversion_to_directory(dc_input_path, dc_output_path)
