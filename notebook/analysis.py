import json
from typing import Tuple

import pandas as pd

class Analysis:

    def __repr__(self):
        return f'analysis_{self.dataset}'

    def __init__(self, filepath: str):
        with open(filepath, 'rt') as f:
            data = json.load(f)
        
        self._columns = set([c['cell'][1] for c in data['samples']])
        self._correctors = set([c['corrector'] for c in data['correctors']])
        self.dataset = data['dataset']
        self.error_stats = data['error_stats']
        self.all_errors = sum([x['errors'] for x in self.error_stats])
        self.samples = data['samples']
        self.correctors_store = data['correctors']

        self.sample_cells = [s['cell'] for s in data['samples']]
        self.column_errors = {c: [s for s in self.sample_cells if s[1] == c] for c in self._columns}

    def error_stats(self):
        return self.error_stats

    def overview(self, column: int):
        if column not in self._columns:
            raise ValueError(f'Column {column} not available. Columns that contain errors are {self._columns}')
        subset_col = [x for x in self.samples if x['column'] == column]
        errors_in_col = [x["errors"] for x in self.error_stats if x["column"] == column][0]
        print(f'{errors_in_col}/{self.all_errors} errors in column {column} ({round(round(errors_in_col / self.all_errors, 3) * 100, 1)}%)')
        df = pd.DataFrame(subset_col)
        return df

    def corrector(self, corrector_name: str, column: int):
        if column not in self._columns:
            raise ValueError(f'Column {column} not available. Columns that contain errors are {self._columns}')
        if corrector_name not in self._correctors:
            raise ValueError(f'{corrector_name} is not an available corrector. Chosse one of {self._correctors}.')
        res = [x for x in self.correctors_store if (x['corrector'] == corrector_name and x['column'] == column)]
        return pd.DataFrame(res)
    
    def cell(self, cell: Tuple[int, int]):
        res = [x for x in self.correctors_store if x['cell'] == cell]
        return pd.DataFrame(res)