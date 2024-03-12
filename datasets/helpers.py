import numpy as np
import random
import pandas as pd

def validate_export(dataset_name: str, df_dirty: pd.DataFrame, df_clean: pd.DataFrame, error_fraction: float):
    """
    Exporting the OpenML datasets can be tricky to do without a bug, which
    is why I test the exported datasets here.
    """
    mask = (df_dirty != df_clean)
    n_errors = mask.sum().sum()
    
    n_rows, n_cols = df_clean.shape

    # numer of errors I expect for imputer_simple_mcar.
    expected_one_col = round(n_rows * error_fraction)

    # and the same for simple_mcar - calculation here looks odd, but is reasonable.
    fractions = [error_fraction / n_cols for _ in range(n_cols)]
    expected_whole_df = sum([round(n_rows * f) for f in fractions])

    if n_errors != expected_one_col and n_errors != expected_whole_df:
        print(f'Dataset {dataset_name} contains {n_errors}, expected {expected_one_col} or {expected_whole_df}.')


def mcar_column(se: pd.Series, fraction: float) -> list:
    """
    Randomly insert missing values into a pandas Series. See docs on
    simple_mcar for more information.

    Returns a copy of se.
    """
    if fraction > 1:
        raise ValueError("Cannot turn more than 100% of the values into errors.")

    n_rows = se.shape[0]
    target_corruptions = round(n_rows * fraction)
    error_positions = random.sample([x for x in range(n_rows)], k=target_corruptions)
    return error_positions


def mar_column(df: pd.DataFrame, se: pd.Series, fraction: float, column: int) -> list:
    n_rows = se.shape[0]
    n_values_to_discard = int(n_rows * fraction)
    perc_lower_start = np.random.randint(0, n_rows - n_values_to_discard)
    perc_idx = range(perc_lower_start, perc_lower_start + n_values_to_discard)
    depends_on_col = np.random.choice(list(set(list(range(df.shape[1]))) - {column}))
    # pick a random percentile of values in other column
    error_positions = list(df.iloc[:, depends_on_col].sort_values().iloc[perc_idx].index)
    return error_positions
        
def mnar_column(se: pd.Series, fraction: float) -> list:
    n_rows = se.shape[0]
    n_values_to_discard = int(n_rows * fraction)
    perc_lower_start = np.random.randint(0, n_rows - n_values_to_discard)
    perc_idx = range(perc_lower_start, perc_lower_start + n_values_to_discard)

    # pick a random percentile of values in this column
    error_positions = list(se.sort_values().iloc[perc_idx].index)
    return error_positions


def corrupt_column(se: pd.Series, error_positions: list, error_token=None, error_token_int=-9999999, error_token_obj='') -> pd.Series:
    """
    Take a series and missing value's positions and insert missing values,
    according to the series' dtype.

    Returns a corrupted copy of the series.
    """
    se_corrupt = se.copy()
    column_dtype = se_corrupt.dtype

    for x in error_positions:
        if str(column_dtype).startswith('int'):
            se_corrupt.iat[x] = error_token_int
        elif column_dtype in ['object', 'str', 'string']:
            se_corrupt.iat[x] = error_token_obj
        else:
            se_corrupt.iat[x] = error_token
    return se_corrupt


def apply_corruption(mechanism: str, df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """
    Randomly insert missing values into a dataframe. Note that specifying the
    three different error_tokens preserves dtypes in the corrupted dataframe,
    as otherwise pandas casts columns to other dtypes.

    Copies df, so that the clean dataframe you pass doesn't get corrupted
    in place.
    """
    df_dirty = df.copy()
    _, n_cols = df.shape


    for col in range(n_cols):
        fraction_col = fraction / n_cols
        se = df_dirty.iloc[:, col]
        if mechanism == 'simple_mcar':
            error_positions = mcar_column(se, fraction_col)
        elif mechanism == 'simple_mar':
            error_positions = mar_column(df_dirty, se, fraction_col, col)
        elif mechanism == 'simple_mnar':
            error_positions = mnar_column(se, fraction_col)
        else:
            raise ValueError(f'Unknown missingness mechanism {mechanism}.')
        se_dirty = corrupt_column(se, error_positions)
        df_dirty.iloc[:, col] = se_dirty

    return df_dirty


def apply_imputer_corruption(mechanism: str, df: pd.DataFrame, se: pd.Series, fraction: float) -> pd.Series:
    """
    Insert missing values into a column `se`. 
    """
    col = df.columns.get_loc(se.name)

    if mechanism == 'imputer_simple_mcar':
        error_positions = mcar_column(se, fraction)
    elif mechanism == 'imputer_simple_mar':
        error_positions = mar_column(df, se, fraction, col)
    elif mechanism == 'imputer_simple_mnar':
        error_positions = mnar_column(se, fraction)
    else:
        raise ValueError(f'Unknown missingness mechanism {mechanism}.')
    se_dirty = corrupt_column(se, error_positions)

    return se_dirty
