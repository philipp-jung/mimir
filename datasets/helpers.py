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

def simple_mcar(df: pd.DataFrame, fraction: float, error_token=None, error_token_int=-9999999, error_token_obj=''):
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
        se_dirty = simple_mcar_column(df_dirty.iloc[:, col], fraction_col, error_token, error_token_int, error_token_obj)
        df_dirty.iloc[:, col] = se_dirty

    return df_dirty


def simple_mcar_column(se: pd.Series, fraction: float, error_token=None, error_token_int=-9999999, error_token_obj=''):
    """
    Randomly insert missing values into a pandas Series. See docs on
    simple_mcar for more information.

    Copies the passed Series `se`, and returns it.
    """
    if fraction > 1:
        raise ValueError("Cannot turn more than 100% of the values into errors.")

    se_corrupt = se.copy()
    column_dtype = se_corrupt.dtype

    n_rows = se.shape[0]
    target_corruptions = round(n_rows * fraction)
    error_positions = random.sample([x for x in range(n_rows)], k=target_corruptions)
    for x in error_positions:
        if str(column_dtype).startswith('int'):
            se_corrupt.iat[x] = error_token_int
        elif column_dtype in ['object', 'str', 'string']:
            se_corrupt.iat[x] = error_token_obj
        else:
            se_corrupt.iat[x] = error_token
    return se_corrupt
