import os
import platform
import subprocess
import pandas as pd
from typing import Tuple, List, Dict, Union
from itertools import combinations
from collections import namedtuple, defaultdict

PdepTuple = namedtuple("PdepTuple", ["pdep", "gpdep", "epdep", "norm_gpdep"])
FDTuple = namedtuple("FDTuple", ["lhs", "rhs"])


def calculate_frequency(df: pd.DataFrame, col: int):
    """
    Calculates the frequency of a value to occur in colum
    col on dataframe df.
    """
    counts = df.iloc[:, col].value_counts()
    return counts.to_dict()


def mine_fd_counts(
    df: pd.DataFrame,
    row_errors: Dict[int, Dict[Tuple, str]],
    fds: List[FDTuple],
) -> Tuple[dict, dict]:
    """
    Calculates a dictionary d that contains the absolute counts of how
    often values in the lhs occur with values in the rhs in the table df,
    for a given set of fds.
    @param df: dataset that is to be cleaned.
    @param row_errors: detected error positions, indexed by row, updated by user input.
    @param fds: known functional dependencies in the dataset.
    @return:
    """
    lhs_values = defaultdict(lambda: defaultdict(dict))
    d = {fd.lhs: {} for fd in fds}
    for fd in fds:
        d[fd.lhs][fd.rhs] = {}

    for row in df.itertuples(index=True):
        i_row, row = row[0], row[1:]
        detected_cells = row_errors[i_row]
        for fd in fds:
            lhs_vals = tuple(row[lhs_col] for lhs_col in fd.lhs)
            rhs_val = row[fd.rhs]

            lhs_contains_error = any(
                [(i_row, lhs_col) in detected_cells for lhs_col in fd.lhs]
            )
            rhs_contains_error = (i_row, fd.rhs) in detected_cells
            if not lhs_contains_error and not rhs_contains_error:
                # update conditional counts
                if lhs_values[fd.lhs][fd.rhs].get(lhs_vals) is None:
                    lhs_values[fd.lhs][fd.rhs][lhs_vals] = 1
                else:
                    lhs_values[fd.lhs][fd.rhs][lhs_vals] += 1

                if d[fd.lhs][fd.rhs].get(lhs_vals) is None:
                    d[fd.lhs][fd.rhs][lhs_vals] = {}
                if d[fd.lhs][fd.rhs][lhs_vals].get(rhs_val) is None:
                    d[fd.lhs][fd.rhs][lhs_vals][rhs_val] = 1.0
                else:
                    d[fd.lhs][fd.rhs][lhs_vals][rhs_val] += 1.0
    return d, lhs_values


def mine_all_counts(
        df: pd.DataFrame,
        detected_cells: Dict[Tuple, str],
        order=1,
        ignore_sign="<<<IGNORE_THIS_VALUE>>>",
) -> Tuple[dict, dict]:
    """
    Calculates a dictionary d that contains the absolute counts of how
    often values in the lhs occur with values in the rhs in the table df.

    The dictionary has the structure
    d[lhs_columns][rhs_column][lhs_values][rhs_value],
    where lhs_columns is a tuple of one or more lhs_columns, and
    rhs_column is the columns whose values are determined by lhs_columns.

    Pass an `order` argument to indicate how many columns in the lhs should
    be investigated. If order=1, only unary relationships are taken into account,
    if order=2, only binary relationships are taken into account, and so on.

    Detected cells is a list of dictionaries whose keys are the coordinates of
    errors in the table. It is used to exclude errors from contributing to the
    value counts. `ignore_sign` is a string that indicates that a cell is erronous,
    too. Cells containing this value are ignored, too.
    """
    i_cols = list(range(df.shape[1]))

    lhs_values = defaultdict(lambda: defaultdict(dict))
    d = {comb: {cc: {} for cc in i_cols} for comb in combinations(i_cols, order)}

    for row in df.itertuples(index=True):
        i_row, row = row[0], row[1:]
        for lhs_cols in combinations(i_cols, order):
            lhs_vals = tuple(row[lhs_col] for lhs_col in lhs_cols)

            lhs_contains_error = any(
                [(i_row, lhs_col) in detected_cells for lhs_col in lhs_cols]
            )
            if ignore_sign not in lhs_vals and not lhs_contains_error:
                # update conditional counts
                for rhs_col in i_cols:
                    if rhs_col not in lhs_cols:
                        rhs_contains_error = (i_row, rhs_col) in detected_cells
                        if ignore_sign not in lhs_vals and not rhs_contains_error:
                            if lhs_values[lhs_cols][rhs_col].get(lhs_vals) is None:
                                lhs_values[lhs_cols][rhs_col][lhs_vals] = 1
                            else:
                                lhs_values[lhs_cols][rhs_col][lhs_vals] += 1

                            rhs_val = row[rhs_col]

                            if d[lhs_cols][rhs_col].get(lhs_vals) is None:
                                d[lhs_cols][rhs_col][lhs_vals] = {}
                            if d[lhs_cols][rhs_col][lhs_vals].get(rhs_val) is None:
                                d[lhs_cols][rhs_col][lhs_vals][rhs_val] = 1.0
                            else:
                                d[lhs_cols][rhs_col][lhs_vals][rhs_val] += 1.0

    return d, lhs_values


def update_vicinity_model(
    counts_dict: dict, lhs_values: dict, clean_sampled_tuple: list, error_positions: dict, row: int
) -> None:
    """
    Update the data structures counts_dict and lhs_values_frequencies with user inputs.
    They are required to calculate gpdeps.
    """
    for lhs_cols in counts_dict:
        lhs_vals = tuple(clean_sampled_tuple[lhs_col] for lhs_col in lhs_cols)
        lhs_contains_error = any(
            [(row, lhs_col) in error_positions for lhs_col in lhs_cols]
        )

        for rhs_col in range(len(clean_sampled_tuple)):  # todo: Only update if there was an error before.
            rhs_contains_error = (row, rhs_col) in error_positions
            any_cell_contains_error = any([lhs_contains_error, rhs_contains_error])
            if rhs_col not in lhs_cols:
                if any_cell_contains_error:  # update only if previously error existed
                    if counts_dict[lhs_cols][rhs_col].get(lhs_vals) is None:
                        counts_dict[lhs_cols][rhs_col][lhs_vals] = {}

                    # Update counts of values in the LHS
                    if lhs_values[lhs_cols][rhs_col].get(lhs_vals) is None:
                        lhs_values[lhs_cols][rhs_col][lhs_vals] = 1
                    else:
                        lhs_values[lhs_cols][rhs_col][lhs_vals] += 1

                    rhs_val = clean_sampled_tuple[rhs_col]
                    if counts_dict[lhs_cols][rhs_col].get(lhs_vals) is None:
                        counts_dict[lhs_cols][rhs_col][lhs_vals] = {}
                    if counts_dict[lhs_cols][rhs_col][lhs_vals].get(rhs_val) is None:
                        counts_dict[lhs_cols][rhs_col][lhs_vals][rhs_val] = 1.0
                    else:
                        counts_dict[lhs_cols][rhs_col][lhs_vals][rhs_val] += 1.0


def expected_pdep(
    n_rows: int,
    counts_dict: dict,
    A: Tuple[int, ...],
    B: int,
) -> Union[float, None]:
    pdep_B = pdep_0(n_rows, counts_dict, B, A)

    if pdep_B is None:
        return None

    if pdep_B == 1:  # division by 0
        return None

    if n_rows == 1:  # division by 0
        return 0

    n_distinct_values_A = len(counts_dict[A][B])
    return pdep_B + (n_distinct_values_A - 1) / (n_rows - 1) * (1 - pdep_B)


def error_corrected_row_count(
    n_rows: int,
    row_errors: Dict[int, Tuple[int, int]],
    A: Tuple[int, ...],
    B: int
) -> int:
    """
    Calculate the number of rows N that do not contain an error in the LHS or RHS.
    @param n_rows: Number of rows in the table including all errors.
    @param A: LHS column
    @param B: RHS column
    @param row_errors: Dictionary with error positions.
    @return: Number of rows without an error
    """
    relevant_cols = list(A) + [B]
    excluded_rows = set()
    for row in row_errors:
        if len(row_errors[row]) == 0:
            pass
        else:
            for _, col in row_errors[row]:
                if col in relevant_cols:
                    excluded_rows.add(row)
    return n_rows - len(excluded_rows)


def pdep_0(
        n_rows: int,
        counts_dict: dict,
        B: int,
        A: Tuple[int, ...]
) -> Union[float, None]:
    """
    Calculate pdep(B), that is the probability that two randomly selected records from B will have the same value.
    Note that in order to calculate pdep(B), you may want to limit the records from B to error-free records in
    context of a left hand side A, for example when calculating pdep(B) to calculate gpdep(A,B).
    """
    if n_rows == 0:  # no tuple exists without an error in lhs and rhs
        return None

    # calculate the frequency of each RHS value, given that the values in all columns covered by LHS and RHS contain
    # no error.
    rhs_abs_frequencies = defaultdict(int)
    for lhs_vals in counts_dict[A][B]:
        for rhs_val in counts_dict[A][B][lhs_vals]:
            rhs_abs_frequencies[rhs_val] += counts_dict[A][B][lhs_vals][rhs_val]

    sum_components = []
    for rhs_frequency in rhs_abs_frequencies.values():
        sum_components.append(rhs_frequency**2)
    return sum(sum_components) / n_rows**2


def pdep(
    n_rows: int,
    counts_dict: dict,
    lhs_values_frequencies: dict,
    A: Tuple[int, ...],
    B: int,
) -> Union[float, None]:
    """
    Calculates the probabilistic dependence pdep(A,B) between a left hand side A,
    which consists of one or more attributes, and a right hand side B,
    which consists of one attribute.

    """
    if n_rows == 0:  # no tuple exists without an error in lhs and rhs
        return None

    sum_components = []

    for lhs_val, rhs_dict in counts_dict[A][B].items():  # lhs_val same as A_i
        lhs_counts = lhs_values_frequencies[A][B][lhs_val]  # same as a_i
        for rhs_val, rhs_counts in rhs_dict.items():  # rhs_counts same as n_ij
            sum_components.append(rhs_counts**2 / lhs_counts)
    return sum(sum_components) / n_rows


def gpdep(
    n_rows: int,
    counts_dict: dict,
    lhs_values_frequencies: dict,
    A: Tuple[int, ...],
    B: int,
) -> Union[PdepTuple, None]:
    """
    Calculates the *genuine* probabilistic dependence (gpdep) between
    a left hand side A, which consists of one or more attributes, and
    a right hand side B, which consists of exactly one attribute.
    """
    if B in A:  # pdep([A,..],A) = 1
        return None

    pdep_A_B = pdep(n_rows, counts_dict, lhs_values_frequencies, A, B)
    epdep_A_B = expected_pdep(n_rows, counts_dict, A, B)

    if pdep_A_B is not None and epdep_A_B is not None:
        gpdep_A_B = pdep_A_B - epdep_A_B
        return PdepTuple(pdep_A_B, gpdep_A_B, epdep_A_B, 0)
    return None


def vicinity_based_corrector_order_n(counts_dict, ed) -> Dict[str, Dict[str, float]]:
    """
    Use Baran's naive strategy to suggest corrections based on higher-order
    vicinity.

    Features generated by this function grow with (n-1)^2 / 2, where n is the
    number of columns of the dataframe. This corrector can be considered to be
    the naive approach to suggesting corrections from higher-order vicinity.

    Notes:
    ed: {   'column': column int,
            'old_value': old error value,
            'vicinity': row that contains the error, including the error
    }
    counts_dict: Dict[Dict[Dict[Dict]]] [lhs][rhs][lhs_value][rhs_value]

    """
    rhs_col = ed["column"]

    results_dictionary = {}
    for lhs_cols in list(counts_dict.keys()):
        if results_dictionary.get(lhs_cols) is None:
            results_dictionary[lhs_cols] = {}
        for lhs_vals in combinations(ed["vicinity"], len(lhs_cols)):
            if rhs_col not in lhs_cols and lhs_vals in counts_dict[lhs_cols][rhs_col]:
                sum_scores = sum(counts_dict[lhs_cols][rhs_col][lhs_vals].values())
                for rhs_val in counts_dict[lhs_cols][rhs_col][lhs_vals]:
                    pr = counts_dict[lhs_cols][rhs_col][lhs_vals][rhs_val] / sum_scores
                    results_dictionary[lhs_cols][rhs_val] = pr
    return results_dictionary


def fd_calc_gpdeps(
    counts_dict: dict, lhs_values_frequencies: dict, shape: Tuple[int, int], row_errors
) -> Dict[Tuple, Dict[int, PdepTuple]]:
    """
    Calculate all gpdeps in a given set of functional dependencies. The difference to calc_all_gpdeps
    is that the counts_dict in this version only contains some FDs, and not all possible FDs in some orders.
    """
    n_rows, n_cols = shape
    lhss = list(counts_dict.keys())

    gpdeps = {lhs: {} for lhs in lhss}
    for lhs in counts_dict:
        for rhs in counts_dict[lhs]:
            N = error_corrected_row_count(n_rows, row_errors, lhs, rhs)
            gpdeps[lhs][rhs] = gpdep(N, counts_dict, lhs_values_frequencies, lhs, rhs)
    return gpdeps


def calc_all_gpdeps(
    counts_dict: dict, lhs_values_frequencies: dict, shape: Tuple[int, int], row_errors, order: int
) -> Dict[Tuple, Dict[int, PdepTuple]]:
    """
    Calculate gpdeps in dataframe df for left hand side values of order `order`.
    """
    n_rows, n_cols = shape
    lhss = set([x for x in counts_dict[order].keys()])
    rhss = list(range(n_cols))

    gpdeps = {lhs: {} for lhs in lhss}
    for lhs in lhss:
        for rhs in rhss:
            N = error_corrected_row_count(n_rows, row_errors, lhs, rhs)
            gpdeps[lhs][rhs] = gpdep(N, counts_dict[order], lhs_values_frequencies[order], lhs, rhs)
    return gpdeps


def invert_and_sort_gpdeps(
    gpdeps: Dict[Tuple, Dict[int, PdepTuple]]
) -> Dict[int, Dict[Tuple, PdepTuple]]:
    """
    Invert the gpdeps dict and sort it. Results in a dict whose first key is
    the rhs, second key is the lhs, value of that is the gpdep score. The second
    key is sorted by the descending gpdep score.
    """
    inverse_gpdeps = {rhs: {} for rhs in list(gpdeps.items())[0][1]}

    for lhs in gpdeps:
        for rhs in gpdeps[lhs]:
            inverse_gpdeps[rhs][lhs] = gpdeps[lhs][rhs]

    for rhs in inverse_gpdeps:
        inverse_gpdeps[rhs] = {
            k: v
            for k, v in sorted(
                inverse_gpdeps[rhs].items(),
                key=lambda x: (x[1] is not None, x[1].gpdep if x[1] is not None else None),
                reverse=True,
            )
        }
    return inverse_gpdeps


def fd_based_corrector(
    inverse_gpdeps: Dict[int, Dict[Tuple, PdepTuple]],
    counts_dict: dict,
    ed: dict,
    feature: str = "pr"
) -> Dict:
    """
    Leverage exact FDs and gpdep to make cleaning suggestions.
    """
    rhs_col = ed["column"]
    gpdeps = inverse_gpdeps.get(rhs_col)

    if gpdeps is None:
        return {}

    results_list = []

    for lhs_cols, pdep_tuple in gpdeps.items():
        lhs_vals = tuple([ed["vicinity"][x] for x in lhs_cols])

        if rhs_col not in lhs_cols and lhs_vals in counts_dict[lhs_cols][rhs_col]:
            sum_scores = sum(counts_dict[lhs_cols][rhs_col][lhs_vals].values())
            for rhs_val in counts_dict[lhs_cols][rhs_col][lhs_vals]:
                pr = counts_dict[lhs_cols][rhs_col][lhs_vals][rhs_val] / sum_scores

                results_list.append(
                    {"correction": rhs_val,
                     "pr": pr,
                     "pdep": pdep_tuple.pdep if pdep_tuple is not None else 0,
                     "gpdep": pdep_tuple.gpdep if pdep_tuple is not None else 0,
                     "epdep": pdep_tuple.epdep if pdep_tuple is not None else 0,
                     "norm_gpdep": pdep_tuple.norm_gpdep if pdep_tuple is not None else 0}
                )

    # sort by gpdep for importance
    sorted_results = sorted(results_list, key=lambda x: x['norm_gpdep'], reverse=True)

    highest_conditional_probabilities = {}

    if feature == "norm_gpdep":
        # We can simply sum up the normalized gpdep score
        for d in sorted_results:
            old_pr = highest_conditional_probabilities.get(d["correction"], 0)
            highest_conditional_probabilities[d["correction"]] = old_pr + d[feature]
    else:
        # Having a sorted dict allows us to only return the highest conditional
        # probability per correction by iterating over all generated corrections
        # like this.
        for d in sorted_results:
            if highest_conditional_probabilities.get(d["correction"]) is None:
                highest_conditional_probabilities[d["correction"]] = d[feature]

    return highest_conditional_probabilities


def pdep_vicinity_based_corrector(
    inverse_sorted_gpdeps: Dict[int, Dict[tuple, PdepTuple]],
    counts_dict: dict,
    ed: dict,
    n_best_pdeps: int = 3,
    features_selection: tuple = ('pr', 'vote'),
    gpdep_threshold: float = 0.5
) -> Dict:
    """
    Leverage gpdep to avoid having correction suggestion feature columns
    grow in number at (n-1)^2 / 2 pace. Only take the `n_best_pdeps`
    highest-scoring dependencies to draw corrections from.
    """
    if len(features_selection) == 0:  # no features to generate
        return {}

    rhs_col = ed["column"]
    gpdeps = inverse_sorted_gpdeps[rhs_col]

    gpdeps_subset = {rhs: gpdeps[rhs] for i, rhs in enumerate(gpdeps) if i < n_best_pdeps}
    results_list = []

    for lhs_cols, pdep_tuple in gpdeps_subset.items():
        lhs_vals = tuple([ed["vicinity"][x] for x in lhs_cols])

        if rhs_col not in lhs_cols and lhs_vals in counts_dict[lhs_cols][rhs_col]:
            sum_scores = sum(counts_dict[lhs_cols][rhs_col][lhs_vals].values())
            for rhs_val in counts_dict[lhs_cols][rhs_col][lhs_vals]:
                pr = counts_dict[lhs_cols][rhs_col][lhs_vals][rhs_val] / sum_scores

                results_list.append(
                    {"correction": rhs_val,
                     "pr": pr,
                     "pdep": pdep_tuple.pdep if pdep_tuple is not None else 0,
                     "gpdep": pdep_tuple.gpdep if pdep_tuple is not None else 0}
                )

    sorted_results = sorted(results_list, key=lambda x: x["pr"], reverse=True)

    highest_conditional_probabilities = {}

    # Having a sorted dict allows us to only return the highest conditional
    # probability per correction by iterating over all generated corrections
    # like this.
    for d in sorted_results:
        if highest_conditional_probabilities.get(d["correction"]) is None:
            if d['gpdep'] > gpdep_threshold:
                highest_conditional_probabilities[d["correction"]] = d["pr"]

    return highest_conditional_probabilities


def cleanest_version(df_dirty: pd.DataFrame, user_input: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the cleanest version of the dataset we know without touching ground-truth.
    @param df_dirty: dirty data
    @param user_input: user input
    @return: cleanest version of the dataset we know without touching ground-truth
    """
    df_clean_iterative = user_input.combine_first(df_dirty)
    return df_clean_iterative


def mine_fds(df_clean_iterative: pd.DataFrame, df_ground_truth: pd.DataFrame) -> List[FDTuple]:
    """
    Mine functional dependencies using HyFD. The function calls a jar file called HyFdMimir-1.3.jar with the following
    parameters:
    @param df_clean_iterative: dirty data, enriched with user input. The cleanest version of the dataset we know without
    touching ground-truth.
    @param df_ground_truth: Ground truth, used by HyFDMimir to determine error positions.
    @return: a dictionary of functional dependencies. The keys are the left-hand-side of the dependency, the values are
    the right-hand-side of the dependency.
    """
    clean_path = "tmp/clean.csv"
    dirty_path = "tmp/dirty.csv"
    fd_path = "tmp/fds.txt"

    # Write dirty data to disk
    df_clean_iterative.to_csv(dirty_path, index=False, encoding="utf-8")
    df_ground_truth.to_csv(clean_path, index=False, encoding="utf-8")

    # Identify system and machine
    system = platform.system()
    machine = platform.machine()

    if machine == 'arm64' and system == 'Darwin':
        binary = "HyFDMimir-1.3-arm64-darwin.jar"
    elif machine == 'x86_64' and system == 'Linux':
        binary = "HyFDMimir-1.3-x86_64-linux.jar"
    else:
        raise ValueError('We have HyFDMimir precompiled for arm64 darwin and x86_64 linux systems. '
        'You will have to compile HyFDMimir for your system and architecture yourself.')

    # Execute HyFDMimir. Do not write output to stdout.
    try:
        subprocess.run(["java", "-jar", binary, dirty_path, clean_path, fd_path],
                       stdout=subprocess.DEVNULL)
    except FileNotFoundError:
        print("HyFDMimir-1.3.jar not found. Please compile it first, following the instructions in the README.")
        exit(1)

    # Read the output of HyFDMimir into memory.

    fds = []
    with open(fd_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            lhs, rhs = line.split("->")
            rhs = rhs.strip()
            if len(lhs) == 0:
                pass
            else:
                lhs = tuple(sorted([int(x) for x in lhs.split(",")]))
                rhs = int(rhs.strip())
                fds.append(FDTuple(lhs, rhs))

    # Remove temporary files.
    os.remove(clean_path)
    os.remove(dirty_path)
    os.remove(fd_path)

    return fds
