import random

from typing import Tuple, List, Dict

import numpy as np
import pandas as pd

import helpers
from pdep import PdepTuple

def generate_pdep_features(cell: Tuple[int, int],
                           counts_dict: Dict,
                           gpdeps: Dict[Tuple[int, ...], PdepTuple],
                           row_values: Tuple,
                           feature: str) -> Dict:

    if gpdeps is None:
        return {}

    rhs_col = cell[1]
    results_list = []

    for lhs_cols, pdep_tuple in gpdeps.items():
        lhs_values = tuple([row_values[x] for x in lhs_cols])

        if rhs_col not in lhs_cols and lhs_values in counts_dict[lhs_cols][rhs_col]:
            for rhs_val in counts_dict[lhs_cols][rhs_col][lhs_values].index:

                results_list.append(
                    {"correction": rhs_val,
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

    if cell == (2,6):
        a =1
    return {'cell': cell, 'corrector': 'fd', 'correction_dict': highest_conditional_probabilities}

def generate_llm_correction_features(cell: Tuple[int, int],
                                     old_value: str,
                                     error_correction_pairs: List[Tuple[str, str]],
                                     dataset_name: str,
                                     error_fraction,
                                     version,
                                     error_class):
    prompt = "You are a data cleaning machine that detects patterns to return a correction. If you do "\
                "not find a correction, you return the token <NULL>. You always follow the example.\n---\n"

    n_pairs = min(10, len(error_correction_pairs))

    for (error, correction) in random.sample(error_correction_pairs, n_pairs):
        prompt = prompt + f"error:{error}" + '\n' + f"correction:{correction}" + '\n'
    prompt = prompt + f"error:{old_value}" + '\n' + "correction:"

    correction, token_logprobs, top_logprobs = helpers.fetch_cached_llm(dataset_name, cell, prompt, 'llm_correction', error_fraction, version, error_class)
    correction_dict = helpers.llm_response_to_corrections(correction, token_logprobs, top_logprobs)

    return {'cell': cell, 'corrector': 'llm_correction', 'correction_dict': correction_dict}


def generate_llm_master_features(cell: Tuple[int, int],
                                 df_error_free_subset: pd.DataFrame,
                                 df_row_with_error: pd.DataFrame,
                                 dataset_name: str,
                                 error_fraction,
                                 version,
                                 error_class):
    prompt = "You are a data cleaning machine that returns a correction, which is a single expression. If "\
                "you do not find a correction, return the token <NULL>. You always follow the example.\n---\n"
    n_pairs = min(5, len(df_error_free_subset))
    rows = random.sample(range(len(df_error_free_subset)), n_pairs)
    for row in rows:
        row_as_string, correction = helpers.error_free_row_to_prompt(df_error_free_subset, row, cell[1])
        prompt = prompt + row_as_string + '\n' + f'correction:{correction}' + '\n'
    final_row_as_string, _ = helpers.error_free_row_to_prompt(df_row_with_error, 0, cell[1])
    prompt = prompt + final_row_as_string + '\n' + 'correction:'

    correction, token_logprobs, top_logprobs = helpers.fetch_cached_llm(dataset_name, cell, prompt, 'llm_master', error_fraction, version, error_class)
    correction_dict = helpers.llm_response_to_corrections(correction, token_logprobs, top_logprobs)

    return {'cell': cell, 'corrector': 'llm_master', 'correction_dict': correction_dict}


def generate_datawig_features(cell: Tuple[int, int], se_probas: pd.Series, error_value):
    prob_d = {key: se_probas.to_dict()[key] for key in se_probas.to_dict()}

    # sometimes autogluon returns np.nan, which the ensemble classifier downstream chokes up on.
    result = {correction: 0.0 if np.isnan(pr) else pr for correction, pr in prob_d.items() if correction != error_value}

    # Sometimes training an auto_instance model fails for a column, while it succeeds on other columns.
    # If training failed for a column, imputer_corrections will be an empty list. Which will lead
    # to one less feature being added to values in that column. Which in turn is bad news in the ensemble.
    # To prevent this, I have imputer_corrections fall back to {}, which has length 1 and will create
    # a feature.
    if len(result) == 0:
       result = {}

    return {'cell': cell, 'corrector': 'auto_instance', 'correction_dict': result}
