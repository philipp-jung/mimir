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

    return {'cell': cell, 'corrector': 'fd', 'correction_dict': highest_conditional_probabilities}

def generate_llm_correction_features(cell: Tuple[int, int],
                                     dataset_name: str,
                                     error_fraction,
                                     version,
                                     error_class,
                                     llm_name: str):

    cache = helpers.fetch_cache(dataset_name, cell, 'llm_correction', error_fraction, version, error_class, llm_name)

    if cache is None:
        return {'cell': cell, 'corrector': 'llm_correction', 'correction_dict': {}}

    correction, token_logprobs, top_logprobs = cache
    correction_dict = helpers.llm_response_to_corrections(correction, token_logprobs, top_logprobs)
    return {'cell': cell, 'corrector': 'llm_correction', 'correction_dict': correction_dict}


def generate_llm_master_features(cell: Tuple[int, int],
                                 dataset_name: str,
                                 error_fraction,
                                 version,
                                 error_class,
                                 llm_name: str):

    cache = helpers.fetch_cache(dataset_name, cell, 'llm_master', error_fraction, version, error_class, llm_name)

    if cache is None:
        return {'cell': cell, 'corrector': 'llm_master', 'correction_dict': {}}

    correction, token_logprobs, top_logprobs = cache
    correction_dict = helpers.llm_response_to_corrections(correction, token_logprobs, top_logprobs)
    return {'cell': cell, 'corrector': 'llm_master', 'correction_dict': correction_dict}


def generate_datawig_features(cell: Tuple[int, int], se_probas: pd.Series, error_value):
    se_subset = se_probas[se_probas >= 0.001]  # cut off corrections with .1% > pr and drop corrections with pr == np.nan
    se_subset = se_subset[se_subset.index != error_value]  # don't suggest the error as a correction

    return {'cell': cell, 'corrector': 'auto_instance', 'correction_dict': se_subset.to_dict()}
