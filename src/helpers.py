import os
import json
import time
import random
import sqlite3
import numpy as np
from typing import Union, Dict, Tuple, List
from dataclasses import dataclass
from collections import defaultdict

import openai
import pandas as pd
import dotenv

dotenv.load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')

@dataclass
class LLMRequest:
    cell: Tuple[int, int]
    corrector_name: str
    prompt: Union[str, None]


@dataclass
class ErrorPositions:
    detected_cells: Dict[Tuple[int, int], Union[str, float, int]]
    table_shape: Tuple[int, int]
    labeled_cells: Dict[Tuple[int, int], Tuple[int, str]]

    def original_column_errors(self) -> Dict[int, List[Tuple[int, int]]]:
        column_errors = {j: [] for j in range(self.table_shape[1])}
        for (row, col), error_value in self.detected_cells.items():
            column_errors[col].append((row, col))
        return column_errors

    @property
    def updated_column_errors(self) -> Dict[int, List[Tuple[int, int]]]:
        column_errors = {j: [] for j in range(self.table_shape[1])}
        for (row, col), error_value in self.detected_cells.items():
            if (row, col) not in self.labeled_cells:
                column_errors[col].append((row, col))
        return column_errors

    def original_row_errors(self) -> Dict[int, List[Tuple[int, int]]]:
        row_errors = {i: [] for i in range(self.table_shape[0])}
        for (row, col), error_value in self.detected_cells.items():
            row_errors[row].append((row, col))
        return row_errors

    def updated_row_errors(self) -> Dict[int, List[Tuple[int, int]]]:
        row_errors = {i: [] for i in range(self.table_shape[0])}
        for (row, col), error_value in self.detected_cells.items():
            if (row, col) not in self.labeled_cells:
                row_errors[row].append((row, col))
        return row_errors


class Corrections:
    """
    Store correction suggestions provided by the correction models in correction_store. In _feature_generator_process
    it is guaranteed that all models return something for each error cell -- if there are no corrections made, that
    will be an empty list. If a correction or multiple corrections has/have been made, there will be a list of
    correction suggestions and feature vectors.
    """

    def __init__(self, model_names: List[str]):
        self.correction_store = {name: dict() for name in model_names}

    def flat_correction_store(self):
        flat_store = {}
        for model in self.correction_store:
            flat_store[model] = self.correction_store[model]
        return flat_store

    @property
    def available_corrections(self) -> List[str]:
        return list(self.correction_store.keys())

    def features(self) -> List[str]:
        """Return a list describing the features the Corrections come from."""
        return list(self.correction_store.keys())

    def get(self, model_name: str) -> Dict:
        """
        For each error-cell, there will be a list of {corrections_suggestion: probability} returned here. If there is no
        correction made for that cell, the list will be empty.
        """
        return self.correction_store[model_name]

    def assemble_pair_features(self) -> Dict[Tuple[int, int], Dict[str, List[float]]]:
        """Return features."""
        flat_corrections = self.flat_correction_store()
        pair_features = defaultdict(dict)
        for mi, model in enumerate(flat_corrections):
            for cell in flat_corrections[model]:
                for correction, pr in flat_corrections[model][cell].items():
                    # interessanter gedanke:
                    # if df_dirty.iloc[cell] == missing_value_token and model == 'llm_value':
                    #   pass
                    if correction not in pair_features[cell]:
                        features = list(flat_corrections.keys())
                        pair_features[cell][correction] = np.zeros(len(features))
                    pair_features[cell][correction][mi] = pr
        return pair_features

    def et_valid_corrections_made(self, corrected_cells: Dict[Tuple[int, int], str], column: int) -> int:
        """
        Per column, return how often the corrector leveaging error transformations mentioned the ground truth
        in its correction suggestions. The result is used to determine if ET models are useful to clean the column
        Depending on the outcome, the inferred_features are discarded.
        """
        if 'llm_correction' not in self.available_corrections or len(corrected_cells) == 0:
            return 0
        
        ground_truth_mentioned = 0
        for error_cell, correction in corrected_cells.items():
            if error_cell[1] == column:
                correction_suggestions = self.correction_store['llm_correction'].get(error_cell, {})
                if correction in list(correction_suggestions.keys()):
                    ground_truth_mentioned += 1
        return ground_truth_mentioned


def connect_to_cache() -> sqlite3.Connection:
    """
    Connect to the cache for LLM prompts.
    @return: a connection to the sqlite3 cache.
    """
    conn = sqlite3.connect('cache.db')
    return conn


def fetch_cached_llm(dataset: str,
                     error_cell: Tuple[int,int],
                     prompt: Union[str, None],
                     correction_model_name: str,
                     error_fraction: Union[None, int] = None,
                     version: Union[None, int] = None,
                     error_class: Union[None, str] = None,
                     llm_name: str = "gpt-3.5-turbo") -> Tuple[dict, dict, dict]:
    """
    Sending requests to LLMs is expensive (time & money). We use caching to mitigate that cost. As primary key for
    a correction serves (dataset_name, error_cell, version, correction_model_name). This is imperfect, but a reasonable
    approximation: Since the prompt-generation itself as well as its dependencies are non-deterministic, the prompt
    cannot serve as part of the primary key.

    If no correction is available in the cache, a request to the LLM is sent.

    @param dataset: name of the dataset that is cleaned.
    @param error_cell: (row, column) position of the error.
    @param prompt: prompt that is sent to the LLM.
    @param correction_model_name: "llm_master" or "llm_correction".
    @param error_fraction: Fraction of errors in the dataset.
    @param version: Version of the dataset. See dataset.py for details.
    @param error_class: Class of the error, e.g. MCAR
    @param llm_name: Name of the LLM used to correct with.
    @return: correction_tokens, token_logprobs, and top_logprobs.
    """
    cache = fetch_cache(dataset, error_cell, correction_model_name, error_fraction, version, error_class, llm_name)
    if cache is not None:
        return cache[0], cache[1], cache[2]
    else:
        return fetch_llm(prompt, dataset, error_cell, correction_model_name, error_fraction, version, error_class, llm_name)


def fetch_cache(dataset: str,
                error_cell: Tuple[int,int],
                correction_model_name: str,
                error_fraction: Union[None, int] = None,
                version: Union[None, int] = None,
                error_class: Union[None, str] = None,
                llm_name: str = "gpt-3.5-turbo") -> Union[None, Tuple[dict, dict, dict]]:
    """
    @param dataset: name of the dataset that is cleaned.
    @param error_cell: (row, column) position of the error.
    @param correction_model_name: "llm_master" or "llm_correction".
    @param error_fraction: Fraction of errors in the dataset.
    @param version: Version of the dataset. See dataset.py for details.
    @param error_class: Class of the error, e.g. MCAR
    @return: correction_tokens, token_logprobs, and top_logprobs.
    """
    dataset_name = dataset

    conn = connect_to_cache()
    cursor = conn.cursor()
    query = """SELECT
                 correction_tokens,
                 token_logprobs,
                 top_logprobs
               FROM cache
               WHERE
                 dataset=?
                 AND row=?
                 AND column=?
                 AND correction_model=?
                 AND llm_name=?"""
    parameters = [dataset_name, error_cell[0], error_cell[1], correction_model_name, llm_name]
    # Add conditions for optional parameters
    if error_fraction is not None:
        query += " AND error_fraction=?"
        parameters.append(error_fraction)
    else:
        query += " AND error_fraction IS NULL"

    if version is not None:
        query += " AND version=?"
        parameters.append(version)
    else:
        query += " AND version IS NULL"

    if error_class is not None:
        query += " AND error_class=?"
        parameters.append(error_class)
    else:
        query += " AND error_class IS NULL"

    cursor.execute(query, tuple(parameters))
    result = cursor.fetchone()
    conn.close()
    if result is not None:
        return json.loads(result[0]), json.loads(result[1]), json.loads(result[2])  # access the correction
    return None


def fetch_llm(prompt: Union[str, None],
              dataset: str,
              error_cell: Tuple[int, int],
              correction_model_name: str,
              error_fraction: Union[None, int] = None,
              version: Union[None, int] = None,
              error_class: Union[None, str] = None,
              llm_name: str = "gpt-3.5-turbo"
              ) -> Tuple[dict, dict, dict]:
    """
    Send request to openai to get a prompt resolved. Write result into cache.
    """
    if prompt is None:
        return {}, {}, {}

    retries = 0
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=llm_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                }],
                logprobs=True,
                top_logprobs=3
            )

            choices = response['choices'][0]
            correction_tokens = [y['token'] for y in choices['logprobs']['content']]
            token_logprobs = [y['logprob'] for y in choices['logprobs']['content']]
            top_logprobs = [{p['token']: p['logprob'] for p in position['top_logprobs']} for position in choices['logprobs']['content']]
            break
        except openai.error.RateLimitError as e:
            if retries > 5:
                print('Exceeded maximum number of retries. Skipping correction. The OpenAI API appears unreachable:')
                print(e)
                return {}, {}, {}
            delay = (2 ** retries) + random.random()
            print(f"Rate limit exceeded, retrying in {delay} seconds.")
            time.sleep(delay)
            retries += 1
        except openai.error.AuthenticationError:
            print(f'Tried sending {correction_model_name} prompt to OpenAI to correct cell {error_cell}.')
            print('However, there is no authentication provided in the .env file. Returning no corrections.')
            return {}, {}, {}

    row, column = error_cell
    conn = connect_to_cache()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO cache
               (dataset, row, column, correction_model, correction_tokens, token_logprobs, top_logprobs, error_fraction, version, error_class, llm_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (dataset, row, column, correction_model_name, json.dumps(correction_tokens), json.dumps(token_logprobs), json.dumps(top_logprobs), error_fraction, version, error_class, llm_name)
    )
    conn.commit()
    conn.close()
    return correction_tokens, token_logprobs, top_logprobs


def construct_llm_corrections(dictionaries: List[Dict[str, float]], current_sentence: str = '',
                        current_probability: float = 0) -> List[Dict[str, float]]:
    """
    Construct all possible sentences from a OpenAI davinci-003 API response. Returns a list of {correction: log_pr}.
    @param dictionaries:
    @param current_sentence:
    @param current_probability:
    @return:
    """
    if not dictionaries:
        # Base case: reached the end of the list, return the constructed correction and its probability as a dictionary
        return [{'correction': current_sentence, 'logprob': current_probability}]
    else:
        # Get the first dictionary in the list
        current_dict = dictionaries[0]
        result = []

        for token, probability in current_dict.items():
            # Recursively call the function for the remaining dictionaries
            sentences = construct_llm_corrections(dictionaries[1:], current_sentence + token,
                                            current_probability + probability)
            result.extend(sentences)

        return result


def llm_response_to_corrections(correction_tokens: dict, token_logprobs: dict, top_logprobs: dict) -> Dict[str, float]:
    # if len(correction_tokens) <= 7:
    #     corrections = construct_llm_corrections(top_logprobs)
    #     # filter out all corrections with pr < 1% <=> logprob > -4.60517.
    #     top_corrections = {c['correction']: np.exp(c['logprob']) for c in corrections if c['logprob'] > -4.60517}
    #     return top_corrections
    correction = ''.join(correction_tokens)
    correction = correction.replace('<MV>', '')  # parse missing value
    if correction.strip() in ['NULL', '<NULL>', 'null', '<null>']:
        return {}
    return {correction: np.exp(sum(token_logprobs))}


def error_free_row_to_prompt(df: pd.DataFrame, row: int, column: int) -> Tuple[str, str]:
    """
    Turn an error-free dataframe-row into a string, and replace the error-column with an <Error> token.
    Return a tuple of (stringified_row, correction). Be mindful that correction is only the correct value if
    the row does not contain an error to begin with.
    """
    if len(df.shape) == 1:  # final row, a series
        correction = ''
        values = df.values
    else:  # dataframe
        correction = df.iloc[row, column]
        values = df.iloc[row, :].values
    row_values = [f"{x}," if i != column else "<Error>," for i, x in enumerate(values)]
    assembled_row_values = ''.join(row_values)[:-1]
    return assembled_row_values, correction


def calc_chunksize(n_tasks, n_cores) -> Union[int, None]:
    """
    Multiprocessing requires the specification of how many tasks are passed to each worker at a time, called
    chunksize. In Mimir, we can assume that each task element takes roughly the same amount of time to finish.
    We call this a Dense Scenario, which makes it reasonable to pass a lot of tasks to the workers at a time,
    because retrieving new tasks from the task queue is expensive relative to how long it takes to carry out one
    task.
    See https://stackoverflow.com/a/54032744 and https://stackoverflow.com/a/43817408 for more info.
    
    Returns an integer that is my best effort to find a good chunksize, or None if it is not beneficial to do
    multiprocessing at all and the computation terminates faster without it.
    """
    pass