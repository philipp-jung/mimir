import os
import json
import yaml
import time
from tqdm import tqdm
import argparse
import itertools
from pathlib import Path

import dotenv
import dataset
import correction

def run_mirmir(c: dict):
    """
    Runs mirmir with a configuration as specified in c.
    @param c: a dictionary that contains all parameters the experiment requires.
    @return: A dictionary containing measurements and the configuration of the measurement.
    """
    version = c["run"] + 1  # RENUVER dataset-versions are 1-indexed
    start_time = time.time()

    try:
        data = dataset.Dataset(c['dataset'], c['error_fraction'], version, c['error_class'], c["n_rows"], c.get("n_cols"))
        data.detected_cells = data.get_errors_dictionary(c['detection_mode'])

        app = correction.Cleaning(
            c["labeling_budget"],
            c["classification_model"],
            c["clean_with_user_input"],
            c["feature_generators"],
            c["vicinity_orders"],
            c["vicinity_feature_generator"],
            c["auto_instance_cache_model"],
            c["n_best_pdeps"],
            c["training_time_limit"],
            c["synth_tuples"],
            c["synth_cleaning_threshold"],
            c["test_synth_data_direction"],
            c["pdep_features"],
            c["gpdep_threshold"],
            c['fd_feature'],
            False,
            c['llm_name_corrfm'],
            c['sampling_technique'],
            c['labeling_error_pct']
        )
        app.VERBOSE = True
        seed = None
        correction_dictionary = app.run(data, seed, synchronous=False)
        end_time = time.time()
        p, r, f = data.get_data_cleaning_evaluation(correction_dictionary)[-3:]
        return {
                "status": 1,
                "result": {
                    "precision": p, "recall": r, "f1": f, "runtime": end_time - start_time
                    },
                "config": c,
                }
    except Exception as e:
        return {"status": 0,
                "result": str(e),
                "config": c}


def combine_configs(ranges: dict, config: dict, runs: int):
    """
    Calculate all possible configurations from combining ranges with the
    static config.
    ranges: dict of lists, where the key identifier the config-parameter,
    and the list contains possible values for that parameter.
    config: dict containing all parameters needed to run Mimir. Keys
    contained in ranges get overwritten.
    runs: integer indicating how often a measurement should be repeated with
    one combination.
    @return: list of config-dicts
    """
    config_combinations = []
    ranges = {**ranges, "run": list(range(runs))}

    range_combinations = itertools.product(*list(ranges.values()))
    for c in range_combinations:
        combination = {}
        for i, key_range in enumerate(ranges.keys()):
            combination[key_range] = c[i]
        config_combinations.append(combination)

    configs = [
        {**config, **range_config} for range_config in config_combinations
    ]

    return configs

def extract_experiments(config:dict) -> set:
    """
    Ensures that every experiment defined in the configuration has corresponding pairs of
    experiment-config.
    Returns a set of experiment names.
    """
    ranges = set()
    configs = set()

    for key in config:
        if key.startswith('ranges_'):
            range = key.split('ranges_')[1]
            if range in ranges:
                raise ValueError(f'Defined range {range} twice.')
            ranges.add(range)
        elif key.startswith('config_'):
            config = key.split('config_')[1]
            if config in configs:
                raise ValueError(f'Defined config {config} twice.')
            configs.add(range)

    orphans = ranges ^ configs
    if len(orphans) > 0:
        raise ValueError(f'The following experiments has either missing range or config definition: {orphans}')
    
    return ranges

def load_dedicated_experiments(saved_config: str):
    """
    Loads a saved configuration yaml-file. All first-level nodes in that yaml-file that begin with
    'ranges_$SOMETHING' are expected to have a corresponding 'config_$SOMETHING'.
    """
    config_path = Path('../infrastructure/saved-experiment-configs') / f'{saved_config}.yaml'
    with open(config_path, 'rt') as f:
        config = yaml.safe_load(f)

    experiments = extract_experiments(config)
    configs = []

    for experiment in experiments:
        combined_config = combine_configs(ranges=config[f'ranges_{experiment}'],
                                          config=config[f'config_{experiment}'],
                                          runs=config['runs'])
        configs.extend(combined_config)

    print(f'Successfully loaded experiment configuration from {saved_config}.')
    return configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start experiment either on a k8s cluster, or on a dedicated machine.')
    parser.add_argument('--saved-config', type=str, help='Name of the saved experiment configuration.', default=None)
    parser.add_argument('--dedicated', action='store_true', help='Whether or not to run a dedicated experiment', default=False)

    args = parser.parse_args()

    # Load OpenAI API-key
    dotenv.load_dotenv()

    results_path = Path('/measurements/')
    results_path.mkdir(parents=True, exist_ok=True)

    if args.dedicated:  # dedicated mode
        if args.saved_config is None:
            raise ValueError('No saved config specified to load experiment from.')

        configs = load_dedicated_experiments(args.saved_config)
        print(f'Start measurement for experiment {args.saved_config}.')

        for i, config in enumerate(tqdm(configs)):
            experiment_id = f'{args.saved_config}-{i}'
            experiment_id = experiment_id.replace('_', '-')
            result = run_mirmir(config)
            with open(results_path / f"{experiment_id}.json", 'wt') as f:
                json.dump(result, f)
    else:  # k8s mode
        if os.getenv('CONFIG') is None or os.getenv('EXPERIMENT_ID') is None:
            raise ValueError('config or experiment_id ENV VARs are not set')
        config = json.loads(os.getenv('CONFIG'))
        experiment_id = os.getenv('EXPERIMENT_ID')
        result = run_mirmir(config)
        with open(results_path / f"{experiment_id}.json", 'wt') as f:
            json.dump(result, f)
