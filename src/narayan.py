import json
from pathlib import Path

import dataset
import correctors

def run_narayan(c: dict):
    print(f"Running Narayan on {c}")
    data = dataset.Dataset(c['dataset'], c['error_fraction'], c['version'], c['error_class'], c["n_rows"], c.get("n_cols"))
    detected_cells = data.get_errors_dictionary('perfect')

    corrections = dict()
    llm_master_args = []
    llm_master_results = []

    for (row, col) in detected_cells:
        llm_master_args.append([(row, col), data.name, data.error_fraction, data.version, data.error_class, 'gpt-3.5-turbo'])

    if len(llm_master_args) > 0:
        llm_master_results = map(correctors.generate_llm_master_features, *zip(*llm_master_args))

    for r in llm_master_results:
        correction = list(r['correction_dict'].keys())

        if len(correction) == 0:
            corrections[r['cell']] = ''
        else:
            corrections[r['cell']] = correction[0]
            
    p, r, f = data.get_data_cleaning_evaluation(corrections)[-3:]

    return {
        "status": 1,
        "result": {
            "precision": p, "recall": r, "f1": f,
            },
        "config": c,
    }

def main(results_path='measurements/'):
    """
    Run Narayan on all datasets three times.
    """
    baran_dataset_ids = ["beers", "flights", "hospital", "rayyan", "tax", "food"]
    renuver_dataset_ids = ["bridges", "cars", "glass", "restaurant"]
    openml_dataset_ids = ["6", "137", "151", "184", "1481", "41027", "43572"]

    for dataset_name in openml_dataset_ids:
        for error_class in ['imputer_simple_mcar']:
            error_fraction = 5
            version = 1
            config = {'dataset': dataset_name,
                        'error_fraction': error_fraction,
                        'version': version,
                        'error_class': error_class,
                        'n_rows': None,
                        'n_cols': None}
            result = run_narayan(config)
            experiment_id = f"{dataset_name}_{error_fraction}_{version}_{error_class}"
            with open(Path(results_path) / f"{experiment_id}.json", 'wt') as f:
                json.dump(result, f)

    three_runs = range(1, 4)

    for dataset_name in renuver_dataset_ids:
        for version in three_runs:
            for error_fraction in [1, 3]:
                error_class = 'imputer_simple_mcar'
                config = {'dataset': dataset_name,
                          'error_fraction': error_fraction,
                          'version': version,
                          'error_class': error_class,
                          'n_rows': None,
                          'n_cols': None}
                result = run_narayan(config)
                experiment_id = f"{dataset_name}_{error_fraction}_{version}_{error_class}"
                with open(Path(results_path) / f"{experiment_id}.json", 'wt') as f:
                    json.dump(result, f)

    for dataset_name in baran_dataset_ids:
        error_fraction = 1
        version = 1
        error_class = 'imputer_simple_mcar'
        config = {'dataset': dataset_name,
                    'error_fraction': error_fraction,
                    'version': version,
                    'error_class': error_class,
                    'n_rows': None,
                    'n_cols': None}
        result = run_narayan(config)
        experiment_id = f"{dataset_name}_{error_fraction}_{version}_{error_class}"
        with open(Path(results_path) / f"{experiment_id}.json", 'wt') as f:
            json.dump(result, f)


if __name__ == '__main__':
    main()
    #res = run_narayan({
    #    'dataset': 'beers',
    #    'error_fraction': '1',
    #    'version': '1',
    #    'error_class': 'simple_mcar',
    #    'n_rows': None,
    #    'n_cols': None
    #})