import argparse
import yaml
import json
import itertools
from pathlib import Path

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

def generate_job(config: dict, experiment_name: str, jobs_path: Path, id: int):
    """
    Generates a kubernetes job config to run a mirmir experiment.
    """

    memory = '64Gi'
    if config['dataset'] in ['tax', 'food']:
        memory = '950Gi'

    template = """apiVersion: batch/v1
kind: Job 
metadata:
  name: {}
spec:
  completions: 1
  template:
    metadata:
      labels:
        app: {}
    spec:
      restartPolicy: Never
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      containers:
        - name: mimir
          image: docker.io/larmor27/mimir:latest
          env:
            - name: CONFIG
              value: '{}'
            - name: EXPERIMENT_ID
              value: {}
          volumeMounts:
            - name: mirmir-results-volume
              mountPath: /measurements  # Mounting the PVC at /app/output directory in the container
            - name: ag-models-volume
              mountPath: /agModels  # Mounting the ephemeral volume at /agModels directory in the container
          resources:
            requests:
              # only start on nodes with 64Gi of RAM available 
              memory: "{}"   
              # only start on nodes with 26 CPU cores available
              cpu: 26
            limits:
              # kill the pod when it uses more than 64Gi of RAM
              memory: "{}"  
              # restrict the pod to never use more than 26 full CPU cores
              cpu: 26
      volumes:
        - name: mirmir-results-volume
          persistentVolumeClaim:
            claimName: mirmir-results-volume
        - name: ag-models-volume
          ephemeral:
            volumeClaimTemplate:
              metadata:
                labels:
                  type: my-frontend-volume
              spec:
                accessModes: [ "ReadWriteOnce" ]
                storageClassName: "cephcsi"
                resources:
                  requests:
                    storage: 30G
    """

    unique_id = f'{experiment_name}-{id}'
    unique_id = unique_id.replace('_', '-')
    job_config = template.format(unique_id, unique_id, json.dumps(config), unique_id, memory, memory)
    with open(jobs_path / f'{unique_id}.yml', 'wt') as f:
        f.write(job_config)

def load_experiment(saved_config: str):
    """
    Load experiment config from a .yaml file in the saved_experiment_configs folder.
    """
    config_path = Path('saved-experiment-configs') / f'{saved_config}.yaml'
    with open(config_path, 'rt') as f:
        config = yaml.safe_load(f)

    experiment_name = config['experiment_name']
    baran_configs, renuver_configs, openml_configs = [], [], []
    if config.get('config_baran'):
        baran_configs = combine_configs(ranges=config['ranges_baran'],
                                        config=config['config_baran'],
                                        runs=config['runs'])
    if config.get('config_renuver'):
        renuver_configs = combine_configs(ranges=config['ranges_renuver'],
                                          config=config['config_renuver'],
                                          runs=config['runs'])
    if config.get('config_openml'):
        openml_configs = combine_configs(ranges=config['ranges_openml'],
                                         config=config['config_openml'],
                                         runs=config['runs'])

    print(f'Successfully loaded experiment configuration from {saved_config}.')

    # merge configs
    configs = [*baran_configs, *renuver_configs, *openml_configs]

    jobs_path = Path('jobs/')
    jobs_path.mkdir(parents=True, exist_ok=True)

    # delete files in jobs/ directory
    for file_path in jobs_path.iterdir():
        if file_path.is_file():
            file_path.unlink()

    i = 0
    for i, config in enumerate(configs):
        generate_job(config, experiment_name, jobs_path, i)

    print(f'Generated {i} jobs and stored them to {jobs_path}/.')

def manual_experiment():
    experiment_name = "2023-11-23-sum-normalize-gpdep"

    baran_configs = combine_configs(
        ranges={
        "dataset": ["beers", "flights", "hospital", "rayyan"],
        "fd_feature": ["gpdep", "norm_gpdep"],
        },
        config={
        "dataset": "1481",
        "error_class": "simple_mcar",
        "error_fraction": 1,
        "labeling_budget": 20,
        "synth_tuples": 100,
        "auto_instance_cache_model": False,
        "clean_with_user_input": True,
        "gpdep_threshold": 0.3,
        "training_time_limit": 90,
        "feature_generators": ['auto_instance', 'domain_instance', 'fd', 'llm_correction', 'llm_master'],
        "classification_model": "ABC",
        "vicinity_orders": [1],
        "vicinity_feature_generator": "naive",
        "n_rows": None,
        "n_best_pdeps": 3,
        "synth_cleaning_threshold": 0.9,
        "test_synth_data_direction": "user_data",
        "pdep_features": ['pr'],
        "fd_feature": "gpdep",
        "domain_model_threshold": 0.01,
        },
        runs=3
    )

    renuver_configs = combine_configs(
        ranges={
        "dataset": ['bridges', 'cars', 'glass', 'restaurant'],
        "fd_feature": ["gpdep", "norm_gpdep"],
        "error_fraction": [1, 3],
        },
        config={
        "dataset": "1481",
        "error_class": "simple_mcar",
        "error_fraction": 1,
        "labeling_budget": 20,
        "synth_tuples": 100,
        "auto_instance_cache_model": False,
        "clean_with_user_input": True,
        "gpdep_threshold": 0.3,
        "training_time_limit": 90,
        "feature_generators": ['auto_instance', 'domain_instance', 'fd', 'llm_correction', 'llm_master'],
        "classification_model": "ABC",
        "vicinity_orders": [1],
        "vicinity_feature_generator": "naive",
        "n_rows": None,
        "n_best_pdeps": 3,
        "synth_cleaning_threshold": 0.9,
        "test_synth_data_direction": "user_data",
        "pdep_features": ['pr'],
        "fd_feature": "gpdep",
        "domain_model_threshold": 0.01,
        },
        runs=3
    )

    openml_configs = combine_configs(
        ranges={
        "dataset": ["6", "137", "184", "1481", "41027", "43572"],
        "fd_feature": ["gpdep", "norm_gpdep"],
        "error_fraction": [1, 5],
        "error_class": ["simple_mcar", "imputer_simple_mcar"],
        },
        config={
        "dataset": "1481",
        "error_class": "simple_mcar",
        "error_fraction": 1,
        "labeling_budget": 20,
        "synth_tuples": 100,
        "auto_instance_cache_model": False,
        "clean_with_user_input": True,
        "gpdep_threshold": 0.3,
        "training_time_limit": 90,
        "feature_generators": ['auto_instance', 'domain_instance', 'fd', 'llm_correction', 'llm_master'],
        "classification_model": "ABC",
        "vicinity_orders": [1],
        "vicinity_feature_generator": "naive",
        "n_rows": 1000,
        "n_best_pdeps": 3,
        "synth_cleaning_threshold": 0.9,
        "test_synth_data_direction": "user_data",
        "pdep_features": ['pr'],
        "fd_feature": "gpdep",
        "domain_model_threshold": 0.01,
        },
        runs=3
    )

    # merge configs
    configs = [*baran_configs, *renuver_configs, *openml_configs]

    jobs_path = Path('jobs/')
    jobs_path.mkdir(parents=True, exist_ok=True)

    # delete files in jobs/ directory
    for file_path in jobs_path.iterdir():
        if file_path.is_file():
            file_path.unlink()

    i = 0
    for i, config in enumerate(configs):
        generate_job(config, experiment_name, jobs_path, i)

    print(f'Generated {i} jobs and stored them to {jobs_path}/.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate k8s job configs, '
                                     'either from a saved file via the -f flag, '
                                     'or manually configured from within the script.')

    parser.add_argument('--from_saved_config', '-f', type=str, default='',
                        help='Name of the yaml file of the saved config.')
    saved_config = parser.parse_args().from_saved_config
    if saved_config == '':
      manual_experiment()
    else:
      load_experiment(saved_config)
