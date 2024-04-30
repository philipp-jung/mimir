import time
import json
from pathlib import Path
import holoclean
from detect import NullDetector, ViolationDetector
from repair.featurize import *

def main():
    """
    Run HoloClean on all datasets three times.
    """
    three_runs = range(1, 4)
    # hospital uses ~100GB of memory and gets killed by the kernel
    # baran_dataset_ids = ["beers", "flights", "hospital", "rayyan"]
    baran_dataset_ids = ["beers", "flights", "rayyan"]
   
    # glass does run into a timeout
    #renuver_dataset_ids = ["bridges", "cars", "glass", "restaurant"]
    renuver_dataset_ids = ["bridges", "cars", "restaurant"]

    # cannot mine DCs for 6, 137 _has not DCs_, 43572 out of Memory 
    openml_dataset_ids = ["151", "184", "1481", "41027",]

    for dataset_name in openml_dataset_ids:
        for error_class in ['imputer_simple_mcar', 'simple_mcar']:
            for error_fraction in [1, 5]:
                for _ in three_runs:
                    dirty_table = f'{dataset_name}_{error_class}_{error_fraction}'
                    clean_table = dataset_name
                    run_hc(dataset_name, clean_table, dirty_table)

    for dataset_name in renuver_dataset_ids:
        clean_table = dataset_name
        for version in three_runs:
            for error_fraction in range(1, 6):
                dirty_table = f'{dataset_name}_{error_fraction}_{version}'
                run_hc(dataset_name, clean_table, dirty_table)

    for dataset_name in baran_dataset_ids:
        for _ in three_runs:
            dirty_table = f'{dataset_name}_dirty'
            clean_table = dataset_name
            run_hc(dataset_name, clean_table, dirty_table)

def run_hc(dataset_name: str, clean_name: str, dirty_name: str):
    """
    Run HoloClean on a single dataset. Write results to the results/ directory,
    which is created if necessary.
    """
    output_path = Path('/root/mirmir/benchmarks/holoclean/src/results/')
    output_path.mkdir(exist_ok=True)

    data_path = Path('/root/mirmir/benchmarks/holoclean/src/data/')
    # 1. Setup a HoloClean session.
    hc = holoclean.HoloClean(
        db_name='holo',
        domain_thresh_1=0,
        domain_thresh_2=0,
        weak_label_thresh=0.99,
        max_domain=10000,
        cor_strength=0.6,
        nb_cor_strength=0.8,
        epochs=10,
        weight_decay=0.01,
        learning_rate=0.001,
        threads=1,
        batch_size=1,
        verbose=False,
        timeout=3*60000,
        feature_norm=False,
        weight_norm=False,
        print_fw=True
    ).session

    # 2. Load training data and denial constraints.
    hc.load_data(dataset_name, data_path/f'datasets/{dirty_name}.csv')
    hc.load_dcs(data_path/f'dcs/hydra_{dataset_name}.txt')
    hc.ds.set_constraints(hc.get_dcs())

    # 3. Detect erroneous cells using these two detectors.
    detectors = [NullDetector(), ViolationDetector()]
    hc.detect_errors(detectors)

    # 4. Repair errors utilizing the defined features.
    hc.setup_domain()
    featurizers = [
        InitAttrFeaturizer(),
        OccurAttrFeaturizer(),
        FreqFeaturizer(),
        ConstraintFeaturizer(),
    ]

    hc.repair_errors(featurizers)

    # 5. Evaluate the correctness of the results.
    result = hc.evaluate(fpath=data_path/f'datasets/{clean_name}.csv',
                       tid_col='tid',
                       attr_col='attribute',
                       val_col='correct_val')

    timestamp = str(int(time.time() * 1e9))
    with open(output_path/f'{dataset_name}_{dirty_name}_{timestamp}.txt', 'w') as f:
        f.write(json.dumps(result._asdict()))

if __name__ == '__main__':
    main()
