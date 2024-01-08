import os
import json

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

    try:
        data = dataset.Dataset(c['dataset'], c['error_fraction'], version, c['error_class'], c["n_rows"])
        data.detected_cells = data.get_errors_dictionary()

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
            c['domain_model_threshold'],
        )
        app.VERBOSE = False
        seed = None
        correction_dictionary = app.run(data, seed)
        p, r, f = data.get_data_cleaning_evaluation(correction_dictionary)[-3:]
        return {
                "status": 1,
                "result": {
                    "precision": p, "recall": r, "f1": f
                    },
                "config": c,
                }
    except Exception as e:
        return {"status": 0,
                "result": str(e),
                "config": c}


if __name__ == "__main__":
    # Load OpenAI API-key
    dotenv.load_dotenv()

    # To use the setup with Kubernetes, read the datasets name from an environment variable.
    config = json.loads(os.environ.get('CONFIG', '{}'))

    if len(config) == 0:
        raise ValueError('No config specified')

    experiment_id = os.environ.get('EXPERIMENT_ID')
    save_path = f"/measurements/{experiment_id}.json"

    result = run_mirmir(config)

    with open(save_path, 'wt') as f:
        json.dump(result, f)
