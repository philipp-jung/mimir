import os
import json
import traceback
from pathlib import Path
import time

from SeqGAN.train import Trainer
from eva import evaluate
from att_reverse import att_reverse
from rule_sample import rule_sample
import config


# To use the setup with Kubernetes, read the datasets name from an environment variable.
dataset = os.environ.get('DATASET', 'DATASET_not_set')

# Otherwise, you can set the dataset name manually here.
# dataset = '184_simple_mcar_1'

path = f"{dataset}_copy"
path_ori = dataset
path_dirty = f"{dataset}_dirty"


# Paths
models_path = Path('models/')
models_base_path = models_path / dataset
path_rules = models_base_path / "rules.txt"

models_base_path.mkdir(parents=True, exist_ok=True)
g_pre_weights_path = models_base_path / 'generator_pre.hdf5'
d_pre_weights_path = models_base_path / 'discriminator_pre.hdf5'
g_weights_path = models_base_path / 'generator.pkl'
d_weights_path = models_base_path / 'discriminator.hdf5'
path_neg = models_base_path / 'generated_sentences.txt'

# Create these files
for file in [g_pre_weights_path, d_pre_weights_path, g_weights_path, d_weights_path, path_neg]:
    file.touch(exist_ok=True)

order = config.order  # Order, 1 for positive order, 0 for negative order

try:
    for order in [1, 0]:  # Changed by Philipp: Run GARF in positive and negative order once each.
        att_reverse(path, order, models_base_path)
        if config.flag == 0 or config.flag == 2:  # 0 for training SeqGAN, 1 for repairing part, 2 for doing it simultaneously
            trainer = Trainer(order,
                            config.batch_size,
                            config.max_length,
                            config.g_e,
                            config.g_h,
                            config.d_e,
                            config.d_h,
                            config.d_dropout,
                            config.generate_samples,
                            path_pos=path,
                            path_neg=str(path_neg),
                            path_rules=path_rules,
                            g_lr=config.g_lr,
                            d_lr=config.d_lr,
                            n_sample=config.n_sample,
                            models_base_path=models_base_path)

            trainer.pre_train(g_epochs=config.g_pre_epochs,
                            d_epochs=config.d_pre_epochs,
                            g_pre_path=str(g_pre_weights_path),
                            d_pre_path=str(d_pre_weights_path),
                            g_lr=config.g_pre_lr,
                            d_lr=config.d_pre_lr)

            trainer.load_pre_train(g_pre_weights_path, d_pre_weights_path)
            trainer.reflect_pre_train()  # Mapping layer weights to agent

            trainer.train(steps=1,
                        g_steps=1,
                        head=10,
                        g_weights_path=g_weights_path,
                        d_weights_path=d_weights_path)

            trainer.save(g_weights_path, d_weights_path)

        if config.flag == 1 or config.flag == 2:
            trainer = Trainer(order,
                            1,
                            config.max_length,
                            config.g_e,
                            config.g_h,
                            config.d_e,
                            config.d_h,
                            config.d_dropout,
                            config.generate_samples,
                            path_pos=path,
                            path_neg=str(path_neg),
                            g_lr=config.g_lr,
                            d_lr=config.d_lr,
                            n_sample=config.n_sample,
                            path_rules=path_rules,
                            models_base_path=models_base_path)
            trainer.load(g_weights_path, d_weights_path)

            rule_len = rule_sample(path_rules, path, order)
            trainer.train_rules(rule_len, path_rules)  # For production rules, generate rules_final.txt from rules.txt
            trainer.filter(path)

            att_reverse(path, 1, models_base_path)
            trainer.repair(path)
    evaluate(path_ori, path, path_dirty)

except Exception as e:
    exception_type = str(type(e).__name__)
    exception_message = str(e)
    exception_traceback = traceback.format_exc()

    # Create a dictionary to store the exception information
    exception_data = {
        "dataset": path_ori,
        "exception_type": exception_type,
        "exception_message": exception_message,
        "exception_traceback": exception_traceback
    }

    # Convert the dictionary to a JSON string
    json_data = json.dumps(exception_data, indent=4)

    # Write the JSON string to a text file
    timestamp = str(time.time_ns())
    with open(f'output/{path_ori}_{timestamp}.txt', 'wt') as file:
        file.write(json_data)
    print('Did not clean data successfully:')
    print(f'{exception_type}: {exception_message}')


