import os
import pandas as pd

# small script to transform .csv files with semicolon-separated values
# into .csv files with comma-separated values. Just move the script in
# a directory containing ;-separted .csv files and run it. Will overwrite
# existing .csv files.

for root, _, files in os.walk('./'):
    for file in files:
        end = file.split('.')[-1]
        if end == 'csv':
            df = pd.read_csv(root+file, sep=';')
            df.to_csv(root+file, index=False)
