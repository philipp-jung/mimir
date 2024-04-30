import ruska
import pandas as pd
from sklearn.datasets import load_iris


def main():
    """
    Load the iris dataset, select the columns 'Petal Length' and 'Target'.
    I looked up that petal length is the most important feature to determine
    the 'Target' class.

    Then, I add missing values completely at random with a 5% chance in the
    Target column, join the clean 'Petal Length' column and save this as the
    dirty version of the data.

    I created this dataset to debug the synth_tuples.
    """
    d = load_iris(as_frame=True)
    df = d["frame"]
    df_corrupted = ruska.simple_mcar(d["frame"], 0.05)

    print('Be careful df_dirty will make a float out of the class value which is an int in df_clean. you have to fix that by hand.')

    df_clean = df.iloc[:, [2, 4]]
    df_corrupted = pd.concat([df.iloc[:, [2]], df_corrupted.iloc[:, [4]]], axis=1)

    df_clean.to_csv('./clean.csv', index=False)
    df_corrupted.to_csv('./dirty.csv', index=False)

    df_clean.to_parquet('./clean.parquet')
    df_corrupted.to_parquet('./dirty.parquet')

    print('Done generating the corrupted data.')


if __name__ == '__main__':
    main()
