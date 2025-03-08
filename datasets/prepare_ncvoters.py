import pandas as pd

from helpers import apply_imputer_corruption

def main():
    dtypes = {
        'voter_id': int,
        'voter_reg_num': str,
        'name_prefix': str,
        'first_name': str,
        'middle_name': str,
        'last_name': str,
        'name_suffix': str,
        'age': int,
        'gender': str,
        'race': str,
        'ethnic': str,
        'street_address': str,
        'city': str,
        'state': str,
        'zip_code': str,
        'full_phone_num': str,
        'birth_place': str,
        'register_date': str,
        'download_month': str,
    }
    target_order = ['voter_id',
                    'voter_reg_num',
                    'zip_code',
                    'city',
                    'name_prefix',
                    'first_name',
                    'middle_name',
                    'last_name',
                    'name_suffix',
                    'age',
                    'gender',
                    'race',
                    'ethnic',
                    'street_address',
                    'state',
                    'full_phone_num',
                    'birth_place',
                    'register_date',
                    'download_month']
    targets = ['zip_code', 'city']
    df = pd.read_csv('ncvoters/ncvoter_1024001r_19c.csv', encoding='latin1', dtype=dtypes)

    df.fillna('', inplace=True)
    df = df[target_order]  # reorder

    df.to_csv('ncvoters/clean.csv', index=False)
    df.to_parquet('ncvoters/clean.parquet', index=False)


    for error_fraction in [0.001]:#[0.01, 0.02, 0.03, 0.04,0.05]:
        df_corrupted = df.copy()
        col_fraction = error_fraction * df.shape[1] / 2
        for target in targets:
            se_target = df_corrupted[target]
            se_target_corrupted = apply_imputer_corruption("imputer_simple_mcar", df, se_target, col_fraction)
            df_corrupted[target] = se_target_corrupted

        df_corrupted.to_csv(f'ncvoters/MCAR/dirty_{int(error_fraction*100)}.csv', index=False)
        df_corrupted.to_parquet(f'ncvoters/MCAR/dirty_{int(error_fraction*100)}.parquet', index=False)
    print('Exported typed and corrupted ncvoters datasets')

if __name__ == '__main__':
    main()

