import pandas as pd

from helpers import apply_corruption

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
    df = pd.read_csv('ncvoters/ncvoter_1024001r_19c.csv', encoding='latin1', dtype=dtypes)

    df.fillna('', inplace=True)
    df_corrupted = apply_corruption("simple_mcar", df, 0.001)

    df.to_csv('ncvoters/clean.csv', index=False)
    df.to_parquet('ncvoters/clean.parquet', index=False)

    df_corrupted.to_csv('ncvoters/dirty.csv', index=False)
    df_corrupted.to_csv('ncvoters/dirty.parquet', index=False)
    print('Exported typed and corrupted ncvoters dataset')

if __name__ == '__main__':
    main()

