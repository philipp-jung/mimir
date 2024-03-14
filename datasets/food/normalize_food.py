import csv
import pandas as pd

def main():
    with open('felix/food_input.csv', 'rt') as f:
        csv_reader = csv.DictReader(f)
        data = {}
        for row in csv_reader:
            col = row['attribute']
            if data.get(col) is None:
                data[col] = []
            data[col].append(row)

    for col in data:
        data[col].sort(key=lambda x: int(x['tupleid']))
        data[col] = [x['value'] for x in data[col]]
        
    df_dirty = pd.DataFrame(data)
    df_dirty = df_dirty.astype({'akaname': 'str',
                     'inspectionid': 'int',
                     'city': 'str',
                     'state': 'category',
                     'results': 'category',
                     'longitude': 'str',
                     'latitude': 'str',
                     'inspectiondate': 'str',
                     'risk': 'category',
                     'location': 'str',
                     'license': 'str',
                     'facilitytype': 'category',
                     'address': 'str',
                     'inspectiontype': 'category',
                     'dbaname': 'str',
                     'zip': 'category'})


    is_same_counter = 0
    same_pairs = []
    with open('felix/labeled_food.csv') as f:
        csv_reader = csv.DictReader(f)
        clean_data = {}
        for row in csv_reader:
            col = row['attribute']
            id = int(row['tupleid']) - 1
            if clean_data.get(col) is None:
                clean_data[col] = []
            clean_data[col].append(row)
            if data[col][id] == row['correct_value']:
                is_same_counter += 1
                same_pairs.append({'old': data[col][id], 'new': row['correct_value']})
            data[col][id] = row['correct_value']
    
    df_clean = pd.DataFrame(data)
    df_clean = df_clean.astype({'akaname': 'str',
                     'inspectionid': 'int',
                     'city': 'str',
                     'state': 'category',
                     'results': 'category',
                     'longitude': 'str',
                     'latitude': 'str',
                     'inspectiondate': 'str',
                     'risk': 'category',
                     'location': 'str',
                     'license': 'str',
                     'facilitytype': 'category',
                     'address': 'str',
                     'inspectiontype': 'category',
                     'dbaname': 'str',
                     'zip': 'category'})
    df_clean.to_csv('clean.csv')
    df_clean.to_parquet('clean.parquet')
    df_dirty.to_csv('dirty.csv')
    df_dirty.to_parquet('dirty.parquet')

if __name__ == '__main__':
    main()