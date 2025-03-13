import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pyarrow.parquet as pq
import joblib
from sklearn.model_selection import train_test_split

# Mapping for seasons
season_mapping = {
    "Adidas" : 11,
    "Under Armour" : 12,
    "Nike" : 13,
    "Puma" : 14,
    "Jansport" : 15,
    "Yes" : 22, 
    "No" : 23,
    "Polyester" : 2,
    "Leather" : 3,
    "Nylon" : 4,
    "Canvas" : 5,
    "Medium" : 6,
    "Large" : 7,
    "Small" : 8,
    "Messenger" : 9,
    "Tote" : 10,
    "Backpack" : 16,
    "Pink" : 17,
    "Gray" : 18,
    "Blue" : 19,
    "Red" : 20,
    "Green" : 21,

}

def convert_strings_to_ascii(df):
    def to_ascii(value):
        if isinstance(value, str):
            return ''.join(str(ord(char)) for char in value)
        return value

    for col in df.columns:
        df[col] = df[col].apply(to_ascii)
    
    return df

def unCacheOrLoad(file):
    # Path for the cache file
    name = file.replace(".csv", '').replace(".parquet", '')
    path = 'cache/' + name + '_cache.joblib'

    # Check if the cache file exists
    if os.path.exists(path):
        # Load from joblib cache
        data = joblib.load(path)
        print(f"Read {file} from joblib cache")
    else:
        # Load from the original source
        if file.endswith(".csv"):
            data = pd.read_csv(file)
        elif file.endswith(".parquet"):
            data = pq.read_table(file).to_pandas()
        else:
            raise ValueError("Unsupported file format")

        # Save to joblib cache
        os.makedirs('cache', exist_ok=True)
        joblib.dump(data, path)
        print(f"Data {file} loaded and cached with joblib")

    return data


def dropUnusedColumns(train, test, y_name, x_name):
    columns = train.columns.difference(test.columns)
    column_list = columns.to_list()
    column_list.append(x_name)
    column_list.remove(y_name)
    print(f'\nDeleting unsililare rows: {column_list}')

    train = train.drop(column_list, axis=1)
    test = test.drop(x_name, axis=1)
    #print(train.isna().sum())
    return train, test


def map_seasons(train, test):
    # Iterate over each column in the DataFrame
    for column in train.columns:
        # Check if any value in the column is a season (e.g., if the column has any value in the mapping keys)
        if train[column].isin(season_mapping.keys()).any():
            # Apply the mapping
            train[column] = train[column].map(season_mapping)
    for column in test.columns:
        # Check if any value in the column is a season (e.g., if the column has any value in the mapping keys)
        if test[column].isin(season_mapping.keys()).any():
            # Apply the mapping
            test[column] = test[column].map(season_mapping)
    return train, test


def fill_NA(train, test, fill):
    for col in train.columns:
        train[col] = train[col].fillna(fill)
    for col in test.columns:
        test[col] = test[col].fillna(fill)
    return train, test

def remove_blank_rows(train, y_name):
    return train.dropna(subset=[y_name])

def makeSNS(train):
    sns.set_style("whitegrid")
    sns.heatmap(train.corr(), annot=True) # This outputs a heatmap of the correlation between all columns
    plt.figure(figsize=(20, 20))
    plt.show()
    print("done")

def traintestslpit(train, y_name):
    x = train.drop(columns=[y_name])
    y = train[y_name]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    y_train = train[y_name]
    X_train = train.drop(y_name, axis=1)
    return X_train, X_test, y_train, y_test

def saveModel(model, file_name):
    joblib.dump(model, file_name)
    print(f"Model saved to {file_name}")
