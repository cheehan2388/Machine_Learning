import pandas as pd
import numpy as np

def drop_sparse_columns(df, threshold=0.5):

    initial_cols = len(df.columns)
    limit = int(len(df) * threshold)
    
    # Drop columns that don't have at least 'limit' non-NaN values
    df_dropped = df.dropna(axis=1, thresh=limit)
    
    dropped_cols = initial_cols - len(df_dropped.columns)
    print(f"Data Cleaning: Dropped {dropped_cols} columns with < {threshold*100}% valid data.")
    
    return df_dropped

if __name__ == "__main__":
    # Example usage / Test
    print("Running data_cleaning.py as main...")
    try:
        df = pd.read_csv('train_data.csv')
        df_clean = drop_sparse_columns(df, 0.5)
        df_clean.to_csv('proccessed_data.csv')
        print("Final shape:", df_clean.shape)
    except FileNotFoundError:
        print("train_data.csv not found for testing.")
