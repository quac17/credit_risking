import pandas as pd
import os

def create_data_subsets(train_path='simplified_train_data.csv', test_path='simplified_test_data.csv', 
                        train_nrows=40000, test_nrows=10000):
    """
    Reads the first N rows from train and test CSV files and saves them as subsets.
    """
    print(f"Sampling {train_nrows} rows from {train_path}...")
    try:
        if os.path.exists(train_path):
            train_subset = pd.read_csv(train_path, nrows=train_nrows)
            train_output = f'subset_train_data.csv'
            train_subset.to_csv(train_output, index=False)
            print(f"Saved {len(train_subset)} rows to {train_output}")
        else:
            print(f"Error: {train_path} not found.")
    except Exception as e:
        print(f"Error processing train data: {e}")

    print(f"Sampling {test_nrows} rows from {test_path}...")
    try:
        if os.path.exists(test_path):
            test_subset = pd.read_csv(test_path, nrows=test_nrows)
            test_output = f'subset_test_data.csv'
            test_subset.to_csv(test_output, index=False)
            print(f"Saved {len(test_subset)} rows to {test_output}")
        else:
            print(f"Error: {test_path} not found.")
    except Exception as e:
        print(f"Error processing test data: {e}")

    print(f"Sampling next 10000 rows from {train_path} for supervised test...")
    try:
        if os.path.exists(train_path):
            # Skip the first 40000 rows (excluding header)
            train2_subset = pd.read_csv(train_path, skiprows=range(1, train_nrows + 1), nrows=10000)
            train2_output = 'subset_train2_data.csv'
            train2_subset.to_csv(train2_output, index=False)
            print(f"Saved {len(train2_subset)} rows to {train2_output}")
    except Exception as e:
        print(f"Error processing train2 data: {e}")

if __name__ == "__main__":
    create_data_subsets()
