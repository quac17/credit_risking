import pandas as pd
import os

def create_data_subsets(
    train_path='simplified_train_data.csv',
    test_path='simplified_test_data.csv',
    train_nrows=40000,
    test_nrows=10000,
    train2_nrows=10000,
    validate_nrows=10000,
):
    """
    Đọc các tập con từ simplified train/test và ghi CSV phục vụ huấn luyện / kiểm định.

    - subset_train_data.csv: train_nrows dòng đầu (huấn luyện).
    - subset_test_data.csv: test_nrows dòng từ simplified test.
    - subset_train2_data.csv: train2_nrows dòng tiếp theo (holdout có nhãn).
    - subset_validate_data.csv: validate_nrows dòng tiếp theo (validation, có TARGET),
      không trùng hai tập trên — dùng làm dữ liệu kiểm định mô hình.
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

    print(f"Sampling next {train2_nrows} rows from {train_path} for supervised test...")
    try:
        if os.path.exists(train_path):
            # Bỏ qua train_nrows dòng dữ liệu đầu (giữ header)
            train2_subset = pd.read_csv(train_path, skiprows=range(1, train_nrows + 1), nrows=train2_nrows)
            train2_output = 'subset_train2_data.csv'
            train2_subset.to_csv(train2_output, index=False)
            print(f"Saved {len(train2_subset)} rows to {train2_output}")
    except Exception as e:
        print(f"Error processing train2 data: {e}")

    print(f"Sampling next {validate_nrows} rows from {train_path} for validation (simplified_validate_data.csv)...")
    try:
        if os.path.exists(train_path):
            skip_after = train_nrows + train2_nrows
            validate_subset = pd.read_csv(
                train_path,
                skiprows=range(1, skip_after + 1),
                nrows=validate_nrows,
            )
            validate_output = 'subset_validate_data.csv'
            validate_subset.to_csv(validate_output, index=False)
            print(f"Saved {len(validate_subset)} rows to {validate_output}")
        else:
            print(f"Error: {train_path} not found.")
    except Exception as e:
        print(f"Error processing validation data: {e}")

if __name__ == "__main__":
    create_data_subsets()
