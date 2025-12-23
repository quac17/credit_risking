import pandas as pd
import os

def create_simplified_data():
    input_path = 'data/application_train.csv'
    output_path = 'simplified_data.csv'

    # Columns to select based on note.txt Part 1.2
    columns_to_keep = [
        # ID
        'SK_ID_CURR', 
        # Target
        'TARGET',
        # External Sources
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        # Financial
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
        # Demographics
        'DAYS_BIRTH', 'CODE_GENDER', 'NAME_EDUCATION_TYPE',
        # Assets
        'FLAG_OWN_CAR', 'FLAG_OWN_REALTY'
    ]

    print(f"Reading from {input_path}...")
    try:
        df = pd.read_csv(input_path)
        
        # Check if all columns exist
        missing_cols = [col for col in columns_to_keep if col not in df.columns]
        if missing_cols:
            print(f"Warning: The following columns were not found in the input data: {missing_cols}")
            # Proceed with available columns
            columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        
        print(f"Selecting {len(columns_to_keep)} columns...")
        simplified_df = df[columns_to_keep]
        
        print(f"Saving to {output_path}...")
        simplified_df.to_csv(output_path, index=False)
        print("Done!")
        
        # Verify
        print("\nVerification:")
        print(f"Output shape: {simplified_df.shape}")
        print("First 5 rows:")
        print(simplified_df.head())

    except FileNotFoundError:
        print(f"Error: File not found at {os.path.abspath(input_path)}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_simplified_data()
