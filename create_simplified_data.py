import pandas as pd
import os

def create_simplified_data_by_iv(iv_csv_path, input_path, output_path):
    print(f"Reading IV values from {iv_csv_path}...")
    if not os.path.exists(iv_csv_path):
        print(f"Error: {iv_csv_path} not found. Please run the analysis script first.")
        return

    # Read top 20 features based on IV
    iv_df = pd.read_csv(iv_csv_path)
    # Filter out target and ID just in case, though they shouldn't be in the top 20 based on IV logic
    top_features = iv_df['Feature'].tolist()[:20]
    
    # Always include ID and Target for the simplified dataset
    columns_to_keep = ['SK_ID_CURR', 'TARGET'] + top_features
    # Deduplicate in case ID or Target were already in top features
    columns_to_keep = list(dict.fromkeys(columns_to_keep))

    print(f"Reading from {input_path}...")
    try:
        # Load data. We use chunking or only necessary columns to save memory if needed,
        # but for this size, reading the whole thing is usually fine if memory allows.
        # To be safe, we check if columns exist first.
        
        # Read only header to check existence
        header = pd.read_csv(input_path, nrows=0).columns.tolist()
        final_cols = [col for col in columns_to_keep if col in header]
        
        if len(final_cols) < len(columns_to_keep):
            missing = set(columns_to_keep) - set(final_cols)
            print(f"Warning: Missing columns in input: {missing}")

        print(f"Selecting {len(final_cols)} columns...")
        df = pd.read_csv(input_path, usecols=final_cols)
        
        print(f"Saving to {output_path}...")
        df.to_csv(output_path, index=False)
        print(f"Done! Created {output_path} with shape {df.shape}")

    except Exception as e:
        print(f"An error occurred while processing {input_path}: {e}")

if __name__ == "__main__":
    iv_results_path = 'filter_output/iv_values_all.csv'
    
    input_train = 'data/application_train.csv'
    input_test = 'data/application_test.csv'
    
    output_train = 'simplified_train_data.csv'
    output_test = 'simplified_test_data.csv'
    
    create_simplified_data_by_iv(iv_results_path, input_train, output_train)
    
    # For test data, 'TARGET' won't be present. Let's handle it gracefully.
    # We can detect presence of TARGET in skip_target logic if needed inside the function, 
    # but the current usecols=final_cols already handles it via the header check.
    create_simplified_data_by_iv(iv_results_path, input_test, output_test)
