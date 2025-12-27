
import pandas as pd
import os
import woe_iv_utils as utils

def main():
    # Load data
    file_path = './data/application_train.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    target_col = 'TARGET'
    id_col = 'SK_ID_CURR'
    
    # Identify feature columns
    feature_cols = [col for col in df.columns if col not in [target_col, id_col]]
    
    print(f"Found {len(feature_cols)} features to analyze.")
    
    all_results = []
    
    # Create output directory
    output_dir = 'filter_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for feature in feature_cols:
        print(f"Analyzing {feature}...")
        try:
            woe_df, iv = utils.calculate_woe_iv(df.copy(), feature, target_col)
            all_results.append({
                'feature': feature,
                'iv': iv,
                'woe_df': woe_df
            })
            print(f"  IV for {feature}: {iv:.4f}")
        except Exception as e:
            print(f"  Error analyzing {feature}: {e}")

    # Sort by IV descending
    all_results = sorted(all_results, key=lambda x: x['iv'], reverse=True)
    
    # Export all IV values to CSV
    iv_all_df = pd.DataFrame([(x['feature'], x['iv']) for x in all_results], columns=['Feature', 'IV'])
    iv_csv_path = os.path.join(output_dir, 'iv_values_all.csv')
    iv_all_df.to_csv(iv_csv_path, index=False)
    print(f"\nAll IV values saved to {iv_csv_path}")

    # Take top 20
    top_20 = all_results[:20]
    
    print("\n--- Top 20 Features by IV ---")
    iv_summary_list = []
    for item in top_20:
        feature = item['feature']
        iv = item['iv']
        woe_df = item['woe_df']
        print(f"{feature}: {iv:.4f}")
        
        # Plot WoE for top 20
        utils.plot_woe(woe_df, feature, output_dir)
        iv_summary_list.append((feature, iv))

    # Plot IV Summary for top 20
    print("\nGenerating IV Summary plot for top 20 features...")
    utils.plot_iv_summary(iv_summary_list, output_dir)
    
    print(f"\nAnalysis complete. Top 20 results in '{output_dir}' directory.")

if __name__ == "__main__":
    main()
