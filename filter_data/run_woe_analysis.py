import os
from pathlib import Path

import pandas as pd
import woe_iv_utils as utils

# Luôn resolve theo gốc dự án (thư mục cha của filter_data/), không phụ thuộc cwd —
# cần khi chạy trong Docker với working_dir=/workspace/filter_data.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_TRAIN = _PROJECT_ROOT / "data" / "application_train.csv"


def main():
    file_path = _DEFAULT_TRAIN
    if not file_path.is_file():
        print(
            f"Error: không tìm thấy {file_path}.\n"
            "Đặt file gốc Home Credit (Kaggle) vào thư mục data/ của dự án, "
            "ví dụ data/application_train.csv trên máy host — Docker mount cả repo nên đường dẫn giống khi chạy local."
        )
        return

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    target_col = 'TARGET'
    id_col = 'SK_ID_CURR'
    
    # Identify feature columns
    feature_cols = [col for col in df.columns if col not in [target_col, id_col]]
    
    print(f"Found {len(feature_cols)} features to analyze.")
    
    all_results = []
    
    # Cùng đường dẫn với create_simplified_data.py: <gốc dự án>/filter_output/
    output_dir = str(_PROJECT_ROOT / "filter_output")
    os.makedirs(output_dir, exist_ok=True)

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
    
    print(f"\nAnalysis complete. Top 20 results in directory:\n  {output_dir}")

if __name__ == "__main__":
    main()
