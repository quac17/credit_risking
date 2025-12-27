
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_woe_iv(df, feature, target):
    """
    Calculates WoE and IV for a given feature and target.
    
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature column.
        target (str): The name of the target column.
        
    Returns:
        pd.DataFrame: DataFrame with WoE and IV details for each bin.
        float: The total Information Value (IV) for the feature.
    """
    lst = []
    
    # Create a copy of the feature series to avoid modifying the original dataframe in place unexpectedly
    s = df[feature].copy()
    
    # Check if numeric-like
    if pd.api.types.is_numeric_dtype(s):
        # Handle Inf
        s = s.replace([np.inf, -np.inf], np.nan)
        
        if s.nunique() > 20:
             # Discretize continuous variables
             try:
                 # usage of bins=10 implies qcut
                 # We use category type to easily add 'Missing' later
                 s_binned = pd.qcut(s, q=10, duplicates='drop').astype('object')
                 # Fill NaN bins with 'Missing'
                 s_binned = s_binned.fillna('Missing').astype(str)
                 df['bin'] = s_binned
             except Exception as e:
                 print(f"Warning: qcut failed for {feature}. Error: {e}")
                 df['bin'] = s.fillna('Missing').astype(str)
        else:
             # Low cardinality numeric
             df['bin'] = s.fillna('Missing').astype(str)
    else:
        # Categorical
        df['bin'] = s.fillna('Missing').astype(str)

    # Calculate statistics using groupby for performance
    grouped = df.groupby(['bin', target]).size().unstack(fill_value=0)
    
    # Ensure both Good (0) and Bad (1) columns exist
    if 0 not in grouped.columns: grouped[0] = 0
    if 1 not in grouped.columns: grouped[1] = 0
    
    # Rename columns and calculate total
    grouped = grouped.rename(columns={0: 'Good', 1: 'Bad'})
    grouped['Total'] = grouped['Good'] + grouped['Bad']
    
    # Reset index to make 'bin' a column
    data = grouped.reset_index().rename(columns={'bin': 'Bin'})
    
    # Calculate totals
    total_good = data['Good'].sum()
    total_bad = data['Bad'].sum()
    
    # Calculate Probabilities
    data['Dist_Good'] = data['Good'] / total_good
    data['Dist_Bad'] = data['Bad'] / total_bad
    
    # Calculate WoE
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    data['WoE'] = np.log((data['Dist_Good'] + epsilon) / (data['Dist_Bad'] + epsilon))
    
    # Calculate IV components
    data['IV_Component'] = (data['Dist_Good'] - data['Dist_Bad']) * data['WoE']
    
    # Calculate Total IV
    total_iv = data['IV_Component'].sum()
    
    # Sort by WoE for better plotting
    data = data.sort_values(by='WoE')
    
    return data, total_iv

def plot_iv_summary(iv_list, output_dir='filter_output'):
    """
    Plots a bar chart of IVs for all variables.
    
    Args:
        iv_list (list of tuples): List of (feature_name, iv_value).
        output_dir (str): Directory to save the plot.
    """
    if not iv_list:
        print("No IV values to plot.")
        return

    iv_df = pd.DataFrame(iv_list, columns=['Variable', 'IV'])
    iv_df = iv_df.sort_values(by='IV', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='IV', y='Variable', data=iv_df, palette='viridis')
    
    # Add threshold line
    plt.axvline(x=0.02, color='r', linestyle='--', label='Threshold (0.02)')
    plt.title('Information Value (IV) by Variable')
    plt.xlabel('IV')
    plt.ylabel('Variable')
    plt.legend()
    plt.tight_layout()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    save_path = os.path.join(output_dir, 'iv_summary.png')
    plt.savefig(save_path)
    plt.close()
    print(f"IV summary plot saved to {save_path}")

def plot_woe(woe_df, feature_name, output_dir='output'):
    """
    Plots WoE for a specific variable.
    
    Args:
        woe_df (pd.DataFrame): The DataFrame returned by calculate_woe_iv.
        feature_name (str): Name of the feature.
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot WoE
    sns.lineplot(x='Bin', y='WoE', data=woe_df, marker='o', color='b', label='WoE')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    plt.title(f'Weight of Evidence (WoE) for {feature_name}')
    plt.xlabel('Bins')
    plt.ylabel('WoE')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, f'woe_{feature_name}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"WoE plot for {feature_name} saved to {save_path}")
