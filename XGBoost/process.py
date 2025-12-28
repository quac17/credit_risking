import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, matthews_corrcoef, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set output directory
OUTPUT_DIR = 'XGBoost/output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def prepare_data(df):
    """
    Standard preprocessing for Home Credit data.
    """
    # Handle DAYS_EMPLOYED anomaly
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    
    # Convert DAYS_BIRTH to positive years
    if 'DAYS_BIRTH' in df.columns:
        df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH']) / 365.25
    
    # Identify categorical and numeric columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(exclude=['object']).columns.tolist()
    
    # Preserve IDs and Target
    if 'SK_ID_CURR' in num_cols: num_cols.remove('SK_ID_CURR')
    if 'TARGET' in num_cols: num_cols.remove('TARGET')
    
    # Impute numeric with median
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # Impute categorical with 'Missing'
    for col in cat_cols:
        df[col] = df[col].fillna('Missing')
        
    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, columns=cat_cols)
    
    return df

def calculate_gmean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        return 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return np.sqrt(sensitivity * specificity)

def optimize_threshold(y_true, y_probs):
    thresholds = np.arange(0.01, 1.0, 0.01)
    gmeans = []
    mccs = []
    
    for thr in thresholds:
        y_pred = (y_probs >= thr).astype(int)
        gmeans.append(calculate_gmean(y_true, y_pred))
        mccs.append(matthews_corrcoef(y_true, y_pred))
        
    best_thr_mcc = thresholds[np.argmax(mccs)]
    return best_thr_mcc, thresholds, gmeans, mccs

def evaluate_and_save(model, df, name, threshold, has_target=True):
    """
    Predicts using the model and saves results to CSV.
    """
    X = df.drop(['SK_ID_CURR', 'TARGET'], axis=1, errors='ignore')
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    # Save results
    result_df = pd.DataFrame({
        'SK_ID_CURR': df['SK_ID_CURR'],
        'Probability': probs,
        'Prediction': preds
    })
    if has_target:
        result_df['Actual'] = df['TARGET']
        
    result_path = os.path.join(OUTPUT_DIR, f'XGBoost_result_{name}.csv')
    result_df.to_csv(result_path, index=False)
    print(f"Saved predictions to {result_path}")
    
    if has_target:
        y_true = df['TARGET']
        metrics = {
            'Accuracy': accuracy_score(y_true, preds),
            'AUC': roc_auc_score(y_true, probs),
            'MCC': matthews_corrcoef(y_true, preds),
            'G-mean': calculate_gmean(y_true, preds)
        }
        return metrics
    return None

def run_pipeline():
    # 1. Load all data
    train1 = pd.read_csv('subset_train_data.csv')
    train2 = pd.read_csv('subset_train2_data.csv')
    test = pd.read_csv('subset_test_data.csv')
    
    # Combine for consistent encoding
    len1, len2 = len(train1), len(train2)
    combined = pd.concat([train1, train2, test], axis=0, sort=False).reset_index(drop=True)
    combined_proc = prepare_data(combined)
    
    # Split back
    train1_proc = combined_proc.iloc[:len1]
    train2_proc = combined_proc.iloc[len1:len1+len2]
    test_proc = combined_proc.iloc[len1+len2:]
    
    # 2. Train on train1
    X_train1 = train1_proc.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    y_train1 = train1_proc['TARGET']
    
    X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.2, stratify=y_train1, random_state=42)
    
    ratio = float(np.sum(y_train1 == 0)) / np.sum(y_train1 == 1)
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, 
        scale_pos_weight=ratio, random_state=42, eval_metric='auc'
    )
    
    print("Training XGBoost on standard train set...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # 3. Find optimal threshold on validation set
    y_val_probs = model.predict_proba(X_val)[:, 1]
    optimal_thr, _, _, _ = optimize_threshold(y_val, y_val_probs)
    print(f"Optimal Threshold (MCC) found: {optimal_thr:.2f}")
    
    # 4. Predict on all subsets
    metrics_list = []
    
    # Train1 Prediction
    m1 = evaluate_and_save(model, train1_proc, 'train', optimal_thr)
    m1['Dataset'] = 'Train (40k)'
    metrics_list.append(m1)
    
    # Train2 Prediction (Supervised Test)
    m2 = evaluate_and_save(model, train2_proc, 'train2_supervised', optimal_thr)
    m2['Dataset'] = 'Supervised Test (10k)'
    metrics_list.append(m2)
    
    # Test Prediction (Unsupervised)
    evaluate_and_save(model, test_proc, 'test', optimal_thr, has_target=False)
    
    # 5. Save Accuracy/Metrics to CSV
    acc_df = pd.DataFrame(metrics_list)
    acc_path = os.path.join(OUTPUT_DIR, 'accuracy.csv')
    acc_df.to_csv(acc_path, index=False)
    print(f"Saved metrics to {acc_path}")
    
    # 6. Plot performance
    plt.figure(figsize=(12, 7))
    acc_melted = acc_df.melt(id_vars='Dataset', var_name='Metric', value_name='Value')
    sns.barplot(data=acc_melted, x='Metric', y='Value', hue='Dataset', palette='viridis')
    plt.ylim(0, 1.1)
    plt.title('XGBoost Multi-Dataset Performance Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for p in plt.gca().patches:
        plt.gca().annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                           textcoords='offset points')
    
    plot_path = os.path.join(OUTPUT_DIR, 'effectiveness_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Effectiveness plot saved to {plot_path}")

    # 7. Plot Threshold Tuning
    best_thr, thrs, gmeans, mccs = optimize_threshold(y_val, y_val_probs)
    plt.figure(figsize=(10, 6))
    plt.plot(thrs, gmeans, label='G-mean', color='blue', lw=2)
    plt.plot(thrs, mccs, label='MCC', color='green', lw=2)
    plt.axvline(best_thr, color='red', linestyle='--', label=f'Optimal Thr: {best_thr:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('XGBoost Threshold Optimization (G-mean & MCC)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    thr_plot_path = os.path.join(OUTPUT_DIR, 'threshold_tuning_visual.png')
    plt.savefig(thr_plot_path)
    plt.close()
    print(f"Threshold tuning plot saved to {thr_plot_path}")

if __name__ == "__main__":
    run_pipeline()
