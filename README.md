# Credit Risk Analysis - Home Credit Default Risk

## Project Overview
Home Credit is a financial group aiming to provide loans to the **unbanked population**—people with little or no credit history. Traditional credit scoring models often fail for this demographic.

The goal of this project is to leverage alternative data (telecommunications, transactions, etc.) to predict a client's repayment ability, thereby safely expanding financial access.

## Problem Definition
*   **Type:** Supervised Binary Classification.
*   **Goal:** Predict whether a client will have difficulty repaying a loan (Target = 1) or will repay successfully (Target = 0).
*   **Key Challenge:** **Imbalanced Dataset**. The "default" class (Target 1) accounts for only ~8% of the data. Standard accuracy metrics are misleading in this context.

## Dataset & Workflow
This project implements an automated data-driven pipeline for credit scoring:

### 1. Influence Analysis (WoE & IV)
Instead of manual feature selection, we use **Weight of Evidence (WoE)** and **Information Value (IV)** to rank 120+ features.
- Run `python run_woe_analysis.py` to analyze global features.
- Outputs are saved in the `filter_output/` directory (IV summary and WoE plots).

### 2. Automated Simplification
The project automatically selects the **top 20 most predictive features** to create a lightweight dataset for modeling.
- Run `python create_simplified_data.py`.
- This generates `simplified_train_data.csv` and `simplified_test_data.csv` based on the latest IV analysis.

### 3. Key Feature Groups (Historically Top Ranked)
*   **EXT_SOURCE_1, 2, 3:** Most powerful external predictors.
*   **Labor & Age:** `DAYS_BIRTH`, `DAYS_EMPLOYED`, `OCCUPATION_TYPE`.
*   **Financials:** `AMT_GOODS_PRICE`, `AMT_CREDIT`, `AMT_ANNUITY`.

## Project Structure
- `woe_iv_utils.py`: Vectorized calculation engine for WoE and IV.
- `run_woe_analysis.py`: Global analysis script.
- `create_simplified_data.py`: Dynamic data extractor.
- `filter_output/`: Visualizations and IV ranking CSV.

## Metrics
Due to the imbalanced nature of the data, we use:
*   **ROC-AUC (Area Under the ROC Curve)**
*   **Gini Coefficient** ($2 \times AUC - 1$)