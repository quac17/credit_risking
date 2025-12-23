# Credit Risk Analysis - Home Credit Default Risk

## Project Overview
Home Credit is a financial group aiming to provide loans to the **unbanked population**—people with little or no credit history. Traditional credit scoring models often fail for this demographic.

The goal of this project is to leverage alternative data (telecommunications, transactions, etc.) to predict a client's repayment ability, thereby safely expanding financial access.

## Problem Definition
*   **Type:** Supervised Binary Classification.
*   **Goal:** Predict whether a client will have difficulty repaying a loan (Target = 1) or will repay successfully (Target = 0).
*   **Key Challenge:** **Imbalanced Dataset**. The "default" class (Target 1) accounts for only ~8% of the data. Standard accuracy metrics are misleading in this context.

## Dataset
This project uses a **simplified version** of the Home Credit dataset, focusing on the primary application information (`application_train.csv`).

Key feature groups include:
*   **External Sources:** Scores from external data providers (often the strongest predictors).
*   **Financial Info:** Income, loan amount, annuity.
*   **Demographics:** Age, gender, education level.
*   **Assets:** Car and real estate ownership.

## Metrics
Due to the imbalanced nature of the data, we avoid standard Accuracy. Instead, we perform evaluation using:
*   **ROC-AUC (Area Under the ROC Curve)**
*   **Gini Coefficient**