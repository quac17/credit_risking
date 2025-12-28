# Logistic Regression Process Note

## 1. Overview
Logistic Regression là mô hình baseline cơ bản, cung cấp khả năng giải thích tốt và tốc độ tính toán nhanh.

## 2. Workflow
1. **Pre-processing**:
   - Tương tự XGBoost (Data cleaning, Median Imputation, One-hot encoding).
2. **Handling Imbalance**:
   - Sử dụng tham số `class_weight='balanced'` để tự động điều chỉnh trọng số các lớp.
3. **Training**:
   - Tối ưu hóa hàm chi phí Log-loss.
4. **Supervised Testing**:
   - Sử dụng file `subset_train2_data.csv` để kiểm nghiệm tính ổn định trên dữ liệu mới có nhãn.
5. **Output**:
   - `LogisticRegression_result_train.csv`: Dự đoán trên tập train.
   - `LogisticRegression_result_train2_supervised.csv`: Dự đoán tập kiểm chứng.
   - `LogisticRegression_result_test.csv`: Dự đoán tập test thực tế.
   - `accuracy.csv`: Bảng chỉ số so sánh hiệu năng.
   - `effectiveness_plot.png`: Biểu đồ so sánh cột.
   - `metrics_explanation.txt`: Giải thích chi tiết kết quả.

## 3. How to Run
```powershell
python LogisticRegression/process.py
```
