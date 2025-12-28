# LightGBM Process Note

## 1. Overview
LightGBM (Light Gradient Boosting Machine) là một framework boosting mạnh mẽ hơn XGBoost về tốc độ và khả năng xử lý các biến phân loại (categorical) trực tiếp.

## 2. Workflow
1. **Pre-processing**:
   - Xử lý giá trị lỗi tương tự các mô hình khác.
   - Điểm khác biệt: Giữ nguyên các biến phân loại và ép kiểu dữ liệu `category`. LightGBM sẽ tự xử lý tối ưu mà không cần One-hot encoding thủ công.
2. **Handling Imbalance**:
   - Sử dụng tham số `is_unbalance=True`.
3. **Training**:
   - Sử dụng cơ chế Early Stopping nếu AUC trên tập validation không cải thiện trong 50 vòng lặp.
4. **Supervised Testing**:
   - Sử dụng file `subset_train2_data.csv` để kiểm tra độ chính xác trên dữ liệu có nhãn mới.
5. **Output**:
   - `LightGBM_result_train.csv`: Dự đoán tập huấn luyện.
   - `LightGBM_result_train2_supervised.csv`: Dự đoán tập kiểm chứng.
   - `LightGBM_result_test.csv`: Dự đoán tập test thực tế.
   - `accuracy.csv`: Thống kê các chỉ số hiệu năng.
   - `effectiveness_plot.png`: Biểu đồ cột so sánh AUC, MCC, G-mean.
   - `metrics_explanation.txt`: Giải thích kết quả chi tiết.

## 3. How to Run
```powershell
python LightGBM/process.py
```
