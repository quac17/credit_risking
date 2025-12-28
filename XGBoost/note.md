# XGBoost Process Note

## 1. Overview
Giải thuật XGBoost (Extreme Gradient Boosting) được triển khai để dự đoán rủi ro tín dụng. Đây là mô hình Ensemble mạnh mẽ dựa trên Gradient Boosting Trees.

## 2. Workflow
1. **Pre-processing**:
   - Xử lý giá trị lỗi của `DAYS_EMPLOYED`.
   - Chuẩn hóa `DAYS_BIRTH`.
   - Imputation: Median cho biến số, "Missing" cho biến phân loại.
   - Mã hóa One-hot encoding.
2. **Handling Imbalance**:
   - Sử dụng tham số `scale_pos_weight` tính toán từ tỷ lệ lớp 0/lớp 1 trong tập huấn luyện.
3. **Training**:
   - Huấn luyện với 100 cây (estimators), độ sâu tối đa là 6.
   - Sử dụng tập validation để theo dõi AUC.
4. **Supervised Testing**:
   - Sử dụng file `subset_train2_data.csv` (10k dòng tiếp theo sau 40k dòng train đầu tiên) làm tập "Supervised Test" để kiểm chứng khả năng tổng quát của mô hình trên dữ liệu mới chưa từng thấy nhưng vẫn có nhãn.
5. **Output**:
   - `XGBoost_result_train.csv`: Dự đoán trên tập 40k train.
   - `XGBoost_result_train2_supervised.csv`: Dự đoán trên tập 10k supervised test.
   - `XGBoost_result_test.csv`: Dự đoán trên tập 10k test (không nhãn).
   - `accuracy.csv`: So sánh các chỉ số hiệu năng (Accuracy, AUC, MCC, G-mean) giữa tập Train và Supervised Test.
   - `effectiveness_plot.png`: Biểu đồ cột so sánh hiệu năng.

## 3. How to Run
```powershell
python XGBoost/process.py
```
