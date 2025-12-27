# Quy trình Phân tích Dữ liệu Tín dụng (Credit Risk Analysis Process)

Quy trình này mô tả các bước từ phân tích độ ảnh hưởng của dữ liệu đến việc tạo ra bộ dữ liệu rút gọn tối ưu và tiền xử lý.

## 1. Phân tích độ ảnh hưởng (Feature Influence Analysis)
Trước khi rút gọn dữ liệu, chúng ta sử dụng kỹ thuật **Weight of Evidence (WoE)** và **Information Value (IV)** để đánh giá mức độ dự báo của từng trường thông tin.

*   **Công cụ:** `run_woe_analysis.py` (sử dụng các hàm từ `woe_iv_utils.py`).
*   **Kết quả:**
    *   `filter_output/iv_values_all.csv`: Danh sách IV của tất cả 120+ trường.
    *   `filter_output/iv_summary.png`: Biểu đồ cột Top 20 trường mạnh nhất.
    *   `filter_output/woe_*.png`: Biểu đồ WoE cho từng trường quan trọng.

## 2. Rút gọn dữ liệu (Data Simplification)
Dựa trên kết quả phân tích, chúng ta tự động hóa việc tạo bộ dữ liệu mới.

*   **Cơ chế:** Lấy 20 trường có IV cao nhất từ file kết quả phân tích.
*   **Công cụ:** `create_simplified_data.py`.
*   **Kết quả:** Tạo ra `simplified_train_data.csv` và `simplified_test_data.csv`.

## 3. Tiền xử lý & Feature Engineering (Pre-processing)
Thực hiện trên bộ dữ liệu Top 20 đã rút gọn.

### 3.1. Xử lý bất thường (Anomalies Handling)
*   **DAYS_EMPLOYED:** Thay thế giá trị `365243` bằng `NaN`.
*   **Inf/NaN:** Đảm bảo các giá trị vô cực được đưa về `NaN`.

### 3.2. Xử lý dữ liệu thiếu (Imputation)
*   **Numeric:** Điền bằng Median.
*   **Categorical:** Điền bằng "Missing".

### 3.3. Feature Engineering
Tạo các tỷ lệ tài chính quan trọng:
*   **Credit_Income_Ratio** = `AMT_CREDIT / AMT_INCOME_TOTAL`.
*   **Annuity_Income_Ratio** = `AMT_ANNUITY / AMT_INCOME_TOTAL`.

## 4. Exploratory Data Analysis (EDA)
Phân tích mối quan hệ giữa các biến và TARGET.

### 4.1. Phân tích biến mục tiêu (Target Distribution)
*   Kiểm tra sự chênh lệch lớp (Imbalance).
*   *Kết luận:* Cần áp dụng Class Weight khi huấn luyện.

### 4.2. Phân tích chi tiết Top Features
*   **EXT_SOURCE:** Biểu đồ phân phối theo Target.
*   **DAYS_BIRTH:** Phân tích rủi ro theo nhóm tuổi.
*   **Categorical Features:** Tỷ lệ nợ xấu theo từng nhóm (Học vấn, Thu nhập).

### 4.3. Ma trận tương quan
*   Kiểm tra đa cộng tuyến giữa các biến trong Top 20.
