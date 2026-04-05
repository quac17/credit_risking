# Credit Risk Analysis - Home Credit Default Risk

## 1. Tổng quan dự án (Project Overview)
Home Credit là tập đoàn tài chính hướng tới việc cung cấp các khoản vay cho những người dân không có hoặc có ít lịch sử tín dụng (**unbanked population**). Các mô hình chấm điểm tín dụng truyền thống thường thất bại với nhóm đối tượng này.
Mục tiêu của dự án là tận dụng các dữ liệu thay thế (viễn thông, giao dịch...) để dự đoán khả năng trả nợ, giúp mở rộng khả năng tiếp cận tài chính một cách an toàn.

## 2. Định nghĩa bài toán (Problem Definition)
*   **Loại bài toán:** Phân loại nhị phân có giám sát (Supervised Binary Classification).
*   **Mục tiêu:** Dự đoán khách hàng gặp khó khăn trả nợ (Target = 1) hoặc trả nợ thành công (Target = 0).
*   **Thách thức chính:** **Dữ liệu mất cân bằng nghiêm trọng**. Tỷ lệ nợ xấu chỉ chiếm khoảng 8%. Do đó, các chỉ số như Accuracy sẽ không phản ánh chính xác hiệu năng mô hình.

## 3. Quy trình thực hiện (Workflow)
Dự án được triển khai theo các bước chuẩn hóa sau:


### Bước 1: Tiền xử lý & Chọn lọc đặc trưng (IV/WoE)
Sử dụng phương pháp Information Value (IV) để tự động chọn ra **Top 20 đặc trưng mạnh nhất** (như EXT_SOURCE, DAYS_BIRTH, AMT_CREDIT...).
- Xử lý các giá trị bất thường (ví dụ: lỗi 365243 ngày làm việc).
- Điền khuyết dữ liệu bằng Median.
- Tạo các tỷ lệ tài chính như `Credit_Income_Ratio`, `Annuity_Income_Ratio`.


### Bước 2: Chuẩn bị dữ liệu mẫu cho huấn luyện (Sampling)
Do dữ liệu gốc rất lớn, chúng ta thực hiện trích xuất các tập con để huấn luyện và kiểm thử nhanh.
- Chạy lệnh: `python sample_data.py`
- Kết quả: Tạo ra các file `subset_train_data.csv` (40k), `subset_train2_data.csv` (10k kiểm chứng), `subset_test_data.csv` (10k dự đoán).

### Bước 3: Huấn luyện mô hình
Đặc tả hướng **deep learning / xếp hạng tín dụng đa cấp** nằm trong `CREDIT_RATING_DEEP_LEARNING.md`. Các pipeline tiền xử lý (IV/WoE, `sample_data.py`) vẫn dùng chung cho bước này.

### Bước 4: Tối ưu hóa ngưỡng & Đánh giá (Threshold Optimization)
Khi có xác suất rủi ro từ mô hình, có thể không dùng ngưỡng mặc định 0.5 mà duyệt ngưỡng từ 0.01 đến 0.99 để cực đại hóa **MCC** và **G-mean** (xem `credit_risk_guide.txt`).

## 4. Cấu trúc thư mục (Project Structure)
- `CREDIT_RATING_DEEP_LEARNING.md`: Mô tả kiến trúc và bài toán xếp hạng tín dụng (deep learning).
- `data/`: Thư mục chứa dữ liệu gốc (nếu có).
- `sample_data.py`: Script lấy mẫu dữ liệu.
- `credit_risk_guide.txt`: Hướng dẫn chuyên sâu về các chỉ số (AUC, MCC, G-mean).
- `report.txt`: Báo cáo nghiên cứu chi tiết.
- `slide.txt`: Khung slide thuyết trình.

## 5. Chỉ số đánh giá (Metrics)
*   **ROC-AUC**: Khả năng phân tách Tốt/Xấu (Chỉnh số chính).
*   **MCC (Matthews Correlation Coefficient)**: Tin cậy nhất cho dữ liệu mất cân bằng.
*   **G-mean**: Cân bằng giữa việc bắt nợ xấu và giữ khách hàng tốt.
*   **Gini Coefficient**: $2 \times AUC - 1$.

---
*Lưu ý: Luôn chạy `sample_data.py` trước khi thực hiện các bước huấn luyện mô hình.*