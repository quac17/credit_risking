# Mô hình xếp hạng tín dụng đa cấp (5 level) dựa trên Deep Learning

Tài liệu này mô tả **bài toán**, **cách định nghĩa nhãn 5 cấp**, **kiến trúc deep learning đề xuất** và **cách đánh giá**, phù hợp để mở rộng dự án Home Credit hiện tại (từ phân loại nhị phân sang xếp hạng rủi ro chi tiết hơn).

---

## 1. Mục tiêu dự án

| Khía cạnh | Mô hình hiện tại (dự án) | Hướng mở rộng đề xuất |
|-----------|--------------------------|------------------------|
| Đầu ra | `TARGET` ∈ {0, 1} (trả nợ / khó khăn trả nợ) | **Hạng tín dụng** ∈ {1,…,5} (ví dụ từ rất thấp đến rất cao) |
| Ứng dụng | Cắt ngưỡng cho vay / từ chối | **Định giá lãi suất, hạn mức, bảo lãnh**, ưu tiên xử lý hồ sơ |
| Thuật toán | (đã gỡ các pipeline ML cổ điển trong repo) | **Mạng nơ-ron** (học biểu diễn phi tuyến + xử lý hỗn hợp số/phân loại) |

**Lý do dùng deep learning (tabular):** dữ liệu tín dụng thường có nhiều biến hỗn hợp (số + phân loại cao bậc), quan hệ phi tuyến và tương tác giữa nhóm biến (thu nhập × tỷ lệ nợ, điểm ngoại…). Các kiến trúc như **MLP có embedding**, **Residual MLP**, **TabNet** hoặc **FT-Transformer** được thiết kế cho bảng dữ liệu, khác với CNN/RNN trên ảnh/chuỗi thời gian thuần.

---

## 2. Định nghĩa 5 cấp xếp hạng (risk bands)

Cần một **thang đo có thứ tự**: hạng 1 = rủi ro thấp nhất (khách tốt nhất), hạng 5 = rủi ro cao nhất (gần với nhóm có khả năng vỡ nợ), hoặc ngược lại — quan trọng là **nhất quán giữa huấn luyện, báo cáo và sản phẩm**.

### 2.1. Khi chỉ có nhãn nhị phân (`TARGET` như trong `application_train.csv`)

Dữ liệu gốc không có “mức độ trễ hạn 30/60/90 ngày”. Có thể **xây nhãn proxy có thứ tự** để huấn luyện mô hình đa lớp:

1. **Bước A — Ước lượng xác suất nợ xấu**  
   Huấn luyện một mô hình tabular baseline (tùy chọn) để có `p_default ∈ [0, 1]` trên tập train.

2. **Bước B — Chia quantile thành 5 nhóm**  
   Trên tập train, sắp xếp `p_default` và cắt thành 5 khoảng (quantile 20%, 40%, …) hoặc cắt **chỉ trên nhóm `TARGET=0`** để tách 4 mức “tốt” và gán `TARGET=1` vào **hạng 5** (một cách làm phổ biến khi thiếu nhãn đa mức thật).

3. **Bước C — Hoặc gán theo điểm tổng hợp**  
   Kết hợp `p_default` với vài biến đã biết mạnh (ví dụ `EXT_SOURCE_*`, tỷ lệ tín dụng/thu nhập) thành một **điểm rủi ro 1 chiều**, rồi quantile hóa thành 5 hạng.

> **Lưu ý nghiên cứu:** Nhãn proxy không thay thế nhãn “trễ hạn thực” nếu sau này có dữ liệu; khi đó chỉ cần thay pipeline gán nhãn, giữ nguyên kiến trúc mạng.

### 2.2. Gợi ý đặt tên nghiệp vụ (ví dụ)

| Hạng | Tên gợi ý | Diễn giải ngắn |
|------|-----------|----------------|
| 1 | Rất thấp | Xác suất vỡ nợ rất thấp, ưu tiên lãi suất tốt |
| 2 | Thấp | Rủi ro thấp |
| 3 | Trung bình | Theo dõi định kỳ |
| 4 | Cao | Tăng biện pháp giảm rủi ro (hạn mức, kỳ hạn) |
| 5 | Rất cao / Từ chối | Gần hoặc tương đương nhóm nợ xấu |

---

## 3. Bài toán học máy: phân loại có thứ tự vs phân loại thường

- **Phân loại 5 lớp độc lập (softmax):** đơn giản, dễ triển khai; có thể vi phạm “thứ tự” (dự đoán nhảy từ hạng 1 sang 5 không qua 2–4).
- **Hồi quy thứ tự (ordinal):** phù hợp thang điểm tín dụng — khuyến nghị khi muốn **hình phạt sai “xa” hơn sai “gần”**.

**Hai hướng triển khai trong deep learning:**

1. **Đầu ra 5 logits + CrossEntropy có trọng số lớp** (xử lý mất cân bằng giữa các hạng).  
2. **CORAL (Cumulative Odds) / ordinal layer:** mạng cho ra `K-1` ranh giới tích lũy; loss phù hợp thứ tự hạng.

Dưới đây mô tả kiến trúc **một backbone chung**; lớp đầu ra có thể hoán đổi giữa softmax và ordinal.

---

## 4. Kiến trúc đề xuất (tổng quan)

```
Đầu vào thô (tabular)
    ↓
[Embedding] — mỗi cột phân loại → vector số chiều d_e
    ↓
[Chuẩn hóa / LayerNorm] — biến số (z-score hoặc đã WoE hóa)
    ↓
[Nối vector] — concat(embedding, numeric)
    ↓
[Khối MLP tích chập sâu] — Linear → (BatchNorm) → GELU/ReLU → Dropout, lặp L lớp
    ↓
[Đầu ra] — Linear → 5 logits (softmax) HOẶC K-1 logits (ordinal CORAL)
```

### 4.1. Nhánh biến phân loại (categorical)

- Mỗi cột category `i` có bảng embedding `E_i ∈ R^{v_i × d_e}` (`v_i` = số giá trị sau khi gom nhóm / Unknown).
- Giảm chiều từ one-hot, học được **tương tự giữa các nhóm** (ví dụ nghề nghiệp gần nhau).

### 4.2. Nhánh biến số (numeric)

- Đưa về cùng thang (chuẩn hóa theo train), hoặc dùng **đặc trưng đã WoE** như trong pipeline IV hiện tại để ổn định gradient.
- Có thể thêm **log1p** cho biến lệch phải (số tiền, thu nhập).

### 4.3. Backbone: Residual MLP (khuyến nghị làm baseline deep)

- Các lớp: `x ← x + F(x)` (residual) giúp huấn luyện sâu hơn trên bảng nhỏ/trung bình.
- Kích thước ẩn ví dụ: `[256, 128, 64]`; `dropout ∈ [0.1, 0.3]`.

### 4.4. Phương án nâng cao (tùy chọn)

| Kiến trúc | Ý tưởng | Khi nào cân nhắc |
|-----------|---------|------------------|
| **TabNet** | Attention theo từng bước, “giải thích” nhóm biến được dùng | Cần interpretability + bảng lớn |
| **FT-Transformer** | Biến mỗi cột thành token, self-attention | Nhiều cột tương tác phức tạp |
| **Deep & Cross Network (DCN)** | Tích chập rõ ràng các tương tác bậc thấp | Baseline mạnh cho CTR/tabular |

Với **~20 biến đã chọn IV** như `simplified_train_data.csv`, **Residual MLP + Embedding** thường đủ mạnh và huấn luyện nhanh; FT-Transformer/TabNet hợp khi mở rộng lên nhiều trăm cột sau khi join bureau.

---

## 5. Chi tiết lớp đầu ra và hàm mất mát

### 5.1. Softmax 5 lớp

- `L = CrossEntropyWithWeights` — trọng số tỉ lệ nghịch tần suất từng hạng trên train.
- Có thể thêm **Focal Loss** nếu một vài hạng cực hiếm.

### 5.2. Ordinal (CORAL)

- Mạng cho `K-1` logits `z_j`; xác suất `P(y > j)` qua sigmoid; suy ra `P(y = k)` bằng hiệu các xác suất tích lũy.
- **Lợi ích:** khuyến khích dự đoán “gần đúng” hơn khi sai hạng.

### 5.3. Điều chuẩn (regularization)

- **Weight decay** (L2), **early stopping** theo validation loss.
- **Dropout** sau các lớp lớn; tránh overfit khi số mẫu subset (ví dụ 40k) nhỏ hơn full Home Credit.

---

## 6. Đánh giá mô hình (không dùng Accuracy làm chỉ số duy nhất)

| Chỉ số | Mục đích |
|--------|----------|
| **Macro-F1 / Weighted-F1** | Cân nhắc từng hạng, tránh bỏ qua lớp hiếm |
| **Quadratic Weighted Kappa (QWK)** | Phạt sai xa trên thang thứ tự — **rất phù hợp xếp hạng** |
| **ROC-AUC (OvR hoặc ordinal extension)** | So sánh với baseline nhị phân nếu map hạng ↔ xác suất |
| **Ma trận nhầm lẫn (5×5)** | Xem có nhầm hệ thống giữa hạng kề nhau hay nhảy cóc |

Tối ưu ngưỡng kinh doanh (như trong `credit_risk_guide.txt`) có thể chuyển sang **ngưỡng trên xác suất từng hạng** hoặc trên **expected loss** theo ma trận chi phí sai khác nhau giữa các hạng.

---

## 7. Luồng dữ liệu gắn với repo hiện tại

1. Giữ **IV/WoE & Top 20 features** (`filter_data/`, `create_simplified_data.py`) làm đầu vào có kiểm soát.  
2. Tách **train / validation / test** theo thời gian hoặc theo `SK_ID_CURR` để tránh rò rỉ.  
3. (Tuỳ chọn) Sinh nhãn 5 hạng theo mục 2 trên **chỉ train**; validation/test dùng cùng quy tắc cắt (quantile cố định từ train).  
4. Huấn luyện PyTorch / TensorFlow theo kiến trúc mục 4–5.  
5. So sánh với một **baseline tabular** (ví dụ boosting đa lớp nếu triển khai ngoài repo) trên cùng nhãn.

---

## 8. Rủi ro và giới hạn

- **Nhãn 5 cấp giả lập** từ nhị phân: thứ hạng phản ánh chủ yếu **mức độ ước lượng rủi ro**, không phải lịch sử trễ hạn chi tiết.  
- **Dữ liệu mất cân bằng:** một số hạng có thể rất ít — cần oversampling nhẹ, loss có trọng số, hoặc SMOTE chỉ trên nhóm đa số (thận trọng với leakage).  
- **Giải thích:** embedding khó giải thích hơn LR; có thể dùng SHAP trên mô hình surrogate hoặc attention weights (TabNet).

---

## 9. Tóm tắt kiến trúc “đề xuất chính”

- **Đầu vào:** bảng đặc trưng đã tiền xử lý (giống pipeline hiện tại).  
- **Mạng:** **Embedding cho category + vector số đã chuẩn hóa → Residual MLP (3–5 khối) → đầu ra 5 lớp (softmax hoặc CORAL)**.  
- **Huấn luyện:** AdamW, learning rate schedule, early stopping; loss có trọng số lớp hoặc ordinal.  
- **Đánh giá:** **QWK + Macro-F1 + ma trận 5×5**, so sánh với baseline GBM.

Tài liệu này có thể làm **đặc tả thiết kế** trước khi implement file huấn luyện (ví dụ `deep_rating/train.py`). Khi triển khai code, nên cố định **random seed**, **phiên bản thư viện** (`requirements.txt`) và **cách sinh nhãn 5 hạng** trong một module riêng để tái lập kết quả.
