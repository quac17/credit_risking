# MLP + CORAL (ordinal) — xếp hạng tín dụng 5 lớp

## Vai trò

**Cùng backbone** với Softmax (embedding + Residual MLP). Khác ở đầu ra: thay vì 5 logits độc lập, CORAL (**COnsistent RAnk Logits**) dùng **K−1 = 4** logits; mỗi logit mô hình hóa xác suất tích lũy \(P(y > j)\) với \(j \in \{0,1,2,3\}\), qua **sigmoid**.

Hàm mất mát: trung bình **BCEWithLogits** trên từng ngưỡng, nhãn nhị phân là \( \mathbb{1}[y > j] \). Sai lệch thứ tự (dự đoán “gần” hạng đúng) thường được phạt nhẹ hơn so với softmax thuần.

## Suy ra lớp

Từ \(P(y>j)\) suy ra phân phối rời rạc trên 5 lớp (hiệu các xác suất tích lũy), rồi **argmax** — xem `coral_predict` trong [`../common/model.py`](../common/model.py).

## Script trong thư mục

| File | Mục đích |
|------|----------|
| `train.py` | Huấn luyện CORAL, lưu checkpoint dưới `outputs/mlp_coral/` (hoặc `--out`) |
| `validate.py` | Metrics trên tập validation, `--checkpoint-dir` trỏ tới thư mục đã train |
| `test.py` | Suy luận; CORAL không xuất vector xác suất đầy đủ như softmax trong file test hiện tại (chỉ lớp dự đoán) |

## Tham chiếu mã

- Loss và biến logits → lớp: [`../common/model.py`](../common/model.py) (`coral_loss`, `coral_predict`, `head='coral'`).
- Engine: [`../common/engine.py`](../common/engine.py).
