# MLP + Softmax — xếp hạng tín dụng 5 lớp

## Vai trò

Mạng học biểu diễn bảng (tabular): mỗi cột phân loại có **embedding**; các cột số được **chuẩn hóa** (StandardScaler sau median/WoE pipeline). Vector nối đi qua **Residual MLP** (khối tuyến tính + BatchNorm + GELU + dropout, có nhánh residual).

Đầu ra: **5 logits** (một cho mỗi hạng 0…4). Chuẩn hóa **softmax** thành xác suất lớp; hàm mất mát **CrossEntropy** với **trọng số lớp** (nghịch tần suất trên tập train) để giảm lệch lớp hiếm.

## Công thức ngắn

- \( \mathbf{z} = \mathrm{MLP}(\mathrm{concat}(\mathrm{emb}_{cat}, \mathbf{x}_{num})) \in \mathbb{R}^5 \)
- \( \mathcal{L} = -\sum_i w_{y_i} \log \frac{e^{z_{i,y_i}}}{\sum_k e^{z_{i,k}}} \)

## Script trong thư mục

| File | Mục đích |
|------|----------|
| `train.py` | Huấn luyện trên CSV train (mặc định `subset_train_data.csv`), lưu `model.pt` + `artifacts.joblib` |
| `validate.py` | Metrics trên CSV validation (mặc định `simplified_validate_data.csv`), cần `--checkpoint-dir` |
| `test.py` | Dự đoán trên CSV test (mặc định `subset_test_data.csv`), ghi `predictions_test.csv` |

Kiểu đầu (`softmax`) được lưu trong checkpoint; `validate`/`test` đọc từ file, không cần truyền `--head`.

## Tham chiếu mã

- Backbone và lớp Softmax: [`../common/model.py`](../common/model.py) (`TabularDeepCreditNet`, `head='softmax'`).
- Luồng huấn luyện: [`../common/engine.py`](../common/engine.py).
