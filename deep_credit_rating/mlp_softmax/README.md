# MLP + Softmax — xếp hạng tín dụng 5 lớp

**Mặc định repo** (`config.py`): `train_sampler=balanced`, `focal_gamma=1.5`, `class_4_sample_weight=10` — ưu tiên học nhãn lớp nguy cơ cao nhất. Để **train kiểu cũ** (shuffle, CE không focal, không nhân mẫu lớp 4):  
`python .../train.py --train-sampler shuffle --focal-gamma 0 --class-4-sample-weight 1`  
hoặc `make docker-train-softmax-extra EXTRA_ARGS="--train-sampler shuffle --focal-gamma 0 --class-4-sample-weight 1"`.

## Vai trò

Mạng học biểu diễn bảng (tabular): mỗi cột phân loại có **embedding**; các cột số được **chuẩn hóa** (StandardScaler sau median/WoE pipeline). Vector nối đi qua **Residual MLP** (khối tuyến tính + BatchNorm + GELU + dropout, có nhánh residual).

Đầu ra: **5 logits** (một cho mỗi hạng 0…4). Chuẩn hóa **softmax** thành xác suất lớp; loss **focal** trên CE (mặc định theo `config.py`); với `train_sampler=shuffle` có thể dùng trọng số lớp trong CE (nghịch tần suất).

## Công thức ngắn

- \( \mathbf{z} = \mathrm{MLP}(\mathrm{concat}(\mathrm{emb}_{cat}, \mathbf{x}_{num})) \in \mathbb{R}^5 \)
- \( \mathcal{L} = -\sum_i w_{y_i} \log \frac{e^{z_{i,y_i}}}{\sum_k e^{z_{i,k}}} \)

## Script trong thư mục

| File | Mục đích |
|------|----------|
| `train.py` | Huấn luyện trên CSV train (mặc định `subset_train_data.csv`), lưu `model.pt` + `artifacts.joblib` |
| `validate.py` | Metrics trên CSV validation (mặc định `subset_validate_data.csv`), cần `--checkpoint-dir` |
| `test.py` | Dự đoán trên CSV test (mặc định `subset_test_data.csv`), ghi `predictions_test.csv` |

Kiểu đầu (`softmax`) được lưu trong checkpoint; `validate`/`test` đọc từ file, không cần truyền `--head`.

## Lớp 4 (nguy cơ cao nhất) — tinh chỉnh thêm

Lớp 4 thường **hiếm** so với 0–3. **Mặc định `config.py`** đã dùng `train_sampler=balanced`, `focal_gamma=1.5`, `class_4_sample_weight=10`. Có thể tăng `--class-4-sample-weight` (vd. 15) nếu recall vẫn thấp, hoặc giảm nếu precision lớp 4 quá yếu.

- **`--train-sampler shuffle` + `--high-risk-weight-boost`:** chỉ khi không dùng `balanced`; nhân trọng số CE theo lớp (ít mạnh hơn so với balanced + nhân mẫu lớp 4).
- **Suy luận không train lại:** `validate.py` / `test.py` — `--high-risk-logit-bias` (thử 0.2–1.5); `validation_metrics.json` ghi giá trị đã dùng.
- **CORAL** (`mlp_coral`): so sánh metrics `high_risk_class` nếu cần thang thứ tự.
- **Giới hạn:** nhãn proxy (LR+quantile), ranh giới 3↔4 trên Top 20 — có thể cần đặc trưng/định nghĩa nhãn, không chỉ chỉnh loss.

## Tham chiếu mã

- Backbone và lớp Softmax: [`../common/model.py`](../common/model.py) (`TabularDeepCreditNet`, `head='softmax'`).
- Luồng huấn luyện: [`../common/engine.py`](../common/engine.py).
