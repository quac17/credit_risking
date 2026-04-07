# Credit Risk — Home Credit (Top 20 IV + Deep Learning)

## 1. Tổng quan

Home Credit cung cấp tín dụng cho nhóm **unbanked / ít lịch sử tín dụng**. Dự án dùng dữ liệu thay thế và **Top 20 đặc trưng** (IV/WoE) để xây **mô hình xếp hạng tín dụng 5 mức** (deep learning: MLP + embedding), bên cạnh bài toán gốc nhị phân `TARGET` (0/1).

**Báo cáo nghiên cứu chi tiết:** [`report.txt`](report.txt).

**Kiến trúc và luồng xử lý (sơ đồ, module, artifact):** [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 2. Hướng dẫn chạy dự án

### Điều kiện

- **Python** 3.10+ (khuyến nghị dùng môi trường ảo). Cài phụ thuộc: `pip install -r requirements-docker.txt` (đủ cho pipeline DL + WoE/simplified) hoặc `requirements.txt` nếu bạn cần đầy đủ gói workspace.
- **Docker** (tùy chọn nhưng nên dùng trên Windows nếu PyTorch/OpenMP gặp lỗi): Docker Desktop + plugin `docker compose`.
- **Dữ liệu gốc:** đặt `application_train.csv` (và `application_test.csv` nếu chạy đủ pipeline test Kaggle) trong thư mục `data/` trước khi chạy IV/WoE và tạo simplified.

### Thứ tự end-to-end (từ thư mục gốc repo)

| Bước | Việc cần làm | Lệnh gợi ý (Python trên máy) | Lệnh gợi ý (Docker / Make) |
|------|----------------|-------------------------------|----------------------------|
| 1 | IV & WoE | `python filter_data/run_woe_analysis.py` (từ gốc repo) | `make docker-woe` hoặc `docker compose run --rm woe-analysis` |
| 2 | Rút gọn Top 20 | `python create_simplified_data.py` | `make docker-simplified` hoặc `docker compose run --rm create-simplified` |
| 3 | Tạo subset & validate/test | `python sample_data.py` | `make docker-sample` hoặc `docker compose run --rm sample-data` |
| 4 | Huấn luyện MLP Softmax | `python deep_credit_rating/mlp_softmax/train.py` | `make docker-train-softmax` |
| 5 | Đánh giá validation | `python deep_credit_rating/mlp_softmax/validate.py --checkpoint-dir deep_credit_rating/outputs/mlp_softmax` | `make docker-validate-softmax` |
| 6 | Inference test | `python deep_credit_rating/mlp_softmax/test.py --checkpoint-dir deep_credit_rating/outputs/mlp_softmax` | `make docker-test-softmax` |

Với mô hình **CORAL**, thay `mlp_softmax` bằng `mlp_coral` và dùng các target `docker-train-coral`, `docker-validate-coral`, `docker-test-coral`.

### Ghi chú thực hành

- **Docker = môi trường chạy code:** `docker-compose.yml` mount toàn bộ repo (`.: /workspace`). File đặt trên máy trong `data/`, `filter_output/`, CSV ở gốc repo… đều là **đầu vào/đầu ra của dự án** — không nằm “trong image”, chỉ cần đúng đường dẫn trong thư mục dự án trước khi chạy `docker compose run`.
- **Build image một lần:** `docker compose build` (hoặc `make docker-build`).
- **Shell trong container (tùy chọn):** `make docker-shell` hoặc `docker compose run --rm app bash` — sau đó chạy các lệnh `python ...` như trên bàn phím.
- **Bọc lệnh từ bash (Linux/macOS/Git Bash):** `bash scripts/docker-compose.sh woe`, `... sample`, `... train-softmax`, v.v. (xem danh sách trong [`scripts/docker-compose.sh`](scripts/docker-compose.sh)).
- **PowerShell:** `.\scripts\docker-compose.ps1 train-softmax` (và các lệnh tương tự).
- **Windows CMD nhanh:** [`docker-sample-data.cmd`](docker-sample-data.cmd), [`docker-train-mlp-softmax.cmd`](docker-train-mlp-softmax.cmd) — có thể mở rộng tương tự cho các bước khác bằng cùng mẫu `docker compose run --rm <service>`.
- **Thử nhanh / debug:** thêm `--max-rows 500` (hoặc số nhỏ hơn) cho `train.py` / `validate.py` / `test.py` để giảm thời gian chạy.
- Checkpoint mặc định nằm dưới `deep_credit_rating/outputs/<mlp_softmax|mlp_coral>/` (đổi bằng `--checkpoint-dir` nếu cần).

Sau bước 3, bạn phải có các file như `subset_train_data.csv`, `simplified_validate_data.csv`, `subset_test_data.csv` (tùy [`sample_data.py`](sample_data.py)) trước khi chạy train DL.

---

## 3. Quy trình dữ liệu (WoE/IV → simplified → subset)

1. **IV & WoE** — `filter_data/run_woe_analysis.py` (cần `data/application_train.csv`). Kết quả: `filter_output/iv_values_all.csv`, biểu đồ IV/WoE.
2. **Rút gọn Top 20** — `create_simplified_data.py` → `simplified_train_data.csv`, `simplified_test_data.csv`.
3. **Lấy mẫu** — `sample_data.py` → `subset_train_data.csv` (40k train), `subset_train2_data.csv` (10k), `simplified_validate_data.csv` (10k validation, có `TARGET`), `subset_test_data.csv` (10k, có thể không có `TARGET`).

Tiền xử lý trên bộ Top 20: sửa `DAYS_EMPLOYED == 365243`, điền median (số) / `"Missing"` (phân loại), có thể thêm tỷ lệ `Credit_Income_Ratio`, `Annuity_Income_Ratio` nếu các cột có trong Top 20.

---

## 4. Bài toán xếp hạng 5 cấp (nhãn proxy)

Từ `TARGET` nhị phân:

- `TARGET = 1` → hạng rủi ro cao nhất (lớp 4).
- `TARGET = 0` → bốn nhóm theo quantile xác suất nợ từ **LogisticRegression** (chỉ để sinh nhãn, không phải mô hình chính).
- Ngưỡng quantile học trên **tập train**; validate/test dùng cùng `labeler` đã lưu trong `artifacts.joblib`.

Đánh giá đa lớp: **QWK**, Macro/Weighted **F1**, ma trận nhầm lẫn (xem [`deep_credit_rating/common/metrics.py`](deep_credit_rating/common/metrics.py)).

---

## 5. Deep learning — cấu trúc & lệnh

| Thư mục | Mô tả |
|---------|--------|
| [`deep_credit_rating/mlp_softmax/`](deep_credit_rating/mlp_softmax/) | Residual MLP + **Softmax** + CE có trọng số — [`README.md`](deep_credit_rating/mlp_softmax/README.md) |
| [`deep_credit_rating/mlp_coral/`](deep_credit_rating/mlp_coral/) | Cùng backbone + **CORAL** (ordinal) — [`README.md`](deep_credit_rating/mlp_coral/README.md) |
| [`deep_credit_rating/common/`](deep_credit_rating/common/) | `pipeline.py`, `engine.py` (train / validate / test), `model.py`, `labels.py`, `config.py` |

**Mặc định CSV** (đổi bằng `--data` / `--checkpoint-dir`):

- Train: `subset_train_data.csv`
- Validate: `simplified_validate_data.csv`
- Test: `subset_test_data.csv`

**Ví dụ (từ thư mục gốc repo):**

```text
python deep_credit_rating/mlp_softmax/train.py
python deep_credit_rating/mlp_softmax/validate.py --checkpoint-dir deep_credit_rating/outputs/mlp_softmax
python deep_credit_rating/mlp_softmax/test.py --checkpoint-dir deep_credit_rating/outputs/mlp_softmax
```

Tương tự thay `mlp_softmax` → `mlp_coral` cho CORAL. Checkpoint gồm `model.pt`, `artifacts.joblib`, `train_meta.json`.

---

## 6. Chỉ số tín dụng (nhị phân / xác suất)

- **Accuracy**: dễ ảo với lớp ~8% nợ xấu; không nên làm chỉ số duy nhất.
- **ROC-AUC**: khả năng phân tách Tốt/Xấu; ~0.7–0.8 thường được coi là khá trong scoring.
- **MCC**: phù hợp dữ liệu mất cân bằng.
- **G-mean**: \(\sqrt{\text{Sensitivity} \times \text{Specificity}}\) — cân bằng bắt nợ xấu / giữ khách tốt.
- **Gini**: \(2 \times \text{AUC} - 1\).

**Tối ưu ngưỡng:** thay vì 0.5 cố định, có thể quét ngưỡng trên xác suất (0.01–0.99) để cực đại hóa MCC hoặc G-mean (ứng dụng khi có mô hình nhị phân hoặc xác suất lớp).

---

## 7. Docker & Compose

Cần Docker và plugin `docker compose`. Image: `Dockerfile` + [`requirements-docker.txt`](requirements-docker.txt) (PyTorch CPU + pandas/sklearn…). Thư mục dự án mount vào `/workspace`.

```text
docker compose build
```

**Lệnh có sẵn** (xem [`docker-compose.yml`](docker-compose.yml), [`Makefile`](Makefile), [`scripts/docker-compose.sh`](scripts/docker-compose.sh)):

- Tiền xử lý: `woe-analysis`, `create-simplified`, `sample-data`
- DL: `train-mlp-softmax`, `validate-mlp-softmax`, `test-mlp-softmax`, và tương tự `*-mlp-coral`

**Shell tương tác:** `docker compose run --rm app bash`

**Windows:** các file `.cmd` ở gốc repo (nếu có) hoặc chạy `docker compose run ...` trong CMD/PowerShell.

Biến môi trường trong image: `KMP_DUPLICATE_LIB_OK`, `OMP_NUM_THREADS` (giảm xung đột OpenMP trên một số máy).

---

## 8. Cấu trúc thư mục (chính)

- `data/` — dữ liệu gốc (thường không commit; cần có khi chạy WoE).
- `filter_data/` — IV/WoE.
- `deep_credit_rating/outputs/` — checkpoint (có thể gitignore).
- `sample_data.py`, `create_simplified_data.py`
- `requirements-docker.txt` — môi trường tối thiểu cho pipeline này; `requirements.txt` có thể chứa gói khác của workspace.

---

## 9. Khung slide / outline báo cáo (rút gọn)

- Bối cảnh: unbanked, mất cân bằng lớp (~8% nợ xấu).
- EDA & WoE/IV → Top 20.
- Mô hình: MLP + Softmax và/hoặc CORAL; train / validate / test tách file.
- Metrics: QWK, F1, AUC/MCC/G-mean khi so sánh nhị phân.
- Kết luận & hướng mở rộng (thêm bureau, v.v.).

---

*Luôn có `simplified_train_data.csv` (và các subset sau `sample_data.py`) trước khi huấn luyện DL. Trên Windows, nếu `import torch` lỗi, ưu tiên chạy trong Docker (Linux).*
