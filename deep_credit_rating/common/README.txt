Thư mục: deep_credit_rating/common/
====================================
Mô-đun dùng chung cho pipeline deep learning xếp hạng tín dụng 5 mức (Softmax và CORAL).
Không phụ thuộc torch ở mức import tệp (trừ model.py); engine.py gọi apply_before_torch() trước khi nạp PyTorch.

Danh sách file và ý nghĩa
--------------------------

__init__.py
  Gói Python "common". Comment ngắn mô tả nội dung gói (tiền xử lý, nhãn, metrics, backbone).

config.py
  Hằng số cấu hình: đường dẫn mặc định tới CSV (train/validate/test), thư mục output,
  NUM_CLASSES, kiến trúc MLP (EMB_DIM, HIDDEN_DIMS, DROPOUT), optimizer (LR, WEIGHT_DECAY),
  EPOCHS, BATCH_SIZE, early stopping, seed, danh sách cột bỏ (DROP_COLS) và cột phân loại
  gợi ý (DEFAULT_CAT_COLS).

env_setup.py
  Hàm apply_before_torch(): đặt biến môi trường (KMP_DUPLICATE_LIB_OK, OMP/MKL_NUM_THREADS)
  trước khi import torch — giảm lỗi/segfault do xung đột OpenMP trên một số môi trường (đặc biệt Windows).

labels.py
  Lớp CreditRatingLabeler: từ TARGET nhị phân + đặc trưng số, sinh nhãn 0..4 bằng
  LogisticRegression + ngưỡng quantile trên nhóm TARGET=0; TARGET=1 → lớp cao nhất.
  Được fit trên train và lưu trong artifacts.joblib; validate/test chỉ transform, không fit lại.

preprocess.py
  fix_days_employed_anomaly: xử lý giá trị 365243 trong DAYS_EMPLOYED.
  TabularPreprocessor: fit/transform median cho số, mã hóa chỉ số cho cột phân loại;
  suy ra danh sách cột cat/num qua infer_feature_columns (nếu có trong tệp).

pipeline.py
  fit_training_pipeline(df): fit preprocessor, labeler, StandardScaler trên tập train;
  trả về tensor đầu vào (đã scale) và nhãn đa lớp.
  transform_eval(df, pre, scaler): transform cho val/test; trả cả X_num_raw (cho labeler)
  và X_num_sc (cho MLP).

model.py
  Định nghĩa PyTorch: ResidualBlock, TabularDeepCreditNet (embedding + MLP + residual + đầu softmax hoặc CORAL),
  coral_loss, coral_predict, softmax_focal_loss (hỗ trợ trọng số theo mẫu cho nhãn lớp cuối). Phụ thuộc torch.

metrics.py
  compute_metrics (cơ bản); compute_extended_metrics (thêm per_class, high_risk_class,
  MAE, balanced_accuracy); format_confusion_matrix_text; format_report (classification_report sklearn).

engine.py
  Luồng chính: parse_train_args / parse_validate_args / parse_test_args; run_train, run_validate, run_test;
  đọc CSV, gọi pipeline, vòng huấn luyện/early stopping, lưu model.pt + artifacts.joblib + metadata;
  load checkpoint cho validate/test. Entry từ mlp_softmax/*.py và mlp_coral/*.py gọi vào đây.

run_training.py
  Tương thích ngược: re-export run_train từ engine; có thể chạy python -m deep_credit_rating.common.run_training.
  Khuyến nghị dùng trực tiếp mlp_softmax/train.py hoặc engine.run_train.

Gợi ý thứ tự phụ thuộc (mức khái niệm)
--------------------------------------
  config, env_setup  →  labels, preprocess  →  pipeline  →  model, metrics  →  engine  →  run_training

Tài liệu tổng thể dự án: README.md và ARCHITECTURE.md ở gốc repo.
