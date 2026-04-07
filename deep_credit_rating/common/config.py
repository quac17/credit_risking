"""Tham số mặc định — có thể ghi đè qua argparse trong train.py."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA = REPO_ROOT / "subset_train_data.csv"
DEFAULT_VALIDATE_DATA = REPO_ROOT / "subset_validate_data.csv"
DEFAULT_TEST_DATA = REPO_ROOT / "subset_test_data.csv"
DEFAULT_OUT = REPO_ROOT / "deep_credit_rating" / "outputs"

NUM_CLASSES = 5
EMB_DIM = 16
HIDDEN_DIMS = (256, 128, 64)
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4
LR = 1e-3
EPOCHS = 80
BATCH_SIZE = 512
EARLY_STOP_PATIENCE = 12
VAL_RATIO = 0.2
SEED = 42

# Nhân thêm trọng số loss cho lớp nguy cơ cao nhất (index NUM_CLASSES-1) khi train Softmax — mặc định 1.0.
# Thử 1.5–3.0 nếu recall lớp 4 thấp (đánh đổi có thể tăng cảnh báo nhầm sang lớp 4).
HIGH_RISK_CLASS_WEIGHT_BOOST = 1.0

# Softmax — mặc định ưu tiên nhãn lớp nguy cơ cao nhất (index NUM_CLASSES-1):
# balanced: WeightedRandomSampler theo nghịch tần suất lớp; với Softmax dùng CE/focal không trọng số lớp.
TRAIN_SAMPLER = "balanced"

# Softmax: focal loss gamma (0 = tắt).
FOCAL_GAMMA = 1.5

# Softmax: nhân loss trên từng mẫu có nhãn lớp cao nhất.
CLASS_4_SAMPLE_WEIGHT = 10.0

# Cột không dùng làm đặc trưng
DROP_COLS = ("SK_ID_CURR", "TARGET")

# Biến phân loại điển hình Home Credit Top-20 (có thể mở rộng)
DEFAULT_CAT_COLS = (
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
)
