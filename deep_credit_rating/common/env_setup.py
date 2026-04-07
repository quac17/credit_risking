"""
Gọi trước khi `import torch` lần đầu — giảm segfault / xung đột OpenMP-MKL trên Windows.
"""


def apply_before_torch() -> None:
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    # Một số bản PyTorch + MKL: giảm đụng độ luồng
    os.environ.setdefault("MKL_NUM_THREADS", "1")
