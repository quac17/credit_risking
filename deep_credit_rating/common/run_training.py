"""
Tương thích ngược: huấn luyện (softmax mặc định nếu gọi trực tiếp).
Ưu tiên dùng mlp_softmax/train.py hoặc common.engine.run_train.
"""
from __future__ import annotations

from deep_credit_rating.common.engine import run_train as main

if __name__ == "__main__":
    main()
