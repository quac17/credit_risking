"""
Train / validate / test cho MLP Softmax và CORAL. Import torch chỉ trong từng hàm sau apply_before_torch.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from deep_credit_rating.common import config as cfg
from deep_credit_rating.common.env_setup import apply_before_torch
from deep_credit_rating.common.pipeline import fit_training_pipeline, transform_eval


def _apply_softmax_high_risk_logit_bias(logits, head: str, bias: float):
    """Cộng bias vào logit lớp nguy cơ cao nhất (index K-1) trước softmax/argmax."""
    if head != "softmax" or abs(bias) < 1e-12:
        return logits
    out = logits.clone()
    out[:, -1] = out[:, -1] + float(bias)
    return out


def _batch_class4_loss_weights(y, multiply: float, num_classes: int):
    """
    Trọng số loss theo từng mẫu (Softmax): nhân `multiply` cho mọi mẫu có nhãn lớp cao nhất (index num_classes-1).
    Các mẫu khác hệ số 1 — ưu tiên gradient trên “đúng là nhãn 4” trước khi tối ưu 0–3.
    """
    if multiply == 1.0:
        return None
    import torch

    hi = num_classes - 1
    return torch.where(
        y == hi,
        torch.full_like(y, multiply, dtype=torch.float32),
        torch.ones_like(y, dtype=torch.float32),
    )


def set_seed(seed: int, np_mod, torch_mod) -> None:
    random.seed(seed)
    np_mod.random.seed(seed)
    torch_mod.manual_seed(seed)
    if torch_mod.cuda.is_available():
        torch_mod.cuda.manual_seed_all(seed)


def parse_train_args(head_default: str = "softmax") -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Huấn luyện xếp hạng tín dụng (deep learning)")
    p.add_argument("--data", type=Path, default=cfg.DEFAULT_DATA, help="CSV train có TARGET")
    p.add_argument("--out", type=Path, default=None, help="Thư mục checkpoint")
    p.add_argument("--head", type=str, default=head_default, choices=("softmax", "coral"))
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=cfg.LR)
    p.add_argument("--weight-decay", type=float, default=cfg.WEIGHT_DECAY)
    p.add_argument("--val-ratio", type=float, default=cfg.VAL_RATIO)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--early-stop", type=int, default=cfg.EARLY_STOP_PATIENCE)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument(
        "--high-risk-weight-boost",
        type=float,
        default=cfg.HIGH_RISK_CLASS_WEIGHT_BOOST,
        help="Softmax + train_sampler=shuffle: nhân thêm trọng số CE cho lớp cuối. Bị bỏ qua khi train_sampler=balanced.",
    )
    p.add_argument(
        "--train-sampler",
        type=str,
        default=cfg.TRAIN_SAMPLER if head_default == "softmax" else "shuffle",
        choices=("shuffle", "balanced"),
        help="balanced = lấy mẫu theo nghịch tần suất lớp (mặc định: softmax=cấu hình repo, coral=shuffle).",
    )
    p.add_argument(
        "--focal-gamma",
        type=float,
        default=cfg.FOCAL_GAMMA if head_default == "softmax" else 0.0,
        help="Softmax: gamma focal loss trên CE (0=tắt). Mặc định softmax theo config.",
    )
    p.add_argument(
        "--class-4-sample-weight",
        type=float,
        default=cfg.CLASS_4_SAMPLE_WEIGHT if head_default == "softmax" else 1.0,
        help="Softmax: nhân loss mẫu nhãn lớp cuối (mặc định softmax theo config). CORAL bỏ qua.",
    )
    args = p.parse_args()
    if args.out is None:
        sub = "mlp_softmax" if args.head == "softmax" else "mlp_coral"
        args.out = cfg.DEFAULT_OUT / sub
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    return args


def parse_validate_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Đánh giá trên tập validation (có TARGET)")
    p.add_argument("--data", type=Path, default=cfg.DEFAULT_VALIDATE_DATA)
    p.add_argument("--checkpoint-dir", type=Path, required=True, help="Thư mục có model.pt + artifacts.joblib")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--out", type=Path, default=None, help="Ghi metrics JSON (mặc định: checkpoint-dir/validation_metrics.json)")
    p.add_argument(
        "--high-risk-logit-bias",
        type=float,
        default=0.0,
        help="Softmax: cộng vào logit lớp cuối trước argmax (vd. 0.3–1.5 tăng recall lớp 4; không cần train lại)",
    )
    args = p.parse_args()
    args.checkpoint_dir = Path(args.checkpoint_dir)
    if args.out is None:
        args.out = args.checkpoint_dir / "validation_metrics.json"
    return args


def parse_test_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Suy luận tập test (TARGET tuỳ chọn)")
    p.add_argument("--data", type=Path, default=cfg.DEFAULT_TEST_DATA)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV dự đoán (mặc định: checkpoint-dir/predictions_test.csv)",
    )
    p.add_argument(
        "--high-risk-logit-bias",
        type=float,
        default=0.0,
        help="Softmax: cộng vào logit lớp cuối trước argmax / softmax (đồng bộ pred và proba)",
    )
    args = p.parse_args()
    args.checkpoint_dir = Path(args.checkpoint_dir)
    if args.out is None:
        args.out = args.checkpoint_dir / "predictions_test.csv"
    return args


def _load_torch_bundle(checkpoint_dir: Path):
    apply_before_torch()
    import torch

    from deep_credit_rating.common.model import TabularDeepCreditNet

    p = checkpoint_dir / "model.pt"
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(p, map_location="cpu")
    artifacts = joblib.load(checkpoint_dir / "artifacts.joblib")
    pre = artifacts["preprocessor"]
    scaler = artifacts["numeric_scaler"]
    labeler = artifacts["labeler"]
    head = ck["head"]
    card = ck["cat_cardinalities"]
    model = TabularDeepCreditNet(
        cat_cardinalities=card,
        num_numeric=len(ck["num_cols"]),
        emb_dim=ck["emb_dim"],
        hidden_dims=tuple(ck["hidden_dims"]),
        num_classes=ck["num_classes"],
        dropout=cfg.DROPOUT,
        head=head,
    )
    model.load_state_dict(ck["state_dict"])
    return torch, model, pre, scaler, labeler, head, ck


def run_train(default_head: str = "softmax") -> None:
    apply_before_torch()
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    from deep_credit_rating.common.metrics import (
        compute_extended_metrics,
        format_confusion_matrix_text,
        format_report,
    )
    from deep_credit_rating.common.model import (
        TabularDeepCreditNet,
        coral_loss,
        coral_predict,
        softmax_focal_loss,
    )
    from deep_credit_rating.common.pipeline import fit_training_pipeline

    args = parse_train_args(default_head)
    set_seed(args.seed, np, torch)

    print(
        "\n=== Cấu hình huấn luyện ===\n"
        f"  data={args.data}\n"
        f"  out={args.out}\n"
        f"  head={args.head}  epochs={args.epochs}  batch_size={args.batch_size}  seed={args.seed}  max_rows={args.max_rows}\n"
    )
    if args.head == "softmax":
        print(
            f"  train_sampler={args.train_sampler}  focal_gamma={args.focal_gamma}\n"
            f"  class_4_sample_weight={args.class_4_sample_weight}  "
            f"high_risk_weight_boost={args.high_risk_weight_boost}\n"
        )
        if (
            args.class_4_sample_weight == 1.0
            and args.focal_gamma == 0
            and args.train_sampler == "shuffle"
        ):
            print(
                "  (Chế độ train softmax không ưu tiên nhãn 4 mạnh — khác mặc định repo; "
                "để mặc định ưu tiên, bỏ các cờ này hoặc dùng config.)\n"
            )

    df = pd.read_csv(args.data, nrows=args.max_rows)
    pre, scaler, labeler, X_cat, X_num_sc, y_labels, y_bin = fit_training_pipeline(df)

    idx = np.arange(len(df))
    tr_idx, va_idx = train_test_split(
        idx, test_size=args.val_ratio, random_state=args.seed, stratify=y_labels
    )

    X_num_t = torch.tensor(X_num_sc, dtype=torch.float32)
    X_cat_t = torch.tensor(X_cat, dtype=torch.long)
    y_t = torch.tensor(y_labels, dtype=torch.long)

    ds_tr = TensorDataset(X_cat_t[tr_idx], X_num_t[tr_idx], y_t[tr_idx])
    ds_va = TensorDataset(X_cat_t[va_idx], X_num_t[va_idx], y_t[va_idx])
    if args.train_sampler == "balanced":
        y_tr_arr = y_labels[tr_idx]
        cnt = np.bincount(y_tr_arr, minlength=cfg.NUM_CLASSES)
        inv = 1.0 / np.maximum(cnt[y_tr_arr], 1)
        samp_w = torch.as_tensor(inv, dtype=torch.double)
        gen = torch.Generator()
        gen.manual_seed(int(args.seed))
        bal_sampler = WeightedRandomSampler(
            samp_w,
            num_samples=len(y_tr_arr),
            replacement=True,
            generator=gen,
        )
        dl_tr = DataLoader(
            ds_tr,
            batch_size=args.batch_size,
            sampler=bal_sampler,
            drop_last=False,
        )
        print(
            "train_sampler=balanced: mỗi epoch lấy mẫu có trọng số nghịch tần suất lớp "
            f"(count/lớp trên split train: {cnt.tolist()})."
        )
    else:
        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA không khả dụng — chuyển sang cpu.")
        device = "cpu"

    card = pre.cat_cardinalities()
    model = TabularDeepCreditNet(
        cat_cardinalities=card,
        num_numeric=len(pre.num_cols),
        emb_dim=cfg.EMB_DIM,
        hidden_dims=cfg.HIDDEN_DIMS,
        num_classes=cfg.NUM_CLASSES,
        dropout=cfg.DROPOUT,
        head=args.head,
    ).to(device)

    y_tr_labels = y_labels[tr_idx]
    counts = np.bincount(y_tr_labels, minlength=cfg.NUM_CLASSES)
    w = len(y_tr_labels) / (cfg.NUM_CLASSES * np.maximum(counts, 1))
    use_balanced_sampler = args.train_sampler == "balanced"
    sw = None
    if args.head == "softmax":
        if use_balanced_sampler:
            print(
                "Softmax + balanced sampler: không dùng trọng số lớp trong CE/focal "
                "(tránh chồng với oversampling theo lớp)."
            )
        else:
            if args.high_risk_weight_boost != 1.0:
                w = w.copy()
                w[-1] *= float(args.high_risk_weight_boost)
                print(
                    f"high_risk_weight_boost={args.high_risk_weight_boost} "
                    f"(trọng số CE lớp {cfg.NUM_CLASSES - 1} sau boost: {w[-1]:.4f})"
                )
            sw = torch.tensor(w, dtype=torch.float32, device=device)
        if args.focal_gamma > 0:
            print(f"focal_gamma={args.focal_gamma} (focal loss trên CE đa lớp)")
        if args.class_4_sample_weight != 1.0:
            print(
                f"class_4_sample_weight={args.class_4_sample_weight} "
                f"(nhân loss mọi mẫu nhãn lớp {cfg.NUM_CLASSES - 1})"
            )
    elif args.head == "coral" and args.class_4_sample_weight != 1.0:
        print("Cảnh báo: --class-4-sample-weight chỉ áp dụng cho head=softmax; bỏ qua.")

    if args.head == "softmax" and not use_balanced_sampler:
        ce_weights_for_meta = sw.detach().cpu().tolist()  # type: ignore[union-attr]
    elif args.head == "softmax":
        ce_weights_for_meta = [1.0] * cfg.NUM_CLASSES
    else:
        ce_weights_for_meta = w.tolist()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)

    best_val = float("inf")
    best_state = None
    bad = 0
    last_epoch = 0

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        model.train()
        total = 0.0
        for xc, xn, y in dl_tr:
            xc, xn, y = xc.to(device), xn.to(device), y.to(device)
            opt.zero_grad()
            logits = model(xn, xc)
            if args.head == "softmax":
                bw = _batch_class4_loss_weights(y, args.class_4_sample_weight, cfg.NUM_CLASSES)
                loss = softmax_focal_loss(logits, y, args.focal_gamma, sw, bw)
            else:
                loss = coral_loss(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(y)

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xc, xn, y in dl_va:
                xc, xn, y = xc.to(device), xn.to(device), y.to(device)
                logits = model(xn, xc)
                if args.head == "softmax":
                    bw = _batch_class4_loss_weights(y, args.class_4_sample_weight, cfg.NUM_CLASSES)
                    loss = softmax_focal_loss(logits, y, args.focal_gamma, sw, bw)
                else:
                    loss = coral_loss(logits, y)
                vloss += loss.item() * len(y)
        vloss /= len(va_idx)
        sched.step(vloss)

        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}  train_loss~{total/len(tr_idx):.4f}  val_loss={vloss:.4f}")
        if bad >= args.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xc, xn, y in dl_va:
            xc, xn = xc.to(device), xn.to(device)
            logits = model(xn, xc)
            if args.head == "softmax":
                pred = logits.argmax(dim=1).cpu().numpy()
            else:
                pred = coral_predict(logits).cpu().numpy()
            all_pred.append(pred)
            all_true.append(y.numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    metrics = compute_extended_metrics(y_true, y_pred, cfg.NUM_CLASSES)
    (args.out / "holdout_confusion_matrix.txt").write_text(
        format_confusion_matrix_text(y_true, y_pred, cfg.NUM_CLASSES), encoding="utf-8"
    )

    ckpt_path = args.out / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "head": args.head,
            "cat_cols": pre.cat_cols,
            "num_cols": pre.num_cols,
            "cat_cardinalities": card,
            "num_classes": cfg.NUM_CLASSES,
            "hidden_dims": cfg.HIDDEN_DIMS,
            "emb_dim": cfg.EMB_DIM,
        },
        ckpt_path,
    )
    meta = {
        "data": str(args.data),
        "head": args.head,
        "train_sampler": args.train_sampler,
        "val_ratio": args.val_ratio,
        "epochs_ran": last_epoch,
        "metrics_holdout": metrics,
        "class_counts_train_split": counts.tolist(),
        "high_risk_weight_boost": getattr(args, "high_risk_weight_boost", 1.0),
        "focal_gamma": getattr(args, "focal_gamma", 0.0),
        "class_4_sample_weight": getattr(args, "class_4_sample_weight", 1.0),
        "class_weights_effective": ce_weights_for_meta,
    }
    (args.out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (args.out / "classification_report_holdout.txt").write_text(
        format_report(y_true, y_pred), encoding="utf-8"
    )
    joblib.dump(
        {"preprocessor": pre, "numeric_scaler": scaler, "labeler": labeler},
        args.out / "artifacts.joblib",
    )
    print("Holdout validation metrics:", json.dumps(metrics, indent=2, ensure_ascii=False))
    print("Saved:", ckpt_path, "artifacts.joblib")


def run_validate() -> None:
    apply_before_torch()
    import torch

    from deep_credit_rating.common.metrics import (
        compute_extended_metrics,
        format_confusion_matrix_text,
        format_report,
    )
    from deep_credit_rating.common.model import coral_predict

    args = parse_validate_args()
    torch_mod, model, pre, scaler, labeler, head, _ = _load_torch_bundle(args.checkpoint_dir)
    device = args.device
    if device == "cuda" and not torch_mod.cuda.is_available():
        device = "cpu"
    model = model.to(device)
    model.eval()

    df = pd.read_csv(args.data, nrows=args.max_rows)
    if "TARGET" not in df.columns:
        raise SystemExit("Validation cần cột TARGET.")
    X_cat, X_num_sc, X_num_raw = transform_eval(df, pre, scaler)
    y_bin = df["TARGET"].values.astype(np.int64)
    y_true = labeler.transform(X_num_raw, y_bin)

    X_num_t = torch.tensor(X_num_sc, dtype=torch.float32)
    X_cat_t = torch.tensor(X_cat, dtype=torch.long)

    all_pred = []
    with torch_mod.no_grad():
        for i in range(0, len(df), 512):
            sl = slice(i, i + 512)
            xc = X_cat_t[sl].to(device)
            xn = X_num_t[sl].to(device)
            logits = model(xn, xc)
            logits = _apply_softmax_high_risk_logit_bias(logits, head, args.high_risk_logit_bias)
            if head == "softmax":
                pred = logits.argmax(dim=1).cpu().numpy()
            else:
                pred = coral_predict(logits).cpu().numpy()
            all_pred.append(pred)
    y_pred = np.concatenate(all_pred)
    metrics = compute_extended_metrics(y_true, y_pred, cfg.NUM_CLASSES)
    out = {
        "data": str(args.data),
        "head": head,
        "checkpoint_dir": str(args.checkpoint_dir),
        "high_risk_logit_bias": args.high_risk_logit_bias,
        "metrics": metrics,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    cm_path = Path(args.out).parent / "validation_confusion_matrix.txt"
    cm_path.write_text(format_confusion_matrix_text(y_true, y_pred, cfg.NUM_CLASSES), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("Confusion matrix:", cm_path)
    rep_path = Path(args.out).parent / "validation_classification_report.txt"
    rep_path.write_text(format_report(y_true, y_pred), encoding="utf-8")
    print("Report:", rep_path)


def run_test() -> None:
    apply_before_torch()
    import torch

    from deep_credit_rating.common.metrics import (
        compute_extended_metrics,
        format_confusion_matrix_text,
        format_report,
    )
    from deep_credit_rating.common.model import coral_predict

    args = parse_test_args()
    torch_mod, model, pre, scaler, labeler, head, _ = _load_torch_bundle(args.checkpoint_dir)
    device = args.device
    if device == "cuda" and not torch_mod.cuda.is_available():
        device = "cpu"
    model = model.to(device)
    model.eval()

    df = pd.read_csv(args.data, nrows=args.max_rows)
    X_cat, X_num_sc, X_num_raw = transform_eval(df, pre, scaler)
    has_target = "TARGET" in df.columns
    y_true = None
    if has_target:
        y_bin = df["TARGET"].values.astype(np.int64)
        y_true = labeler.transform(X_num_raw, y_bin)

    X_num_t = torch.tensor(X_num_sc, dtype=torch.float32)
    X_cat_t = torch.tensor(X_cat, dtype=torch.long)

    rows = []
    with torch_mod.no_grad():
        for i in range(0, len(df), 512):
            sl = slice(i, i + 512)
            xc = X_cat_t[sl].to(device)
            xn = X_num_t[sl].to(device)
            logits = model(xn, xc)
            logits = _apply_softmax_high_risk_logit_bias(logits, head, args.high_risk_logit_bias)
            if head == "softmax":
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                pred = logits.argmax(dim=1).cpu().numpy()
            else:
                pred = coral_predict(logits).cpu().numpy()
                proba = None
            for j in range(len(pred)):
                r = {"pred_class": int(pred[j]), "rating_1_to_5": int(pred[j]) + 1}
                if "SK_ID_CURR" in df.columns:
                    r["SK_ID_CURR"] = df["SK_ID_CURR"].iloc[i + j]
                if proba is not None:
                    for k in range(proba.shape[1]):
                        r[f"proba_class_{k}"] = float(proba[j, k])
                rows.append(r)

    out_df = pd.DataFrame(rows)
    if has_target and y_true is not None:
        out_df["y_true_class"] = y_true
        metrics = compute_extended_metrics(y_true, out_df["pred_class"].values, cfg.NUM_CLASSES)
        print("Test metrics (có TARGET):", json.dumps(metrics, indent=2, ensure_ascii=False))
        (args.checkpoint_dir / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (args.checkpoint_dir / "test_confusion_matrix.txt").write_text(
            format_confusion_matrix_text(y_true, out_df["pred_class"].values, cfg.NUM_CLASSES),
            encoding="utf-8",
        )
        print(format_report(y_true, out_df["pred_class"].values))

    out_df.to_csv(args.out, index=False)
    print("Saved predictions:", args.out)
