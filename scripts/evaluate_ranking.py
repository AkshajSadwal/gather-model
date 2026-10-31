"""Evaluate a ranking checkpoint on what matters for a feed, not only mean ROC-AUC.

Reports, on one chronological split:
  1. Per-target ROC-AUC, PR-AUC, per-user AUC (GAUC, with 95% CI over users), and calibration.
  2. Watch-time fit (MAE, RMSE, R^2 in log-seconds).
  3. NDCG@K when re-ranking each user's logged impressions (blocks of consecutive
     impressions), for the current hand-weighted score, every single predicted
     probability, a popularity baseline and random order. Differences against the
     hand-weighted score carry paired bootstrap CIs.

Validation is the default. The test split is only read with --allow-test, so it is not
used for tuning by accident.

    python -m scripts.evaluate_ranking \
        --model-path artifacts/two_stage_runs_parallel/parallel_finetune10_r42_m1s43_20261031_203323/model2_ranking/best_model.pt \
        --output reports/experiments/ranking_eval_finetune10.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common.data_version import schema_fingerprint
from common.eval_metrics import relevance_gains
from common.ranking_eval import evaluate_ranking_predictions
from model2_ranking.model_v2 import ReelsRankerV2
from model2_ranking.recommend_v2 import SCORE_WEIGHTS


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in keys}


@torch.no_grad()
def predict(model: ReelsRankerV2, categorical: np.ndarray, numeric: np.ndarray,
            device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    n = categorical.shape[0]
    probs = np.empty((n, model.action_head.out_features), dtype=np.float32)
    watch = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        logits, watch_pred = model(torch.from_numpy(categorical[start:end]).to(device),
                                   torch.from_numpy(numeric[start:end]).to(device))
        probs[start:end] = torch.sigmoid(logits).cpu().numpy()
        watch[start:end] = watch_pred.cpu().numpy()
        if start == 0 or (start // batch_size) % 40 == 0:
            print(f"  scored {end:,}/{n:,} rows", flush=True)
    return probs, watch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/processed/ranking/1k_v2")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--allow-test", action="store_true", help="Required to read the held-out test split")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=30,
                        help="Consecutive impressions of one user treated as one re-ranking pool")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-rows", type=int, default=0, help="Use only the first N rows (quick smoke test)")
    parser.add_argument("--skip-popularity", action="store_true",
                        help="Do not load train.npz for the popularity baseline (saves ~2 GB RAM)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("Reading the held-out test split needs --allow-test. Tune on validation, report on test once.")

    started = time.perf_counter()
    data_dir = Path(args.data_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    targets = list(metadata["binary_targets"])
    device = choose_device()
    print(f"Device: {device}")

    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    if checkpoint.get("dataset") != metadata.get("dataset"):
        raise ValueError("Checkpoint dataset does not match the prepared data")
    saved_schema = checkpoint.get("schema_fingerprint")
    if saved_schema is not None and saved_schema != schema_fingerprint(data_dir):
        raise ValueError("Checkpoint feature schema differs from the prepared data")
    if checkpoint.get("binary_targets") != targets:
        raise ValueError("Checkpoint targets differ from the prepared data")
    model = ReelsRankerV2(checkpoint["categorical_sizes"], checkpoint["num_numeric_features"],
                          num_targets=len(targets), embedding_dim=checkpoint["embedding_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    split = load_split(data_dir / f"{args.split}.npz", ("categorical", "numeric", "labels", "watch_log_seconds"))
    if args.max_rows > 0:
        split = {k: v[: args.max_rows] for k, v in split.items()}
    categorical = split["categorical"].astype(np.int32, copy=False)
    numeric = split["numeric"].astype(np.float32, copy=False)
    labels = split["labels"].astype(np.float32, copy=False)
    watch_true = split["watch_log_seconds"].astype(np.float32, copy=False)
    user = categorical[:, 0]
    video = categorical[:, 1]
    print(f"Scoring {categorical.shape[0]:,} {args.split} rows with {Path(args.model_path).parent.name}...")
    probs, watch_pred = predict(model, categorical, numeric, device, args.batch_size)

    popularity = None
    if not args.skip_popularity:
        print("Loading train.npz for the popularity baseline...")
        train = load_split(data_dir / "train.npz", ("categorical", "labels"))
        popularity = np.bincount(train["categorical"][:, 1], weights=relevance_gains(train["labels"], targets),
                                 minlength=int(metadata["categorical_sizes"][1]))
        del train

    print("Computing metrics and bootstrap intervals...")
    report: dict = {
        "model_path": str(args.model_path), "split": args.split, "rows": int(categorical.shape[0]),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_validation_mean_auc": checkpoint.get("validation_score"),
    }
    report.update(evaluate_ranking_predictions(
        probs, watch_pred, labels, watch_true, user, video, targets, SCORE_WEIGHTS,
        popularity_by_video=popularity, block_size=args.block_size, ks=tuple(args.ks),
        n_boot=args.bootstrap, seed=args.seed,
    ))
    per_target = report["per_target"]
    ndcg_report = report["ndcg_reranking_logged_impressions"]["scores"]
    ks = tuple(args.ks)

    # ---- readable summary
    print(f"\n=== Per-target quality on {args.split} ===")
    print(f"{'target':<18}{'AUC':>8}{'GAUC':>8}{'GAUC 95% CI':>18}{'PR-AUC':>9}{'base rate':>11}{'pred/base':>11}{'ECE':>8}")
    for name, v in per_target.items():
        print(f"{name:<18}{v['auc']:>8.4f}{v['gauc']:>8.4f}   [{v['gauc_ci95'][0]:.3f},{v['gauc_ci95'][1]:.3f}]"
              f"{v['pr_auc']:>9.4f}{v['base_rate']:>11.4f}{v['prediction_over_base']:>11.2f}{v['ece']:>8.4f}")
    print(f"mean AUC {report['mean_auc']:.4f} | mean per-user AUC {report['mean_gauc']:.4f}")
    print(f"\nwatch time (log-seconds): {report['watch_time_log_seconds']}")
    print(f"\n=== NDCG, re-ranking each user's logged impressions (higher is better) ===")
    header = "".join(f"{'NDCG@' + str(k):>12}" for k in ks)
    print(f"{'score':<34}{header}   difference vs current hand-weighted score (95% CI)")
    for name, entry in ndcg_report.items():
        cells = "".join(f"{entry[f'@{k}']['ndcg']:>12.4f}" for k in ks)
        diff = ""
        if "diff_vs_hand_weighted" in entry[f"@{ks[0]}"]:
            d = entry[f"@{ks[0]}"]["diff_vs_hand_weighted"]
            flag = " *" if d["ci95"][0] > 0 or d["ci95"][1] < 0 else ""
            diff = f"   {d['mean']:+.4f} [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}]{flag}"
        print(f"{name:<34}{cells}{diff}")
    print("(* = difference at the first K is clearly different from zero)")

    report["elapsed_seconds"] = time.perf_counter() - started
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, allow_nan=True, default=float), encoding="utf-8")
        print(f"\nSaved: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
