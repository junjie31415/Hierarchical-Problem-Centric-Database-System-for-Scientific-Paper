import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from sklearn.ensemble import ExtraTreesClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from weak_methods.run_weak_methods import (
    LABELS,
    LABEL_TO_ID,
    build_paper_cache,
    compute_metrics,
    feature_vector,
    fit_idf,
    pair_feature_dict,
    read_jsonl,
    standardize,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train static ExtraTrees pair model.")
    parser.add_argument("--train-papers", type=Path, default=Path("dataset/train_papers.jsonl"))
    parser.add_argument("--train-labels", type=Path, default=Path("dataset/train_labels.jsonl"))
    parser.add_argument("--valid-papers", type=Path, default=Path("dataset/valid_papers.jsonl"))
    parser.add_argument("--valid-labels", type=Path, default=Path("dataset/valid_labels.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("dbsystem/outputs/pair_model_static_extratrees"))
    parser.add_argument("--max-reference-titles", type=int, default=80)
    parser.add_argument("--tree-estimators", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def build_examples(
    labels: list[dict[str, Any]],
    paper_cache: dict[str, dict[str, Any]],
    idf: dict[str, float],
    default_idf: float,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    rows: list[dict[str, Any]] = []
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    for row in labels:
        features = pair_feature_dict(
            paper_cache[row["paper_a_id"]],
            paper_cache[row["paper_b_id"]],
            idf,
            default_idf,
        )
        rows.append(
            {
                "pair_id": row["pair_id"],
                "paper_a_id": row["paper_a_id"],
                "paper_b_id": row["paper_b_id"],
                "true_label": row["label"],
            }
        )
        x_rows.append(feature_vector(features, "static"))
        y_rows.append(LABEL_TO_ID[row["label"]])
    return rows, torch.tensor(x_rows, dtype=torch.float), torch.tensor(y_rows, dtype=torch.long)


def predict_rows(model: ExtraTreesClassifier, x_tensor: torch.Tensor, rows: list[dict[str, Any]]) -> list[str]:
    predictions = model.predict(x_tensor.detach().cpu().numpy()).tolist()
    return [LABELS[int(index)] for index in predictions]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    train_papers = read_jsonl(args.train_papers)
    train_labels = read_jsonl(args.train_labels)
    valid_papers = read_jsonl(args.valid_papers)
    valid_labels = read_jsonl(args.valid_labels)

    idf = fit_idf(train_papers, args.max_reference_titles)
    default_idf = torch.log(torch.tensor((1.0 + len(train_papers)) / 1.0)).item() + 1.0
    train_cache = build_paper_cache(train_papers, idf, default_idf, args.max_reference_titles)
    valid_cache = build_paper_cache(valid_papers, idf, default_idf, args.max_reference_titles)
    train_rows, train_x, train_y = build_examples(train_labels, train_cache, idf, default_idf)
    valid_rows, valid_x, valid_y = build_examples(valid_labels, valid_cache, idf, default_idf)
    train_x_std, valid_x_std, mean, std = standardize(train_x, valid_x)

    model = ExtraTreesClassifier(
        n_estimators=args.tree_estimators,
        min_samples_leaf=2,
        n_jobs=args.num_workers,
        random_state=args.seed,
    )
    fit_start = time.perf_counter()
    model.fit(train_x_std.detach().cpu().numpy(), train_y.detach().cpu().numpy())
    fit_time = time.perf_counter() - fit_start

    train_pred = predict_rows(model, train_x_std, train_rows)
    valid_pred = predict_rows(model, valid_x_std, valid_rows)
    train_metrics = compute_metrics(train_rows, train_pred)
    valid_metrics = compute_metrics(valid_rows, valid_pred)

    artifact = {
        "model": model,
        "idf": idf,
        "default_idf": default_idf,
        "mean": mean.detach().cpu().numpy(),
        "std": std.detach().cpu().numpy(),
        "labels": LABELS,
        "label_to_id": LABEL_TO_ID,
        "max_reference_titles": args.max_reference_titles,
        "feature_set": "static",
        "feature_names": [
            "title_jaccard",
            "abstract_jaccard",
            "reference_jaccard",
            "reference_overlap",
            "content_cosine",
            "title_a_in_b",
            "title_b_in_a",
            "year_diff_a_minus_b",
            "reference_count_diff_a_minus_b",
            "title_len_diff_a_minus_b",
            "venue_match",
        ],
        "train_paper_cache": train_cache,
    }
    model_path = args.output_dir / "model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(artifact, handle)

    summary = {
        "model_path": str(model_path),
        "pair_model": "ExtraTreesClassifier",
        "feature_set": "static_11",
        "train_papers": len(train_papers),
        "train_examples": len(train_rows),
        "valid_examples": len(valid_rows),
        "label_counts": dict(Counter(row["true_label"] for row in train_rows)),
        "tree_estimators": args.tree_estimators,
        "fit_time_seconds": fit_time,
        "total_time_seconds": time.perf_counter() - start,
        "train": {
            "accuracy": train_metrics["accuracy"],
            "macro_f1": train_metrics["macro_f1"],
            "per_class": train_metrics["per_class"],
        },
        "valid": {
            "accuracy": valid_metrics["accuracy"],
            "macro_f1": valid_metrics["macro_f1"],
            "per_class": valid_metrics["per_class"],
        },
    }
    write_json(args.output_dir / "model_summary.json", summary)
    print(json.dumps({"event": "model_saved", **summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
