import argparse
import csv
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from weak_methods.run_weak_methods import (
    LABELS,
    compute_metrics,
    package_available,
    parse_backend_list,
    tabular_backend_package,
    train_backend,
    train_tabular_backend,
    write_json,
    write_jsonl,
)


DEFAULT_BACKENDS = "mlp,random_forest,extra_trees"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train backends on BERT chunk features.")
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("baseline/outputs/fixed_5x5_bert_features"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("baseline/outputs/bert_fixed5x5_backend_comparison"),
    )
    parser.add_argument(
        "--static-metrics",
        type=Path,
        default=Path("weak_methods/outputs/default/all_metrics.json"),
    )
    parser.add_argument("--backends", default=DEFAULT_BACKENDS)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tree-estimators", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument(
        "--write-train-predictions",
        action="store_true",
        help="Also write large train_predictions.jsonl files.",
    )
    return parser.parse_args()


def load_feature_split(path: Path) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    data = torch.load(path, map_location="cpu")
    x = data["x"].reshape(data["x"].shape[0], -1).float()
    y = data["y"].long()
    rows = [
        {
            "pair_id": pair_id,
            "split": path.stem,
            "paper_a_id": paper_a_id,
            "paper_b_id": paper_b_id,
            "paper_a_title": None,
            "paper_b_title": None,
            "true_label": label,
        }
        for pair_id, paper_a_id, paper_b_id, label in zip(
            data["pair_ids"],
            data["paper_a_ids"],
            data["paper_b_ids"],
            data["labels"],
        )
    ]
    return x, y, rows


def standardize_in_place(
    train_x: torch.Tensor,
    valid_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_(min=1e-6)
    train_x.sub_(mean).div_(std)
    valid_x.sub_(mean).div_(std)
    return mean.squeeze(0), std.squeeze(0)


def annotate_predictions(
    rows: list[dict[str, Any]],
    predictions: list[str],
    probability_rows: list[dict[str, float]],
    method: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row, pred, probs in zip(rows, predictions, probability_rows):
        item = {
            "pair_id": row["pair_id"],
            "split": row["split"],
            "paper_a_id": row["paper_a_id"],
            "paper_b_id": row["paper_b_id"],
            "true_label": row["true_label"],
            "predicted_label": pred,
            "method": method,
        }
        for label in LABELS:
            item[f"prob_{label}"] = float(probs.get(label, 0.0))
        out.append(item)
    return out


def write_method_result(
    output_dir: Path,
    method_name: str,
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    train_predictions: list[str],
    train_probs: list[dict[str, float]],
    valid_predictions: list[str],
    valid_probs: list[dict[str, float]],
    extra: dict[str, Any],
    write_train_predictions: bool,
) -> dict[str, Any]:
    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        method_dir / "valid_predictions.jsonl",
        annotate_predictions(valid_rows, valid_predictions, valid_probs, method_name),
    )
    if write_train_predictions:
        write_jsonl(
            method_dir / "train_predictions.jsonl",
            annotate_predictions(train_rows, train_predictions, train_probs, method_name),
        )
    result = {
        "method": method_name,
        "train": compute_metrics(train_rows, train_predictions),
        "valid": compute_metrics(valid_rows, valid_predictions),
        "extra": extra,
    }
    write_json(method_dir / "metrics.json", result)
    return result


def load_static_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("methods", {})


def write_comparison(
    output_dir: Path,
    bert_methods: dict[str, Any],
    static_methods: dict[str, Any],
) -> None:
    rows: list[dict[str, Any]] = []
    for method_name, bert_result in sorted(bert_methods.items()):
        backend = method_name
        static_result = static_methods.get(method_name)
        row = {
            "backend": backend,
            "bert_valid_accuracy": bert_result["valid"]["accuracy"],
            "bert_valid_macro_f1": bert_result["valid"]["macro_f1"],
            "static_valid_accuracy": None,
            "static_valid_macro_f1": None,
            "macro_f1_delta_bert_minus_static": None,
        }
        if static_result:
            row["static_valid_accuracy"] = static_result["valid"]["accuracy"]
            row["static_valid_macro_f1"] = static_result["valid"]["macro_f1"]
            row["macro_f1_delta_bert_minus_static"] = (
                row["bert_valid_macro_f1"] - row["static_valid_macro_f1"]
            )
        rows.append(row)

    with (output_dir / "bert_vs_static_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "backend",
                "bert_valid_accuracy",
                "bert_valid_macro_f1",
                "static_valid_accuracy",
                "static_valid_macro_f1",
                "macro_f1_delta_bert_minus_static",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "bert_vs_static_summary.json", rows)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y, train_rows = load_feature_split(args.feature_dir / "train.pt")
    valid_x, valid_y, valid_rows = load_feature_split(args.feature_dir / "valid.pt")
    mean, std = standardize_in_place(train_x, valid_x)

    backend_args = SimpleNamespace(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        tree_estimators=args.tree_estimators,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )
    results: dict[str, Any] = {
        "config": {
            "feature_dir": str(args.feature_dir),
            "output_dir": str(args.output_dir),
            "static_metrics": str(args.static_metrics),
            "train_examples": len(train_rows),
            "valid_examples": len(valid_rows),
            "input_shape_after_flatten": list(train_x.shape),
            "source_feature_shape": "[N, 10, 768]",
            "backend_input_note": "Backends receive flattened BERT chunk features [N, 7680].",
            "labels": list(LABELS),
            "backends": parse_backend_list(args.backends),
            "tree_estimators": args.tree_estimators,
            "epochs": args.epochs,
            "device": args.device,
        },
        "methods": {},
        "skipped_methods": {},
    }

    write_json(
        args.output_dir / "feature_standardization_summary.json",
        {
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
            "std_min": float(std.min().item()),
            "std_max": float(std.max().item()),
        },
    )

    for backend in parse_backend_list(args.backends):
        method_name = backend
        try:
            if backend == "mlp":
                train_pred, train_prob, valid_pred, valid_prob, backend_summary = train_backend(
                    backend,
                    train_x,
                    train_y,
                    valid_x,
                    backend_args,
                )
            else:
                package_name = tabular_backend_package(backend)
                if not package_available(package_name):
                    raise ImportError(f"Missing package: {package_name}")
                train_pred, train_prob, valid_pred, valid_prob, backend_summary = (
                    train_tabular_backend(
                        backend,
                        train_x,
                        train_y,
                        valid_x,
                        backend_args,
                    )
                )
            result = write_method_result(
                args.output_dir,
                method_name,
                train_rows,
                valid_rows,
                train_pred,
                train_prob,
                valid_pred,
                valid_prob,
                backend_summary,
                args.write_train_predictions,
            )
            results["methods"][method_name] = result
            print(
                json.dumps(
                    {
                        "event": "method_done",
                        "method": method_name,
                        "valid_accuracy": result["valid"]["accuracy"],
                        "valid_macro_f1": result["valid"]["macro_f1"],
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            results["skipped_methods"][method_name] = repr(exc)
            print(
                json.dumps(
                    {
                        "event": "method_failed",
                        "method": method_name,
                        "error": repr(exc),
                    }
                ),
                flush=True,
            )
        write_json(args.output_dir / "all_metrics.partial.json", results)

    write_json(args.output_dir / "all_metrics.json", results)
    static_methods = load_static_metrics(args.static_metrics)
    write_comparison(args.output_dir, results["methods"], static_methods)
    print(json.dumps({"event": "done", "output_dir": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
