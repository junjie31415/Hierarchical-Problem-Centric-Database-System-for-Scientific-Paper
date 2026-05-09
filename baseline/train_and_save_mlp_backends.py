import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Union

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from weak_methods.run_weak_methods import (
    LABELS,
    FeatureClassifier,
    build_paper_cache,
    class_weights,
    compute_metrics,
    fit_idf,
    load_split_examples,
    read_jsonl,
    standardize,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and save MLP backend models.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--bert-feature-dir",
        type=Path,
        default=Path("baseline/outputs/fixed_5x5_bert_features"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("baseline/outputs/saved_mlp_backends"),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-reference-titles", type=int, default=80)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_mlp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    valid_x: torch.Tensor,
    valid_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[FeatureClassifier, dict[str, Any]]:
    model = FeatureClassifier(train_x.shape[1], "mlp", args.hidden_size, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_y, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_x_device = train_x.to(device)
    train_y_device = train_y.to(device)
    history: list[dict[str, Union[float, int]]] = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(train_x_device.shape[0], device=device)
        total_loss = 0.0
        total_seen = 0
        for batch_start in range(0, train_x_device.shape[0], args.batch_size):
            indices = permutation[batch_start : batch_start + args.batch_size]
            logits = model(train_x_device[indices])
            targets = train_y_device[indices]
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item()) * int(indices.shape[0])
            total_seen += int(indices.shape[0])
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            history.append({"epoch": epoch, "train_loss": total_loss / total_seen})
    train_seconds = time.perf_counter() - start

    model.eval()
    predictions: list[str] = []
    with torch.inference_mode():
        for batch_start in range(0, valid_x.shape[0], args.batch_size):
            logits = model(valid_x[batch_start : batch_start + args.batch_size].to(device))
            pred_ids = torch.argmax(logits, dim=-1).detach().cpu().tolist()
            predictions.extend(LABELS[int(index)] for index in pred_ids)
    metrics = compute_metrics(valid_rows, predictions)
    summary = {
        "train_seconds": train_seconds,
        "history": history,
        "valid_metrics": metrics,
        "parameter_count": count_parameters(model),
        "input_dim": int(train_x.shape[1]),
    }
    return model, summary


def save_artifact(
    output_dir: Path,
    model: FeatureClassifier,
    mean: torch.Tensor,
    std: torch.Tensor,
    summary: dict[str, Any],
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "mean": mean.cpu(),
            "std": std.cpu(),
            "label_order": list(LABELS),
            "config": config,
            "summary": summary,
        },
        output_dir / "model.pt",
    )
    write_json(output_dir / "summary.json", {"config": config, **summary})


def train_static(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    train_papers = read_jsonl(args.dataset_dir / "train_papers.jsonl")
    valid_papers = read_jsonl(args.dataset_dir / "valid_papers.jsonl")
    train_labels = read_jsonl(args.dataset_dir / "train_labels.jsonl")
    valid_labels = read_jsonl(args.dataset_dir / "valid_labels.jsonl")
    idf = fit_idf(train_papers, args.max_reference_titles)
    default_idf = torch.log(torch.tensor((1.0 + len(train_papers)) / 1.0)).item() + 1.0
    train_cache = build_paper_cache(train_papers, idf, default_idf, args.max_reference_titles)
    valid_cache = build_paper_cache(valid_papers, idf, default_idf, args.max_reference_titles)
    _, train_x, train_y = load_split_examples(train_labels, train_cache, idf, default_idf, "static")
    valid_rows, valid_x, _ = load_split_examples(valid_labels, valid_cache, idf, default_idf, "static")
    train_x_std, valid_x_std, mean, std = standardize(train_x, valid_x)
    model, summary = train_mlp(train_x_std, train_y, valid_x_std, valid_rows, args, device)
    config = {
        "feature_source": "static_11_features",
        "input_dim": 11,
        "backend": "mlp",
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_reference_titles": args.max_reference_titles,
        "note": "Static model also requires idf/default_idf for raw feature extraction; this artifact stores only MLP and standardization.",
    }
    save_artifact(args.output_dir / "static_mlp", model, mean, std, summary, config)
    return summary


def feature_rows_from_pt(path: Path) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    data = torch.load(path, map_location="cpu")
    x = data["x"].reshape(data["x"].shape[0], -1).float()
    y = data["y"].long()
    rows = [
        {
            "pair_id": pair_id,
            "split": path.stem,
            "paper_a_id": paper_a_id,
            "paper_b_id": paper_b_id,
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


def train_bert(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    train_x, train_y, _ = feature_rows_from_pt(args.bert_feature_dir / "train.pt")
    valid_x, _, valid_rows = feature_rows_from_pt(args.bert_feature_dir / "valid.pt")
    train_x_std, valid_x_std, mean, std = standardize(train_x, valid_x)
    model, summary = train_mlp(train_x_std, train_y, valid_x_std, valid_rows, args, device)
    config = {
        "feature_source": "bert_fixed_5x5",
        "source_shape": "[N, 10, 768]",
        "backend_input_shape": "[N, 7680]",
        "input_dim": 7680,
        "backend": "mlp",
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    save_artifact(args.output_dir / "bert_fixed5x5_mlp", model, mean, std, summary, config)
    return summary


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    static_summary = train_static(args, device)
    print(
        json.dumps(
            {
                "event": "static_saved",
                "parameter_count": static_summary["parameter_count"],
                "valid_macro_f1": static_summary["valid_metrics"]["macro_f1"],
            }
        ),
        flush=True,
    )
    bert_summary = train_bert(args, device)
    print(
        json.dumps(
            {
                "event": "bert_saved",
                "parameter_count": bert_summary["parameter_count"],
                "valid_macro_f1": bert_summary["valid_metrics"]["macro_f1"],
            }
        ),
        flush=True,
    )
    write_json(
        args.output_dir / "summary.json",
        {
            "static_mlp": static_summary,
            "bert_fixed5x5_mlp": bert_summary,
            "same_architecture": "Both are one-hidden-layer MLPs, but parameter counts differ because input dimensions differ.",
        },
    )


if __name__ == "__main__":
    main()
