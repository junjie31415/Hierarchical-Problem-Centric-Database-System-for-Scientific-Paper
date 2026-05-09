import argparse
import importlib.util
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import nn


LABELS = ("A_TO_B", "B_TO_A", "PEER", "NONE")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "via",
    "with",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".cache" / "matplotlib"),
)


class FeatureClassifier(nn.Module):
    def __init__(self, input_size: int, backend: str, hidden_size: int, dropout: float) -> None:
        super().__init__()
        if backend == "mlp":
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, len(LABELS)),
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run weak/non-embedding methods.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("weak_methods/outputs/default"))
    parser.add_argument("--max-reference-titles", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--tabular-backends",
        type=str,
        default="random_forest,extra_trees",
        help="Comma-separated tabular backends to run after MLP.",
    )
    parser.add_argument("--tree-estimators", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize(text: Any) -> list[str]:
    if text is None:
        return []
    return [
        token
        for token in TOKEN_RE.findall(str(text).lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def unique_tokens(text: Any) -> set[str]:
    return set(tokenize(text))


def clean_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_year(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def reference_text(paper: dict[str, Any], max_reference_titles: int) -> str:
    titles = paper.get("reference_titles") or []
    if not isinstance(titles, list):
        titles = []
    if max_reference_titles > 0:
        titles = titles[:max_reference_titles]
    return " ".join(str(title) for title in titles)


def metadata_text(paper: dict[str, Any], max_reference_titles: int) -> str:
    return " ".join(
        [
            str(paper.get("title") or ""),
            str(paper.get("abstract") or ""),
            str(paper.get("venue") or ""),
            reference_text(paper, max_reference_titles),
        ]
    )


def fit_idf(train_papers: list[dict[str, Any]], max_reference_titles: int) -> dict[str, float]:
    df: Counter[str] = Counter()
    for paper in train_papers:
        df.update(set(tokenize(metadata_text(paper, max_reference_titles))))
    document_count = len(train_papers)
    return {
        token: math.log((1.0 + document_count) / (1.0 + count)) + 1.0
        for token, count in df.items()
    }


def weighted_norm(counts: Counter[str], idf: dict[str, float], default_idf: float) -> float:
    return math.sqrt(sum((count * idf.get(token, default_idf)) ** 2 for token, count in counts.items()))


def build_paper_cache(
    papers: list[dict[str, Any]],
    idf: dict[str, float],
    default_idf: float,
    max_reference_titles: int,
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for paper in papers:
        ref_text = reference_text(paper, max_reference_titles)
        content_counts = Counter(tokenize(metadata_text(paper, max_reference_titles)))
        cache[paper["paper_id"]] = {
            "paper_id": paper["paper_id"],
            "title": paper.get("title"),
            "node_path": paper.get("node_path"),
            "title_tokens": unique_tokens(paper.get("title")),
            "abstract_tokens": unique_tokens(paper.get("abstract")),
            "ref_tokens": unique_tokens(ref_text),
            "content_counts": content_counts,
            "content_norm": weighted_norm(content_counts, idf, default_idf),
            "title_len": len(unique_tokens(paper.get("title"))),
            "ref_count": clean_number(paper.get("reference_count")),
            "year": safe_year(paper.get("publication_year")),
            "venue": str(paper.get("venue") or "").lower().strip(),
        }
    return cache


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def containment(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left) if left else 0.0


def cosine(
    left_counts: Counter[str],
    left_norm: float,
    right_counts: Counter[str],
    right_norm: float,
    idf: dict[str, float],
    default_idf: float,
) -> float:
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    if len(left_counts) > len(right_counts):
        left_counts, right_counts = right_counts, left_counts
    dot = 0.0
    for token, left_count in left_counts.items():
        right_count = right_counts.get(token)
        if right_count:
            weight = idf.get(token, default_idf)
            dot += left_count * weight * right_count * weight
    return dot / (left_norm * right_norm)


def pair_feature_dict(
    paper_a: dict[str, Any],
    paper_b: dict[str, Any],
    idf: dict[str, float],
    default_idf: float,
) -> dict[str, Any]:
    title_j = jaccard(paper_a["title_tokens"], paper_b["title_tokens"])
    abstract_j = jaccard(paper_a["abstract_tokens"], paper_b["abstract_tokens"])
    ref_overlap = len(paper_a["ref_tokens"] & paper_b["ref_tokens"])
    ref_j = jaccard(paper_a["ref_tokens"], paper_b["ref_tokens"])
    content_cos = cosine(
        paper_a["content_counts"],
        paper_a["content_norm"],
        paper_b["content_counts"],
        paper_b["content_norm"],
        idf,
        default_idf,
    )
    a_in_b = containment(paper_a["title_tokens"], paper_b["title_tokens"])
    b_in_a = containment(paper_b["title_tokens"], paper_a["title_tokens"])
    year_a = paper_a["year"]
    year_b = paper_b["year"]
    year_diff = None if year_a is None or year_b is None else year_a - year_b
    ref_diff = paper_a["ref_count"] - paper_b["ref_count"]
    title_len_diff = paper_a["title_len"] - paper_b["title_len"]
    venue_match = bool(paper_a["venue"] and paper_a["venue"] == paper_b["venue"])

    return {
        "title_jaccard": title_j,
        "abstract_jaccard": abstract_j,
        "reference_jaccard": ref_j,
        "reference_overlap": ref_overlap,
        "content_cosine": content_cos,
        "title_a_in_b": a_in_b,
        "title_b_in_a": b_in_a,
        "year_diff_a_minus_b": year_diff,
        "reference_count_diff_a_minus_b": ref_diff,
        "title_len_diff_a_minus_b": title_len_diff,
        "venue_match": venue_match,
    }


STATIC_FEATURE_NAMES = (
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
)

def feature_names(feature_set: str) -> list[str]:
    if feature_set == "static":
        return list(STATIC_FEATURE_NAMES)
    raise ValueError(f"Unsupported feature set: {feature_set}")


def feature_vector(features: dict[str, Any], feature_set: str) -> list[float]:
    year_diff = features["year_diff_a_minus_b"]
    values = [
        float(features["title_jaccard"]),
        float(features["abstract_jaccard"]),
        float(features["reference_jaccard"]),
        min(float(features["reference_overlap"]), 30.0) / 30.0,
        float(features["content_cosine"]),
        float(features["title_a_in_b"]),
        float(features["title_b_in_a"]),
        0.0 if year_diff is None else max(min(float(year_diff), 20.0), -20.0) / 20.0,
        max(min(float(features["reference_count_diff_a_minus_b"]), 100.0), -100.0) / 100.0,
        max(min(float(features["title_len_diff_a_minus_b"]), 20.0), -20.0) / 20.0,
        1.0 if features["venue_match"] else 0.0,
    ]
    if feature_set == "static":
        return values
    raise ValueError(f"Unsupported feature set: {feature_set}")


def load_split_examples(
    labels: list[dict[str, Any]],
    paper_cache: dict[str, dict[str, Any]],
    idf: dict[str, float],
    default_idf: float,
    feature_set: str,
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
        out = {
            "pair_id": row["pair_id"],
            "split": row.get("split"),
            "paper_a_id": row["paper_a_id"],
            "paper_b_id": row["paper_b_id"],
            "paper_a_title": row.get("paper_a_title"),
            "paper_b_title": row.get("paper_b_title"),
            "paper_a_node_path": row.get("paper_a_node_path"),
            "paper_b_node_path": row.get("paper_b_node_path"),
            "true_label": row["label"],
            "features": features,
        }
        rows.append(out)
        x_rows.append(feature_vector(features, feature_set))
        y_rows.append(LABEL_TO_ID[row["label"]])
    return rows, torch.tensor(x_rows, dtype=torch.float), torch.tensor(y_rows, dtype=torch.long)


def standardize(train_x: torch.Tensor, valid_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (train_x - mean) / std, (valid_x - mean) / std, mean.squeeze(0), std.squeeze(0)


def compute_metrics(rows: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    for row, pred_label in zip(rows, predictions):
        matrix[LABEL_TO_ID[row["true_label"]]][LABEL_TO_ID[pred_label]] += 1

    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(len(LABELS)))
    per_class: dict[str, dict[str, Union[float, int]]] = {}
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for index, label in enumerate(LABELS):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(len(LABELS)) if row != index)
        fn = sum(matrix[index][col] for col in range(len(LABELS)) if col != index)
        support = sum(matrix[index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_precision": sum(precision_values) / len(precision_values),
        "macro_recall": sum(recall_values) / len(recall_values),
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
        "confusion_matrix": {"labels": list(LABELS), "matrix": matrix},
    }


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
            "paper_a_title": row.get("paper_a_title"),
            "paper_b_title": row.get("paper_b_title"),
            "true_label": row["true_label"],
            "predicted_label": pred,
            "method": method,
        }
        for label in LABELS:
            item[f"prob_{label}"] = float(probs.get(label, 0.0))
        out.append(item)
    return out


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def class_weights(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    counts = Counter(labels.detach().cpu().tolist())
    total = int(labels.shape[0])
    return torch.tensor(
        [total / (len(LABELS) * counts[index]) if counts.get(index) else 0.0 for index in range(len(LABELS))],
        dtype=torch.float,
        device=device,
    )


def train_backend(
    backend: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    valid_x: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[list[str], list[dict[str, float]], list[str], list[dict[str, float]], dict[str, Any]]:
    device = resolve_device(args.device)
    model = FeatureClassifier(train_x.shape[1], backend, args.hidden_size, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_x_device = train_x.to(device)
    train_y_device = train_y.to(device)
    history: list[dict[str, Union[float, int]]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(train_x_device.shape[0], device=device)
        total_loss = 0.0
        total_seen = 0
        for start in range(0, train_x_device.shape[0], args.batch_size):
            indices = permutation[start : start + args.batch_size]
            logits = model(train_x_device[indices])
            targets = train_y_device[indices]
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item()) * int(indices.shape[0])
            total_seen += int(indices.shape[0])
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            history.append({"epoch": epoch, "train_loss": total_loss / total_seen if total_seen else 0.0})
    def predict(x_tensor: torch.Tensor) -> tuple[list[str], list[dict[str, float]]]:
        model.eval()
        with torch.no_grad():
            logits = model(x_tensor.to(device)).detach().cpu()
            probs_tensor = torch.softmax(logits, dim=-1)
        predictions = [ID_TO_LABEL[int(index)] for index in torch.argmax(probs_tensor, dim=-1).tolist()]
        probabilities = [
            {label: float(row[LABEL_TO_ID[label]]) for label in LABELS}
            for row in probs_tensor.tolist()
        ]
        return predictions, probabilities

    train_predictions, train_probabilities = predict(train_x)
    valid_predictions, valid_probabilities = predict(valid_x)
    return (
        train_predictions,
        train_probabilities,
        valid_predictions,
        valid_probabilities,
        {
            "backend": backend,
            "history": history,
            "device": str(device),
        },
    )


def parse_backend_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def probability_rows_from_matrix(prob_matrix: Any, classes: Optional[Any] = None) -> tuple[list[str], list[dict[str, float]]]:
    import numpy as np

    values = np.asarray(prob_matrix, dtype=float)
    full = np.zeros((values.shape[0], len(LABELS)), dtype=float)
    if classes is None:
        classes = list(range(values.shape[1]))
    for source_index, class_id in enumerate(classes):
        full[:, int(class_id)] = values[:, source_index]
    predictions = [ID_TO_LABEL[int(index)] for index in np.argmax(full, axis=1).tolist()]
    probabilities = [
        {label: float(row[LABEL_TO_ID[label]]) for label in LABELS}
        for row in full.tolist()
    ]
    return predictions, probabilities


def tabular_backend_package(backend: str) -> str:
    return {
        "random_forest": "sklearn",
        "extra_trees": "sklearn",
    }[backend]


def train_tabular_backend(
    backend: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    valid_x: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[list[str], list[dict[str, float]], list[str], list[dict[str, float]], dict[str, Any]]:
    import numpy as np

    train_x_np = train_x.detach().cpu().numpy()
    valid_x_np = valid_x.detach().cpu().numpy()
    train_y_np = train_y.detach().cpu().numpy()
    package_name = tabular_backend_package(backend)
    summary: dict[str, Any] = {
        "backend": backend,
        "package": package_name,
        "tree_estimators": args.tree_estimators,
    }

    if backend == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=args.tree_estimators,
            min_samples_leaf=2,
            n_jobs=args.num_workers,
            random_state=args.seed,
        )
        model.fit(train_x_np, train_y_np)
    elif backend == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        model = ExtraTreesClassifier(
            n_estimators=args.tree_estimators,
            min_samples_leaf=2,
            n_jobs=args.num_workers,
            random_state=args.seed,
        )
        model.fit(train_x_np, train_y_np)
    else:
        raise ValueError(f"Unsupported tabular backend: {backend}")

    train_predictions, train_probabilities = probability_rows_from_matrix(
        model.predict_proba(train_x_np),
        getattr(model, "classes_", None),
    )
    valid_predictions, valid_probabilities = probability_rows_from_matrix(
        model.predict_proba(valid_x_np),
        getattr(model, "classes_", None),
    )
    return train_predictions, train_probabilities, valid_predictions, valid_probabilities, summary


def predictions_from_probabilities(probability_rows: list[dict[str, float]]) -> list[str]:
    return [max(LABELS, key=lambda label: row.get(label, 0.0)) for row in probability_rows]


def run_and_write(
    method_name: str,
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    train_predictions: list[str],
    train_probs: list[dict[str, float]],
    valid_predictions: list[str],
    valid_probs: list[dict[str, float]],
    output_dir: Path,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    method_dir = output_dir / method_name
    write_jsonl(method_dir / "valid_predictions.jsonl", annotate_predictions(valid_rows, valid_predictions, valid_probs, method_name))
    write_jsonl(method_dir / "train_predictions.jsonl", annotate_predictions(train_rows, train_predictions, train_probs, method_name))
    result = {
        "method": method_name,
        "train": compute_metrics(train_rows, train_predictions),
        "valid": compute_metrics(valid_rows, valid_predictions),
        "extra": extra or {},
    }
    write_json(method_dir / "metrics.json", result)
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_papers = read_jsonl(args.dataset_dir / "train_papers.jsonl")
    valid_papers = read_jsonl(args.dataset_dir / "valid_papers.jsonl")
    train_labels = read_jsonl(args.dataset_dir / "train_labels.jsonl")
    valid_labels = read_jsonl(args.dataset_dir / "valid_labels.jsonl")
    idf = fit_idf(train_papers, args.max_reference_titles)
    default_idf = math.log((1.0 + len(train_papers)) / 1.0) + 1.0
    train_cache = build_paper_cache(train_papers, idf, default_idf, args.max_reference_titles)
    valid_cache = build_paper_cache(valid_papers, idf, default_idf, args.max_reference_titles)
    train_rows, train_x, train_y = load_split_examples(train_labels, train_cache, idf, default_idf, "static")
    valid_rows, valid_x, _ = load_split_examples(valid_labels, valid_cache, idf, default_idf, "static")
    train_x_std, valid_x_std, mean, std = standardize(train_x, valid_x)

    results: dict[str, Any] = {
        "config": {
            "dataset_dir": str(args.dataset_dir),
            "output_dir": str(args.output_dir),
            "train_examples": len(train_rows),
            "valid_examples": len(valid_rows),
            "labels": list(LABELS),
            "feature_set": "static",
            "feature_names": feature_names("static"),
            "device": args.device,
            "max_reference_titles": args.max_reference_titles,
            "backends": ["mlp", *parse_backend_list(args.tabular_backends)],
            "tree_estimators": args.tree_estimators,
        },
        "methods": {},
        "skipped_methods": {},
    }

    train_pred, train_prob, valid_pred, valid_prob, backend_summary = train_backend(
        "mlp",
        train_x_std,
        train_y,
        valid_x_std,
        args,
    )
    result = run_and_write(
        "mlp",
        train_rows,
        valid_rows,
        train_pred,
        train_prob,
        valid_pred,
        valid_prob,
        args.output_dir,
        backend_summary,
    )
    results["methods"]["mlp"] = result
    print(json.dumps({"event": "method_done", "method": "mlp", "valid_macro_f1": result["valid"]["macro_f1"]}), flush=True)

    for backend in parse_backend_list(args.tabular_backends):
        method_name = backend
        try:
            package_name = tabular_backend_package(backend)
        except KeyError:
            results["skipped_methods"][method_name] = f"Unsupported backend: {backend}"
            continue
        if not package_available(package_name):
            results["skipped_methods"][method_name] = f"Missing package: {package_name}"
            continue
        train_pred, train_prob, valid_pred, valid_prob, backend_summary = train_tabular_backend(
            backend,
            train_x_std,
            train_y,
            valid_x_std,
            args,
        )
        result = run_and_write(
            method_name,
            train_rows,
            valid_rows,
            train_pred,
            train_prob,
            valid_pred,
            valid_prob,
            args.output_dir,
            backend_summary,
        )
        results["methods"][method_name] = result
        print(json.dumps({"event": "method_done", "method": method_name, "valid_macro_f1": result["valid"]["macro_f1"]}), flush=True)

    summary_rows = []
    for method, result in results["methods"].items():
        summary_rows.append(
            {
                "method": method,
                "valid_accuracy": result["valid"]["accuracy"],
                "valid_macro_f1": result["valid"]["macro_f1"],
                "valid_A_TO_B_recall": result["valid"]["per_class"]["A_TO_B"]["recall"],
                "valid_B_TO_A_recall": result["valid"]["per_class"]["B_TO_A"]["recall"],
                "valid_PEER_recall": result["valid"]["per_class"]["PEER"]["recall"],
                "valid_NONE_recall": result["valid"]["per_class"]["NONE"]["recall"],
            }
        )
    results["summary"] = sorted(summary_rows, key=lambda row: row["valid_macro_f1"], reverse=True)
    write_json(args.output_dir / "all_metrics.json", results)
    print(json.dumps({"event": "done", "summary": results["summary"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
