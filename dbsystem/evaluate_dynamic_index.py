import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


LABELS = ("A_TO_B", "B_TO_A", "PEER", "NONE")
RELATED_LABELS = {"A_TO_B", "B_TO_A", "PEER"}


@dataclass
class ClusterNode:
    node_id: str
    level: int
    paper_ids: list[str]
    child_ids: list[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    centroid: Optional[np.ndarray] = None
    representative_paper_id: Optional[str] = None
    representative_vector: Optional[np.ndarray] = None

    @property
    def is_leaf(self) -> bool:
        return not self.child_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dynamic hierarchical insertion.")
    parser.add_argument("--papers", type=Path, default=Path("dataset/train_papers.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/train_labels.jsonl"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("weak_methods/outputs/default/extra_trees/train_predictions.jsonl"),
        help="Optional pair predictor outputs. If absent, evaluation uses ground-truth labels for candidate analysis.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dbsystem/outputs/leaf16_base2"))
    parser.add_argument("--leaf-size", type=int, default=16)
    parser.add_argument("--base", type=float, default=2.0)
    parser.add_argument("--beam-sizes", type=str, default="1,2,3")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="Optional paper limit for smoke tests.")
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


def compute_four_class_metrics(true_labels: list[str], pred_labels: list[str]) -> dict[str, Any]:
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    for true_label, pred_label in zip(true_labels, pred_labels):
        matrix[label_to_id[true_label]][label_to_id[pred_label]] += 1

    total = len(true_labels)
    correct = sum(matrix[index][index] for index in range(len(LABELS)))
    per_class: dict[str, dict[str, Union[float, int]]] = {}
    f1_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
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


def metadata_text(paper: dict[str, Any]) -> str:
    reference_titles = paper.get("reference_titles") or []
    if not isinstance(reference_titles, list):
        reference_titles = []
    return " ".join(
        [
            str(paper.get("title") or ""),
            str(paper.get("abstract") or ""),
            str(paper.get("venue") or ""),
            " ".join(str(title) for title in reference_titles),
        ]
    )


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def level_schedule(n_papers: int, leaf_size: int, base: float) -> list[int]:
    if n_papers <= 0:
        return []
    if n_papers <= leaf_size:
        return [1]
    counts = [max(1, math.ceil(n_papers / leaf_size))]
    while counts[-1] > 1:
        next_count = math.ceil(math.log(counts[-1], base))
        if next_count >= counts[-1]:
            next_count = counts[-1] - 1
        counts.append(max(1, next_count))
    return counts


def cluster_labels(vectors: np.ndarray, n_clusters: int, seed: int, n_init: int) -> np.ndarray:
    n_items = vectors.shape[0]
    if n_clusters <= 1 or n_items <= 1:
        return np.zeros(n_items, dtype=np.int64)
    n_clusters = min(n_clusters, n_items)
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
    return model.fit_predict(vectors)


def capacity_limited_cluster_labels(vectors: np.ndarray, n_clusters: int, seed: int, n_init: int) -> np.ndarray:
    n_items = vectors.shape[0]
    if n_clusters <= 1 or n_items <= 1:
        return np.zeros(n_items, dtype=np.int64)
    n_clusters = min(n_clusters, n_items)
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
    model.fit(vectors)
    centers = model.cluster_centers_.astype(np.float32)
    base_capacity = n_items // n_clusters
    extras = n_items % n_clusters
    capacities = np.array(
        [base_capacity + (1 if cluster_id < extras else 0) for cluster_id in range(n_clusters)],
        dtype=np.int64,
    )
    distances = ((vectors[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    preferences = np.argsort(distances, axis=1)
    if n_clusters == 1:
        margins = np.zeros(n_items, dtype=np.float32)
    else:
        margins = distances[np.arange(n_items), preferences[:, 1]] - distances[np.arange(n_items), preferences[:, 0]]
    order = sorted(range(n_items), key=lambda index: float(margins[index]), reverse=True)
    labels = np.full(n_items, -1, dtype=np.int64)
    remaining = capacities.copy()
    for item_index in order:
        for cluster_id in preferences[item_index]:
            if remaining[cluster_id] > 0:
                labels[item_index] = int(cluster_id)
                remaining[cluster_id] -= 1
                break
    if np.any(labels < 0):
        open_clusters = [int(cluster_id) for cluster_id, capacity in enumerate(remaining) for _ in range(int(capacity))]
        for item_index in np.flatnonzero(labels < 0):
            labels[item_index] = open_clusters.pop()
    return labels


def load_label_map(path: Path) -> dict[tuple[str, str], str]:
    label_map: dict[tuple[str, str], str] = {}
    for row in read_jsonl(path):
        label_map[(row["paper_a_id"], row["paper_b_id"])] = row["label"]
    return label_map


def load_prediction_map(path: Path, label_map: dict[tuple[str, str], str]) -> dict[tuple[str, str], str]:
    if not path.exists():
        return dict(label_map)
    prediction_map: dict[tuple[str, str], str] = {}
    for row in read_jsonl(path):
        prediction_map[(row["paper_a_id"], row["paper_b_id"])] = row["predicted_label"]
    return prediction_map


class DynamicHierarchicalIndex:
    def __init__(self, *, leaf_size: int, base: float, seed: int, max_features: int, ngram_max: int, n_init: int) -> None:
        self.leaf_size = leaf_size
        self.base = base
        self.seed = seed
        self.max_features = max_features
        self.ngram_max = ngram_max
        self.n_init = n_init
        self.paper_store: dict[str, dict[str, Any]] = {}
        self.inserted_ids: list[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.paper_vectors: dict[str, np.ndarray] = {}
        self.nodes: dict[str, ClusterNode] = {}
        self.levels: list[list[str]] = []
        self.root_id: Optional[str] = None
        self.schedule: list[int] = []

    def rebuild(self) -> None:
        self.nodes = {}
        self.levels = []
        self.root_id = None
        self.paper_vectors = {}
        self.schedule = level_schedule(len(self.inserted_ids), self.leaf_size, self.base)
        if not self.inserted_ids:
            self.vectorizer = None
            return

        texts = [metadata_text(self.paper_store[paper_id]) for paper_id in self.inserted_ids]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            min_df=1,
            max_df=1.0,
            ngram_range=(1, self.ngram_max),
            norm="l2",
            max_features=self.max_features,
        )
        matrix = self.vectorizer.fit_transform(texts).astype(np.float32).toarray()
        for paper_id, vector in zip(self.inserted_ids, matrix):
            self.paper_vectors[paper_id] = normalize_vector(vector)

        current_items = list(self.inserted_ids)
        current_vectors = np.vstack([self.paper_vectors[paper_id] for paper_id in current_items])
        for level_index, count in enumerate(self.schedule):
            if level_index == 0:
                labels = capacity_limited_cluster_labels(current_vectors, count, self.seed + level_index, self.n_init)
            else:
                labels = cluster_labels(current_vectors, count, self.seed + level_index, self.n_init)
            grouped: dict[int, list[str]] = defaultdict(list)
            for item_id, label in zip(current_items, labels):
                grouped[int(label)].append(item_id)

            level_node_ids: list[str] = []
            next_items: list[str] = []
            next_vectors: list[np.ndarray] = []
            for local_index, (_, item_ids) in enumerate(sorted(grouped.items())):
                node_id = f"L{level_index}_{local_index}"
                if level_index == 0:
                    paper_ids = list(item_ids)
                    child_ids: list[str] = []
                else:
                    child_ids = list(item_ids)
                    paper_ids = []
                    for child_id in child_ids:
                        paper_ids.extend(self.nodes[child_id].paper_ids)
                        self.nodes[child_id].parent_id = node_id
                node = ClusterNode(node_id=node_id, level=level_index, paper_ids=paper_ids, child_ids=child_ids)
                self._refresh_node_vectors(node)
                self.nodes[node_id] = node
                level_node_ids.append(node_id)
                next_items.append(node_id)
                next_vectors.append(node.centroid if node.centroid is not None else np.zeros(current_vectors.shape[1], dtype=np.float32))

            self.levels.append(level_node_ids)
            current_items = next_items
            current_vectors = np.vstack(next_vectors)

        self.root_id = self.levels[-1][0] if self.levels else None

    def _refresh_node_vectors(self, node: ClusterNode) -> None:
        if not node.paper_ids:
            return
        vectors = np.vstack([self.paper_vectors[paper_id] for paper_id in node.paper_ids])
        node.centroid = normalize_vector(vectors.mean(axis=0))
        if node.is_leaf:
            scores = vectors @ node.centroid
            best_index = int(np.argmax(scores))
            node.representative_paper_id = node.paper_ids[best_index]
            node.representative_vector = self.paper_vectors[node.representative_paper_id]
        else:
            child_ids = node.child_ids
            child_centroids = np.vstack([self.nodes[child_id].centroid for child_id in child_ids])
            scores = child_centroids @ node.centroid
            best_child_id = child_ids[int(np.argmax(scores))]
            node.representative_paper_id = self.nodes[best_child_id].representative_paper_id
            node.representative_vector = self.nodes[best_child_id].representative_vector

    def vectorize_query(self, paper: dict[str, Any]) -> np.ndarray:
        if self.vectorizer is None:
            return np.zeros(1, dtype=np.float32)
        vector = self.vectorizer.transform([metadata_text(paper)]).astype(np.float32).toarray()[0]
        return normalize_vector(vector)

    def route(self, paper: dict[str, Any], beam_size: int) -> dict[str, Any]:
        if not self.inserted_ids:
            return {"candidate_ids": [], "leaf_ids": [], "visited_nodes": 0, "best_leaf_id": None}
        if self.root_id is None or self.vectorizer is None:
            return {
                "candidate_ids": list(self.inserted_ids),
                "leaf_ids": [self.root_id] if self.root_id else [],
                "visited_nodes": len(self.inserted_ids),
                "best_leaf_id": self.root_id,
            }

        query_vector = self.vectorize_query(paper)
        root = self.nodes[self.root_id]
        if root.is_leaf:
            return {
                "candidate_ids": list(root.paper_ids),
                "leaf_ids": [root.node_id],
                "visited_nodes": 1,
                "best_leaf_id": root.node_id,
            }

        beam: list[tuple[str, float, int]] = [(self.root_id, 0.0, 0)]
        visited_nodes = 0
        while beam and not all(self.nodes[node_id].is_leaf for node_id, _, _ in beam):
            candidates: list[tuple[str, float, int]] = []
            for node_id, score_sum, depth in beam:
                node = self.nodes[node_id]
                if node.is_leaf:
                    candidates.append((node_id, score_sum, depth))
                    continue
                for child_id in node.child_ids:
                    child = self.nodes[child_id]
                    rep_vector = child.representative_vector
                    sim = float(query_vector @ rep_vector) if rep_vector is not None and rep_vector.shape == query_vector.shape else 0.0
                    candidates.append((child_id, score_sum + sim, depth + 1))
                    visited_nodes += 1
            candidates.sort(key=lambda item: (item[1] / max(1, item[2]), item[1]), reverse=True)
            beam = candidates[: max(1, beam_size)]

        leaf_ids = [node_id for node_id, _, _ in beam if self.nodes[node_id].is_leaf]
        candidate_ids: list[str] = []
        for leaf_id in leaf_ids:
            candidate_ids.extend(self.nodes[leaf_id].paper_ids)
        candidate_ids = sorted(set(candidate_ids), key=candidate_ids.index)
        best_leaf_id = beam[0][0] if beam else None
        return {
            "candidate_ids": candidate_ids,
            "leaf_ids": leaf_ids,
            "visited_nodes": visited_nodes,
            "best_leaf_id": best_leaf_id,
        }

    def insert(self, paper: dict[str, Any], best_leaf_id: Optional[str], old_schedule: list[int]) -> tuple[bool, float]:
        paper_id = paper["paper_id"]
        self.paper_store[paper_id] = paper
        self.inserted_ids.append(paper_id)
        new_schedule = level_schedule(len(self.inserted_ids), self.leaf_size, self.base)
        should_rebuild = self.vectorizer is None or new_schedule != old_schedule or best_leaf_id is None
        start = time.perf_counter()
        if should_rebuild:
            self.rebuild()
            return True, time.perf_counter() - start

        vector = self.vectorize_query(paper)
        self.paper_vectors[paper_id] = vector
        node_id = best_leaf_id
        while node_id is not None:
            node = self.nodes[node_id]
            node.paper_ids.append(paper_id)
            self._refresh_node_vectors(node)
            node_id = node.parent_id
        if best_leaf_id in self.nodes and len(self.nodes[best_leaf_id].paper_ids) > self.leaf_size:
            self.rebuild()
            return True, time.perf_counter() - start
        return False, time.perf_counter() - start

    def stats(self) -> dict[str, Any]:
        leaf_ids = self.levels[0] if self.levels else []
        leaf_sizes = [len(self.nodes[node_id].paper_ids) for node_id in leaf_ids]
        return {
            "paper_count": len(self.inserted_ids),
            "schedule_bottom_up": self.schedule,
            "level_counts_bottom_up": [len(level) for level in self.levels],
            "leaf_count": len(leaf_ids),
            "leaf_size_min": min(leaf_sizes) if leaf_sizes else 0,
            "leaf_size_max": max(leaf_sizes) if leaf_sizes else 0,
            "leaf_size_mean": statistics.mean(leaf_sizes) if leaf_sizes else 0.0,
            "leaf_size_median": statistics.median(leaf_sizes) if leaf_sizes else 0.0,
        }


def evaluate_stream(
    papers: list[dict[str, Any]],
    label_map: dict[tuple[str, str], str],
    prediction_map: dict[tuple[str, str], str],
    args: argparse.Namespace,
    beam_size: int,
) -> dict[str, Any]:
    index = DynamicHierarchicalIndex(
        leaf_size=args.leaf_size,
        base=args.base,
        seed=args.seed,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        n_init=args.n_init,
    )
    baseline_correct = 0
    system_correct = 0
    total_pairs = 0
    related_total = 0
    system_related_hits = 0
    query_related_total = 0
    query_related_hit = 0
    baseline_pair_predictions = 0
    system_pair_predictions = 0
    baseline_true_labels: list[str] = []
    baseline_pred_labels: list[str] = []
    system_true_labels: list[str] = []
    system_pred_labels: list[str] = []
    visited_nodes_total = 0
    candidate_sizes: list[int] = []
    reductions: list[float] = []
    speedups: list[float] = []
    route_time = 0.0
    rebuild_time = 0.0
    update_time = 0.0
    rebuilds = 0
    insertion_records: list[dict[str, Any]] = []

    wall_start = time.perf_counter()
    for step, paper in enumerate(papers):
        paper_id = paper["paper_id"]
        existing_ids = list(index.inserted_ids)
        old_schedule = level_schedule(len(existing_ids), args.leaf_size, args.base)

        route_start = time.perf_counter()
        route_result = index.route(paper, beam_size)
        route_time += time.perf_counter() - route_start

        candidate_ids = set(route_result["candidate_ids"])
        candidate_size = len(candidate_ids)
        existing_count = len(existing_ids)
        baseline_pair_predictions += existing_count
        system_pair_predictions += candidate_size
        visited_nodes_total += int(route_result["visited_nodes"])
        if existing_count:
            candidate_sizes.append(candidate_size)
            reductions.append(1.0 - candidate_size / existing_count)
            speedups.append(existing_count / candidate_size if candidate_size else float("inf"))

        query_has_related = False
        query_hit = False
        for existing_id in existing_ids:
            true_label = label_map.get((paper_id, existing_id), "NONE")
            pair_prediction = prediction_map.get((paper_id, existing_id), true_label)
            baseline_correct += int(pair_prediction == true_label)
            baseline_true_labels.append(true_label)
            baseline_pred_labels.append(pair_prediction)

            if existing_id in candidate_ids:
                system_prediction = pair_prediction
            else:
                system_prediction = "NONE"
            system_correct += int(system_prediction == true_label)
            system_true_labels.append(true_label)
            system_pred_labels.append(system_prediction)
            total_pairs += 1

            if true_label in RELATED_LABELS:
                related_total += 1
                query_has_related = True
                if existing_id in candidate_ids:
                    system_related_hits += 1
                    query_hit = True

        if query_has_related:
            query_related_total += 1
            query_related_hit += int(query_hit)

        rebuilt, insert_update_time = index.insert(paper, route_result["best_leaf_id"], old_schedule)
        if rebuilt:
            rebuilds += 1
            rebuild_time += insert_update_time
        else:
            update_time += insert_update_time

        if step < 5 or rebuilt:
            insertion_records.append(
                {
                    "step": step + 1,
                    "paper_id": paper_id,
                    "existing_count": existing_count,
                    "candidate_size": candidate_size,
                    "rebuilt": rebuilt,
                    "old_schedule": old_schedule,
                    "new_schedule": level_schedule(len(index.inserted_ids), args.leaf_size, args.base),
                }
            )

    wall_time = time.perf_counter() - wall_start
    avg_candidates = statistics.mean(candidate_sizes) if candidate_sizes else 0.0
    median_candidates = statistics.median(candidate_sizes) if candidate_sizes else 0.0
    return {
        "beam_size": beam_size,
        "paper_count": len(papers),
        "total_ordered_insert_pairs": total_pairs,
        "baseline_linear_scan": {
            "pair_predictions": baseline_pair_predictions,
            "four_class_accuracy": baseline_correct / total_pairs if total_pairs else 1.0,
            "related_pair_recall": 1.0,
            "four_class_metrics_including_none": compute_four_class_metrics(
                baseline_true_labels,
                baseline_pred_labels,
            ),
        },
        "dynamic_index": {
            "pair_predictions": system_pair_predictions,
            "visited_cluster_representatives": visited_nodes_total,
            "pair_prediction_reduction": 1.0 - system_pair_predictions / baseline_pair_predictions if baseline_pair_predictions else 0.0,
            "speedup_by_pair_predictions": baseline_pair_predictions / system_pair_predictions if system_pair_predictions else float("inf"),
            "four_class_accuracy_if_uncandidates_none": system_correct / total_pairs if total_pairs else 1.0,
            "four_class_metrics_including_none": compute_four_class_metrics(
                system_true_labels,
                system_pred_labels,
            ),
            "related_pair_recall": system_related_hits / related_total if related_total else 1.0,
            "query_related_hit_rate": query_related_hit / query_related_total if query_related_total else 1.0,
            "avg_candidate_size": avg_candidates,
            "median_candidate_size": median_candidates,
            "avg_reduction_per_insert": statistics.mean(reductions) if reductions else 0.0,
            "median_reduction_per_insert": statistics.median(reductions) if reductions else 0.0,
            "avg_speedup_per_insert": statistics.mean(speedups) if speedups else 0.0,
            "median_speedup_per_insert": statistics.median(speedups) if speedups else 0.0,
            "rebuild_count": rebuilds,
            "route_time_seconds": route_time,
            "rebuild_time_seconds": rebuild_time,
            "local_update_time_seconds": update_time,
            "wall_time_seconds": wall_time,
        },
        "final_index": index.stats(),
        "sample_insertions": insertion_records[:40],
    }


def main() -> None:
    args = parse_args()
    papers = read_jsonl(args.papers)
    papers.sort(key=lambda row: int(row.get("split_index", 0)))
    if args.limit > 0:
        papers = papers[: args.limit]
    label_map = load_label_map(args.labels)
    prediction_map = load_prediction_map(args.predictions, label_map)
    beam_sizes = [int(item.strip()) for item in args.beam_sizes.split(",") if item.strip()]

    results = {
        "config": {
            "papers": str(args.papers),
            "labels": str(args.labels),
            "predictions": str(args.predictions),
            "prediction_file_exists": args.predictions.exists(),
            "leaf_size": args.leaf_size,
            "base": args.base,
            "beam_sizes": beam_sizes,
            "seed": args.seed,
            "max_features": args.max_features,
            "ngram_max": args.ngram_max,
            "n_init": args.n_init,
            "paper_count": len(papers),
            "schedule_for_final_n": level_schedule(len(papers), args.leaf_size, args.base),
        },
        "runs": [],
    }
    for beam_size in beam_sizes:
        run = evaluate_stream(papers, label_map, prediction_map, args, beam_size)
        results["runs"].append(run)
        print(
            json.dumps(
                {
                    "event": "beam_done",
                    "beam_size": beam_size,
                    "pair_prediction_reduction": run["dynamic_index"]["pair_prediction_reduction"],
                    "system_accuracy": run["dynamic_index"]["four_class_accuracy_if_uncandidates_none"],
                    "related_pair_recall": run["dynamic_index"]["related_pair_recall"],
                    "avg_candidate_size": run["dynamic_index"]["avg_candidate_size"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "dynamic_insertion_eval.json", results)
    print(json.dumps({"event": "done", "output": str(args.output_dir / "dynamic_insertion_eval.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
