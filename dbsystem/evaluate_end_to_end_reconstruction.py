import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_dynamic_index import (
    DynamicHierarchicalIndex,
    LABELS,
    compute_four_class_metrics,
    level_schedule,
    read_jsonl,
    write_json,
)
from weak_methods.run_weak_methods import (
    build_paper_cache,
    feature_vector,
    pair_feature_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end reconstruction timing.")
    parser.add_argument("--papers", type=Path, default=Path("dataset/train_papers.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/train_labels.jsonl"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("dbsystem/outputs/pair_model_static_extratrees/model.pkl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dbsystem/outputs/end_to_end_static_extratrees_leaf16_beam2"),
    )
    parser.add_argument("--leaf-size", type=int, default=16)
    parser.add_argument("--base", type=float, default=2.0)
    parser.add_argument("--beam-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--limit-papers", type=int, default=0)
    parser.add_argument("--log-every-pairs", type=int, default=1000)
    parser.add_argument(
        "--reuse-linear-from",
        type=Path,
        default=None,
        help="Optional previous end_to_end_reconstruction.json to reuse its linear_scan result.",
    )
    return parser.parse_args()


class RawStaticExtraTreesPredictor:
    def __init__(self, model_path: Path, papers: list[dict[str, Any]]) -> None:
        with model_path.open("rb") as handle:
            artifact = pickle.load(handle)
        self.model = artifact["model"]
        self.idf = artifact["idf"]
        self.default_idf = artifact["default_idf"]
        self.mean = torch.tensor(artifact["mean"], dtype=torch.float32)
        self.std = torch.tensor(artifact["std"], dtype=torch.float32)
        self.labels = list(artifact["labels"])
        self.feature_set = artifact.get("feature_set", "static")
        self.max_reference_titles = int(artifact.get("max_reference_titles", 80))
        self.paper_by_id = {paper["paper_id"]: paper for paper in papers}
        self.calls = 0
        self.feature_seconds = 0.0
        self.model_seconds = 0.0

    def reset_timer(self) -> None:
        self.calls = 0
        self.feature_seconds = 0.0
        self.model_seconds = 0.0

    def predict(self, paper_a_id: str, paper_b_id: str) -> str:
        feature_start = time.perf_counter()
        paper_cache = build_paper_cache(
            [self.paper_by_id[paper_a_id], self.paper_by_id[paper_b_id]],
            self.idf,
            self.default_idf,
            self.max_reference_titles,
        )
        features = pair_feature_dict(
            paper_cache[paper_a_id],
            paper_cache[paper_b_id],
            self.idf,
            self.default_idf,
        )
        x = torch.tensor([feature_vector(features, self.feature_set)], dtype=torch.float32)
        x = ((x - self.mean) / self.std).detach().cpu().numpy()
        self.feature_seconds += time.perf_counter() - feature_start

        model_start = time.perf_counter()
        pred_id = int(self.model.predict(x)[0])
        self.model_seconds += time.perf_counter() - model_start
        self.calls += 1
        return self.labels[pred_id]

    def timing(self, wall_seconds: float) -> dict[str, Any]:
        return {
            "pair_predictions": self.calls,
            "feature_seconds": self.feature_seconds,
            "model_seconds": self.model_seconds,
            "feature_plus_model_seconds": self.feature_seconds + self.model_seconds,
            "wall_seconds": wall_seconds,
            "avg_ms_per_pair_feature": self.feature_seconds * 1000 / self.calls if self.calls else 0.0,
            "avg_ms_per_pair_model": self.model_seconds * 1000 / self.calls if self.calls else 0.0,
            "avg_ms_per_pair_feature_plus_model": (self.feature_seconds + self.model_seconds) * 1000 / self.calls if self.calls else 0.0,
            "avg_ms_per_pair_wall": wall_seconds * 1000 / self.calls if self.calls else 0.0,
            "pairs_per_second_wall": self.calls / wall_seconds if wall_seconds else 0.0,
        }


def load_label_map(path: Path) -> dict[tuple[str, str], str]:
    return {
        (row["paper_a_id"], row["paper_b_id"]): row["label"]
        for row in read_jsonl(path)
    }


def add_predicted_graph_item(
    edges: list[dict[str, str]],
    peer_pairs: list[dict[str, str]],
    paper_a_id: str,
    paper_b_id: str,
    pred_label: str,
) -> None:
    if pred_label == "A_TO_B":
        edges.append({"source": paper_a_id, "target": paper_b_id, "label": pred_label})
    elif pred_label == "B_TO_A":
        edges.append({"source": paper_b_id, "target": paper_a_id, "label": pred_label})
    elif pred_label == "PEER":
        peer_pairs.append({"paper_a_id": paper_a_id, "paper_b_id": paper_b_id, "label": pred_label})


def run_linear(
    papers: list[dict[str, Any]],
    label_map: dict[tuple[str, str], str],
    predictor: RawStaticExtraTreesPredictor,
    log_every_pairs: int,
) -> dict[str, Any]:
    predictor.reset_timer()
    true_labels: list[str] = []
    pred_labels: list[str] = []
    edges: list[dict[str, str]] = []
    peers: list[dict[str, str]] = []
    inserted: list[str] = []
    start = time.perf_counter()
    for paper in papers:
        paper_id = paper["paper_id"]
        for existing_id in inserted:
            true_label = label_map.get((paper_id, existing_id), "NONE")
            pred_label = predictor.predict(paper_id, existing_id)
            true_labels.append(true_label)
            pred_labels.append(pred_label)
            add_predicted_graph_item(edges, peers, paper_id, existing_id, pred_label)
            if log_every_pairs > 0 and predictor.calls % log_every_pairs == 0:
                elapsed = time.perf_counter() - start
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "system": "linear",
                            "pairs_done": predictor.calls,
                            "elapsed_seconds": elapsed,
                            "pairs_per_second": predictor.calls / elapsed if elapsed else 0.0,
                        }
                    ),
                    flush=True,
                )
        inserted.append(paper_id)
    wall_seconds = time.perf_counter() - start
    return {
        "system": "linear_scan",
        "paper_count": len(papers),
        "graph": {
            "directed_edge_count": len(edges),
            "peer_pair_count": len(peers),
            "sample_directed_edges": edges[:20],
            "sample_peer_pairs": peers[:20],
        },
        "timing": predictor.timing(wall_seconds),
        "metrics_including_none": compute_four_class_metrics(true_labels, pred_labels),
    }


def run_dynamic(
    papers: list[dict[str, Any]],
    label_map: dict[tuple[str, str], str],
    predictor: RawStaticExtraTreesPredictor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    predictor.reset_timer()
    index = DynamicHierarchicalIndex(
        leaf_size=args.leaf_size,
        base=args.base,
        seed=args.seed,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        n_init=args.n_init,
    )
    true_labels: list[str] = []
    pred_labels: list[str] = []
    edges: list[dict[str, str]] = []
    peers: list[dict[str, str]] = []
    visited_nodes = 0
    candidate_sizes: list[int] = []
    route_seconds = 0.0
    rebuild_seconds = 0.0
    update_seconds = 0.0
    rebuild_count = 0
    start = time.perf_counter()
    for paper in papers:
        paper_id = paper["paper_id"]
        existing_ids = list(index.inserted_ids)
        old_schedule = level_schedule(len(existing_ids), args.leaf_size, args.base)
        route_start = time.perf_counter()
        route_result = index.route(paper, args.beam_size)
        route_seconds += time.perf_counter() - route_start
        candidate_ids = set(route_result["candidate_ids"])
        visited_nodes += int(route_result["visited_nodes"])
        if existing_ids:
            candidate_sizes.append(len(candidate_ids))
        for existing_id in existing_ids:
            true_label = label_map.get((paper_id, existing_id), "NONE")
            if existing_id in candidate_ids:
                pred_label = predictor.predict(paper_id, existing_id)
                add_predicted_graph_item(edges, peers, paper_id, existing_id, pred_label)
                if args.log_every_pairs > 0 and predictor.calls % args.log_every_pairs == 0:
                    elapsed = time.perf_counter() - start
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "system": "dynamic",
                                "pairs_done": predictor.calls,
                                "elapsed_seconds": elapsed,
                                "pairs_per_second": predictor.calls / elapsed if elapsed else 0.0,
                            }
                        ),
                        flush=True,
                    )
            else:
                pred_label = "NONE"
            true_labels.append(true_label)
            pred_labels.append(pred_label)

        rebuilt, insert_seconds = index.insert(paper, route_result["best_leaf_id"], old_schedule)
        if rebuilt:
            rebuild_count += 1
            rebuild_seconds += insert_seconds
        else:
            update_seconds += insert_seconds
    wall_seconds = time.perf_counter() - start
    baseline_pairs = len(papers) * (len(papers) - 1) // 2
    pair_predictions = predictor.calls
    return {
        "system": "dbsystem_dynamic",
        "paper_count": len(papers),
        "beam_size": args.beam_size,
        "leaf_size": args.leaf_size,
        "graph": {
            "directed_edge_count": len(edges),
            "peer_pair_count": len(peers),
            "sample_directed_edges": edges[:20],
            "sample_peer_pairs": peers[:20],
        },
        "timing": {
            **predictor.timing(wall_seconds),
            "route_seconds": route_seconds,
            "rebuild_seconds": rebuild_seconds,
            "local_update_seconds": update_seconds,
            "visited_cluster_representatives": visited_nodes,
        },
        "efficiency": {
            "linear_pair_count": baseline_pairs,
            "dynamic_pair_predictions": pair_predictions,
            "pair_prediction_reduction": 1.0 - pair_predictions / baseline_pairs if baseline_pairs else 0.0,
            "speedup_by_pair_predictions": baseline_pairs / pair_predictions if pair_predictions else float("inf"),
            "avg_candidate_size": sum(candidate_sizes) / len(candidate_sizes) if candidate_sizes else 0.0,
            "median_candidate_size": sorted(candidate_sizes)[len(candidate_sizes) // 2] if candidate_sizes else 0.0,
            "rebuild_count": rebuild_count,
        },
        "metrics_including_none": compute_four_class_metrics(true_labels, pred_labels),
        "final_index": index.stats(),
    }


def main() -> None:
    args = parse_args()
    papers = read_jsonl(args.papers)
    papers.sort(key=lambda row: int(row.get("split_index", 0)))
    if args.limit_papers > 0:
        papers = papers[: args.limit_papers]
    label_map = load_label_map(args.labels)
    predictor = RawStaticExtraTreesPredictor(args.model_path, papers)

    print(json.dumps({"event": "start", "papers": len(papers), "model_path": str(args.model_path)}), flush=True)
    if args.reuse_linear_from is not None:
        cached = json.loads(args.reuse_linear_from.read_text())
        linear = cached["linear_scan"]
        print(
            json.dumps(
                {
                    "event": "linear_reused",
                    "source": str(args.reuse_linear_from),
                    "pairs": linear["timing"]["pair_predictions"],
                    "wall_seconds": linear["timing"]["wall_seconds"],
                    "accuracy": linear["metrics_including_none"]["accuracy"],
                    "macro_f1": linear["metrics_including_none"]["macro_f1"],
                }
            ),
            flush=True,
        )
    else:
        linear = run_linear(papers, label_map, predictor, args.log_every_pairs)
        print(
            json.dumps(
                {
                    "event": "linear_done",
                    "pairs": linear["timing"]["pair_predictions"],
                    "wall_seconds": linear["timing"]["wall_seconds"],
                    "accuracy": linear["metrics_including_none"]["accuracy"],
                    "macro_f1": linear["metrics_including_none"]["macro_f1"],
                }
            ),
            flush=True,
        )
    dynamic = run_dynamic(papers, label_map, predictor, args)
    print(
        json.dumps(
            {
                "event": "dynamic_done",
                "pairs": dynamic["timing"]["pair_predictions"],
                "wall_seconds": dynamic["timing"]["wall_seconds"],
                "accuracy": dynamic["metrics_including_none"]["accuracy"],
                "macro_f1": dynamic["metrics_including_none"]["macro_f1"],
                "reduction": dynamic["efficiency"]["pair_prediction_reduction"],
            }
        ),
        flush=True,
    )

    result = {
        "config": {
            "papers": str(args.papers),
            "labels": str(args.labels),
            "paper_count": len(papers),
            "pair_predictor": "static_11_feature_ExtraTrees_valid_best_backend",
            "model_path": str(args.model_path),
            "leaf_size": args.leaf_size,
            "base": args.base,
            "beam_size": args.beam_size,
            "none_included_in_metrics": True,
            "schedule_for_final_n": level_schedule(len(papers), args.leaf_size, args.base),
        },
        "linear_scan": linear,
        "dbsystem_dynamic": dynamic,
        "comparison": {
            "pair_prediction_reduction": dynamic["efficiency"]["pair_prediction_reduction"],
            "pair_prediction_speedup": dynamic["efficiency"]["speedup_by_pair_predictions"],
            "wall_time_speedup_linear_over_dynamic": (
                linear["timing"]["wall_seconds"] / dynamic["timing"]["wall_seconds"]
                if dynamic["timing"]["wall_seconds"]
                else None
            ),
            "linear_accuracy": linear["metrics_including_none"]["accuracy"],
            "dynamic_accuracy": dynamic["metrics_including_none"]["accuracy"],
            "linear_macro_f1": linear["metrics_including_none"]["macro_f1"],
            "dynamic_macro_f1": dynamic["metrics_including_none"]["macro_f1"],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "end_to_end_reconstruction.json", result)
    print(json.dumps({"event": "done", "output": str(args.output_dir / "end_to_end_reconstruction.json")}), flush=True)


if __name__ == "__main__":
    main()
