import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine BERT feature shards.")
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("baseline/outputs/fixed_5x5_bert_features"),
    )
    parser.add_argument("--num-shards", type=int, default=4)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def combine_split(feature_dir: Path, split: str, num_shards: int) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    for shard_id in range(num_shards):
        path = feature_dir / f"{split}_shard{shard_id:02d}of{num_shards:02d}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(torch.load(path, map_location="cpu"))

    x = torch.cat([part["x"] for part in parts], dim=0)
    y = torch.cat([part["y"] for part in parts], dim=0)
    chunk_mask = torch.cat([part["chunk_mask"] for part in parts], dim=0)

    labels: list[str] = []
    pair_ids: list[str] = []
    paper_a_ids: list[str] = []
    paper_b_ids: list[str] = []
    for part in parts:
        labels.extend(part["labels"])
        pair_ids.extend(part["pair_ids"])
        paper_a_ids.extend(part["paper_a_ids"])
        paper_b_ids.extend(part["paper_b_ids"])

    combined = {
        "x": x,
        "y": y,
        "chunk_mask": chunk_mask,
        "labels": labels,
        "pair_ids": pair_ids,
        "paper_a_ids": paper_a_ids,
        "paper_b_ids": paper_b_ids,
        "label_order": parts[0]["label_order"],
        "feature_shape": list(x.shape),
        "feature_description": parts[0]["feature_description"],
        "source_shards": [
            f"{split}_shard{shard_id:02d}of{num_shards:02d}.pt"
            for shard_id in range(num_shards)
        ],
    }
    output_path = feature_dir / f"{split}.pt"
    torch.save(combined, output_path)

    size_bytes = int(output_path.stat().st_size)
    summary = {
        "split": split,
        "output": str(output_path),
        "source_shards": combined["source_shards"],
        "x_shape": list(x.shape),
        "x_dtype": str(x.dtype),
        "y_shape": list(y.shape),
        "chunk_mask_shape": list(chunk_mask.shape),
        "num_pairs": int(x.shape[0]),
        "file_size_bytes": size_bytes,
        "file_size_gib": size_bytes / 1024**3,
        "label_order": combined["label_order"],
        "label_counts": {label: labels.count(label) for label in combined["label_order"]},
    }
    write_json(feature_dir / f"{split}.summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        split: combine_split(args.feature_dir, split, args.num_shards)
        for split in ("train", "valid")
    }
    write_json(args.feature_dir / "combined_summary.json", summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
