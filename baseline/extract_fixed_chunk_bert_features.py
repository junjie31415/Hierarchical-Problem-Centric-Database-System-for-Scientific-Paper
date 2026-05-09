import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


LABELS = ("A_TO_B", "B_TO_A", "PEER", "NONE")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract [5 A chunks + 5 B chunks, hidden] BERT pair features."
    )
    parser.add_argument("--split", choices=("train", "valid"), required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("baseline/outputs/fixed_5x5_bert_features"),
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--chunks-per-side", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--pooling", choices=("mean", "cls"), default="mean")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument(
        "--include-reference-count",
        dest="include_reference_count",
        action="store_true",
    )
    parser.add_argument(
        "--no-reference-count",
        dest="include_reference_count",
        action="store_false",
    )
    parser.set_defaults(include_reference_count=True)
    parser.add_argument("--no-reference-titles", action="store_true")
    parser.add_argument(
        "--max-reference-titles",
        type=int,
        default=0,
        help="Maximum reference titles per paper. Use 0 to include all available titles.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def paper_text(
    paper: dict[str, Any],
    include_reference_count: bool,
    include_reference_titles: bool,
    max_reference_titles: int,
) -> str:
    parts = [
        ("Title", paper.get("title")),
        ("Abstract", paper.get("abstract")),
        ("Year", paper.get("publication_year")),
        ("Venue", paper.get("venue")),
    ]
    if include_reference_count:
        parts.append(("Reference count", paper.get("reference_count")))
    if include_reference_titles:
        reference_titles = paper.get("reference_titles") or []
        if not isinstance(reference_titles, list):
            reference_titles = []
        if max_reference_titles > 0:
            reference_titles = reference_titles[:max_reference_titles]
        reference_text = " | ".join(
            clean_text(title) for title in reference_titles if clean_text(title)
        )
        parts.append(("Reference titles", reference_text))

    text_parts: list[str] = []
    for key, value in parts:
        text = clean_text(value)
        if text:
            text_parts.append(f"{key}: {text}")
    return " ".join(text_parts)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


class ChunkDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        papers: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        chunks_per_side: int,
        include_reference_count: bool,
        include_reference_titles: bool,
        max_reference_titles: int,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        content_length = max_length - 2
        if content_length <= 0:
            raise ValueError("--max-length must be at least 3.")

        for paper_index, paper in enumerate(papers):
            text = paper_text(
                paper,
                include_reference_count=include_reference_count,
                include_reference_titles=include_reference_titles,
                max_reference_titles=max_reference_titles,
            )
            token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
            chunks = [
                token_ids[start : start + content_length]
                for start in range(0, len(token_ids), content_length)
            ]
            if not chunks and token_ids:
                chunks = [token_ids]
            used_chunks = chunks[:chunks_per_side]
            for chunk_index, chunk in enumerate(used_chunks):
                encoded = tokenizer.prepare_for_model(
                    chunk,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                )
                self.rows.append(
                    {
                        "paper_index": paper_index,
                        "chunk_index": chunk_index,
                        "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
                        "attention_mask": torch.tensor(
                            encoded["attention_mask"], dtype=torch.long
                        ),
                    }
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def collate_chunks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "paper_index": torch.tensor([row["paper_index"] for row in rows], dtype=torch.long),
        "chunk_index": torch.tensor([row["chunk_index"] for row in rows], dtype=torch.long),
        "input_ids": torch.stack([row["input_ids"] for row in rows], dim=0),
        "attention_mask": torch.stack([row["attention_mask"] for row in rows], dim=0),
    }


def pool_tokens(outputs: Any, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    hidden = outputs.last_hidden_state
    if pooling == "cls":
        return hidden[:, 0, :]
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def extract_paper_features(
    papers: list[dict[str, Any]],
    tokenizer: Any,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    hidden_size = int(model.config.hidden_size)
    features = torch.zeros(
        (len(papers), args.chunks_per_side, hidden_size),
        dtype=torch.float16 if args.dtype == "fp16" else torch.float32,
    )
    chunk_mask = torch.zeros((len(papers), args.chunks_per_side), dtype=torch.bool)
    diagnostics: list[dict[str, Any]] = []

    content_length = args.max_length - 2
    for paper in papers:
        text = paper_text(
            paper,
            include_reference_count=args.include_reference_count,
            include_reference_titles=not args.no_reference_titles,
            max_reference_titles=args.max_reference_titles,
        )
        token_count = len(tokenizer.encode(text, add_special_tokens=False, truncation=False))
        needed_chunks = max(1, math.ceil(token_count / content_length)) if token_count else 0
        diagnostics.append(
            {
                "paper_id": str(paper["paper_id"]),
                "title": paper.get("title"),
                "token_count_no_special": token_count,
                "needed_chunks": needed_chunks,
                "used_chunks": min(needed_chunks, args.chunks_per_side),
                "truncated": needed_chunks > args.chunks_per_side,
            }
        )

    dataset = ChunkDataset(
        papers=papers,
        tokenizer=tokenizer,
        max_length=args.max_length,
        chunks_per_side=args.chunks_per_side,
        include_reference_count=args.include_reference_count,
        include_reference_titles=not args.no_reference_titles,
        max_reference_titles=args.max_reference_titles,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_chunks,
    )

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = pool_tokens(outputs, attention_mask, args.pooling).detach().cpu()
            if args.dtype == "fp16":
                pooled = pooled.to(torch.float16)
            else:
                pooled = pooled.to(torch.float32)
            for row_index, paper_index in enumerate(batch["paper_index"].tolist()):
                chunk_index = int(batch["chunk_index"][row_index].item())
                features[paper_index, chunk_index] = pooled[row_index]
                chunk_mask[paper_index, chunk_index] = True

    return features, chunk_mask, diagnostics


def shard_rows(rows: list[dict[str, Any]], shard_id: int, num_shards: int) -> list[dict[str, Any]]:
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("--shard-id must be in [0, num_shards).")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_id]


def build_pair_tensor(
    pair_rows: list[dict[str, Any]],
    papers_by_id: dict[str, int],
    paper_features: torch.Tensor,
    paper_chunk_mask: torch.Tensor,
) -> dict[str, Any]:
    if pair_rows:
        hidden_size = int(paper_features.shape[-1])
        dtype = paper_features.dtype
    else:
        hidden_size = int(paper_features.shape[-1])
        dtype = paper_features.dtype
    x = torch.zeros((len(pair_rows), 10, hidden_size), dtype=dtype)
    chunk_mask = torch.zeros((len(pair_rows), 10), dtype=torch.bool)
    y = torch.zeros((len(pair_rows),), dtype=torch.long)
    pair_ids: list[str] = []
    paper_a_ids: list[str] = []
    paper_b_ids: list[str] = []
    labels: list[str] = []

    for row_index, row in enumerate(pair_rows):
        label = str(row["label"])
        if label not in LABEL_TO_ID:
            raise ValueError(f"Unsupported label: {label}")
        paper_a_id = str(row["paper_a_id"])
        paper_b_id = str(row["paper_b_id"])
        paper_a_index = papers_by_id[paper_a_id]
        paper_b_index = papers_by_id[paper_b_id]
        chunks_per_side = int(paper_features.shape[1])
        x[row_index, :chunks_per_side] = paper_features[paper_a_index]
        x[row_index, chunks_per_side : chunks_per_side * 2] = paper_features[paper_b_index]
        chunk_mask[row_index, :chunks_per_side] = paper_chunk_mask[paper_a_index]
        chunk_mask[row_index, chunks_per_side : chunks_per_side * 2] = paper_chunk_mask[
            paper_b_index
        ]
        y[row_index] = LABEL_TO_ID[label]
        pair_ids.append(str(row["pair_id"]))
        paper_a_ids.append(paper_a_id)
        paper_b_ids.append(paper_b_id)
        labels.append(label)

    return {
        "x": x,
        "chunk_mask": chunk_mask,
        "y": y,
        "labels": labels,
        "pair_ids": pair_ids,
        "paper_a_ids": paper_a_ids,
        "paper_b_ids": paper_b_ids,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    serializable_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    papers_path = args.dataset_dir / f"{args.split}_papers.jsonl"
    labels_path = args.dataset_dir / f"{args.split}_labels.jsonl"
    all_papers = read_jsonl(papers_path)
    labels = read_jsonl(labels_path)
    if args.limit_pairs:
        labels = labels[: args.limit_pairs]
    shard_labels = shard_rows(labels, args.shard_id, args.num_shards)
    needed_paper_ids = {
        str(row[key]) for row in shard_labels for key in ("paper_a_id", "paper_b_id")
    }
    papers = [
        paper for paper in all_papers if str(paper["paper_id"]) in needed_paper_ids
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    if args.dtype == "fp16" and device.type == "cuda":
        model = model.half()

    paper_features, paper_chunk_mask, paper_diagnostics = extract_paper_features(
        papers=papers,
        tokenizer=tokenizer,
        model=model,
        args=args,
        device=device,
    )

    papers_by_id = {str(paper["paper_id"]): index for index, paper in enumerate(papers)}
    pair_data = build_pair_tensor(
        pair_rows=shard_labels,
        papers_by_id=papers_by_id,
        paper_features=paper_features,
        paper_chunk_mask=paper_chunk_mask,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_name = f"{args.split}_shard{args.shard_id:02d}of{args.num_shards:02d}"
    output_path = args.output_dir / f"{shard_name}.pt"
    summary_path = args.output_dir / f"{shard_name}.summary.json"

    payload = {
        **pair_data,
        "label_order": list(LABELS),
        "feature_shape": list(pair_data["x"].shape),
        "feature_description": (
            "x has shape [num_pairs, 10, hidden]. Slots 0-4 are paper A chunks; "
            "slots 5-9 are paper B chunks. Each chunk is one pooled BERT vector."
        ),
        "config": serializable_config,
    }
    torch.save(payload, output_path)

    feature_bytes = int(pair_data["x"].numel() * pair_data["x"].element_size())
    summary = {
        "split": args.split,
        "output": str(output_path),
        "num_pairs_in_shard": len(shard_labels),
        "num_pairs_before_sharding": len(labels),
        "num_papers_in_shard_pairs": len(papers),
        "num_papers_in_split": len(all_papers),
        "x_shape": list(pair_data["x"].shape),
        "x_dtype": str(pair_data["x"].dtype),
        "x_size_bytes": feature_bytes,
        "x_size_gib": feature_bytes / 1024**3,
        "chunk_mask_shape": list(pair_data["chunk_mask"].shape),
        "label_order": list(LABELS),
        "paper_truncated_count": sum(1 for row in paper_diagnostics if row["truncated"]),
        "paper_max_needed_chunks": max(
            (int(row["needed_chunks"]) for row in paper_diagnostics), default=0
        ),
        "pooling": args.pooling,
        "note": "These are BERT chunk features, not trained 4-class logits.",
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
