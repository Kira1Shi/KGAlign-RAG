from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

RANDOM_SEED = 42

TASKS = ("Data2txt", "QA", "Summary", "XSum")

SPLIT_FILES = {
    "train": ("ragtruth_train.csv", "xsum_train.csv"),
    "test": ("ragtruth_test.csv", "xsum_test.csv"),
}


def norm(text: Any) -> str:
    """Normalize an entity name before comparing it with another name."""
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def read_texts(path: Path) -> pd.DataFrame:
    """Read raw texts accepted by extract.py."""
    frame = pd.read_csv(path)
    required = {"sample_id", "task_type", "context", "response"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["sample_id"] = frame.sample_id.astype(str)
    frame["task_type"] = frame.task_type.astype(str).str.strip()
    frame["context"] = frame.context.fillna("").astype(str)
    frame["response"] = frame.response.fillna("").astype(str)
    if frame.sample_id.duplicated().any():
        raise ValueError(f"Duplicate sample_id in {path}")
    if not frame.task_type.isin(TASKS).all():
        raise ValueError(f"Unknown task_type in {path}")
    if frame.context.str.strip().eq("").any() or frame.response.str.strip().eq("").any():
        raise ValueError(f"Context and response must be non-empty in {path}")
    return frame


def _read_dataset(path: Path, split: str, dataset: str) -> pd.DataFrame:
    """Read one prepared RAGTruth or XSum file used by train.py."""
    frame = pd.read_csv(path)
    required = {
        "sample_id", "source_id", "task_type", "context", "response",
        "generator_model", "hallucinated",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["sample_id"] = dataset + "::" + frame.sample_id.astype(str)
    frame["source_id"] = dataset + "::" + frame.source_id.astype(str)
    frame["task_type"] = frame.task_type.astype(str).str.strip()
    frame["generator_model"] = frame.generator_model.fillna("unknown").astype(str)
    frame["hallucinated"] = pd.to_numeric(frame.hallucinated, errors="raise").astype(int)
    frame["split"] = split
    if not frame.task_type.isin(TASKS).all():
        raise ValueError(f"Unknown task_type in {path}")
    if not frame.hallucinated.isin([0, 1]).all():
        raise ValueError(f"Hallucination labels in {path} must be 0 or 1")
    return frame


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    """Combine the prepared RAGTruth and XSum files for train or test."""
    frames = []
    for filename in SPLIT_FILES[split]:
        dataset = "ragtruth" if filename.startswith("ragtruth_") else "xsum"
        frames.append(_read_dataset(data_dir / filename, split, dataset))
    return pd.concat(frames, ignore_index=True)


def check_no_overlap(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Prevent sample or source leakage from train into test."""
    if set(train.sample_id) & set(test.sample_id):
        raise ValueError("Train and test contain the same sample_id")
    if set(train.source_id) & set(test.source_id):
        raise ValueError("Train and test contain the same source_id")


def _check_graph(item: Any, path: Path, line_number: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"{path}:{line_number}: graph row must be an object")
    if not item.get("sample_id") or item.get("task_type") not in TASKS:
        raise ValueError(f"{path}:{line_number}: invalid sample_id or task_type")
    for name in ("context_graph", "response_graph"):
        graph = item.get(name)
        if not isinstance(graph, dict):
            raise ValueError(f"{path}:{line_number}: missing {name}")
        if not isinstance(graph.get("entities"), list):
            raise ValueError(f"{path}:{line_number}: malformed {name} entities")
        if not isinstance(graph.get("relations"), list):
            raise ValueError(f"{path}:{line_number}: malformed {name} relations")
    item["sample_id"] = str(item["sample_id"])
    return item


def iter_graphs(path: Path) -> Iterator[dict[str, Any]]:
    """Yield precomputed graph pairs one at a time from JSONL."""
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                yield _check_graph(json.loads(line), path, line_number)


def graph_batches(path: Path, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    """Read a graph JSONL file in bounded batches."""
    batch = []
    for graph in iter_graphs(path):
        batch.append(graph)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def read_graphs(
    path: Path, sample_ids: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Load all graphs, or only the requested sample IDs, into memory."""
    graphs = {}
    for graph in iter_graphs(path):
        sample_id = graph["sample_id"]
        if sample_ids is not None and sample_id not in sample_ids:
            continue
        if sample_id in graphs:
            raise ValueError(f"Duplicate graph for sample {sample_id}")
        graphs[sample_id] = graph
    return graphs


def write_graphs(graphs: Iterable[dict[str, Any]], path: Path) -> None:
    """Write graph pairs as JSONL, one input example per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for graph in graphs:
            file.write(json.dumps(graph, ensure_ascii=False) + "\n")
