from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


RAGTRUTH_REVISION = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
XSUM_ANNOTATIONS_REVISION = "4abdbb2c807cb93c1df1188a1992d20a55cdcc34"
XSUM_REVISION = "5b1805b1bc7ec93a1a03e5a81d80142b3b931a4e"

RAW_URLS = {
    "response.jsonl": (
        "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/"
        f"{RAGTRUTH_REVISION}/dataset/response.jsonl"
    ),
    "source_info.jsonl": (
        "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/"
        f"{RAGTRUTH_REVISION}/dataset/source_info.jsonl"
    ),
    "hallucination_annotations_xsum_summaries.csv": (
        "https://raw.githubusercontent.com/google-research-datasets/"
        "xsum_hallucination_annotations/"
        f"{XSUM_ANNOTATIONS_REVISION}/hallucination_annotations_xsum_summaries.csv"
    ),
    "eval_scores_xsum_summaries.csv": (
        "https://raw.githubusercontent.com/google-research-datasets/"
        "xsum_hallucination_annotations/"
        f"{XSUM_ANNOTATIONS_REVISION}/eval_scores_xsum_summaries.csv"
    ),
}

XSUM_SYSTEM_TO_SCORE_ID = {
    "BERTS2S": "bert_withckpt",
    "TConvS2S": "tconvs2s",
    "PtGen": "ptgen",
    "TranS2S": "bert_nockpt",
}

EXPECTED_ROWS = {
    "ragtruth": {"train": 10668, "val": 3564, "test": 3558},
    "xsum": {"train": 1196, "val": 388, "test": 408},
}


def download(url: str, destination: Path, force: bool = False) -> None:
    if destination.is_file() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "KGAlign-RAG"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as file:
        shutil.copyfileobj(response, file)
    temporary.replace(destination)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return rows


def ragtruth_label(labels: list[dict[str, Any]]) -> int:
    return int(any(not bool(label.get("implicit_true", False)) for label in labels))


def normalize_text(value: Any) -> str:
    return " ".join(str(value).split())


def serialize_source(task: str, source: Any) -> tuple[str, str]:
    if task == "QA":
        return (
            normalize_text(source.get("passages", "")),
            normalize_text(source.get("question", "")),
        )
    if task == "Summary":
        return normalize_text(source), ""
    if task == "Data2txt":
        serialized = json.dumps(
            source, ensure_ascii=False, separators=(",", ":")
        )
        return normalize_text(serialized), ""
    raise ValueError(f"Unsupported RAGTruth task: {task!r}")


def prepare_ragtruth(raw_dir: Path) -> pd.DataFrame:
    sources = {
        str(row["source_id"]): row
        for row in load_jsonl(raw_dir / "source_info.jsonl")
    }
    rows = []
    for response in load_jsonl(raw_dir / "response.jsonl"):
        source_id = str(response["source_id"])
        source = sources.get(source_id)
        if source is None:
            raise ValueError(f"Missing RAGTruth source {source_id}")
        task = str(source["task_type"])
        context, query = serialize_source(task, source["source_info"])
        labels = response.get("labels") or []
        rows.append({
            "sample_id": str(response["id"]),
            "source_id": source_id,
            "task_type": task,
            "context": context,
            "query": query,
            "response": normalize_text(response["response"]),
            "generator_model": str(response.get("model", "unknown")),
            "quality": str(response.get("quality", "good")),
            "hallucinated": ragtruth_label(labels),
            "raw_labels": json.dumps(
                labels, ensure_ascii=False, separators=(",", ":")
            ),
            "original_split": str(response.get("split", "")),
        })
    return pd.DataFrame(rows)


def normalize_bbcid(value: Any) -> str:
    return re.sub(r"\.0$", "", str(value).strip())


def prepare_xsum(raw_dir: Path) -> pd.DataFrame:
    from datasets import load_dataset

    annotations = pd.read_csv(
        raw_dir / "hallucination_annotations_xsum_summaries.csv"
    )
    scores = pd.read_csv(raw_dir / "eval_scores_xsum_summaries.csv")

    annotations["system_key"] = annotations["system"].map(
        XSUM_SYSTEM_TO_SCORE_ID
    )
    annotations["bbcid_norm"] = annotations["bbcid"].map(normalize_bbcid)
    annotations = (
        annotations
        .dropna(subset=["system_key", "bbcid_norm", "summary"])
        .drop_duplicates(subset=["system_key", "bbcid_norm", "summary"])
        .copy()
    )

    score_ids = scores["system_bbcid"].astype(str).str.rsplit(
        "_", n=1, expand=True
    )
    if score_ids.shape[1] != 2:
        raise ValueError("Expected XSum score IDs in '<system>_<bbcid>' format")
    scores["system_key"] = score_ids[0]
    scores["bbcid_norm"] = score_ids[1].map(normalize_bbcid)
    scores = (
        scores.groupby(["system_key", "bbcid_norm"], as_index=False)
        .agg(Faithful=("Faithful", "mean"), Factual=("Factual", "mean"))
    )

    frame = annotations.merge(
        scores,
        on=["system_key", "bbcid_norm"],
        how="inner",
        validate="many_to_one",
    )
    frame["Faithful"] = pd.to_numeric(frame["Faithful"], errors="coerce")
    frame["Factual"] = pd.to_numeric(frame["Factual"], errors="coerce")
    frame = frame.dropna(subset=["Faithful"]).copy()
    frame["hallucinated"] = (frame["Faithful"] < 1.0 - 1e-12).astype(int)
    frame["source_id"] = frame["bbcid_norm"].astype(str)
    frame["sample_id"] = frame["system_key"] + "::" + frame["source_id"]

    xsum = load_dataset(
        "EdinburghNLP/xsum",
        split="test",
        revision=XSUM_REVISION,
    )
    source_by_id = {
        normalize_bbcid(row["id"]): row["document"]
        for row in xsum
    }
    frame["context"] = frame["source_id"].map(source_by_id)
    frame = frame.dropna(subset=["context", "summary"]).copy()
    frame["query"] = ""
    frame["response"] = frame["summary"]
    frame["task_type"] = "XSum"
    frame["generator_model"] = frame["system"]

    leading = [
        "sample_id", "source_id", "task_type", "context", "query",
        "response", "generator_model", "hallucinated",
    ]
    trailing = [column for column in frame.columns if column not in leading]
    return frame[leading + trailing]


def source_digest(source_ids: set[str]) -> str:
    payload = "\n".join(sorted(source_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_frame(
    frame: pd.DataFrame,
    dataset: str,
    manifest: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    spec = manifest["datasets"][dataset]
    source_ids = set(frame["source_id"].astype(str))
    if len(source_ids) != spec["source_count"]:
        raise ValueError(
            f"{dataset}: expected {spec['source_count']} sources, "
            f"found {len(source_ids)}"
        )
    if source_digest(source_ids) != spec["source_ids_sha256"]:
        raise ValueError(f"{dataset}: source IDs differ from the paper manifest")

    validation = set(spec["validation"])
    test = set(spec["test"])
    if validation & test:
        raise ValueError(f"{dataset}: validation and test sources overlap")
    train = source_ids - validation - test
    assignments = {
        **{source_id: "train" for source_id in train},
        **{source_id: "val" for source_id in validation},
        **{source_id: "test" for source_id in test},
    }

    split_names = frame["source_id"].astype(str).map(assignments)
    if split_names.isna().any():
        raise ValueError(f"{dataset}: manifest does not cover every source")
    return {
        split: frame.loc[split_names.eq(split)].reset_index(drop=True)
        for split in ("train", "val", "test")
    }


def write_splits(
    datasets: dict[str, pd.DataFrame],
    output_dir: Path,
    manifest_path: Path,
    force: bool,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [
        output_dir / f"{dataset}_{split}.csv"
        for dataset in datasets
        for split in ("train", "val", "test")
    ]
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Output files already exist: {names}; use --force")

    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, frame in datasets.items():
        for split, part in split_frame(frame, dataset, manifest).items():
            expected = EXPECTED_ROWS[dataset][split]
            if len(part) != expected:
                raise ValueError(
                    f"{dataset}_{split}: expected {expected} rows, found {len(part)}"
                )
            path = output_dir / f"{dataset}_{split}.csv"
            part.to_csv(path, index=False)
            print(f"{path}: {len(part)} rows")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=root / "datasets")
    parser.add_argument(
        "--manifest", type=Path, default=root / "split_manifest.json"
    )
    parser.add_argument(
        "--force", action="store_true", help="replace existing generated CSVs"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="download the pinned raw files again",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for filename, url in RAW_URLS.items():
        download(url, args.raw_dir / filename, args.force_download)
    datasets = {
        "ragtruth": prepare_ragtruth(args.raw_dir),
        "xsum": prepare_xsum(args.raw_dir),
    }
    write_splits(datasets, args.output_dir, args.manifest, args.force)


if __name__ == "__main__":
    main()
