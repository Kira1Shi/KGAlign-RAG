import csv
import json
import random

DATA_DIR = "data"
SEED = 17


def flatten_ctx(si):
    if isinstance(si, str):
        return si
    parts = []
    if "question" in si:
        parts += [str(si.get("question", "")), str(si.get("passages", ""))]
    else:
        for k in ("name", "address", "city", "state", "categories"):
            if si.get(k):
                parts.append(f"{k}: {si[k]}")
        if isinstance(si.get("hours"), dict):
            parts.append("hours: " + "; ".join(f"{d} {h}" for d, h in si["hours"].items()))
        if isinstance(si.get("attributes"), dict):
            parts.append("attributes: " + "; ".join(
                f"{k} {v}" for k, v in si["attributes"].items() if v not in (None, "None")))
        if si.get("business_stars") is not None:
            parts.append(f"business stars: {si['business_stars']}")
        for r in si.get("review_info", []) or []:
            if isinstance(r, dict) and r.get("review_text"):
                parts.append(str(r["review_text"]))
    return "\n".join(p for p in parts if p)


def _ragtruth_test(data_dir):
    src = {r["source_id"]: r for r in
           (json.loads(l) for l in open(f"{data_dir}/RAGTruth/dataset/source_info.jsonl"))}
    resp = [json.loads(l) for l in open(f"{data_dir}/RAGTruth/dataset/response.jsonl")]
    test = [r for r in resp if r["split"] == "test"]
    random.shuffle(test)
    out = []
    for r in test:
        s = src[r["source_id"]]
        out.append({
            "ctx": flatten_ctx(s["source_info"]),
            "ans": r["response"],
            "task": s["task_type"],
            "y": 1 if r.get("labels") else 0,
        })
    return out


def _xsum_summaries(data_dir):
    path = (f"{data_dir}/xsum_hallucination_annotations/"
            "factuality_annotations_xsum_summaries.csv")
    seen, summaries = set(), []
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["bbcid"], row["system"])
            if key not in seen:
                seen.add(key)
                summaries.append(row["summary"])
    random.shuffle(summaries)
    return summaries


def build_data(n_rag=600, n_xsum=400, data_dir=DATA_DIR):
    random.seed(SEED)
    items = _ragtruth_test(data_dir)[:n_rag]
    xsum = _xsum_summaries(data_dir)[:n_xsum]
    return items, xsum
