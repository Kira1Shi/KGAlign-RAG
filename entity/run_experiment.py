import argparse
import json
import os
import statistics
import sys
import time

from sklearn.metrics import roc_auc_score, average_precision_score

from data import build_data
from extractors import REGISTRY, get_extractor

TASKS = ("Summary", "QA", "Data2txt")


def parse_args():
    p = argparse.ArgumentParser(
        description="Entity extractor comparison on RAGTruth and XSum")
    p.add_argument("method", choices=sorted(REGISTRY))
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--n-xsum", type=int, default=400)
    p.add_argument("--budget", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--out", default="results/ner_results.json")
    return p.parse_args()


def load_cache(path):
    cache = {}
    if os.path.exists(path):
        for line in open(path):
            rec = json.loads(line)
            cache[rec["k"]] = set(rec["e"])
    return cache


def extract_all(fn, tasks, cache, path, budget):
    started = time.time()
    with open(path, "a") as f:
        for key, text in tasks:
            if key in cache:
                continue
            if budget and time.time() - started > budget:
                break
            t0 = time.time()
            ents = fn(text)
            cache[key] = ents
            f.write(json.dumps({"k": key, "e": sorted(ents),
                                "t": round(time.time() - t0, 4)}) + "\n")
            f.flush()
    return sum(1 for key, _ in tasks if key not in cache)


def entity_grounding(ans, ctx):
    return len(ans & ctx) / len(ans)


def evaluate(items, xsum, cache, path):
    ctx_sets = [cache[f"ctx{i}"] for i in range(len(items))]
    ans_sets = [cache[f"ans{i}"] for i in range(len(items))]
    xsum_sets = [cache[f"x{i}"] for i in range(len(xsum))]
    ctx_times = [json.loads(l)["t"] for l in open(path)
                 if json.loads(l)["k"].startswith("ctx")]

    ctx_counts = [len(s) for s in ctx_sets]
    ans_counts = [len(s) for s in ans_sets]

    risk, y, task, eg = [], [], [], []
    for i, (ctx, ans) in enumerate(zip(ctx_sets, ans_sets)):
        if not ans:
            continue
        score = entity_grounding(ans, ctx)
        eg.append(score)
        risk.append(1 - score)
        y.append(items[i]["y"])
        task.append(items[i]["task"])

    per_task = {}
    for t in TASKS:
        idx = [j for j, tt in enumerate(task) if tt == t]
        ys, rs = [y[j] for j in idx], [risk[j] for j in idx]
        per_task[t] = [round(roc_auc_score(ys, rs), 3) if len(set(ys)) > 1 else None,
                       len(idx)]

    xsum_counts = [len(s) for s in xsum_sets]
    return {
        "n": len(items),
        "ctx_med": statistics.median(ctx_counts),
        "ctx_mean": round(statistics.mean(ctx_counts), 1),
        "ctx_below20": round(100 * sum(v < 20 for v in ctx_counts) / len(ctx_counts)),
        "ans_med": statistics.median(ans_counts),
        "ans_mean": round(statistics.mean(ans_counts), 1),
        "ans_empty": round(100 * sum(v == 0 for v in ans_counts) / len(ans_counts), 1),
        "kept": len(y),
        "dropped": len(ans_counts) - len(y),
        "mean_eg": round(statistics.mean(eg), 3),
        "auroc": round(roc_auc_score(y, risk), 3),
        "auprc": round(average_precision_score(y, risk), 3),
        "per_task": per_task,
        "ms_per_ctx": round(1000 * statistics.mean(ctx_times)),
        "x_med": statistics.median(xsum_counts),
        "x_mean": round(statistics.mean(xsum_counts), 1),
        "x_empty": round(100 * sum(v == 0 for v in xsum_counts) / len(xsum_counts), 1),
        "x_below5": round(100 * sum(v < 5 for v in xsum_counts) / len(xsum_counts), 1),
    }


def save(out_path, method, res):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    all_res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    all_res[method] = res
    json.dump(all_res, open(out_path, "w"), indent=1, ensure_ascii=False)


def main():
    args = parse_args()
    items, xsum = build_data(args.n, args.n_xsum, args.data_dir)
    fn = get_extractor(args.method)

    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, f"{args.method}.jsonl")
    cache = load_cache(cache_path)

    tasks = []
    for i, item in enumerate(items):
        tasks.append((f"ctx{i}", item["ctx"]))
        tasks.append((f"ans{i}", item["ans"]))
    for i, summary in enumerate(xsum):
        tasks.append((f"x{i}", summary))

    remaining = extract_all(fn, tasks, cache, cache_path, args.budget)
    if remaining:
        print(f"[{args.method}] remaining: {remaining}, rerun to continue")
        sys.exit(3)

    res = evaluate(items, xsum, cache, cache_path)
    save(args.out, args.method, res)
    print(f"{args.method:14s} ctx_med={res['ctx_med']:5.1f} ans_med={res['ans_med']:5.1f} "
          f"empty={res['ans_empty']:4.1f}% EG={res['mean_eg']:.2f} "
          f"AUROC={res['auroc']:.3f} AUPRC={res['auprc']:.3f} | "
          f"XSum med={res['x_med']:4.1f} empty={res['x_empty']:4.1f}% "
          f"<5={res['x_below5']:4.1f}% | {res['ms_per_ctx']} ms/ctx")


if __name__ == "__main__":
    main()
