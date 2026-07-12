import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["spacy_sm", "spacy_md", "nltk_ne", "noun_chunks",
         "nc_clean", "content_nouns", "hybrid"]
LABELS = ["spaCy sm", "spaCy md", "NLTK NE", "noun_chunks",
          "nc_clean", "content_nouns", "hybrid"]
COLORS = ["#b0b7c3", "#b0b7c3", "#b0b7c3", "#7fa8d9",
          "#4e79a7", "#2f5c8a", "#f28e2b"]


def main(results_path="results/ner_results.json",
         out_path="assets/fig_ner_comparison.png"):
    r = json.load(open(results_path))
    methods = [m for m in ORDER if m in r]
    labels = [LABELS[ORDER.index(m)] for m in methods]
    colors = [COLORS[ORDER.index(m)] for m in methods]
    x = np.arange(len(methods))
    width = 0.38

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    ax[0].bar(x, [r[m]["auroc"] for m in methods], color=colors)
    ax[0].axhline(0.5, ls="--", c="r", lw=1, label="случайный уровень")
    ax[0].axhline(0.89, ls=":", c="g", lw=1, label="HalluGraph, юр. тексты")
    ax[0].set_ylim(0.4, 1.0)
    ax[0].set_title("AUROC метрики Entity Grounding (RAGTruth)")
    ax[0].legend(fontsize=8)
    for i, m in enumerate(methods):
        ax[0].text(i, r[m]["auroc"] + 0.008, f"{r[m]['auroc']:.3f}",
                   ha="center", fontsize=8)

    ax[1].bar(x - width / 2, [r[m]["ctx_med"] for m in methods], width,
              label="контекст", color="#4e79a7")
    ax[1].bar(x + width / 2, [r[m]["ans_med"] for m in methods], width,
              label="ответ", color="#f28e2b")
    ax[1].axhline(20, ls="--", c="r", lw=1, label="порог 20 сущностей")
    ax[1].set_title("Медиана уникальных сущностей (RAGTruth)")
    ax[1].legend(fontsize=8)

    ax[2].bar(x - width / 2, [r[m]["ans_empty"] for m in methods], width,
              label="пустой граф ответа, RAGTruth", color="#4e79a7")
    ax[2].bar(x + width / 2, [r[m]["x_empty"] for m in methods], width,
              label="пустой граф реферата, XSum", color="#e15759")
    ax[2].set_title("Доля пустых графов, %")
    ax[2].legend(fontsize=8)

    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    fig.suptitle("Сравнение методов выделения сущностей на реальных данных "
                 "RAGTruth и XSum")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)


if __name__ == "__main__":
    main()
