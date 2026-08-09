# KGAlign-RAG

Official implementation of **Using Knowledge Graph Alignment to Measure Answer
Support in RAG**.

Ivan Kosmynin, Kira Shilovskaya, Elizaveta Ivanova, Lyubov Razumova, Daria
Kotova, and Alexey Zaytsev.

## Overview

KGAlign-RAG is a black-box hallucination detector for retrieval-augmented
generation. It compares knowledge graphs extracted from the retrieved context
and the generated response, then estimates response-level hallucination risk
from a small set of graph-alignment features.

The graph extractor is
[GLiNER-Relex](https://arxiv.org/abs/2605.10108), a state-of-the-art unified
zero-shot model for joint named-entity recognition and relation extraction.
Entity support, relation endpoints, graph size, and extraction confidence are
converted into two to six task-specific features and scored by logistic
regression.

## Results

Held-out performance:

| Task | ROC AUC | PR AUC | Balanced accuracy | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Data2Text | 0.749 | 0.825 | 0.721 | 0.717 | 0.772 |
| QA | 0.743 | 0.454 | 0.666 | 0.652 | 0.475 |
| RAGTruth Summary | 0.578 | 0.339 | 0.566 | 0.562 | 0.419 |
| XSum | 0.662 | 0.953 | 0.640 | 0.613 | 0.747 |

## Installation

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data

The hallucination detector is trained and evaluated on two datasets:

- [RAGTruth](https://github.com/ParticleMedia/RAGTruth) contains nearly 18,000
  responses produced by six language models for question answering,
  summarization, and data-to-text generation.
- [XSum hallucination annotations](https://github.com/google-research-datasets/xsum_hallucination_annotations)
  contain human faithfulness judgments for system-generated, one-sentence
  summaries of BBC articles.

Raw data are downloaded from the original repositories and converted locally:

```bash
python prepare_data.py
```

The script uses pinned upstream revisions, joins the XSum annotations with the
corresponding source articles, and applies the source-disjoint partitions in
`split_manifest.json`. It creates:

```text
datasets/
|-- ragtruth_train.csv
|-- ragtruth_val.csv
|-- ragtruth_test.csv
|-- xsum_train.csv
|-- xsum_val.csv
`-- xsum_test.csv
```

## Training

```bash
python train.py \
  --data-dir datasets \
  --output-dir artifacts/model \
  --save-graphs artifacts/train.graphs.jsonl \
  --device cuda:0
```

Training fits one classifier for each of Data2Text, QA, RAGTruth Summary, and
XSum. The output directory contains the fitted models, extracted features,
standardized coefficients, and model settings.

## Evaluation

```bash
python test.py \
  --data-dir datasets \
  --model-dir artifacts/model \
  --output-dir artifacts/evaluation \
  --save-graphs artifacts/test.graphs.jsonl \
  --device cuda:0 \
  --plots
```

Evaluation writes per-example risk scores, task-level metrics, coefficients,
and figures. Graph extraction can be skipped in subsequent runs by replacing
`--save-graphs PATH` with `--graphs PATH`.
