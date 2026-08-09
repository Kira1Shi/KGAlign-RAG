from __future__ import annotations

import argparse
import json
from pathlib import Path

from extract import DEFAULT_BATCH_SIZE, get_graphs
from model import (
    ENTITY_MATCH_THRESHOLD,
    FEATURES,
    REGULARIZATION,
    RISK_THRESHOLD,
    coefficients,
    dataset_features,
    fit_models,
    save_models,
)
from utils import TASKS, read_split


def train(
    data_dir: Path,
    output_dir: Path,
    graph_path: Path | None = None,
    save_graph_path: Path | None = None,
    device: str | None = None,
    extract_batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Fit one model per task using the prepared training split."""
    train_rows = read_split(data_dir, "train")
    graphs = get_graphs(
        train_rows,
        graph_path,
        device,
        extract_batch_size,
        save_graph_path,
    )

    train_features = dataset_features(train_rows, graphs)
    models = fit_models(train_features)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_features.to_csv(output_dir / "train_features.csv", index=False)
    coefficients(models).to_csv(output_dir / "coefficients.csv", index=False)
    save_models(models, output_dir / "models.joblib")
    (output_dir / "settings.json").write_text(
        json.dumps(
            {
                "training_split": "train",
                "entity_match_threshold": ENTITY_MATCH_THRESHOLD,
                "risk_threshold": RISK_THRESHOLD,
                "features": {task: list(FEATURES[task]) for task in TASKS},
                "regularization": REGULARIZATION,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved four task models to {output_dir / 'models.joblib'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    graph_options = parser.add_mutually_exclusive_group()
    graph_options.add_argument(
        "--graphs",
        type=Path,
        help="Precomputed graph JSONL. If omitted, graphs are extracted from text.",
    )
    graph_options.add_argument(
        "--save-graphs",
        type=Path,
        help="Save newly extracted graphs for reuse.",
    )
    parser.add_argument(
        "--device",
        help="GLiNER device when --graphs is omitted, for example cpu or cuda:0.",
    )
    parser.add_argument(
        "--extract-batch-size", type=int, default=DEFAULT_BATCH_SIZE
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        args.data_dir,
        args.output_dir,
        args.graphs,
        args.save_graphs,
        args.device,
        args.extract_batch_size,
    )


if __name__ == "__main__":
    main()
