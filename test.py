from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from extract import DEFAULT_BATCH_SIZE, get_graphs
from model import coefficients, dataset_features, load_models, metrics, predict
from utils import TASKS, read_split, write_graphs


DEFAULT_PREDICT_BATCH_SIZE = 1024


def evaluate(
    data_dir: Path,
    model_dir: Path,
    output_dir: Path,
    graph_path: Path | None = None,
    save_graph_path: Path | None = None,
    device: str | None = None,
    extract_batch_size: int = DEFAULT_BATCH_SIZE,
    predict_batch_size: int = DEFAULT_PREDICT_BATCH_SIZE,
    make_plots: bool = False,
) -> None:
    """Evaluate saved models using the prepared test split."""
    test_rows = read_split(data_dir, "test")
    graphs = get_graphs(test_rows, graph_path, device, extract_batch_size)
    if save_graph_path is not None and graph_path is None:
        write_graphs(graphs.values(), save_graph_path)

    test_features = dataset_features(test_rows, graphs)
    models = load_models(model_dir / "models.joblib")
    predictions = predict(test_features, models, batch_size=predict_batch_size)
    test_metrics = pd.DataFrame(
        [
            {
                "task_type": task,
                **metrics(predictions[predictions.task_type.eq(task)]),
            }
            for task in TASKS
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    test_features.to_csv(output_dir / "test_features.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    test_metrics.to_csv(output_dir / "test_metrics.csv", index=False)
    coefficients(models).to_csv(output_dir / "coefficients.csv", index=False)

    if make_plots:
        from plots import make_all_plots

        make_all_plots(output_dir)
    print(test_metrics.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
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
    parser.add_argument(
        "--predict-batch-size", type=int, default=DEFAULT_PREDICT_BATCH_SIZE
    )
    parser.add_argument(
        "--plots", action="store_true", help="Create evaluation plots after testing."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        args.data_dir,
        args.model_dir,
        args.output_dir,
        args.graphs,
        args.save_graphs,
        args.device,
        args.extract_batch_size,
        args.predict_batch_size,
        args.plots,
    )


if __name__ == "__main__":
    main()
