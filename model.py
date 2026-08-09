from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import TASKS, norm, RANDOM_SEED


ENTITY_MATCH_THRESHOLD = 0.86
RISK_THRESHOLD = 0.5

# These values have been empirically obrained
CONTAINED_NAME_SIMILARITY = 0.90
ACRONYM_SIMILARITY = 0.92

# Entity support measures how well response entities match the
# context. Novel endpoint features measure relations whose head or tail is not
# grounded. Count, ratio, and confidence features describe graph size and the
# GLiNER-Relex's confidence of the extracted entities.
FEATURES = {
    "Data2txt": (
        "weighted_ungrounded_entity_rate",
        "response_context_entity_ratio",
        "log_response_relation_count",
        "fuzzy_novel_endpoint_relation_rate",
        "fuzzy_ungrounded_entity_rate",
        "entity_confidence_gap",
    ),
    "QA": (
        "ungrounded_response_entity_count_log",
        "min_entity_support_similarity",
    ),
    "Summary": (
        "mean_entity_support_similarity",
        "mean_response_entity_confidence",
    ),
    "XSum": (
        "ungrounded_response_entity_count_log",
        "novel_endpoint_relation_rate",
    ),
}

REGULARIZATION = {
    "Data2txt": 10.0,
    "QA": 0.1,
    "Summary": 0.1,
    "XSum": 0.1,
}


def canonical_triple(relation: dict[str, Any]) -> tuple[str, str, str]:
    """Create the key used to remove duplicate relation predictions."""
    head = norm(relation["head_text"])
    tail = norm(relation["tail_text"])
    relation_id = str(relation["relation_id"])
    if relation_id == "spouse_of" and tail < head:
        head, tail = tail, head
    return head, relation_id, tail


def text_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", norm(text), flags=re.UNICODE))


@lru_cache(maxsize=500_000)
def name_similarity(left: str, right: str) -> float:
    """Compare two entity names using simple, inspectable string heuristics."""
    left = norm(left)
    right = norm(right)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    # Similar-looking dates and numbers must not be treated as aliases.
    left_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", left))
    right_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", right))
    if (left_numbers or right_numbers) and left_numbers != right_numbers:
        return 0.0

    left_tokens = text_tokens(left)
    right_tokens = text_tokens(right)
    left_set = set(left_tokens)
    right_set = set(right_tokens)

    overlap = len(left_set & right_set)
    token_f1 = 0.0
    if left_set and right_set:
        token_f1 = 2 * overlap / (len(left_set) + len(right_set))

    # "the United States" contains the full name "United States".
    containment = 0.0
    shorter, longer = sorted((left_tokens, right_tokens), key=len)
    if shorter and set(shorter).issubset(set(longer)):
        # Do not match a long name merely because both names contain a short
        # generic token such as "the", "of", or "in".
        if len(shorter) > 1 or len(shorter[0]) >= 4:
            containment = CONTAINED_NAME_SIMILARITY

    # An exact initials match, such as "US" and "United States", is an alias.
    acronym_match = 0.0
    if len(left_tokens) == 1 and len(right_tokens) > 1:
        initials = "".join(token[0] for token in right_tokens)
        if left_tokens[0] == initials:
            acronym_match = ACRONYM_SIMILARITY
    elif len(right_tokens) == 1 and len(left_tokens) > 1:
        initials = "".join(token[0] for token in left_tokens)
        if right_tokens[0] == initials:
            acronym_match = ACRONYM_SIMILARITY

    character_similarity = SequenceMatcher(None, left, right).ratio()
    return max(token_f1, containment, acronym_match, character_similarity)


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def confidence_weighted_bad_rate(confidences: list[float], bad: list[bool]) -> float:
    """Fraction of entity confidence assigned to ungrounded entities."""
    if not confidences:
        return 0.0
    total_confidence = sum(confidences)
    if total_confidence == 0:
        return sum(bad) / len(bad)
    bad_confidence = sum(
        confidence for confidence, is_bad in zip(confidences, bad) if is_bad
    )
    return bad_confidence / total_confidence


def graph_entities(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Return unique entities, including relation endpoints missed by NER."""
    entities = {}
    for entity in graph["entities"]:
        key = norm(entity["text"])
        confidence = float(entity.get("score", 0.0))
        if key and (key not in entities or confidence > entities[key]["score"]):
            entities[key] = {"text": str(entity["text"]), "score": confidence}

    for relation in graph["relations"]:
        for endpoint in ("head", "tail"):
            text = str(relation[f"{endpoint}_text"])
            key = norm(text)
            if key and key not in entities:
                entities[key] = {"text": text, "score": 0.0}
    return list(entities.values())


def feature_row(graph_pair: dict[str, Any]) -> dict[str, Any]:
    """Compute all features used by at least one of the four final models."""
    sample_id = str(graph_pair["sample_id"])
    task = str(graph_pair["task_type"])
    if task not in FEATURES:
        raise ValueError(f"Unknown task {task!r} for sample {sample_id}")

    context_graph = graph_pair["context_graph"]
    response_graph = graph_pair["response_graph"]
    context_entities = graph_entities(context_graph)
    response_entities = graph_entities(response_graph)
    context_names = [entity["text"] for entity in context_entities]
    context_name_set = {norm(name) for name in context_names}
    response_name_set = {
        norm(entity["text"]) for entity in response_entities
    }

    support = [
        max(
            (name_similarity(entity["text"], name) for name in context_names),
            default=0.0,
        )
        for entity in response_entities
    ]
    fuzzy_ungrounded = [value < ENTITY_MATCH_THRESHOLD for value in support]
    response_confidences = [float(entity["score"]) for entity in response_entities]
    positive_response_confidences = [value for value in response_confidences if value > 0]
    positive_context_confidences = [
        float(entity["score"]) for entity in context_entities if entity["score"] > 0
    ]

    response_relations = response_graph["relations"]
    unique_relations = {
        canonical_triple(relation): relation for relation in response_relations
    }

    fuzzy_novel_relations = 0
    for relation in response_relations:
        head_supported = max(
            (name_similarity(relation["head_text"], name) for name in context_names),
            default=0.0,
        ) >= ENTITY_MATCH_THRESHOLD
        tail_supported = max(
            (name_similarity(relation["tail_text"], name) for name in context_names),
            default=0.0,
        ) >= ENTITY_MATCH_THRESHOLD
        if not head_supported or not tail_supported:
            fuzzy_novel_relations += 1

    strict_novel_relations = 0
    for relation in unique_relations.values():
        head = norm(relation["head_text"])
        tail = norm(relation["tail_text"])
        if head not in context_name_set or tail not in context_name_set:
            strict_novel_relations += 1

    response_entity_count = len(response_name_set)
    context_entity_count = len(context_name_set)
    response_relation_count = len(unique_relations)
    ungrounded_entity_count = len(response_name_set - context_name_set)

    status = "predicted"
    if context_entity_count == 0:
        status = "no_context_entities"
    elif response_entity_count == 0:
        status = "no_response_entities"

    return {
        "sample_id": sample_id,
        "task_type": task,
        "status": status,
        "can_predict": int(status == "predicted"),
        "context_entity_count": context_entity_count,
        "response_entity_count": response_entity_count,
        "response_relation_count": response_relation_count,
        "weighted_ungrounded_entity_rate": confidence_weighted_bad_rate(
            response_confidences, fuzzy_ungrounded
        ),
        "response_context_entity_ratio": (
            response_entity_count / max(context_entity_count, 1)
        ),
        "log_response_relation_count": math.log1p(response_relation_count),
        "fuzzy_novel_endpoint_relation_rate": (
            fuzzy_novel_relations / len(response_relations)
            if response_relations else 0.0
        ),
        "fuzzy_ungrounded_entity_rate": (
            sum(fuzzy_ungrounded) / len(fuzzy_ungrounded)
            if fuzzy_ungrounded else 0.0
        ),
        "entity_confidence_gap": (
            mean_or_zero(positive_response_confidences)
            - mean_or_zero(positive_context_confidences)
        ),
        "ungrounded_response_entity_count_log": math.log1p(ungrounded_entity_count),
        "min_entity_support_similarity": min(support, default=0.0),
        "mean_entity_support_similarity": mean_or_zero(support),
        "mean_response_entity_confidence": mean_or_zero(positive_response_confidences),
        "novel_endpoint_relation_rate": (
            strict_novel_relations / response_relation_count
            if response_relation_count else 0.0
        ),
    }


def make_features(graphs: list[dict[str, Any]] | dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Public interface for computing features from precomputed graphs."""
    graph_rows = graphs.values() if isinstance(graphs, dict) else graphs
    rows = [feature_row(graph) for graph in graph_rows]
    return pd.DataFrame(rows)


def dataset_features(
    frame: pd.DataFrame, graphs: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    """Attach dataset labels and metadata to precomputed-graph features."""
    missing_graphs = sorted(set(frame.sample_id) - set(graphs))
    if missing_graphs:
        raise ValueError(f"Missing graphs for {len(missing_graphs)} dataset rows")

    selected_graphs = {sample_id: graphs[sample_id] for sample_id in frame.sample_id}
    features = make_features(selected_graphs)
    metadata_columns = [
        "sample_id", "task_type", "source_id", "generator_model",
        "hallucinated", "split",
    ]
    metadata = frame[metadata_columns]
    return features.merge(
        metadata,
        on=["sample_id", "task_type"],
        how="left",
        validate="one_to_one",
    )


def create_model(task: str):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=REGULARIZATION[task],
            penalty="l2",
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=RANDOM_SEED,
        ),
    )


def fit_models(features: pd.DataFrame) -> dict[str, Any]:
    models = {}
    for task in TASKS:
        task_rows = features[
            features.task_type.eq(task) & features.can_predict.eq(1)
        ]
        model = create_model(task)
        model.fit(task_rows[list(FEATURES[task])], task_rows.hallucinated)
        models[task] = model
    return models


def predict(features: pd.DataFrame, models: dict[str, Any], batch_size: int = 1024) -> pd.DataFrame:
    """Predict in batches so large graph files do not create large model inputs."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    scored = features.copy()
    risk_scores = np.full(len(scored), np.nan)
    predicted_classes = np.full(len(scored), np.nan)

    for task in TASKS:
        rows = scored.task_type.eq(task) & scored.can_predict.eq(1)
        if not rows.any():
            continue
        positions = np.flatnonzero(rows.to_numpy())
        for start in range(0, len(positions), batch_size):
            batch_positions = positions[start : start + batch_size]
            batch = scored.iloc[batch_positions][list(FEATURES[task])]
            risk = models[task].predict_proba(batch)[:, 1]
            risk_scores[batch_positions] = risk
            predicted_classes[batch_positions] = (risk >= RISK_THRESHOLD).astype(int)

    scored["risk_score"] = risk_scores
    scored["prediction"] = pd.array(predicted_classes, dtype="Int64")
    return scored


def metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    predicted = predictions[predictions.risk_score.notna()]
    result = {
        "n_samples": len(predictions),
        "n_predicted": len(predicted),
        "coverage": len(predicted) / len(predictions),
        "hallucination_rate": float(predictions.hallucinated.mean()),
    }
    if predicted.hallucinated.nunique() < 2:
        return result

    labels = predicted.hallucinated.astype(int)
    risk = predicted.risk_score.astype(float)
    classes = predicted.prediction.astype(int)
    result.update({
        "roc_auc": roc_auc_score(labels, risk),
        "pr_auc": average_precision_score(labels, risk),
        "accuracy": accuracy_score(labels, classes),
        "balanced_accuracy": balanced_accuracy_score(labels, classes),
        "precision": precision_score(labels, classes, zero_division=0),
        "recall": recall_score(labels, classes, zero_division=0),
        "f1": f1_score(labels, classes, zero_division=0),
    })
    return result


def save_models(models: dict[str, Any], path: Path) -> None:
    """Save fitted models together with the feature names they expect."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "features": FEATURES,
            "models": models,
        },
        path,
    )


def load_models(path: Path) -> dict[str, Any]:
    """Load models and reject files created for a different feature format."""
    saved = joblib.load(path)
    saved_features = {
        task: tuple(names) for task, names in saved.get("features", {}).items()
    }
    if saved_features != FEATURES:
        raise ValueError("The model file uses a different feature set")
    return saved["models"]


def coefficients(models: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        coefficients = models[task].named_steps["logisticregression"].coef_[0]
        for feature, coefficient in zip(FEATURES[task], coefficients):
            rows.append({
                "task_type": task,
                "feature": feature,
                "standardized_coefficient": float(coefficient),
            })
    return pd.DataFrame(rows)
