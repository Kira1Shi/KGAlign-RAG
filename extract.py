from __future__ import annotations

import gc
import warnings
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from tqdm.auto import tqdm

from utils import TASKS, norm, read_graphs


MODEL_NAME = "knowledgator/gliner-relex-multi-v1.0"
DEFAULT_BATCH_SIZE = 8

ENTITY_LABELS = (
    "person",
    "organization",
    "location",
    "date or time",
    "event",
    "work or product",
    "role or title",
    "language",
    "numeric value",
    "other named entity",
)

# Each pair is (stable canonical ID, model-facing GLiNER label).
# Separated relation schemes for different tasks proved to be beneficial
# due to the natural variety of the used lexics
RELATIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "Data2txt": (
        ("part_of", "part of"),
        ("located_in", "located in"),
        ("capital_of", "capital of"),
        ("born_in", "born in"),
        ("died_in", "died in"),
        ("citizen_of", "citizen of"),
        ("date_of_birth", "date of birth"),
        ("date_of_death", "date of death"),
        ("started_on", "started on"),
        ("ended_on", "ended on"),
        ("occurred_on", "occurred on"),
        ("member_of", "member of"),
        ("employed_by", "employed by"),
        ("educated_at", "educated at"),
        ("holds_role_in", "holds role in"),
        ("founded", "founded"),
        ("owns", "owns"),
        ("acquired", "acquired"),
        ("operates", "operates"),
        ("parent_of", "parent of"),
        ("spouse_of", "spouse of"),
        ("created", "created"),
        ("published", "published"),
        ("uses_language", "uses language"),
        ("participated_in", "participated in"),
        ("won", "won"),
        ("defeated", "defeated"),
        ("supports", "supports"),
        ("opposes", "opposes"),
        ("causes", "causes"),
    ),
    "QA": (
        ("located_in", "entity is located in place"),
        ("part_of", "component is part of whole"),
        ("made_of", "object is made of material"),
        ("has_property", "entity has a non-numeric property, state, amenity, or evaluation"),
        ("has_value", "entity has a numeric, monetary, rating, or measured value"),
        ("used_for", "item or method is used for purpose"),
        ("requires", "action or process requires a condition, input, or prerequisite"),
        ("causes", "cause produces outcome"),
        ("prevents", "action prevents outcome"),
        ("increases", "factor increases quantity or state"),
        ("decreases", "factor decreases quantity or state"),
        ("depends_on", "quantity or state depends on factor"),
        ("exceeds", "first entity is greater, higher, larger, faster, or more than second"),
        ("applies_to", "rule or policy applies to group or case"),
        ("permits", "rule or system permits action"),
        ("procedure_step", "procedure includes step"),
        ("uses_parameter", "action uses a tool, material, setting, order, or duration"),
        ("treats", "treatment treats condition"),
        ("supported_by_document", "claim is supported or mentioned by document"),
        ("unsupported_by_document", "claim is absent or unsupported by document"),
        ("occurs_at", "event occurs at time"),
        ("experiences", "entity undergoes an event or incident"),
        ("communicates", "source states, reports, announces, predicts, warns, or discusses claim"),
        ("should_do", "actor should or is recommended to perform action"),
        ("risk_or_exposure", "entity faces risk or exposure from hazard or outcome"),
    ),
    "Summary": (
        ("located_in", "entity is located in place"),
        ("part_of", "component is part of whole"),
        ("has_property", "entity has a non-numeric property, state, amenity, or evaluation"),
        ("has_value", "entity has a numeric, monetary, rating, or measured value"),
        ("has_count", "entity has a count"),
        ("offers", "provider offers or performs an item, service, or process"),
        ("causes", "cause produces outcome"),
        ("decreases", "factor decreases quantity or state"),
        ("forbids", "authority or rule forbids action or denies a right"),
        ("occurs_at", "event occurs at time"),
        ("experiences", "entity undergoes an event or incident"),
        ("communicates", "source states, reports, announces, predicts, warns, or discusses claim"),
        ("plans", "actor intends or plans a future action"),
        ("holds_role", "person holds role or position in organization"),
        ("arrests", "authority arrests person"),
        ("charges", "authority charges person with offense"),
        ("convicts", "court convicts person of offense"),
        ("challenges", "actor challenges or appeals person, action, or decision"),
        ("enacts_policy", "institution passes, waives, or changes law or policy"),
        ("attacks", "actor attacks target"),
        ("kills", "actor kills person"),
        ("captures", "actor captures place or target"),
        ("wins", "participant wins competition, award, election, or legal victory"),
        ("risk_or_exposure", "entity faces risk or exposure from hazard or outcome"),
        ("pays_for", "actor pays for item or service"),
        ("parent_of", "person is parent of person"),
        ("born_in", "person is born in place"),
        ("dies_in", "person dies in place"),
        ("moves_to", "entity moves toward or to place"),
        ("resolves", "actor resolves dispute or problem"),
        ("creates", "creator creates, launches, produces, or photographs work or product"),
        ("supports", "actor supports proposal, policy, claim, or person"),
        ("opposes", "actor opposes proposal, policy, claim, or person"),
    ),
    "XSum": (
        ("located_in", "entity is located in place"),
        ("has_value", "entity has a numeric, monetary, rating, or measured value"),
        ("decreases", "factor decreases quantity or state"),
        ("forbids", "authority or rule forbids action or denies a right"),
        ("experiences", "entity undergoes an event or incident"),
        ("communicates", "source states, reports, announces, predicts, warns, or discusses claim"),
        ("plans", "actor intends or plans a future action"),
        ("holds_role", "person holds role or position in organization"),
        ("member_of", "person or organization is member of organization"),
        ("recruits", "organization recruits or signs person"),
        ("resigns", "person resigns or leaves role or organization"),
        ("arrests", "authority arrests person"),
        ("charges", "authority charges person with offense"),
        ("challenges", "actor challenges or appeals person, action, or decision"),
        ("kills", "actor kills person"),
        ("wins", "participant wins competition, award, election, or legal victory"),
        ("scores", "player scores goal or point"),
        ("draws", "team draws with team"),
        ("plays_against", "team or participant plays against opponent"),
        ("risk_or_exposure", "entity faces risk or exposure from hazard or outcome"),
        ("moves_to", "entity moves toward or to place"),
        ("resolves", "actor resolves dispute or problem"),
        ("creates", "creator creates, launches, produces, or photographs work or product"),
    ),
}

RELATION_THRESHOLDS = {
    "Data2txt": 0.70,
    "QA": 0.70,
    "Summary": 0.40,
    "XSum": 0.30,
}


def relation_labels(task: str) -> tuple[list[str], dict[str, str]]:
    """Return the GLiNER labels and their shorter IDs for one task."""
    if task not in RELATIONS:
        raise ValueError(f"Unknown task {task!r}. Expected one of {TASKS}.")
    pairs = RELATIONS[task]
    return (
        [label for _, label in pairs],
        {norm(label): relation_id for relation_id, label in pairs},
    )


def parse_entity(value: Any) -> dict[str, Any]:
    """Convert one raw GLiNER entity to the format saved in graph JSONL."""
    if not isinstance(value, dict):
        raise TypeError("GLiNER entity output must be a dictionary")
    item = value
    missing = {"start", "end", "text"} - set(item)
    if missing:
        raise ValueError(f"entity is missing {sorted(missing)}")
    start, end = int(item["start"]), int(item["end"])
    if start < 0 or end < start:
        raise ValueError(f"invalid entity offsets ({start}, {end})")
    return {
        "start": start,
        "end": end,
        "text": str(item["text"]),
        "label": str(item.get("label", item.get("type", ""))),
        "score": float(item.get("score", 0.0)),
    }


def parse_relation(value: Any, task: str) -> dict[str, Any]:
    """Convert a raw GLiNER relation label to our stable relation ID."""
    if not isinstance(value, dict):
        raise TypeError("GLiNER relation output must be a dictionary")
    item = value
    head = item.get("head")
    tail = item.get("tail")
    if not isinstance(head, dict) or not isinstance(tail, dict):
        raise TypeError("GLiNER relation endpoints must be dictionaries")
    for endpoint_name, endpoint in (("head", head), ("tail", tail)):
        missing = {"start", "end", "text"} - set(endpoint)
        if missing:
            raise ValueError(f"relation {endpoint_name} is missing {sorted(missing)}")
    raw_label = item.get("relation", item.get("label"))
    if raw_label is None:
        raise ValueError("relation has no label")
    _, label_to_id = relation_labels(task)
    relation_id = label_to_id.get(norm(raw_label))
    if relation_id is None:
        raise ValueError(f"unexpected relation label for {task}: {raw_label!r}")
    return {
        "head_start": int(head["start"]),
        "head_end": int(head["end"]),
        "head_text": str(head["text"]),
        "tail_start": int(tail["start"]),
        "tail_end": int(tail["end"]),
        "tail_text": str(tail["text"]),
        "relation_id": relation_id,
        "score": float(item.get("score", 0.0)),
    }


def clean_graph(
    entities: Sequence[Any], relations: Sequence[Any], task: str
) -> dict[str, Any]:
    """Parse GLiNER output and keep the highest-scoring duplicate prediction."""
    entity_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw_entity in entities:
        entity = parse_entity(raw_entity)
        key = (
            entity["start"], entity["end"], norm(entity["text"]),
            norm(entity["label"]),
        )
        if key not in entity_by_key or entity["score"] > entity_by_key[key]["score"]:
            entity_by_key[key] = entity

    relation_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw_relation in relations:
        relation = parse_relation(raw_relation, task)
        key = (
            relation["head_start"], relation["head_end"],
            relation["relation_id"], relation["tail_start"], relation["tail_end"],
        )
        if key not in relation_by_key or relation["score"] > relation_by_key[key]["score"]:
            relation_by_key[key] = relation
    return {
        "entities": sorted(entity_by_key.values(), key=lambda x: (x["start"], x["end"])),
        "relations": sorted(
            relation_by_key.values(),
            key=lambda x: (x["head_start"], x["tail_start"], x["relation_id"]),
        ),
    }


class GraphExtractor:
    """Load GLiNER-Relex and construct graphs from batches of text."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise RuntimeError(
                "GLiNER is not installed. Install the project dependencies "
                "with: pip install -r requirements.txt"
            ) from exc
        kwargs = {"map_location": device} if device else {}
        self.model = GLiNER.from_pretrained(model_name, **kwargs)
        self.batch_size = batch_size

    @staticmethod
    def _is_cuda_oom(error: RuntimeError) -> bool:
        return "cuda out of memory" in str(error).casefold()

    @staticmethod
    def _clear_cuda_cache() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    def _inference(self, texts: Sequence[str], task: str) -> tuple[Any, Any]:
        labels, _ = relation_labels(task)
        try:
            return self.model.inference(
                texts=list(texts),
                labels=list(ENTITY_LABELS),
                relations=labels,
                relation_threshold=RELATION_THRESHOLDS[task],
                batch_size=min(self.batch_size, len(texts)),
                return_relations=True,
            )
        except RuntimeError as exc:
            if not self._is_cuda_oom(exc):
                raise
            if len(texts) == 1:
                raise RuntimeError(
                    "CUDA ran out of memory while extracting one example. "
                    "Free GPU memory or rerun with --device cpu."
                ) from exc

        smaller_batch_size = max(1, len(texts) // 2)
        warnings.warn(
            "CUDA ran out of memory during graph extraction; retrying the "
            f"failed batch in chunks of at most {smaller_batch_size}.",
            RuntimeWarning,
            stacklevel=2,
        )
        self._clear_cuda_cache()
        entity_batches: list[Any] = []
        relation_batches: list[Any] = []
        for start in range(0, len(texts), smaller_batch_size):
            entities, relations = self._inference(
                texts[start : start + smaller_batch_size], task
            )
            entity_batches.extend(entities)
            relation_batches.extend(relations)
        return entity_batches, relation_batches

    def extract_batch(self, texts: Sequence[str], task: str) -> list[dict[str, Any]]:
        output = self._inference(texts, task)
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("GLiNER-Relex must return (entities, relations)")
        entity_batches, relation_batches = output
        if len(entity_batches) != len(texts) or len(relation_batches) != len(texts):
            raise RuntimeError("GLiNER-Relex returned a wrong batch length")
        return [
            clean_graph(entities, relations, task)
            for entities, relations in zip(entity_batches, relation_batches)
        ]


def extract_graphs(
    frame: pd.DataFrame,
    model_name: str = MODEL_NAME,
    device: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Construct one context graph and one response graph for every row."""
    required = {"sample_id", "task_type", "context", "response"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"extraction frame is missing columns {missing}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    extractor = GraphExtractor(model_name, device, batch_size)
    graphs: dict[str, Any] = {}
    for task in TASKS:
        part = frame[frame["task_type"].eq(task)].reset_index(drop=True)
        if part.empty:
            continue
        with tqdm(total=len(part), desc=f"Extract {task}", unit="examples") as bar:
            for start in range(0, len(part), batch_size):
                batch = part.iloc[start : start + batch_size]
                context_graphs = extractor.extract_batch(batch["context"].tolist(), task)
                response_graphs = extractor.extract_batch(batch["response"].tolist(), task)
                for row, context_graph, response_graph in zip(
                    batch.itertuples(index=False), context_graphs, response_graphs
                ):
                    sample_id = str(row.sample_id)
                    if sample_id in graphs:
                        raise ValueError(f"duplicate sample_id during extraction: {sample_id}")
                    graphs[sample_id] = {
                        "sample_id": sample_id,
                        "task_type": task,
                        "context_graph": context_graph,
                        "response_graph": response_graph,
                    }
                bar.update(len(batch))
    return graphs


def get_graphs(
    frame: pd.DataFrame,
    graph_path: Path | None = None,
    device: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Load precomputed graphs, or extract them when no graph file is given."""
    if graph_path is not None:
        return read_graphs(graph_path, set(frame.sample_id.astype(str)))
    return extract_graphs(frame, device=device, batch_size=batch_size)
