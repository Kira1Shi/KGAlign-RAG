from collections import OrderedDict
from typing import Iterable

import numpy as np


class RelationCanonicalizer:
    """Definition-based relation canonicalization adapted from EDC.

    EDC source: https://github.com/clear-nus/edc
    The implementation is adapted to this project's dictionary triplets and
    HuggingFaceChatModel interface.
    """

    def __init__(self, model, prompt_builder, embedder, top_k: int = 5):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        self.model = model
        self.prompt_builder = prompt_builder
        self.embedder = embedder
        self.top_k = top_k

    def canonicalize(
        self,
        text: str,
        triples: list[dict],
        relation_definitions: dict[str, str],
        schema: dict[str, str] | None = None,
        allow_new: bool = True,
    ) -> tuple[list[dict], dict[str, str], dict[str, str]]:
        """Canonicalize relations and return triples, schema, and mappings."""

        canonical_schema = dict(schema or {})
        schema_embeddings = self._embed_schema(canonical_schema)
        mappings: dict[str, str] = {}
        canonicalized_triples: list[dict] = []

        for triple in triples:
            raw_relation = triple["relation"]

            if raw_relation in canonical_schema:
                canonical_relation = raw_relation
            elif raw_relation in mappings:
                canonical_relation = mappings[raw_relation]
            else:
                definition = relation_definitions.get(raw_relation)
                canonical_relation = self._canonicalize_relation(
                    text=text,
                    triple=triple,
                    relation_definition=definition,
                    schema=canonical_schema,
                    schema_embeddings=schema_embeddings,
                )

                if canonical_relation is None:
                    canonical_relation = raw_relation
                    if allow_new and definition:
                        canonical_schema[raw_relation] = definition
                        schema_embeddings[raw_relation] = self._encode(definition)

                mappings[raw_relation] = canonical_relation

            canonicalized = dict(triple)
            canonicalized["relation"] = canonical_relation
            canonicalized["raw_relation"] = raw_relation
            canonicalized_triples.append(canonicalized)

        return (
            self._deduplicate(canonicalized_triples),
            canonical_schema,
            mappings,
        )

    def _canonicalize_relation(
        self,
        text: str,
        triple: dict,
        relation_definition: str | None,
        schema: dict[str, str],
        schema_embeddings: dict[str, np.ndarray],
    ) -> str | None:
        if not relation_definition or not schema:
            return None

        candidates = self._retrieve_candidates(
            relation_definition,
            schema,
            schema_embeddings,
        )
        prompt = self.prompt_builder.build(
            text=text,
            triple=triple,
            relation_definition=relation_definition,
            candidates=candidates,
        )
        return self._parse_choice(self.model.generate(prompt), candidates)

    def _retrieve_candidates(
        self,
        relation_definition: str,
        schema: dict[str, str],
        schema_embeddings: dict[str, np.ndarray],
    ) -> list[tuple[str, str]]:
        query_embedding = self._encode(relation_definition)
        scored = [
            (relation, float(query_embedding @ schema_embeddings[relation]))
            for relation in schema
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            (relation, schema[relation])
            for relation, _ in scored[: self.top_k]
        ]

    def _embed_schema(
        self,
        schema: dict[str, str],
    ) -> dict[str, np.ndarray]:
        if not schema:
            return {}

        relations = list(schema)
        embeddings = self.embedder.encode(
            [schema[relation] for relation in relations],
            normalize_embeddings=True,
        )
        return dict(zip(relations, embeddings))

    def _encode(self, definition: str) -> np.ndarray:
        return self.embedder.encode(
            [definition],
            normalize_embeddings=True,
        )[0]

    @staticmethod
    def _parse_choice(
        response: str,
        candidates: list[tuple[str, str]],
    ) -> str | None:
        choice = response.strip().upper()
        if not choice:
            return None

        choice_index = ord(choice[0]) - ord("A")
        if 0 <= choice_index < len(candidates):
            return candidates[choice_index][0]
        return None

    @staticmethod
    def _deduplicate(triples: Iterable[dict]) -> list[dict]:
        deduplicated: OrderedDict[tuple[str, str, str], dict] = OrderedDict()

        for triple in triples:
            key = (
                triple["subject"].casefold(),
                triple["relation"].casefold(),
                triple["object"].casefold(),
            )
            if key not in deduplicated:
                canonical_triple = dict(triple)
                canonical_triple["raw_relations"] = [triple["raw_relation"]]
                canonical_triple.pop("raw_relation")
                deduplicated[key] = canonical_triple
            elif (
                triple["raw_relation"]
                not in deduplicated[key]["raw_relations"]
            ):
                deduplicated[key]["raw_relations"].append(
                    triple["raw_relation"]
                )

        return list(deduplicated.values())
