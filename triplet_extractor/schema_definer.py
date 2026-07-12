class RelationSchemaDefiner:
    """Generate contextual relation definitions in the EDC Define phase."""

    def __init__(self, model, prompt_builder):
        self.model = model
        self.prompt_builder = prompt_builder

    def define(self, text: str, triples: list[dict]) -> dict[str, str]:
        relations = list(dict.fromkeys(
            triple["relation"]
            for triple in triples
        ))
        if not relations:
            return {}

        prompt = self.prompt_builder.build(
            text=text,
            triples=triples,
            relations=relations,
        )
        response = self.model.generate(prompt)
        return self._parse_definitions(response, relations)

    @staticmethod
    def _parse_definitions(
        response: str,
        expected_relations: list[str],
    ) -> dict[str, str]:
        expected_lookup = {
            relation.casefold(): relation
            for relation in expected_relations
        }
        definitions: dict[str, str] = {}

        for line in response.splitlines():
            if ":" not in line:
                continue
            raw_relation, raw_definition = line.split(":", 1)
            relation = raw_relation.strip().strip("`\"'")
            definition = raw_definition.strip()
            canonical_name = expected_lookup.get(relation.casefold())
            if canonical_name and definition:
                definitions[canonical_name] = definition

        return definitions
