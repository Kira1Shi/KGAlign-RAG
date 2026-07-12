class SchemaDefinitionPromptBuilder:
    def __init__(self, template_path: str):
        with open(template_path, "r", encoding="utf-8") as file:
            self.template = file.read()

    def build(
        self,
        text: str,
        triples: list[dict],
        relations: list[str],
    ) -> str:
        formatted_triples = [
            [triple["subject"], triple["relation"], triple["object"]]
            for triple in triples
        ]
        return self.template.format(
            text=text,
            triples=formatted_triples,
            relations=relations,
        )


class CanonicalizationPromptBuilder:
    def __init__(self, template_path: str):
        with open(template_path, "r", encoding="utf-8") as file:
            self.template = file.read()

    def build(
        self,
        text: str,
        triple: dict,
        relation_definition: str,
        candidates: list[tuple[str, str]],
    ) -> str:
        formatted_triple = [
            triple["subject"],
            triple["relation"],
            triple["object"],
        ]
        choices = []
        for index, (relation, definition) in enumerate(candidates):
            letter = chr(ord("A") + index)
            choices.append(f"{letter}. '{relation}': {definition}")
        none_letter = chr(ord("A") + len(candidates))
        choices.append(f"{none_letter}. None of the above.")

        return self.template.format(
            text=text,
            triple=formatted_triple,
            relation=triple["relation"],
            relation_definition=relation_definition,
            choices="\n".join(choices),
        )
