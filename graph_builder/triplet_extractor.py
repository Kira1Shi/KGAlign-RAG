
class RelationExtractor:
    def __init__(
        self,
        model,
        prompt_builder
    ):

        self.model = model
        self.prompt_builder = prompt_builder



    def extract(
        self,
        text: str,
        entities: list[str]
    ) -> list[dict]:

        triples = []

        pairs = self._create_entity_pairs(
            entities
        )

        for subject, object_ in pairs:
            prompt = self.prompt_builder.build(
                text=text,
                subject=subject,
                object_=object_
            )

            relation = self.model.generate(
                prompt
            )


            relation = self._clean_relation(
                relation
            )


            if relation == "NONE":
                continue

            triple = {

                "subject": subject,

                "relation": relation,

                "object": object_

            }


            triples.append(
                triple
            )


        return triples




    def _create_entity_pairs(
        self,
        entities: list[str]
    ) -> list[tuple]:



        pairs = []
        for i in range(len(entities)):
            for j in range(
                i + 1,
                len(entities)
            ):


                pairs.append(
                    (
                        entities[i],
                        entities[j]
                    )
                )


        return pairs




    def _clean_relation(
        self,
        relation: str
    ) -> str:

        relation = relation.strip()

        relation = relation.lower()

        if relation.startswith(
            "relation:"
        ):

            relation = relation.replace(
                "relation:",
                ""
            ).strip()

        if relation in [
            "none",
            "no relation",
            "no",
            "null"
        ]:

            return "NONE"


        return relation