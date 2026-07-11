from model import HuggingFaceChatModel
from prompt_builder import PromptBuilder
from extractor import RelationExtractor
from canonicalizer import RelationCanonicalizer
from canonicalization_prompt_builder import (
    CanonicalizationPromptBuilder,
    SchemaDefinitionPromptBuilder,
)
from schema_definer import RelationSchemaDefiner
from config import (
    CANONICALIZATION_EMBEDDING_MODEL,
    CANONICALIZATION_TOP_K,
)
from sentence_transformers import SentenceTransformer



def main():

    text = """
      Elon Musk founded Tesla in 2003 and became its CEO.
      Tesla is an electric vehicle manufacturer headquartered in Austin, Texas.
      The company produces electric cars, battery energy storage systems, and solar products.
      In 2016, Tesla acquired SolarCity, a solar energy services company founded by Lyndon Rive and Peter Rive.
      SpaceX was founded by Elon Musk in 2002 and is headquartered in Hawthorne, California.
      Elon Musk is also the CEO of SpaceX.
      """

    entities = [
          "Elon Musk",
          "Tesla",
          "Austin",
          "Texas",
          "electric cars",
          "battery energy storage systems",
          "solar products",
          "SolarCity",
          "Lyndon Rive",
          "Peter Rive",
          "SpaceX",
          "Hawthorne",
          "California"
      ]

    print("Loading language model...")

    model = HuggingFaceChatModel()

    print("Loading prompt builder...")

    prompt_builder = PromptBuilder(
        template_path="../prompts/pairwise.txt"
    )

    print("Initializing extractor...")

    extractor = RelationExtractor(
        model=model,
        prompt_builder=prompt_builder
    )

    print("Extracting relations...")

    raw_triples = extractor.extract(
        text=text,
        entities=entities
    )

    print("Loading relation canonicalizer...")

    canonicalization_prompt_builder = CanonicalizationPromptBuilder(
        template_path="../prompts/canonicalize_relation.txt"
    )

    schema_definition_prompt_builder = SchemaDefinitionPromptBuilder(
        template_path="../prompts/define_relations.txt"
    )

    schema_definer = RelationSchemaDefiner(
        model=model,
        prompt_builder=schema_definition_prompt_builder,
    )

    embedder = SentenceTransformer(
        CANONICALIZATION_EMBEDDING_MODEL
    )

    canonicalizer = RelationCanonicalizer(
        model=model,
        prompt_builder=canonicalization_prompt_builder,
        embedder=embedder,
        top_k=CANONICALIZATION_TOP_K,
    )

    relation_definitions = schema_definer.define(
        text=text,
        triples=raw_triples,
    )

    triples, relation_schema, mappings = canonicalizer.canonicalize(
        text=text,
        triples=raw_triples,
        relation_definitions=relation_definitions,
    )

    print("\nRaw triples:")

    for triple in raw_triples:

        print(triple)

    print("\nCanonical triples:")

    for triple in triples:

        print(triple)

    print("\nRelation schema:")

    for relation, definition in relation_schema.items():

        print(f"{relation}: {definition}")

    print("\nRelation mappings:")

    for raw_relation, canonical_relation in mappings.items():

        print(f"{raw_relation} -> {canonical_relation}")



if __name__ == "__main__":

    main()
