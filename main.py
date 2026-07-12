from config import *

from graph_builder.graph_builder import build_graph


text = """
      Elon Musk founded Tesla in 2003 and became its CEO.
      Tesla is an electric vehicle manufacturer headquartered in Austin, Texas.
      The company produces electric cars, battery energy storage systems, and solar products.
      In 2016, Tesla acquired SolarCity, a solar energy services company founded by Lyndon Rive and Peter Rive.
      SpaceX was founded by Elon Musk in 2002 and is headquartered in Hawthorne, California.
      Elon Musk is also the CEO of SpaceX.
      """

entities, triples, schema, mappings = build_graph(
    text=text,

    entity_method=ENTITY_EXTRACTION_METHOD,

    model_name=TRIPLET_MODEL_NAME,
    use_4bit=TRIPLET_USE_4BIT,
    system_prompt=TRIPLET_SYSTEM_PROMPT,
    max_new_tokens=TRIPLET_MAX_NEW_TOKENS,
    temperature=TRIPLET_TEMPERATURE,
    top_p=TRIPLET_TOP_P,
    do_sample=TRIPLET_DO_SAMPLE,

    embedding_model=CANONICALIZATION_EMBEDDING_MODEL,
    canonicalization_top_k=CANONICALIZATION_TOP_K,

    pairwise_prompt_path="prompts/pairwise.txt",
    canonicalization_prompt_path="prompts/canonicalize_relation.txt",
    schema_prompt_path="prompts/define_relations.txt",
)

print("entities:")

for entity in entities:
    print(entity)

print("Graph:")

for triple in triples:
    print(triple)
