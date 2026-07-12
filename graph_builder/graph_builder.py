from sentence_transformers import SentenceTransformer

from .model import HuggingFaceChatModel
from .entity_extraction import EntityExtractor
from .prompt_builder import PromptBuilder
from .triplet_extractor import RelationExtractor
from .canonicalizer import RelationCanonicalizer
from .canonicalization_prompt_builder import (
    CanonicalizationPromptBuilder,
    SchemaDefinitionPromptBuilder,
)
from .schema_definer import RelationSchemaDefiner


def build_graph(
    text: str,

    # Entity extraction
    entity_method: str,

    # LLM
    model_name: str,
    use_4bit: bool,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,

    # Canonicalization
    embedding_model: str,
    canonicalization_top_k: int,

    # Prompt templates
    pairwise_prompt_path: str,
    canonicalization_prompt_path: str,
    schema_prompt_path: str,
):
    """
    Builds a knowledge graph from the input text.

    Parameters
    ----------
    text : str
        Input text.

    entity_method : str
        Entity extraction method.

    model_name : str
        HuggingFace model name.

    use_4bit : bool
        Whether to use 4-bit quantization.

    system_prompt : str
        System prompt for the LLM.

    max_new_tokens : int
        Maximum number of generated tokens.

    temperature : float
        Generation temperature.

    top_p : float
        Top-p sampling parameter.

    do_sample : bool
        Whether to sample during generation.

    embedding_model : str
        SentenceTransformer model used for canonicalization.

    canonicalization_top_k : int
        Number of nearest neighbours during canonicalization.

    pairwise_prompt_path : str
        Path to pairwise extraction prompt.

    canonicalization_prompt_path : str
        Path to relation canonicalization prompt.

    schema_prompt_path : str
        Path to schema definition prompt.

    Returns
    -------
    tuple
        (
            triples,
            relation_schema,
            relation_mappings
        )
    """

    # ------------------------------------------------------
    # Language model
    # ------------------------------------------------------

    model = HuggingFaceChatModel(
        model_name=model_name,
        use_4bit=use_4bit,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )

    # ------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------

    entity_extractor = EntityExtractor(
        method=entity_method
    )

    entities = entity_extractor.extract(text)

    # ------------------------------------------------------
    # Relation extraction
    # ------------------------------------------------------

    prompt_builder = PromptBuilder(
        template_path=pairwise_prompt_path
    )

    relation_extractor = RelationExtractor(
        model=model,
        prompt_builder=prompt_builder,
    )

    raw_triples = relation_extractor.extract(
        text=text,
        entities=entities,
    )

    # ------------------------------------------------------
    # Schema definition
    # ------------------------------------------------------

    schema_prompt_builder = SchemaDefinitionPromptBuilder(
        template_path=schema_prompt_path
    )

    schema_definer = RelationSchemaDefiner(
        model=model,
        prompt_builder=schema_prompt_builder,
    )

    relation_definitions = schema_definer.define(
        text=text,
        triples=raw_triples,
    )

    # ------------------------------------------------------
    # Canonicalization
    # ------------------------------------------------------

    canonicalization_prompt_builder = CanonicalizationPromptBuilder(
        template_path=canonicalization_prompt_path
    )

    embedder = SentenceTransformer(
        embedding_model
    )

    canonicalizer = RelationCanonicalizer(
        model=model,
        prompt_builder=canonicalization_prompt_builder,
        embedder=embedder,
        top_k=canonicalization_top_k,
    )

    triples, relation_schema, relation_mappings = (
        canonicalizer.canonicalize(
            text=text,
            triples=raw_triples,
            relation_definitions=relation_definitions,
        )
    )

    return (
        entities,
        triples,
        relation_schema,
        relation_mappings,
    )