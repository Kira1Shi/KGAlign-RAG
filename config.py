# ==========================================================
# Entity Extraction
# ==========================================================

ENTITY_EXTRACTION_METHOD = "hybrid"

# ==========================================================
# Triplet Extraction Model
# ==========================================================

TRIPLET_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" #Llama 3.1-8B-Instruct, Mistral-7B-Instruct-v0.3

TRIPLET_USE_4BIT = True

TRIPLET_SYSTEM_PROMPT = (
    "You are an expert in relation extraction."
)

TRIPLET_MAX_NEW_TOKENS = 256

TRIPLET_TEMPERATURE = 0.0

TRIPLET_TOP_P = 1.0

TRIPLET_DO_SAMPLE = False

# ==========================================================
# Relation Canonicalization
# ==========================================================

CANONICALIZATION_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

CANONICALIZATION_TOP_K = 5
