import functools

GENERIC = {"thing", "things", "way", "ways", "time", "times", "one", "ones",
           "kind", "sort", "part", "parts", "lot", "lots", "example", "examples",
           "question", "questions", "answer", "answers", "case", "cases",
           "fact", "facts", "issue", "issues", "point", "points", "number"}


@functools.lru_cache(maxsize=None)
def _sm():
    import spacy
    return spacy.load("en_core_web_sm")


@functools.lru_cache(maxsize=None)
def _md():
    import spacy
    return spacy.load("en_core_web_md")


def _clean_chunk(chunk):
    toks = [t for t in chunk
            if t.pos_ in ("NOUN", "PROPN", "ADJ", "NUM")
            and not t.is_stop and not t.is_punct]
    if not toks:
        return None
    head = toks[-1]
    if head.lemma_.lower() in GENERIC:
        return None
    txt = " ".join(t.lemma_.lower() if t is head else t.text.lower() for t in toks).strip()
    return txt or None


def ext_spacy_sm(text):
    return {e.text.lower().strip() for e in _sm()(text).ents if e.text.strip()}


def ext_spacy_md(text):
    return {e.text.lower().strip() for e in _md()(text).ents if e.text.strip()}


def ext_nltk(text):
    import nltk
    from nltk import word_tokenize, pos_tag, ne_chunk
    out = set()
    for sent in nltk.sent_tokenize(text):
        tree = ne_chunk(pos_tag(word_tokenize(sent)))
        for node in tree:
            if hasattr(node, "label"):
                out.add(" ".join(w for w, _ in node.leaves()).lower())
    return out


def ext_noun_chunks(text):
    return {c.text.lower().strip() for c in _sm()(text).noun_chunks if c.text.strip()}


def ext_nc_clean(text):
    return {v for v in (_clean_chunk(c) for c in _sm()(text).noun_chunks) if v}


def ext_content_nouns(text):
    return {t.lemma_.lower() for t in _sm()(text)
            if t.pos_ in ("NOUN", "PROPN") and not t.is_stop
            and t.lemma_.lower() not in GENERIC and len(t.lemma_) > 2}


def ext_hybrid(text):
    doc = _sm()(text)
    out = {e.text.lower().strip() for e in doc.ents if e.text.strip()}
    out |= {v for v in (_clean_chunk(c) for c in doc.noun_chunks) if v}
    return out


REGISTRY = {
    "spacy_sm": ext_spacy_sm,
    "spacy_md": ext_spacy_md,
    "nltk_ne": ext_nltk,
    "noun_chunks": ext_noun_chunks,
    "nc_clean": ext_nc_clean,
    "content_nouns": ext_content_nouns,
    "hybrid": ext_hybrid,
}


def get_extractor(name):
    return REGISTRY[name]
