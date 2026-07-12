from .entity_extractors import get_extractor


class EntityExtractor:
    def __init__(self, method: str):
        self.method = method
        self._extractor = get_extractor(method)

    def extract(self, text: str):
        return sorted(self._extractor(text))