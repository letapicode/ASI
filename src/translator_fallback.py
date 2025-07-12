class CrossLingualTranslator:  # type: ignore
    """Fallback translator if real dependency is missing."""

    def __init__(self, languages):
        self.languages = list(languages)

    def translate(self, text: str, lang: str) -> str:
        if lang not in self.languages:
            raise ValueError("unsupported language")
        return text

    def translate_all(self, text: str):
        return {l: text for l in self.languages}

__all__ = ["CrossLingualTranslator"]
