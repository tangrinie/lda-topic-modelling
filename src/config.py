from __future__ import annotations

from dataclasses import dataclass, field


# Add future file readers here after implementing them in src/file_handling.py.
DEFAULT_SUPPORTED_FILE_TYPES = ["txt", "csv", "pdf"]

STOPWORD_LANGUAGE_OPTIONS = {
    "english_italian": "English + Italian",
    "english": "English",
    "italian": "Italian",
    "none": "None",
}


@dataclass(frozen=True)
class PreprocessingConfig:
    lowercase: bool = True
    normalize_accents: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = False
    remove_stopwords: bool = True
    stopword_language: str = "english_italian"
    use_stemming: bool = False
    min_token_length: int = 3
    custom_stopwords: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ModelConfig:
    num_topics: int = 5
    words_per_topic: int = 10
    min_df: int = 1
    max_df: float = 1.0
    random_state: int = 42
    max_iter: int = 20
    max_features: int | None = 5000
    learning_method: str = "batch"
