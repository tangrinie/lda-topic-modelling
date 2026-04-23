from __future__ import annotations

import re
import string

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.config import PreprocessingConfig
from src.file_handling import Document


TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z']*")
PUNCTUATION_TRANSLATION = str.maketrans({character: " " for character in string.punctuation})


def parse_custom_stopwords(raw_text: str) -> set[str]:
    if not raw_text:
        return set()

    normalized = raw_text.replace(",", "\n")
    return {word.strip().lower() for word in normalized.splitlines() if word.strip()}


def preprocess_documents(documents: list[Document], config: PreprocessingConfig) -> pd.DataFrame:
    records = []

    for document in documents:
        tokens = preprocess_text(document.text, config)
        records.append(
            {
                "document_name": document.name,
                "source_type": document.source_type,
                "original_char_count": document.char_count,
                "token_count": len(tokens),
                "tokens": tokens,
                "processed_text": " ".join(tokens),
            }
        )

    return pd.DataFrame(records)


def preprocess_text(text: str, config: PreprocessingConfig) -> list[str]:
    if config.lowercase:
        text = text.lower()

    if config.remove_punctuation:
        text = text.translate(PUNCTUATION_TRANSLATION)

    if config.remove_numbers:
        text = re.sub(r"\d+", " ", text)

    tokens = TOKEN_PATTERN.findall(text)
    stopwords = set(ENGLISH_STOP_WORDS).union(config.custom_stopwords)
    # Future extension: replace stemming with lemmatization if you add spaCy or WordNet.
    stemmer = get_stemmer() if config.use_stemming else None

    cleaned_tokens: list[str] = []
    for token in tokens:
        token = token.strip("'")
        if len(token) < config.min_token_length:
            continue
        if config.remove_stopwords and token.lower() in stopwords:
            continue
        if stemmer is not None:
            token = stemmer.stem(token)
        cleaned_tokens.append(token)

    return cleaned_tokens


def get_stemmer():
    # This uses NLTK's PorterStemmer, which does not require downloading corpora.
    from nltk.stem import PorterStemmer

    return PorterStemmer()


def build_preprocessing_summary(preprocessed_df: pd.DataFrame) -> dict:
    processed_documents = int((preprocessed_df["token_count"] > 0).sum())
    total_tokens = int(preprocessed_df["token_count"].sum())
    empty_documents = int((preprocessed_df["token_count"] == 0).sum())

    return {
        "processed_documents": processed_documents,
        "total_tokens": total_tokens,
        "empty_documents": empty_documents,
    }
