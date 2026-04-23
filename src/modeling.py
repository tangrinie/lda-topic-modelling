from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from src.config import ModelConfig


class TopicModelError(ValueError):
    """Raised when the uploaded corpus cannot produce a usable topic model."""


@dataclass
class TopicModelResult:
    topics: pd.DataFrame
    topic_terms: pd.DataFrame
    document_topics: pd.DataFrame
    document_topic_distribution: pd.DataFrame
    vocabulary_size: int
    topic_count: int


def fit_lda_model(preprocessed_df: pd.DataFrame, config: ModelConfig) -> TopicModelResult:
    usable_df = preprocessed_df[preprocessed_df["token_count"] > 0].copy()
    if usable_df.empty:
        raise TopicModelError("No documents contain tokens after preprocessing.")

    try:
        vectorizer = CountVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
            min_df=config.min_df,
            max_df=config.max_df,
            max_features=config.max_features,
        )
        document_term_matrix = vectorizer.fit_transform(usable_df["processed_text"])
    except ValueError as exc:
        raise TopicModelError(
            "The vectorizer removed all terms. Try lowering minimum document frequency, "
            "raising maximum document frequency, or relaxing preprocessing settings."
        ) from exc

    if document_term_matrix.shape[1] == 0:
        raise TopicModelError("No vocabulary terms are available for modeling.")

    lda_model = LatentDirichletAllocation(
        n_components=config.num_topics,
        max_iter=config.max_iter,
        learning_method=config.learning_method,
        random_state=config.random_state,
        evaluate_every=-1,
    )
    document_topic_matrix = lda_model.fit_transform(document_term_matrix)
    feature_names = vectorizer.get_feature_names_out()

    topic_terms = build_topic_terms(
        lda_model.components_,
        feature_names,
        top_n=config.words_per_topic,
    )
    topics = build_topic_summary(topic_terms)
    document_topics = build_document_topic_mapping(
        usable_df["document_name"].tolist(),
        document_topic_matrix,
    )
    document_topic_distribution = build_document_topic_distribution(
        usable_df["document_name"].tolist(),
        document_topic_matrix,
    )

    return TopicModelResult(
        topics=topics,
        topic_terms=topic_terms,
        document_topics=document_topics,
        document_topic_distribution=document_topic_distribution,
        vocabulary_size=len(feature_names),
        topic_count=config.num_topics,
    )


def build_topic_terms(
    topic_term_matrix: np.ndarray,
    feature_names: np.ndarray,
    top_n: int,
) -> pd.DataFrame:
    records = []

    for topic_index, term_weights in enumerate(topic_term_matrix):
        top_indices = term_weights.argsort()[::-1][:top_n]
        top_weights = term_weights[top_indices]
        weight_total = top_weights.sum()

        for rank, term_index in enumerate(top_indices, start=1):
            weight = float(term_weights[term_index])
            records.append(
                {
                    "topic_id": topic_index + 1,
                    "topic_label": f"Topic {topic_index + 1}",
                    "rank": rank,
                    "term": str(feature_names[term_index]),
                    "weight": weight,
                    "relative_weight": float(weight / weight_total) if weight_total else 0.0,
                }
            )

    return pd.DataFrame(records)


def build_topic_summary(topic_terms: pd.DataFrame) -> pd.DataFrame:
    summaries = []

    for topic_id, group in topic_terms.groupby("topic_id", sort=True):
        ordered = group.sort_values("rank")
        summaries.append(
            {
                "topic_id": int(topic_id),
                "topic_label": f"Topic {int(topic_id)}",
                "keywords": ", ".join(ordered["term"].tolist()),
            }
        )

    return pd.DataFrame(summaries)


def build_document_topic_mapping(
    document_names: list[str],
    document_topic_matrix: np.ndarray,
) -> pd.DataFrame:
    dominant_topic_indices = document_topic_matrix.argmax(axis=1)
    dominant_scores = document_topic_matrix.max(axis=1)

    records = []
    for row_index, document_name in enumerate(document_names):
        record = {
            "document_name": document_name,
            "dominant_topic": f"Topic {dominant_topic_indices[row_index] + 1}",
            "dominant_topic_probability": round(float(dominant_scores[row_index]), 4),
        }
        for topic_index, score in enumerate(document_topic_matrix[row_index], start=1):
            record[f"Topic {topic_index}"] = round(float(score), 4)
        records.append(record)

    return pd.DataFrame(records)


def build_document_topic_distribution(
    document_names: list[str],
    document_topic_matrix: np.ndarray,
) -> pd.DataFrame:
    records = []

    for row_index, document_name in enumerate(document_names):
        for topic_index, score in enumerate(document_topic_matrix[row_index], start=1):
            records.append(
                {
                    "document_name": document_name,
                    "topic": f"Topic {topic_index}",
                    "topic_probability": float(score),
                }
            )

    return pd.DataFrame(records)
