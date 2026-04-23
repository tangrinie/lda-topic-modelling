from __future__ import annotations

import json

import pandas as pd


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")


def results_to_json_bytes(result) -> bytes:
    payload = {
        "topics": result.topics.to_dict(orient="records"),
        "topic_terms": result.topic_terms.to_dict(orient="records"),
        "document_topics": result.document_topics.to_dict(orient="records"),
        "document_topic_distribution": result.document_topic_distribution.to_dict(orient="records"),
        "vocabulary_size": result.vocabulary_size,
        "topic_count": result.topic_count,
    }
    return json.dumps(payload, indent=2).encode("utf-8")
