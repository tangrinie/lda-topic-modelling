from __future__ import annotations

import altair as alt
import pandas as pd


# Future extension: add a pyLDAvis-style component here if you choose that dependency.
def make_topic_terms_chart(topic_terms: pd.DataFrame, topic_id: int) -> alt.Chart:
    chart_data = topic_terms[topic_terms["topic_id"] == topic_id].copy()
    chart_data = chart_data.sort_values("weight", ascending=True)

    return (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("weight:Q", title="Term weight"),
            y=alt.Y("term:N", sort=None, title="Keyword"),
            tooltip=[
                alt.Tooltip("term:N", title="Keyword"),
                alt.Tooltip("weight:Q", title="Weight", format=".2f"),
                alt.Tooltip("relative_weight:Q", title="Share of top terms", format=".1%"),
            ],
        )
        .properties(height=360)
    )


def make_document_topic_chart(document_topic_distribution: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(document_topic_distribution)
        .mark_bar()
        .encode(
            x=alt.X("document_name:N", title="Document", sort=None),
            y=alt.Y("topic_probability:Q", title="Topic probability", stack="normalize"),
            color=alt.Color("topic:N", title="Topic"),
            tooltip=[
                alt.Tooltip("document_name:N", title="Document"),
                alt.Tooltip("topic:N", title="Topic"),
                alt.Tooltip("topic_probability:Q", title="Probability", format=".1%"),
            ],
        )
        .properties(height=360)
    )


def make_dominant_topic_chart(document_topics: pd.DataFrame) -> alt.Chart:
    topic_counts = (
        document_topics["dominant_topic"]
        .value_counts()
        .rename_axis("topic")
        .reset_index(name="document_count")
    )

    return (
        alt.Chart(topic_counts)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("document_count:Q", title="Documents"),
            y=alt.Y("topic:N", title="Dominant topic", sort="-x"),
            tooltip=[
                alt.Tooltip("topic:N", title="Topic"),
                alt.Tooltip("document_count:Q", title="Documents"),
            ],
        )
        .properties(height=260)
    )
