from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import DEFAULT_SUPPORTED_FILE_TYPES, ModelConfig, PreprocessingConfig
from src.export import dataframe_to_csv_bytes, results_to_json_bytes
from src.file_handling import Document, read_uploaded_files
from src.modeling import TopicModelError, fit_lda_model
from src.preprocessing import build_preprocessing_summary, parse_custom_stopwords, preprocess_documents
from src.utils import format_file_size
from src.visualization import (
    make_document_topic_chart,
    make_dominant_topic_chart,
    make_topic_terms_chart,
)


SAMPLE_DOCUMENTS_DIR = Path("sample_documents")


def main() -> None:
    st.set_page_config(
        page_title="LDA Topic Modeling Studio",
        page_icon=":page_facing_up:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_styles()

    st.title("LDA Topic Modeling Studio")
    st.caption(
        "Upload text-based documents, tune preprocessing and modeling settings, "
        "then explore the topics discovered by Latent Dirichlet Allocation."
    )

    sidebar_settings = render_sidebar()

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=sidebar_settings["supported_file_types"],
        accept_multiple_files=True,
        help="Start with .txt and .csv files. PDF extraction is available when the file contains selectable text.",
    )

    use_sample_documents = st.checkbox(
        "Include sample documents",
        value=False,
        help="Adds small bundled text files so you can test the app before uploading your own documents.",
    )

    documents, read_errors = read_uploaded_files(
        uploaded_files=uploaded_files or [],
        supported_file_types=sidebar_settings["supported_file_types"],
        csv_mode=sidebar_settings["csv_mode"],
        max_file_size_mb=sidebar_settings["max_file_size_mb"],
    )

    if use_sample_documents:
        documents.extend(load_sample_documents())

    render_file_errors(read_errors)
    render_document_overview(documents)

    run_disabled = not documents
    run_clicked = st.button(
        "Run topic modeling",
        type="primary",
        use_container_width=True,
        disabled=run_disabled,
    )

    if run_disabled:
        st.info("Upload at least one supported document or enable the sample documents to begin.")
        return

    if not run_clicked:
        st.info("Review your documents and settings, then run the model when ready.")
        return

    preprocessing_config = sidebar_settings["preprocessing_config"]
    model_config = sidebar_settings["model_config"]

    with st.spinner("Preprocessing documents..."):
        preprocessed_df = preprocess_cached(to_document_records(documents), preprocessing_config)

    preprocessing_summary = build_preprocessing_summary(preprocessed_df)
    if preprocessing_summary["processed_documents"] == 0:
        st.error(
            "No usable tokens were found after preprocessing. Try lowering the minimum token length, "
            "turning off stopword removal, or uploading longer documents."
        )
        return

    with st.spinner("Fitting the LDA topic model..."):
        try:
            result = fit_lda_cached(tuple(preprocessed_df.to_dict(orient="records")), model_config)
        except TopicModelError as exc:
            st.error(str(exc))
            return

    render_results(result, preprocessed_df, preprocessing_summary)


def render_sidebar() -> dict:
    with st.sidebar:
        st.header("Configuration")

        st.subheader("Files")
        supported_file_types = st.multiselect(
            "Supported upload formats",
            options=DEFAULT_SUPPORTED_FILE_TYPES,
            default=DEFAULT_SUPPORTED_FILE_TYPES,
            help="Customize this list later in src/config.py if you add more readers.",
        )
        if not supported_file_types:
            st.warning("Select at least one file format.")
            supported_file_types = ["txt"]

        csv_mode_label = st.radio(
            "CSV handling",
            options=["One document per file", "One document per row"],
            index=0,
            help="Row mode works well when each row contains a separate article, review, or note.",
        )
        csv_mode = "rows" if csv_mode_label == "One document per row" else "file"

        max_file_size_mb = st.slider(
            "Maximum file size per upload (MB)",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
        )

        st.subheader("Preprocessing")
        lowercase = st.checkbox("Lowercase text", value=True)
        remove_punctuation = st.checkbox("Remove punctuation", value=True)
        remove_numbers = st.checkbox("Remove numbers", value=False)
        remove_stopwords = st.checkbox("Remove stopwords", value=True)
        use_stemming = st.checkbox(
            "Apply Porter stemming",
            value=False,
            help="Optional normalization. Leave off if you want cleaner, more readable keywords.",
        )
        min_token_length = st.slider("Minimum token length", 1, 10, 3)
        custom_stopwords_text = st.text_area(
            "Custom stopwords",
            placeholder="Add words separated by commas or new lines",
            help="Useful for domain words that appear everywhere, such as company names or boilerplate terms.",
        )

        preprocessing_config = PreprocessingConfig(
            lowercase=lowercase,
            remove_punctuation=remove_punctuation,
            remove_numbers=remove_numbers,
            remove_stopwords=remove_stopwords,
            use_stemming=use_stemming,
            min_token_length=min_token_length,
            custom_stopwords=frozenset(parse_custom_stopwords(custom_stopwords_text)),
        )

        st.subheader("LDA model")
        num_topics = st.slider("Number of topics", 2, 20, 5)
        words_per_topic = st.slider("Words per topic", 3, 25, 10)
        min_df = st.slider(
            "Minimum document frequency",
            1,
            10,
            1,
            help="A term must appear in at least this many documents to be included.",
        )
        max_df = st.slider(
            "Maximum document frequency",
            0.50,
            1.00,
            1.00,
            0.05,
            help="Lower this to remove words that appear in most documents.",
        )
        max_iter = st.slider("Iterations", 5, 100, 20, 5)
        random_state = st.number_input("Random state", min_value=0, value=42, step=1)
        max_features = st.number_input(
            "Maximum vocabulary size",
            min_value=0,
            value=5000,
            step=500,
            help="Set to 0 for no explicit vocabulary cap.",
        )
        learning_method = st.selectbox(
            "Learning method",
            options=["batch", "online"],
            index=0,
            help="Batch is stable for smaller corpora. Online can be faster for larger datasets.",
        )

        model_config = ModelConfig(
            num_topics=num_topics,
            words_per_topic=words_per_topic,
            min_df=min_df,
            max_df=max_df,
            random_state=int(random_state),
            max_iter=max_iter,
            max_features=None if max_features == 0 else int(max_features),
            learning_method=learning_method,
        )

    return {
        "supported_file_types": supported_file_types,
        "csv_mode": csv_mode,
        "max_file_size_mb": max_file_size_mb,
        "preprocessing_config": preprocessing_config,
        "model_config": model_config,
    }


def render_file_errors(read_errors: list[str]) -> None:
    if not read_errors:
        return

    with st.expander("Files that could not be processed", expanded=True):
        for error in read_errors:
            st.error(error)


def render_document_overview(documents: list[Document]) -> None:
    if not documents:
        return

    total_chars = sum(document.char_count for document in documents)
    total_size = sum(document.size_bytes for document in documents)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Documents ready", f"{len(documents):,}")
    metric_cols[1].metric("Total characters", f"{total_chars:,}")
    metric_cols[2].metric("Approx. upload size", format_file_size(total_size))

    overview_df = pd.DataFrame(
        {
            "Document": [document.name for document in documents],
            "Type": [document.source_type.upper() for document in documents],
            "Characters": [document.char_count for document in documents],
        }
    )
    st.dataframe(overview_df, use_container_width=True, hide_index=True)

    with st.expander("Document preview"):
        selected_name = st.selectbox(
            "Preview document",
            options=[document.name for document in documents],
        )
        selected_doc = next(document for document in documents if document.name == selected_name)
        st.text_area("Extracted text", selected_doc.text[:3000], height=220, disabled=True)


def render_results(result, preprocessed_df: pd.DataFrame, preprocessing_summary: dict) -> None:
    st.success("Topic model completed.")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Processed documents", f"{preprocessing_summary['processed_documents']:,}")
    metric_cols[1].metric("Total tokens", f"{preprocessing_summary['total_tokens']:,}")
    metric_cols[2].metric("Vocabulary size", f"{result.vocabulary_size:,}")
    metric_cols[3].metric("Topics", f"{result.topic_count:,}")

    st.subheader("Extracted topics")
    topic_cards = st.columns(2)
    for index, row in result.topics.iterrows():
        with topic_cards[index % 2]:
            with st.container(border=True):
                st.markdown(f"#### {row['topic_label']}")
                st.write(row["keywords"])

    st.subheader("Explore results")
    topic_tab, distribution_tab, mapping_tab, preprocessing_tab, download_tab = st.tabs(
        [
            "Topic keywords",
            "Document distribution",
            "Document mapping",
            "Preprocessing",
            "Downloads",
        ]
    )

    with topic_tab:
        selected_topic = st.selectbox("Topic", options=result.topics["topic_label"].tolist())
        selected_topic_id = int(
            result.topics.loc[result.topics["topic_label"] == selected_topic, "topic_id"].iloc[0]
        )
        st.altair_chart(
            make_topic_terms_chart(result.topic_terms, selected_topic_id),
            use_container_width=True,
        )
        st.dataframe(
            result.topic_terms[result.topic_terms["topic_id"] == selected_topic_id],
            use_container_width=True,
            hide_index=True,
        )

    with distribution_tab:
        if result.document_topic_distribution.empty:
            st.info("Document distribution is unavailable for this run.")
        else:
            st.altair_chart(
                make_document_topic_chart(result.document_topic_distribution),
                use_container_width=True,
            )
            st.altair_chart(make_dominant_topic_chart(result.document_topics), use_container_width=True)

    with mapping_tab:
        st.dataframe(result.document_topics, use_container_width=True, hide_index=True)

    with preprocessing_tab:
        display_columns = ["document_name", "token_count", "processed_text"]
        st.dataframe(preprocessed_df[display_columns], use_container_width=True, hide_index=True)

    with download_tab:
        st.download_button(
            "Download topics as CSV",
            dataframe_to_csv_bytes(result.topics),
            file_name="lda_topics.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download document mapping as CSV",
            dataframe_to_csv_bytes(result.document_topics),
            file_name="document_topic_mapping.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download full results as JSON",
            results_to_json_bytes(result),
            file_name="lda_results.json",
            mime="application/json",
            use_container_width=True,
        )


@st.cache_data(show_spinner=False)
def preprocess_cached(
    document_records: tuple[tuple[str, str, str, int], ...],
    config: PreprocessingConfig,
) -> pd.DataFrame:
    documents = [
        Document(name=name, text=text, source_type=source_type, size_bytes=size_bytes)
        for name, text, source_type, size_bytes in document_records
    ]
    return preprocess_documents(documents, config)


@st.cache_data(show_spinner=False)
def fit_lda_cached(preprocessed_records: tuple[dict, ...], config: ModelConfig):
    preprocessed_df = pd.DataFrame(preprocessed_records)
    return fit_lda_model(preprocessed_df, config)


def to_document_records(documents: list[Document]) -> tuple[tuple[str, str, str, int], ...]:
    return tuple(
        (document.name, document.text, document.source_type, document.size_bytes)
        for document in documents
    )


def load_sample_documents() -> list[Document]:
    sample_documents: list[Document] = []

    for path in sorted(SAMPLE_DOCUMENTS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        sample_documents.append(
            Document(
                name=f"sample/{path.name}",
                text=text,
                source_type="txt",
                size_bytes=path.stat().st_size,
            )
        )

    return sample_documents


def apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #e5e7eb;
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
