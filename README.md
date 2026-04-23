# LDA Topic Modeling Studio

A professional Streamlit web application for uploading text-based documents, preprocessing their contents, running Latent Dirichlet Allocation (LDA), and exploring extracted topics through tables and charts.

The base version is designed to work reliably with `.txt` and `.csv` files. PDF support is included through `pypdf` for PDFs that contain selectable text.

## Features

- Upload one or more `.txt`, `.csv`, or `.pdf` documents
- Read CSV files as one document per file or one document per row
- Safe file handling with clear per-file error messages
- Configurable preprocessing:
  - lowercasing
  - punctuation removal
  - number removal
  - stopword removal
  - custom stopwords
  - optional Porter stemming
  - minimum token length
- Configurable LDA parameters:
  - number of topics
  - words per topic
  - minimum document frequency
  - maximum document frequency
  - random state
  - iterations
  - vocabulary size
  - learning method
- Clear topic keyword summaries
- Topic distribution across documents
- Document-to-topic mapping table
- CSV and JSON download buttons for results
- Bundled sample documents for quick testing
- Streamlit Community Cloud ready

## Repository Structure

```text
lda-topic-modelling/
|-- .streamlit/
|   `-- config.toml
|-- sample_documents/
|   |-- climate_policy.txt
|   |-- fintech_security.txt
|   `-- healthcare_ai.txt
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- export.py
|   |-- file_handling.py
|   |-- modeling.py
|   |-- preprocessing.py
|   |-- utils.py
|   `-- visualization.py
|-- .gitignore
|-- lda.py
|-- README.md
|-- requirements.txt
`-- streamlit_app.py
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/lda-topic-modelling.git
cd lda-topic-modelling
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS or Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Local Usage

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

You can test the app immediately by checking **Include sample documents** in the interface.

## Streamlit Community Cloud Deployment

1. Push this project to a GitHub repository.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Create a new app and connect your GitHub account.
4. Select this repository.
5. Set the main file path to:

```text
streamlit_app.py
```

6. Deploy the app.

Streamlit Community Cloud will install the packages listed in `requirements.txt`.

## Customization Guide

- Supported file types are defined in `src/config.py`.
- File readers live in `src/file_handling.py`.
- Preprocessing steps live in `src/preprocessing.py`.
- LDA modeling logic lives in `src/modeling.py`.
- Charts live in `src/visualization.py`.
- Download/export helpers live in `src/export.py`.
- UI layout and Streamlit controls live in `streamlit_app.py`.

Good places for future improvements:

- Add `.docx` support in `src/file_handling.py`.
- Add lemmatization with spaCy or NLTK WordNet.
- Add a pyLDAvis-style visualization page.
- Add language-specific stopword lists.
- Add automated tests for file readers and preprocessing.
- Store previous modeling runs in a database or cloud storage.

## Notes on PDF Support

PDF extraction works only when a PDF contains selectable text. Scanned PDFs usually require OCR, which is intentionally not included in the base version to keep deployment simple.

## Performance Tips

- Start with smaller files while tuning preprocessing settings.
- Increase maximum vocabulary size only when needed.
- Use CSV row mode when each row is a separate document.
- For large corpora, try the `online` learning method and reduce the vocabulary cap.
