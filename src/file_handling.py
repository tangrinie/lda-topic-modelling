from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
import re
from typing import Iterable, Literal

import pandas as pd


CsvMode = Literal["file", "rows"]
WORD_PATTERN = re.compile(r"\S+")


@dataclass(frozen=True)
class Document:
    name: str
    text: str
    source_type: str
    size_bytes: int = 0

    @property
    def char_count(self) -> int:
        return len(self.text)


def read_uploaded_files(
    uploaded_files: Iterable,
    supported_file_types: list[str],
    csv_mode: CsvMode = "file",
    max_file_size_mb: int = 10,
) -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    errors: list[str] = []
    supported_extensions = {file_type.lower().lstrip(".") for file_type in supported_file_types}
    max_file_size_bytes = max_file_size_mb * 1024 * 1024

    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, "name", "uploaded_file")
        extension = Path(filename).suffix.lower().lstrip(".")

        try:
            file_bytes = uploaded_file.getvalue()
            if len(file_bytes) > max_file_size_bytes:
                raise ValueError(f"File is larger than the configured {max_file_size_mb} MB limit.")

            if extension not in supported_extensions:
                raise ValueError(f".{extension or 'unknown'} files are not enabled in the sidebar.")

            if extension == "txt":
                text = decode_text(file_bytes).strip()
                documents.append(build_document(filename, text, extension, len(file_bytes)))
            elif extension == "csv":
                documents.extend(read_csv_documents(filename, file_bytes, csv_mode))
            elif extension == "pdf":
                text = read_pdf_text(file_bytes).strip()
                documents.append(build_document(filename, text, extension, len(file_bytes)))
            else:
                # Future extension: add .docx, .html, or OCR-based readers here.
                raise ValueError(f"No reader is implemented for .{extension} files yet.")
        except Exception as exc:
            errors.append(f"{filename}: {exc}")

    return documents, errors


def decode_text(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError("Could not decode the file as text.")


def build_document(filename: str, text: str, extension: str, size_bytes: int) -> Document:
    if not text.strip():
        raise ValueError("No readable text was found.")

    return Document(name=filename, text=text, source_type=extension, size_bytes=size_bytes)


def read_csv_documents(filename: str, file_bytes: bytes, csv_mode: CsvMode) -> list[Document]:
    text = decode_text(file_bytes)

    try:
        dataframe = pd.read_csv(StringIO(text))
    except Exception as exc:
        raise ValueError(f"Could not parse CSV file. Details: {exc}") from exc

    text_columns = [
        column
        for column in dataframe.columns
        if dataframe[column].dtype == "object" or pd.api.types.is_string_dtype(dataframe[column])
    ]
    if not text_columns:
        raise ValueError("The CSV does not contain text columns.")

    text_frame = dataframe[text_columns].fillna("").astype(str)

    if csv_mode == "rows":
        row_documents: list[Document] = []
        for row_index, row in text_frame.iterrows():
            row_text = " ".join(value.strip() for value in row.tolist() if value.strip())
            if row_text:
                row_documents.append(
                    Document(
                        name=f"{filename} row {row_index + 1}",
                        text=row_text,
                        source_type="csv",
                        size_bytes=len(file_bytes) if not row_documents else 0,
                    )
                )

        if not row_documents:
            raise ValueError("The CSV rows did not contain readable text.")
        return row_documents

    combined_text = "\n".join(
        " ".join(value.strip() for value in row.tolist() if value.strip())
        for _, row in text_frame.iterrows()
    ).strip()
    return [build_document(filename, combined_text, "csv", len(file_bytes))]


def read_pdf_text(file_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError("PDF support requires the optional pypdf package.") from exc

    try:
        reader = PdfReader(BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise ValueError(f"Could not extract text from PDF. Details: {exc}") from exc

    text = "\n\n".join(pages).strip()
    if not text:
        raise ValueError("No selectable text was found in the PDF.")

    return text


def split_documents_into_chunks(
    documents: list[Document],
    enabled: bool = True,
    target_words: int = 300,
    min_words_for_chunking: int = 450,
) -> list[Document]:
    if not enabled:
        return documents

    chunked_documents: list[Document] = []

    for document in documents:
        words = WORD_PATTERN.findall(document.text)
        if len(words) < min_words_for_chunking:
            chunked_documents.append(document)
            continue

        chunks = split_words(words, target_words)
        for chunk_index, chunk_words in enumerate(chunks, start=1):
            chunked_documents.append(
                Document(
                    name=f"{document.name} chunk {chunk_index}",
                    text=" ".join(chunk_words),
                    source_type=f"{document.source_type} chunk",
                    size_bytes=document.size_bytes if chunk_index == 1 else 0,
                )
            )

    return chunked_documents


def split_words(words: list[str], target_words: int) -> list[list[str]]:
    target_words = max(target_words, 50)
    return [words[start : start + target_words] for start in range(0, len(words), target_words)]
