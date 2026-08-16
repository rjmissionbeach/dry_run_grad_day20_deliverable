import hashlib
import io
import json
import re
import time
from datetime import datetime, timezone

import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Research Assistant", page_icon="📄", layout="wide")

SYSTEM_PROMPT = """You are a research assistant that answers strictly from the provided context.

Rules:
- Use ONLY the CONTEXT below. Do not use outside knowledge, even if you are confident.
- If the context does not contain the answer, reply exactly: "I don't know based on the provided documents."
- If the context only partially answers, say which part you can support and which you cannot.
- Cite the bracketed source numbers you used, e.g. [1], [2]. Cite only numbers that appear in CONTEXT.
- Do not estimate, extrapolate, or infer figures that are not stated.
- Be concise and factual."""

REFUSAL = "I don't know based on the provided documents."


# ============================== document ingestion ==============================
def strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<!--.*?-->", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    for entity, char in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
                         ("&gt;", ">"), ("&#8217;", "'"), ("&#8220;", '"'),
                         ("&#8221;", '"'), ("&quot;", '"')]:
        raw = raw.replace(entity, char)
    return raw


def read_file(name: str, data: bytes) -> str:
    """Extract plain text from one uploaded file."""
    lower = name.lower()

    if lower.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:
            st.warning(f"Could not read {name}: {exc}")
            return ""

    text = data.decode("utf-8", errors="ignore")

    # .md is plain text -> treat exactly like .txt (no manual conversion needed)
    if lower.endswith((".html", ".htm")):
        text = strip_html(text)

    return text


def clean(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return [c for c in chunks if len(c.split()) > 20]


@st.cache_data(show_spinner=False)
def build_corpus(files: tuple):
    """files: tuple of (name, bytes). Cached so re-runs don't re-parse."""
    documents, seen_hashes, skipped = [], {}, []

    for name, data in files:
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen_hashes:
            skipped.append((name, seen_hashes[digest]))
            continue
        seen_hashes[digest] = name

        text = clean(read_file(name, data))
        if not text:
            skipped.append((name, "no extractable text (scanned PDF? needs OCR)"))
            continue

        for j, chunk in enumerate(chunk_text(text)):
            documents.append({"text": chunk, "source": name, "chunk_id": j})

    return documents, skipped


# ================================== retrieval ==================================
# SWAP POINT 1 of 2: replace this function to use embeddings instead
