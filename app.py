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

REFUSAL = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You are a research assistant that answers strictly from the provided context.

Rules:
- Use ONLY the CONTEXT below. Do not use outside knowledge, even if confident.
- If the context does not contain the answer, reply exactly: "{REFUSAL}"
- If the context only partly answers, say what you can support and what you cannot.
- Cite the bracketed source numbers you used, e.g. [1], [2]. Only cite numbers shown in CONTEXT.
- Never estimate or infer figures that are not explicitly stated.
- Be concise and factual."""


def strip_html(raw):
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<!--.*?-->", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    for a, b in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&quot;", '"'), ("&#8217;", "'"), ("&#8220;", '"'), ("&#8221;", '"')]:
        raw = raw.replace(a, b)
    return raw


def read_file(name, data):
    lower = name.lower()
    if lower.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception as exc:
            st.warning(f"Could not read {name}: {exc}")
            return ""
    text = data.decode("utf-8", errors="ignore")
    # .md and .txt are plain text, no conversion needed; .htm/.html get tags stripped
    if lower.endswith((".html", ".htm")):
        text = strip_html(text)
    return text


def clean(text):
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def chunk_text(text, chunk_size=900, overlap=150):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return [c for c in chunks if len(c.split()) > 20]


@st.cache_data(show_spinner=False)
def build_corpus(files):
    documents, seen, notes = [], {}, []
    for name, data in files:
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            notes.append(f"Skipped **{name}** — identical to *{seen[digest]}*")
            continue
        seen[digest] = name
        text = clean(read_file(name, data))
        if not text:
            notes.append(f"Skipped **{name}** — no extractable text (scanned PDF?)")
            continue
        for j, ch in enumerate(chunk_text(text)):
            documents.append({"text": ch, "source": name, "chunk_id": j})
    return documents, notes


@st.cache_resource(show_spinner=False)
def build_index(texts):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                          sublinear_tf=True, max_df=0.9)
    return vec, vec.fit_transform(texts)


def retrieve(query, vec, matrix, documents, k=5, min_score=0.03):
    sims = cosine_similarity(vec.transform([query]), matrix).ravel()
    order = np.argsort(-sims)[:k]
    return [{**documents[i], "score": float(sims[i])} for i in order if sims[i] >= min_score]


def build_context(hits):
    return "\n\n".join(
        f"[{n}] (source: {h['source']}, chunk {h['chunk_id']})\n{h['text']}"
        for n, h in enumerate(hits, 1)
    )
for key, default in [("messages", []), ("log", [])]:
    st.session_state.setdefault(key, default)

st.title("📄 Document Research Assistant")
st.caption("Answers come only from the documents you upload, with citations.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("OpenRouter API key", type="password",
                            placeholder="sk-or-v1-...")
    model = st.text_input("Model", value="openrouter/free")
    uploads = st.file_uploader("Reference documents",
                               type=["pdf", "txt", "md", "html", "htm"],
                               accept_multiple_files=True)
    st.divider()
    k = st.slider("Chunks retrieved", 3, 10, 5)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_tokens = st.slider("Max answer tokens", 400, 3000, 1500, 100)
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    if st.session_state.log:
        st.download_button("Download session log (JSON)",
                           data=json.dumps(st.session_state.log, indent=2),
                           file_name="session_log.json", mime="application/json")

documents, vec, matrix = [], None, None
if uploads:
    files = tuple((f.name, f.getvalue()) for f in uploads)
    documents, notes = build_corpus(files)
    for note in notes:
        st.sidebar.warning(note)
    if documents:
        vec, matrix = build_index(tuple(d["text"] for d in documents))
        st.sidebar.success(f"{len(set(d['source'] for d in documents))} docs → "
                           f"{len(documents)} chunks")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("hits"):
            with st.expander("Sources"):
                for n, h in enumerate(msg["hits"], 1):
                    st.markdown(f"**[{n}] {h['source']}** — chunk {h['chunk_id']}, "
                                f"score {h['score']:.3f}")
                    st.caption(h["text"][:400] + "…")

question = st.chat_input("Ask a question about your documents")

if question:
    if not api_key:
        st.error("Paste your OpenRouter API key in the sidebar.")
        st.stop()
    if not documents:
        st.error("Upload at least one document first.")
                st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    hits = retrieve(question, vec, matrix, documents, k=k)

    with st.chat_message("assistant"):
        start = time.time()
        if not hits:
            answer = REFUSAL
            st.markdown(answer)
        else:
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            payload = [{"role": "system", "content": SYSTEM_PROMPT}]
            for m in st.session_state.messages[-7:-1]:
                payload.append({"role": m["role"], "content": m["content"]})
            payload.append({
                "role": "user",
                "content": f"CONTEXT:\n{build_context(hits)}\n\nQUESTION: {question}",
            })
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=payload,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                answer = st.write_stream(
                    (c.choices[0].delta.content or "") for c in stream
                )
            except Exception as exc:
                answer = f"**API error:** {exc}"
                st.error(answer)
        elapsed = time.time() - start

        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer or "")})
        bad = [n for n in cited if n < 1 or n > len(hits)]
        if bad:
            st.warning(f"Answer cites source number(s) {bad} that were not provided.")

        if hits:
            with st.expander(f"Sources ({len(hits)} chunks retrieved)"):
                for n, h in enumerate(hits, 1):
                    used = "✅ cited" if n in cited else "— not cited"
                    st.markdown(
                        f"**[{n}] {h['source']}** · chunk {h['chunk_id']} · "
                        f"score {h['score']:.3f} · {used}"
                    )
                    st.caption(h["text"][:500] + ("…" if len(h["text"]) > 500 else ""))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "hits": hits}
    )
    st.session_state.log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "question": question,
        "answer": answer,
        "refused": (answer or "").strip().startswith(REFUSAL),
        "latency_seconds": round(elapsed, 2),
        "temperature": temperature,
        "k": k,
        "retrieved": [
            {"source": h["source"], "chunk_id": h["chunk_id"],
             "score": round(h["score"], 4), "cited": (i in cited)}
            for i, h in enumerate(hits, 1)
        ],
        "invalid_citations": bad,
    })
