"""
Document Processor: Handles PDF/TXT ingestion, chunking, TF-IDF embedding, and FAISS indexing.
100% LOCAL — no external document APIs, no HuggingFace model downloads required.
Uses TF-IDF + SVD (LSA) for embeddings: fast, offline, great for domain-specific docs.
"""

import pickle
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Callable, Optional

import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from pypdf import PdfReader

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
DOCS_DIR        = BASE_DIR / "data" / "docs"
INDEX_DIR       = BASE_DIR / "data" / "index"
INDEX_FILE      = INDEX_DIR / "faiss.index"
META_FILE       = INDEX_DIR / "metadata.pkl"
VECTORIZER_FILE = INDEX_DIR / "vectorizer.pkl"

INDEX_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

EMBED_DIM = 256   # SVD output dimensions

# ─────────────────────────────────────────────────────────────────────────────
# 1. Text Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_txt(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".txt", ".md"):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Chunking
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\s*\n\s*', '\n', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 3. Embedding — TF-IDF + Truncated SVD (LSA) — fully local
# ─────────────────────────────────────────────────────────────────────────────

class LocalEmbedder:
    """
    TF-IDF + Truncated SVD (Latent Semantic Analysis).
    Completely offline. No model downloads. Works great on domain-specific text.
    """
    def __init__(self, n_components: int = EMBED_DIM):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50_000,
            sublinear_tf=True,
            min_df=1,
        )
        self.svd    = None
        self.fitted = False

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        tfidf  = self.vectorizer.fit_transform(texts)
        n_comp = min(self.n_components, tfidf.shape[0] - 1, tfidf.shape[1])
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        vecs   = self.svd.fit_transform(tfidf)
        self.fitted = True
        return normalize(vecs, norm="l2").astype(np.float32)

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Embedder not fitted. Index documents first.")
        tfidf = self.vectorizer.transform(texts)
        vecs  = self.svd.transform(tfidf)
        return normalize(vecs, norm="l2").astype(np.float32)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "svd": self.svd, "fitted": self.fitted}, f)

    @classmethod
    def load(cls, path: str) -> "LocalEmbedder":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj.vectorizer = state["vectorizer"]
        obj.svd        = state["svd"]
        obj.fitted     = state["fitted"]
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# 4. Public API
# ─────────────────────────────────────────────────────────────────────────────

def ingest_documents(file_paths: List[str], status_cb: Optional[Callable] = None) -> Dict:
    """
    Ingest a list of file paths into the FAISS index.
    All processing is local. Returns a summary dict.
    """
    all_chunks: List[str] = []
    all_meta:   List[Dict] = []

    for fp in file_paths:
        fname = Path(fp).name
        if status_cb:
            status_cb(f"📄 Extracting: {fname}")
        try:
            raw   = extract_text(fp)
            clean = clean_text(raw)
            chunks = chunk_text(clean)
            for i, ch in enumerate(chunks):
                all_chunks.append(ch)
                all_meta.append({"source": fname, "chunk_id": i, "text": ch})
        except Exception as e:
            if status_cb:
                status_cb(f"   ⚠ Skipped {fname}: {e}")

    if not all_chunks:
        return {"status": "error", "message": "No text could be extracted from documents."}

    if status_cb:
        status_cb(f"🔢 Embedding {len(all_chunks)} chunks (TF-IDF + SVD)…")

    embedder = LocalEmbedder()
    vectors  = embedder.fit_transform(all_chunks)

    if status_cb:
        status_cb("🗄️ Building FAISS index…")

    dim   = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner-product = cosine sim on L2-normalized vecs
    index.add(vectors)

    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "wb") as f:
        pickle.dump(all_meta, f)
    embedder.save(str(VECTORIZER_FILE))

    return {
        "status":       "ok",
        "total_chunks": len(all_chunks),
        "documents":    list({m["source"] for m in all_meta}),
        "embed_dim":    dim,
    }


def index_exists() -> bool:
    return INDEX_FILE.exists() and META_FILE.exists() and VECTORIZER_FILE.exists()


def load_index():
    index    = faiss.read_index(str(INDEX_FILE))
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    embedder = LocalEmbedder.load(str(VECTORIZER_FILE))
    return index, meta, embedder


def get_indexed_documents() -> List[str]:
    if not index_exists():
        return []
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    return sorted({m["source"] for m in meta})
