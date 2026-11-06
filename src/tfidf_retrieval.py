"""
=============================================================================
 Modul TF-IDF Retrieval & Vector Space Model (Tugas 2 IR)
=============================================================================
 Implementasi:
   - TF (Term Frequency)   : 1 + log10(tf) jika tf > 0, else 0
   - IDF (Inverse Doc Freq) : log10(N / df)
   - TF-IDF                : tf_weight * idf
   - Vector Space Model    : dokumen & query direpresentasikan sebagai vektor
   - Cosine Similarity     : cos(q, d) = (q · d) / (|q| * |d|)
=============================================================================
"""

import os
import sys
import math
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.indexing import load_index
from src.preprocessing import preprocess_text


def compute_tf(term: str, doc_tokens: list) -> float:
    """
    TF weight menggunakan log-normalisasi:
        w(t,d) = 1 + log10(tf)   jika tf > 0
                 0                jika tf = 0
    """
    tf = doc_tokens.count(term) if isinstance(doc_tokens, list) else 0
    if tf > 0:
        return 1 + math.log10(tf)
    return 0.0


def compute_idf(df_term: int, N: int) -> float:
    """
    IDF menggunakan log:
        idf(t) = log10(N / df)
    """
    if df_term <= 0 or N <= 0:
        return 0.0
    return math.log10(N / df_term)


def compute_tfidf(tf_weight: float, idf: float) -> float:
    """TF-IDF = tf_weight * idf."""
    return tf_weight * idf


def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """
    Cosine similarity antara dua vektor (dict: term → weight).

        cos(a, b) = Σ(a_i * b_i) / (|a| * |b|)
    """
    # Dot product (hanya term yang ada di kedua vektor)
    common_terms = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[t] * vec_b[t] for t in common_terms)

    # Magnitudes
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


def tfidf_search(query: str, index_data: dict, top_k: int = 10) -> list:
    """
    Pencarian menggunakan TF-IDF + Cosine Similarity.

    Parameters
    ----------
    query : str
        Query pengguna (teks mentah, akan dipreprocess).
    index_data : dict
        Inverted index.
    top_k : int
        Jumlah hasil teratas.

    Returns
    -------
    list of (doc_id, score)
        Daftar dokumen terurut berdasarkan skor cosine similarity.
    """
    N = index_data["doc_count"]
    idx = index_data["index"]

    # --- Preprocess query ---
    query_processed = preprocess_text(query)
    query_tokens = query_processed.split()

    if not query_tokens:
        return []

    # --- Bangun vektor query (TF-IDF) ---
    query_tf = Counter(query_tokens)
    query_vector = {}

    for term in query_tf:
        if term in idx:
            tf_w = 1 + math.log10(query_tf[term])
            idf  = compute_idf(idx[term]["df"], N)
            query_vector[term] = tf_w * idf

    if not query_vector:
        return []

    # --- Cari kandidat dokumen (hanya yang punya term dari query) ---
    candidate_docs = set()
    for term in query_vector:
        if term in idx:
            for posting in idx[term]["postings"]:
                candidate_docs.add(posting["doc_id"])

    # --- Hitung cosine similarity tiap kandidat ---
    scores = []
    for doc_id in candidate_docs:
        doc_vector = {}
        for term in query_vector:
            if term in idx:
                # Cari TF term ini di dokumen doc_id
                tf = 0
                for posting in idx[term]["postings"]:
                    if posting["doc_id"] == doc_id:
                        tf = posting["tf"]
                        break
                if tf > 0:
                    tf_w = 1 + math.log10(tf)
                    idf  = compute_idf(idx[term]["df"], N)
                    doc_vector[term] = tf_w * idf

        if doc_vector:
            score = cosine_similarity(query_vector, doc_vector)
            if score > 0:
                scores.append((doc_id, score))

    # --- Urutkan dari terbesar ---
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ---------------------------------------------------------------------------
# Entry point — demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(BASE_DIR, "index", "inverted_index.json")
    data_path  = os.path.join(BASE_DIR, "data", "dataset_preprocessed.csv")

    if not os.path.exists(index_path):
        print("ERROR: Jalankan indexing.py terlebih dahulu!")
    else:
        index_data = load_index(index_path)
        df = pd.read_csv(data_path)

        print("=" * 60)
        print("TF-IDF + COSINE SIMILARITY RETRIEVAL")
        print("=" * 60)

        query = "Indonesia juara ASEAN Cup"
        print(f"\nQuery: '{query}'")
        results = tfidf_search(query, index_data, top_k=10)

        for rank, (doc_id, score) in enumerate(results, 1):
            row = df[df["doc_id"] == doc_id].iloc[0]
            print(f"\n  [{rank}] {row['title']}")
            print(f"      Skor  : {score:.4f}")
            print(f"      URL   : {row['url']}")
            print(f"      Snippet: {row['content'][:100]}...")

