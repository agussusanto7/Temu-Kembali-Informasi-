"""
=============================================================================
 Modul Okapi BM25 Ranking (Tugas 2 IR)
=============================================================================
 Implementasi BM25 murni (tanpa ketergantungan library luar):
   - k1 : parameter term frequency saturation (default: 1.2)
   - b  : parameter document length normalization (default: 0.75)
   - IDF: log( (N - df + 0.5) / (df + 0.5) + 1 )   [versi Robertson-Spärck Jones]
   - BM25 score = Σ IDF(q_i) * [ tf * (k1 + 1) ] / [ tf + k1 * (1 - b + b * (dl / avgdl)) ]
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


def compute_bm25_idf(df: int, N: int) -> float:
    """
    IDF versi BM25 (Robertson-Spärck Jones):
        IDF(t) = log( (N - df + 0.5) / (df + 0.5) + 1 )
    """
    return math.log((N - df + 0.5) / (df + 0.5) + 1.0)


def bm25_search(query: str,
                index_data: dict,
                k1: float = 1.2,
                b: float = 0.75,
                top_k: int = 10) -> list:
    """
    Pencarian menggunakan Okapi BM25.

    Parameters
    ----------
    query : str
        Query pengguna (teks mentah).
    index_data : dict
        Inverted index.
    k1 : float
        Parameter saturasi TF (default: 1.2).
    b : float
        Parameter normalisasi panjang dokumen (default: 0.75).
    top_k : int
        Jumlah dokumen teratas.

    Returns
    -------
    list of (doc_id, score)
        Daftar dokumen terurut berdasarkan skor BM25.
    """
    N = index_data["doc_count"]
    avgdl = index_data["avgdl"]
    doc_lens = index_data["doc_lens"]
    idx = index_data["index"]

    # --- Preprocess query ---
    query_processed = preprocess_text(query)
    query_tokens = query_processed.split()

    if not query_tokens:
        return []

    # Frekuensi term di query
    q_counter = Counter(query_tokens)

    # --- Hitung skor BM25 per dokumen ---
    doc_scores = {}

    for term, q_tf in q_counter.items():
        if term not in idx:
            continue

        term_data = idx[term]
        df_t = term_data["df"]
        idf = compute_bm25_idf(df_t, N)

        # Loop semua dokumen yang mengandung term ini
        for posting in term_data["postings"]:
            doc_id = posting["doc_id"]
            tf = posting["tf"]
            dl = doc_lens.get(doc_id, avgdl)

            # Rumus BM25 untuk term ini
            # Saturation term + Length normalization
            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * (dl / avgdl))
            term_score = idf * (numerator / denominator)

            # Tambahkan ke skor akumulasi dokumen
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + term_score

    # --- Urutkan dari terbesar ---
    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


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
        print("OKAPI BM25 RANKING")
        print("=" * 60)

        query = "Indonesia juara ASEAN Cup"
        print(f"\nQuery: '{query}'")
        results = bm25_search(query, index_data, top_k=10)

        for rank, (doc_id, score) in enumerate(results, 1):
            row = df[df["doc_id"] == doc_id].iloc[0]
            print(f"\n  [{rank}] {row['title']}")
            print(f"      Skor BM25 : {score:.4f}")
            print(f"      URL       : {row['url']}")
            print(f"      Snippet   : {row['content'][:100]}...")

