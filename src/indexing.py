"""
=============================================================================
 Modul Indexing — Inverted Index (Tugas 2 IR)
=============================================================================
 Membangun inverted index: term → postings list
 Setiap entry postings menyimpan:
   - doc_id    : ID dokumen
   - tf        : Term Frequency (jumlah kemunculan term di dokumen tsb)

 Disimpan juga:
   - df        : Document Frequency per term
   - doc_count : total jumlah dokumen (N)
   - doc_lens  : panjang (jumlah token) tiap dokumen (untuk BM25)
   - avgdl     : rata-rata panjang dokumen
=============================================================================
"""

import os
import json
import math
import pandas as pd
from collections import defaultdict, Counter


def build_inverted_index(df: pd.DataFrame,
                         text_column: str = "content_preprocessed") -> dict:
    """
    Bangun inverted index dari DataFrame.

    Returns
    -------
    index_data : dict
        {
          "doc_count": int,
          "avgdl": float,
          "doc_lens": { doc_id: int, ... },
          "index": {
              term: {
                  "df": int,
                  "postings": [ {"doc_id": str, "tf": int}, ... ]
              },
              ...
          }
        }
    """
    N = len(df)
    inverted = defaultdict(lambda: {"df": 0, "postings": []})
    doc_lens = {}

    for _, row in df.iterrows():
        doc_id = row["doc_id"]
        tokens = str(row[text_column]).split()
        doc_lens[doc_id] = len(tokens)

        # Hitung TF per term dalam dokumen ini
        tf_counter = Counter(tokens)

        for term, tf in tf_counter.items():
            inverted[term]["postings"].append({"doc_id": doc_id, "tf": tf})

    # Hitung DF
    for term in inverted:
        inverted[term]["df"] = len(inverted[term]["postings"])

    avgdl = sum(doc_lens.values()) / N if N > 0 else 0

    index_data = {
        "doc_count": N,
        "avgdl": avgdl,
        "doc_lens": doc_lens,
        "index": dict(inverted),
    }

    return index_data


def save_index(index_data: dict, output_path: str):
    """Simpan inverted index ke file JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)
    print(f"Inverted index disimpan ke: {output_path}")
    print(f"  Jumlah term unik : {len(index_data['index'])}")
    print(f"  Jumlah dokumen   : {index_data['doc_count']}")
    print(f"  Rata-rata panjang: {index_data['avgdl']:.1f} token")


def load_index(index_path: str) -> dict:
    """Muat inverted index dari file JSON."""
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def show_index_sample(index_data: dict, n_terms: int = 10):
    """Tampilkan contoh index, postings, TF, dan DF."""
    idx = index_data["index"]
    # Ambil term dengan DF tertinggi (paling umum)
    sorted_terms = sorted(idx.items(), key=lambda x: x[1]["df"], reverse=True)

    print("\n" + "=" * 60)
    print(f"CONTOH INVERTED INDEX ({n_terms} term teratas berdasarkan DF)")
    print("=" * 60)
    print(f"{'Term':<20} {'DF':>5}  {'Contoh Postings (doc_id: tf)'}")
    print("-" * 60)

    for term, data in sorted_terms[:n_terms]:
        postings_sample = data["postings"][:5]
        postings_str = ", ".join(
            f"{p['doc_id']}:{p['tf']}" for p in postings_sample
        )
        if len(data["postings"]) > 5:
            postings_str += f", ... (+{len(data['postings'])-5} lagi)"
        print(f"{term:<20} {data['df']:>5}  [{postings_str}]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preprocessed_csv = os.path.join(BASE_DIR, "data", "dataset_preprocessed.csv")
    index_path = os.path.join(BASE_DIR, "index", "inverted_index.json")

    if not os.path.exists(preprocessed_csv):
        print("ERROR: Jalankan preprocessing.py terlebih dahulu!")
    else:
        df = pd.read_csv(preprocessed_csv)
        print("Membangun inverted index...")
        index_data = build_inverted_index(df)
        save_index(index_data, index_path)
        show_index_sample(index_data)

