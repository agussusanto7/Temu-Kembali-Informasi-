"""
=============================================================================
 Modul Mesin Pencari (Search Engine) — Tugas 2 IR
=============================================================================
 Mengintegrasikan:
   - Inverted Index
   - Boolean Retrieval
   - TF-IDF + Cosine Similarity
   - Okapi BM25
   - Pembuatan snippet/cuplikan dinamis
=============================================================================
"""

import os
import sys
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.indexing import load_index
from src.preprocessing import preprocess_text
from src.boolean_retrieval import boolean_search
from src.tfidf_retrieval import tfidf_search
from src.bm25 import bm25_search


class SearchEngine:
    """Mesin pencari sederhana yang menggabungkan seluruh komponen IR."""

    def __init__(self, data_path: str, index_path: str):
        self.data_path = data_path
        self.index_path = index_path

        # Muat dataset & index
        print("Memuat dataset dan index...")
        self.df = pd.read_csv(data_path).set_index("doc_id", drop=False)
        self.index_data = load_index(index_path)
        print("Search Engine siap digunakan!")

    def make_snippet(self, text: str, query: str, length: int = 150) -> str:
        """
        Buat cuplikan teks yang relevan dengan query.
        Mencari kalimat pertama yang mengandung term dari query.
        """
        if not isinstance(text, str):
            return ""

        query_words = [w.lower() for w in query.split() if len(w) > 2]
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Cari kalimat yang cocok
        best_sentence = ""
        for s in sentences:
            s_lower = s.lower()
            if any(w in s_lower for w in query_words):
                best_sentence = s
                break

        # Jika tidak ada kalimat yang cocok, ambil awal teks
        if not best_sentence:
            snippet = text[:length]
        else:
            snippet = best_sentence[:length]

        return snippet.strip() + "..." if len(text) > len(snippet) else snippet

    def search(self, query: str, method: str = "bm25", top_k: int = 10) -> list:
        """
        Lakukan pencarian.

        Parameters
        ----------
        query : str
            Query pencarian.
        method : str
            "bm25" / "tfidf" / "boolean"
        top_k : int
            Jumlah hasil teratas.

        Returns
        -------
        list of dict
            [
              {
                "rank": int,
                "doc_id": str,
                "title": str,
                "url": str,
                "score": float,
                "snippet": str,
                "sumber": str
              }, ...
            ]
        """
        if not query.strip():
            return []

        method = method.lower()
        results = []

        if method == "bm25":
            raw_results = bm25_search(query, self.index_data, top_k=top_k)
        elif method == "tfidf":
            raw_results = tfidf_search(query, self.index_data, top_k=top_k)
        elif method == "boolean":
            doc_ids = boolean_search(query, self.index_data)
            # Boolean tidak punya score numerik (0/1), kita buat skor 1.0
            raw_results = [(d, 1.0) for d in doc_ids[:top_k]]
        else:
            raise ValueError(f"Metode tidak dikenali: {method}. Pilih: 'bm25', 'tfidf', 'boolean'")

        # Susun output lengkap
        for rank, (doc_id, score) in enumerate(raw_results, 1):
            if doc_id in self.df.index:
                row = self.df.loc[doc_id]
                snippet = self.make_snippet(row["content"], query)
                results.append({
                    "rank": rank,
                    "doc_id": doc_id,
                    "title": row["title"],
                    "url": row["url"],
                    "score": round(score, 4),
                    "snippet": snippet,
                    "sumber": row.get("sumber", "-"),
                })

        return results

    def print_results(self, query: str, results: list, method: str):
        """Format tampilan hasil pencarian di terminal."""
        print("\n" + "=" * 70)
        print(f"HASIL PENCARIAN [{method.upper()}]: '{query}'")
        print("=" * 70)

        if not results:
            print("❌ Tidak ditemukan dokumen yang sesuai.")
            print("   Saran: coba gunakan kata kunci lain yang lebih umum.")
            return

        print(f"Ditemukan {len(results)} dokumen teratas:\n")
        for r in results:
            print(f"[{r['rank']}] {r['title']}")
            print(f"    Doc ID : {r['doc_id']} | Sumber: {r['sumber']} | Skor: {r['score']}")
            print(f"    URL    : {r['url']}")
            print(f"    Snippet: {r['snippet']}")
            print("-" * 70)


# ---------------------------------------------------------------------------
# CLI Interaktif
# ---------------------------------------------------------------------------
def interactive_cli():
    """Mode interaktif di terminal."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path  = os.path.join(BASE_DIR, "data", "dataset_preprocessed.csv")
    index_path = os.path.join(BASE_DIR, "index", "inverted_index.json")

    if not os.path.exists(index_path) or not os.path.exists(data_path):
        print("ERROR: Pastikan dataset_preprocessed.csv dan inverted_index.json sudah ada!")
        print("Jalankan preprocessing.py dan indexing.py terlebih dahulu.")
        return

    engine = SearchEngine(data_path, index_path)

    print("\n" + "=" * 70)
    print(" SELAMAT DATANG DI MESIN PENCARI TEMU KEMBALI INFORMASI (TUGAS 2)")
    print(" Topik: Timnas Indonesia Juara FIFA ASEAN Cup 2026")
    print("=" * 70)
    print("Metode yang tersedia:")
    print("  1. bm25    - Okapi BM25 (Rekomendasi)")
    print("  2. tfidf   - TF-IDF + Cosine Similarity")
    print("  3. boolean - Boolean Retrieval (AND/OR/NOT)")
    print("Ketik 'exit' atau 'keluar' untuk mengakhiri.\n")

    current_method = "bm25"
    top_k = 5

    while True:
        try:
            q = input(f"\n[Metode: {current_method} | Top-{top_k}] Masukkan query: ").strip()
            if q.lower() in ("exit", "keluar", "q"):
                print("Sampai jumpa!")
                break

            # Ganti metode via perintah khusus
            if q.lower().startswith(":method "):
                new_m = q.split()[1].lower()
                if new_m in ("bm25", "tfidf", "boolean"):
                    current_method = new_m
                    print(f"Metode diubah ke: {current_method}")
                else:
                    print("Metode tidak valid! Pilih: bm25, tfidf, boolean")
                continue

            if q.lower().startswith(":top "):
                top_k = int(q.split()[1])
                print(f"Top-k diubah ke: {top_k}")
                continue

            if not q:
                continue

            results = engine.search(q, method=current_method, top_k=top_k)
            engine.print_results(q, results, current_method)

        except (KeyboardInterrupt, EOFError):
            print("\nSampai jumpa!")
            break


if __name__ == "__main__":
    interactive_cli()

