"""
=============================================================================
 Modul Boolean Retrieval (Tugas 2 IR)
=============================================================================
 Mendukung query Boolean sederhana:
   - AND : irisan (kedua term harus ada)
   - OR  : gabungan (salah satu term ada)
   - NOT : pengurangan (term tidak boleh ada)
   - Default: AND (jika tidak ada operator)
=============================================================================
"""

import os
import sys
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.indexing import load_index


def _get_postings_set(index_data: dict, term: str) -> set:
    """Ambil set doc_id yang mengandung term."""
    idx = index_data["index"]
    if term in idx:
        return {p["doc_id"] for p in idx[term]["postings"]}
    return set()


def boolean_search(query: str, index_data: dict) -> list:
    """
    Pencarian Boolean sederhana.

    Mendukung operator: AND, OR, NOT
    Operator default jika tidak disebutkan: AND

    Parameters
    ----------
    query : str
        Query pengguna, misal: "timnas AND indonesia" atau "asean NOT thailand"
    index_data : dict
        Inverted index.

    Returns
    -------
    list of str
        Daftar doc_id yang memenuhi query.
    """
    # Tokenize query dengan mempertahankan operator
    tokens = query.lower().split()

    if not tokens:
        return []

    # Parse menjadi list operasi: [(term/operator, ...), ...]
    all_docs = set(index_data["doc_lens"].keys())

    result = None
    current_op = "AND"

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token in ("and", "or", "not"):
            current_op = token.upper()
            i += 1
            continue

        # Dapatkan postings untuk term ini
        term_docs = _get_postings_set(index_data, token)

        if result is None:
            if current_op == "NOT":
                result = all_docs - term_docs
            else:
                result = term_docs
        else:
            if current_op == "AND":
                result = result & term_docs
            elif current_op == "OR":
                result = result | term_docs
            elif current_op == "NOT":
                result = result - term_docs

        # Reset operator ke AND (default)
        current_op = "AND"
        i += 1

    return sorted(result) if result else []


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(BASE_DIR, "index", "inverted_index.json")

    if not os.path.exists(index_path):
        print("ERROR: Jalankan indexing.py terlebih dahulu!")
    else:
        index_data = load_index(index_path)
        print("=" * 60)
        print("BOOLEAN RETRIEVAL")
        print("=" * 60)

        queries = [
            "timnas AND indonesia",
            "asean OR aff",
            "indonesia NOT thailand",
            "juara AND final",
        ]

        for q in queries:
            results = boolean_search(q, index_data)
            print(f"\nQuery: '{q}'")
            print(f"  Hasil: {len(results)} dokumen")
            if results:
                print(f"  Contoh: {results[:10]}")

