"""
=============================================================================
 Modul Evaluasi Retrieval (Tugas 2 IR)
=============================================================================
 Melakukan evaluasi retrieval:
   - 10 query pengujian
   - Ground truth manual/relevansi per query
   - Precision@K, Recall@K, F1@K (K = 5 dan 10)
   - Perbandingan TF-IDF + Cosine vs Okapi BM25
   - Analisis perbedaan ranking
=============================================================================
"""

import os
import sys
import json
import pandas as pd

# Pastikan root direktori tugas2 ada di sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.indexing import load_index
from src.tfidf_retrieval import tfidf_search
from src.bm25 import bm25_search


# ---------------------------------------------------------------------------
# 10 Query Pengujian beserta Kriteria Relevansi & Ground Truth
# ---------------------------------------------------------------------------
# Query dirancang mencakup variasi:
#   1. Query pendek & spesifik (nama orang, skor final)
#   2. Query umum (juara, perayaan)
#   3. Query tentang lawan (Thailand, Malaysia)
#   4. Query tentang taktik / pelatih
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    {
        "query_id": "Q1",
        "query": "Indonesia juara final Thailand penalti",
        "deskripsi": "Berita kemenangan Indonesia atas Thailand lewat adu penalti di final",
        "keywords_must": ["indonesia", "thailand", "penalti"],
    },
    {
        "query_id": "Q2",
        "query": "Emil Audero kiper pahlawan penalti",
        "deskripsi": "Peran kiper Emil Audero menepis penalti di final",
        "keywords_must": ["audero"],
    },
    {
        "query_id": "Q3",
        "query": "John Herdman pelatih taktik strategi",
        "deskripsi": "Kinerja dan taktik pelatih John Herdman",
        "keywords_must": ["herdman"],
    },
    {
        "query_id": "Q4",
        "query": "Elkan Baggott gol penyeimbang",
        "deskripsi": "Gol krusial Elkan Baggott di babak perpanjangan waktu",
        "keywords_must": ["baggott"],
    },
    {
        "query_id": "Q5",
        "query": "Kevin Diks kapten kado ulang tahun",
        "deskripsi": "Kevin Diks sebagai kapten dan momen ulang tahunnya",
        "keywords_must": ["diks"],
    },
    {
        "query_id": "Q6",
        "query": "Chanathip Songkrasin kapten Thailand komentar",
        "deskripsi": "Pernyataan Chanathip Songkrasin pasca kekalahan Thailand",
        "keywords_must": ["chanathip"],
    },
    {
        "query_id": "Q7",
        "query": "Malaysia clean sheet gagal lolos final",
        "deskripsi": "Nasib Timnas Malaysia yang tidak kebobolan tapi gagal ke final",
        "keywords_must": ["malaysia"],
    },
    {
        "query_id": "Q8",
        "query": "Prabowo Subianto Presiden nonton selebrasi GBK",
        "deskripsi": "Kehadiran dan selebrasi Presiden Prabowo di SUGBK",
        "keywords_must": ["prabowo"],
    },
    {
        "query_id": "Q9",
        "query": "Justin Hubner bek preman",
        "deskripsi": "Aksi dan komentar Justin Hubner selama turnamen",
        "keywords_must": ["hubner"],
    },
    {
        "query_id": "Q10",
        "query": "Erick Thohir PSSI evaluasi timnas",
        "deskripsi": "Pernyataan Ketua Umum PSSI Erick Thohir",
        "keywords_must": ["erick"],
    },
]


def determine_ground_truth(df: pd.DataFrame, queries: list) -> dict:
    """
    Menentukan himpunan dokumen relevan (ground truth) untuk setiap query
    berdasarkan kriteria kata kunci wajib pada judul/konten.
    """
    ground_truth = {}

    for q in queries:
        qid = q["query_id"]
        must_words = q["keywords_must"]

        relevant_docs = []
        for _, row in df.iterrows():
            text = (str(row["title"]) + " " + str(row["content"])).lower()
            # Dokumen relevan jika mengandung SEMUA kata kunci wajib
            if all(w in text for w in must_words):
                relevant_docs.append(row["doc_id"])

        ground_truth[qid] = relevant_docs

    return ground_truth


# ---------------------------------------------------------------------------
# Metrik Evaluasi: Precision@K, Recall@K, F1@K
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Precision@K = (jumlah dokumen relevan di top-k) / k."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    rel_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant)
    return rel_in_top_k / k


def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Recall@K = (jumlah dokumen relevan di top-k) / (total dokumen relevan)."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    rel_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant)
    return rel_in_top_k / len(relevant)


def f1_at_k(p: float, r: float) -> float:
    """F1@K = 2 * P * R / (P + R)."""
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


# ---------------------------------------------------------------------------
# Jalankan Evaluasi Lengkap
# ---------------------------------------------------------------------------

def run_evaluation(data_path: str, index_path: str, output_csv: str):
    """Jalankan evaluasi komparatif antara TF-IDF dan BM25."""
    df = pd.read_csv(data_path)
    index_data = load_index(index_path)

    print("=" * 70)
    print("MENENTUKAN GROUND TRUTH UNTUK 10 QUERY PENGUJIAN")
    print("=" * 70)

    gt = determine_ground_truth(df, TEST_QUERIES)

    # Simpan queries & ground truth ke file CSV
    gt_rows = []
    for q in TEST_QUERIES:
        qid = q["query_id"]
        rel_docs = gt[qid]
        print(f"[{qid}] '{q['query']}' → {len(rel_docs)} dokumen relevan")
        gt_rows.append({
            "query_id": qid,
            "query": q["query"],
            "deskripsi": q["deskripsi"],
            "jumlah_relevan": len(rel_docs),
            "relevant_doc_ids": ",".join(rel_docs[:20]) + ("..." if len(rel_docs) > 20 else "")
        })

    queries_df = pd.DataFrame(gt_rows)
    queries_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\nGround truth disimpan ke: {output_csv}")

    # --- Evaluasi Per Query ---
    print("\n" + "=" * 70)
    print("EVALUASI PERFORMA: TF-IDF + COSINE vs OKAPI BM25")
    print("=" * 70)

    results_table = []

    for q in TEST_QUERIES:
        qid = q["query_id"]
        query_text = q["query"]
        relevant_set = set(gt[qid])

        # Retrieval dengan kedua metode (ambil top 10)
        tfidf_res = [doc_id for doc_id, _ in tfidf_search(query_text, index_data, top_k=10)]
        bm25_res  = [doc_id for doc_id, _ in bm25_search(query_text, index_data, top_k=10)]

        # Hitung metrik @5
        p5_tfidf = precision_at_k(tfidf_res, relevant_set, 5)
        r5_tfidf = recall_at_k(tfidf_res, relevant_set, 5)
        f1_5_tfidf = f1_at_k(p5_tfidf, r5_tfidf)

        p5_bm25 = precision_at_k(bm25_res, relevant_set, 5)
        r5_bm25 = recall_at_k(bm25_res, relevant_set, 5)
        f1_5_bm25 = f1_at_k(p5_bm25, r5_bm25)

        # Hitung metrik @10
        p10_tfidf = precision_at_k(tfidf_res, relevant_set, 10)
        r10_tfidf = recall_at_k(tfidf_res, relevant_set, 10)
        f1_10_tfidf = f1_at_k(p10_tfidf, r10_tfidf)

        p10_bm25 = precision_at_k(bm25_res, relevant_set, 10)
        r10_bm25 = recall_at_k(bm25_res, relevant_set, 10)
        f1_10_bm25 = f1_at_k(p10_bm25, r10_bm25)

        results_table.append({
            "Query ID": qid,
            "Query": query_text[:28],
            "P@5 TFIDF": p5_tfidf,
            "P@5 BM25": p5_bm25,
            "R@5 TFIDF": r5_tfidf,
            "R@5 BM25": r5_bm25,
            "F1@5 TFIDF": f1_5_tfidf,
            "F1@5 BM25": f1_5_bm25,
            "P@10 TFIDF": p10_tfidf,
            "P@10 BM25": p10_bm25,
            "R@10 TFIDF": r10_tfidf,
            "R@10 BM25": r10_bm25,
            "F1@10 TFIDF": f1_10_tfidf,
            "F1@10 BM25": f1_10_bm25,
        })

    eval_df = pd.DataFrame(results_table)

    # Format cetak
    print("\n--- HASIL EVALUASI TOP-5 ---")
    print(eval_df[["Query ID", "Query", "P@5 TFIDF", "P@5 BM25", "R@5 TFIDF", "R@5 BM25", "F1@5 TFIDF", "F1@5 BM25"]].to_string(index=False))

    print("\n--- HASIL EVALUASI TOP-10 ---")
    print(eval_df[["Query ID", "Query", "P@10 TFIDF", "P@10 BM25", "R@10 TFIDF", "R@10 BM25", "F1@10 TFIDF", "F1@10 BM25"]].to_string(index=False))

    # --- Rata-rata Keseluruhan (Mean) ---
    print("\n" + "=" * 70)
    print("RATA-RATA KESELURUHAN (MEAN)")
    print("=" * 70)
    print(f"Precision@5   : TF-IDF = {eval_df['P@5 TFIDF'].mean():.4f}  |  BM25 = {eval_df['P@5 BM25'].mean():.4f}")
    print(f"Recall@5      : TF-IDF = {eval_df['R@5 TFIDF'].mean():.4f}  |  BM25 = {eval_df['R@5 BM25'].mean():.4f}")
    print(f"F1@5          : TF-IDF = {eval_df['F1@5 TFIDF'].mean():.4f}  |  BM25 = {eval_df['F1@5 BM25'].mean():.4f}")
    print("-" * 70)
    print(f"Precision@10  : TF-IDF = {eval_df['P@10 TFIDF'].mean():.4f}  |  BM25 = {eval_df['P@10 BM25'].mean():.4f}")
    print(f"Recall@10     : TF-IDF = {eval_df['R@10 TFIDF'].mean():.4f}  |  BM25 = {eval_df['R@10 BM25'].mean():.4f}")
    print(f"F1@10         : TF-IDF = {eval_df['F1@10 TFIDF'].mean():.4f}  |  BM25 = {eval_df['F1@10 BM25'].mean():.4f}")

    # Simpan hasil perbandingan
    summary_path = os.path.join(os.path.dirname(output_csv), "evaluation_results.csv")
    eval_df.to_csv(summary_path, index=False)
    print(f"\nHasil evaluasi lengkap disimpan ke: {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path  = os.path.join(BASE_DIR, "data", "dataset_preprocessed.csv")
    index_path = os.path.join(BASE_DIR, "index", "inverted_index.json")
    queries_csv = os.path.join(BASE_DIR, "evaluation", "queries.csv")

    if not os.path.exists(index_path):
        print("ERROR: Jalankan indexing.py terlebih dahulu!")
    else:
        run_evaluation(data_path, index_path, queries_csv)

