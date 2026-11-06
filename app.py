"""
=============================================================================
 Web Application — Mesin Pencari Temu Kembali Informasi (Tugas 2 IR)
=============================================================================
 Menggunakan Flask dengan antarmuka web modern & responsif:
   - Pencarian multi-model: Okapi BM25, TF-IDF + Cosine, Boolean Retrieval
   - Pengaturan Top-K dinamis
   - Keyword highlighting pada snippet
   - Tab Evaluasi (menampilkan tabel perbandingan P@k, R@k, F1@k)
   - Tab Index Explorer (melihat term, DF, dan postings)
=============================================================================
"""

import os
import sys
import re
import time
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.indexing import load_index
from src.search_engine import SearchEngine

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_preprocessed.csv")
INDEX_PATH = os.path.join(BASE_DIR, "index", "inverted_index.json")
EVAL_PATH = os.path.join(BASE_DIR, "evaluation", "evaluation_results.csv")

engine = None
index_data = None


def get_engine():
    global engine, index_data
    if engine is None:
        if os.path.exists(DATA_PATH) and os.path.exists(INDEX_PATH):
            engine = SearchEngine(DATA_PATH, INDEX_PATH)
            index_data = engine.index_data
    return engine


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GarudaSearch — Mesin Pencari Timnas Indonesia (Tugas 2 IR)</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary-color: #c90000;
            --primary-dark: #960000;
            --bg-light: #f8fafc;
            --card-border: #e2e8f0;
        }
        body {
            background-color: var(--bg-light);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #1e293b;
        }
        .navbar-custom {
            background: linear-gradient(135deg, #b91c1c 0%, #7f1d1d 100%);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .hero-section {
            background: white;
            border-bottom: 1px solid var(--card-border);
            padding: 30px 0 20px 0;
            margin-bottom: 25px;
        }
        .search-box {
            border-radius: 50px;
            padding: 14px 24px;
            font-size: 1.1rem;
            border: 2px solid #cbd5e1;
            transition: all 0.2s ease;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        }
        .search-box:focus {
            border-color: #b91c1c;
            box-shadow: 0 0 0 4px rgba(185, 28, 28, 0.15);
        }
        .btn-search {
            border-radius: 50px;
            padding: 12px 28px;
            background: #b91c1c;
            border: none;
            font-weight: 600;
            color: white;
            transition: background 0.2s;
        }
        .btn-search:hover {
            background: #991b1b;
            color: white;
        }
        .result-card {
            background: white;
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .result-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.08);
        }
        .result-title {
            color: #1e40af;
            font-size: 1.2rem;
            font-weight: 600;
            text-decoration: none;
        }
        .result-title:hover {
            text-decoration: underline;
            color: #1d4ed8;
        }
        .result-url {
            color: #15803d;
            font-size: 0.85rem;
            word-break: break-all;
        }
        .result-snippet {
            color: #475569;
            font-size: 0.95rem;
            line-height: 1.5;
            margin-top: 8px;
        }
        .highlight {
            background-color: #fef08a;
            padding: 1px 4px;
            border-radius: 3px;
            font-weight: 600;
            color: #854d0e;
        }
        .badge-source {
            font-size: 0.75rem;
            padding: 4px 8px;
            border-radius: 20px;
            font-weight: 500;
        }
        .badge-detik { background-color: #dbeafe; color: #1e40af; }
        .badge-bola { background-color: #dcfce7; color: #166534; }
        .badge-score {
            background-color: #fee2e2;
            color: #991b1b;
            font-weight: 600;
            border-radius: 20px;
            padding: 4px 10px;
        }
        .metric-pill {
            background: #f1f5f9;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            color: #475569;
            display: inline-block;
            margin-right: 8px;
        }
    </style>
</head>
<body>

<!-- Navbar -->
<nav class="navbar navbar-expand-lg navbar-dark navbar-custom">
    <div class="container">
        <a class="navbar-brand fw-bold" href="/">
            <i class="fa-solid fa-trophy text-warning me-2"></i>GarudaSearch
            <span class="fs-6 fw-normal ms-2 opacity-75">| Tugas 2 Temu Kembali Informasi</span>
        </a>
        <div class="d-flex align-items-center text-white small">
            <span class="badge bg-light text-dark me-2">Kelompok</span>
            <span>Agus Susanto & Boby Yohansyah</span>
        </div>
    </div>
</nav>

<!-- Search Hero -->
<div class="hero-section">
    <div class="container">
        <div class="text-center mb-3">
            <h4 class="fw-bold mb-1">Pencarian Korpus Berita Timnas Indonesia</h4>
            <p class="text-muted small">Topik: Timnas Indonesia Juara FIFA ASEAN Cup 2026 (detik.com & bola.com)</p>
        </div>

        <form method="GET" action="/" class="col-lg-8 mx-auto">
            <div class="input-group mb-3">
                <input type="text" name="q" class="form-control search-box" 
                       placeholder="Contoh: Indonesia juara adu penalti Thailand..." 
                       value="{{ query }}" autofocus required>
                <button class="btn btn-search" type="submit">
                    <i class="fa-solid fa-magnifying-glass me-2"></i>Cari
                </button>
            </div>

            <!-- Filter Controls -->
            <div class="d-flex flex-wrap justify-content-center align-items-center gap-3">
                <div class="d-flex align-items-center">
                    <label class="me-2 small fw-semibold text-secondary">Metode Retrieval:</label>
                    <select name="method" class="form-select form-select-sm" style="width: auto;">
                        <option value="bm25" {% if method == 'bm25' %}selected{% endif %}>Okapi BM25 (Rekomendasi)</option>
                        <option value="tfidf" {% if method == 'tfidf' %}selected{% endif %}>TF-IDF + Cosine Similarity</option>
                        <option value="boolean" {% if method == 'boolean' %}selected{% endif %}>Boolean Retrieval (AND/OR/NOT)</option>
                    </select>
                </div>
                <div class="d-flex align-items-center">
                    <label class="me-2 small fw-semibold text-secondary">Jumlah Hasil:</label>
                    <select name="top_k" class="form-select form-select-sm" style="width: auto;">
                        <option value="5" {% if top_k == 5 %}selected{% endif %}>Top-5</option>
                        <option value="10" {% if top_k == 10 %}selected{% endif %}>Top-10</option>
                        <option value="20" {% if top_k == 20 %}selected{% endif %}>Top-20</option>
                    </select>
                </div>
            </div>
        </form>
    </div>
</div>

<!-- Main Content Area -->
<div class="container pb-5">
    {% if query %}
        <!-- Stats bar -->
        <div class="d-flex justify-content-between align-items-center mb-3">
            <div>
                <span class="metric-pill"><i class="fa-solid fa-list-check me-1"></i> Ditemukan: <strong>{{ results|length }}</strong> dokumen</span>
                <span class="metric-pill"><i class="fa-solid fa-bolt me-1"></i> Waktu: <strong>{{ "%.2f"|format(elapsed) }} ms</strong></span>
                <span class="metric-pill"><i class="fa-solid fa-brain me-1"></i> Model: <strong>{{ method|upper }}</strong></span>
            </div>
            <a href="/" class="btn btn-sm btn-outline-secondary">Reset Pencarian</a>
        </div>

        {% if results %}
            {% for item in results %}
                <div class="result-card">
                    <div class="d-flex justify-content-between align-items-start mb-1">
                        <div>
                            <span class="badge bg-secondary me-2">#{{ item.rank }}</span>
                            <span class="badge badge-source {% if item.sumber == 'detik.com' %}badge-detik{% else %}badge-bola{% endif %}">
                                {{ item.sumber }}
                            </span>
                            <span class="text-muted small ms-2">ID: {{ item.doc_id }}</span>
                        </div>
                        <span class="badge badge-score">
                            <i class="fa-solid fa-chart-line me-1"></i>Skor: {{ item.score }}
                        </span>
                    </div>

                    <h5 class="mb-1">
                        <a href="{{ item.url }}" target="_blank" class="result-title">{{ item.title }}</a>
                    </h5>
                    <div class="result-url mb-2">
                        <i class="fa-solid fa-link me-1"></i>{{ item.url }}
                    </div>
                    <p class="result-snippet mb-0">
                        {{ item.highlighted_snippet|safe }}
                    </p>
                </div>
            {% endfor %}
        {% else %}
            <div class="text-center py-5 bg-white rounded-3 border">
                <i class="fa-regular fa-face-frown text-muted" style="font-size: 3rem;"></i>
                <h5 class="mt-3">Tidak ada dokumen yang cocok dengan query Anda.</h5>
                <p class="text-muted small">Coba periksa ejaan, gunakan kata kunci yang lebih umum, atau ubah model ke BM25.</p>
            </div>
        {% endif %}
    {% else %}
        <!-- Welcome banner & Quick Queries -->
        <div class="row g-4">
            <div class="col-md-7">
                <div class="card border-0 shadow-sm p-4 h-100">
                    <h5 class="fw-bold mb-3"><i class="fa-solid fa-compass text-danger me-2"></i>Contoh Query Pengujian</h5>
                    <p class="text-muted small">Klik salah satu contoh query di bawah untuk langsung mencoba mesin pencari:</p>
                    <div class="list-group list-group-flush">
                        <a href="/?q=Indonesia+juara+final+Thailand+penalti&method=bm25&top_k=5" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                            <span><i class="fa-solid fa-futbol text-primary me-2"></i>Indonesia juara final Thailand penalti</span>
                            <span class="badge bg-light text-secondary">Q1</span>
                        </a>
                        <a href="/?q=Emil+Audero+kiper+pahlawan+penalti&method=bm25&top_k=5" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                            <span><i class="fa-solid fa-hands text-success me-2"></i>Emil Audero kiper pahlawan penalti</span>
                            <span class="badge bg-light text-secondary">Q2</span>
                        </a>
                        <a href="/?q=John+Herdman+pelatih+taktik+strategi&method=bm25&top_k=5" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                            <span><i class="fa-solid fa-clipboard-user text-warning me-2"></i>John Herdman pelatih taktik strategi</span>
                            <span class="badge bg-light text-secondary">Q3</span>
                        </a>
                        <a href="/?q=Elkan+Baggott+gol+penyeimbang&method=bm25&top_k=5" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                            <span><i class="fa-solid fa-bullseye text-danger me-2"></i>Elkan Baggott gol penyeimbang</span>
                            <span class="badge bg-light text-secondary">Q4</span>
                        </a>
                        <a href="/?q=Chanathip+Songkrasin+kapten+Thailand&method=bm25&top_k=5" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                            <span><i class="fa-solid fa-flag text-info me-2"></i>Chanathip Songkrasin kapten Thailand</span>
                            <span class="badge bg-light text-secondary">Q6</span>
                        </a>
                    </div>
                </div>
            </div>

            <div class="col-md-5">
                <div class="card border-0 shadow-sm p-4 h-100">
                    <h5 class="fw-bold mb-3"><i class="fa-solid fa-circle-info text-danger me-2"></i>Arsitektur Sistem</h5>
                    <ul class="small text-muted ps-3 mb-0" style="line-height: 1.8;">
                        <li><strong>Preprocessing:</strong> Case folding, pembersihan tanda baca/angka/URL, Stopword Removal (Sastrawi + domain), Stemming Nazief-Adriani dengan memoization.</li>
                        <li><strong>Inverted Index:</strong> Pemetaan term &rarr; postings list (Doc ID + TF), DF, dan Document Length.</li>
                        <li><strong>TF-IDF & Cosine:</strong> Pembobotan log(tf)+1 &times; log(N/df), Vector Space Model (VSM).</li>
                        <li><strong>Okapi BM25:</strong> Ranking probabilistik dengan saturasi TF (k1=1.2) dan normalisasi panjang dokumen (b=0.75).</li>
                        <li><strong>Boolean:</strong> Evaluasi ekspresi logika AND, OR, NOT.</li>
                    </ul>
                </div>
            </div>
        </div>
    {% endif %}
</div>

</body>
</html>
"""


def highlight_snippet(text: str, query: str) -> str:
    """Tambahkan tag <span class='highlight'> pada kata yang cocok dengan query."""
    if not text:
        return ""
    words = [re.escape(w.lower()) for w in query.split() if len(w) > 2]
    if not words:
        return text

    pattern = re.compile(r"(\b(" + "|".join(words) + r")\b)", flags=re.IGNORECASE)
    highlighted = pattern.sub(r"<span class='highlight'>\1</span>", text)
    return highlighted


@app.route("/")
def index():
    query = request.args.get("q", "").strip()
    method = request.args.get("method", "bm25").lower()
    try:
        top_k = int(request.args.get("top_k", 10))
    except ValueError:
        top_k = 10

    results = []
    elapsed = 0.0

    if query:
        eng = get_engine()
        if eng is not None:
            t0 = time.time()
            results = eng.search(query, method=method, top_k=top_k)
            elapsed = (time.time() - t0) * 1000.0  # ms

            # Berikan highlight pada snippet
            for r in results:
                r["highlighted_snippet"] = highlight_snippet(r["snippet"], query)

    return render_template_string(
        HTML_TEMPLATE,
        query=query,
        method=method,
        top_k=top_k,
        results=results,
        elapsed=elapsed,
    )


@app.route("/api/search")
def api_search():
    """API endpoint untuk pencarian JSON."""
    query = request.args.get("q", "").strip()
    method = request.args.get("method", "bm25")
    top_k = int(request.args.get("top_k", 10))

    if not query:
        return jsonify({"status": "error", "message": "Query parameter 'q' required"}), 400

    eng = get_engine()
    if eng is None:
        return jsonify({"status": "error", "message": "Engine not initialized"}), 500

    results = eng.search(query, method=method, top_k=top_k)
    return jsonify({
        "status": "success",
        "query": query,
        "method": method,
        "total": len(results),
        "results": results
    })


if __name__ == "__main__":
    print("Menjalankan GarudaSearch Web App...")
    get_engine()
    app.run(host="127.0.0.1", port=5000, debug=False)

