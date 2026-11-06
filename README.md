# 🏆 Tugas 2 — Retrieval Data & Pembuatan Mesin Pencari Sederhana

**Mata Kuliah:** Temu Kembali Informasi (Information Retrieval)  
**Kelompok:**
- **Agus Susanto** (NIM: 202310370311373)
- **Boby Yohansyah** (NIM: 202310370311374)

**Topik Dataset:** Berita Timnas Indonesia Juara FIFA ASEAN Cup 2026 (Sumber: `detik.com` & `bola.com`)  
**Tugas Lanjutan dari:** Tugas 1 — Crawling Data

---

## 📖 Ringkasan Proyek

Proyek ini membangun sistem **Temu Kembali Informasi Klasik (Classical Information Retrieval)** end-to-end berbasis Python, mulai dari prapemrosesan teks, indexing, retrieval model (Boolean, TF-IDF + Cosine Similarity, dan Okapi BM25), antarmuka mesin pencari web interaktif (Flask), hingga evaluasi retrieval komparatif (Precision@k, Recall@k, F1@k).

### Alur Sistem

```
Koleksi Dokumen (CSV)
       │
       ▼
[1] Prapemrosesan Teks (Case folding, Cleaning, Tokenisasi, Stopword Removal, Stemming Nazief-Adriani)
       │
       ▼
[2] Pembangunan Inverted Index (term ➔ postings: doc_id + tf, DF, doc length, avgdl)
       │
       ▼
[3] Query Processing (Metode identik dengan prapemrosesan dokumen)
       │
       ▼
[4] Retrieval & Scoring Model:
    ├── (A) Boolean Retrieval (AND, OR, NOT)
    ├── (B) TF-IDF & Vector Space Model (Cosine Similarity)
    └── (C) Okapi BM25 (k1=1.2, b=0.75, term saturation & length normalization)
       │
       ▼
[5] Ranking & Top-K Retrieval
       │
       ▼
[6] Antarmuka Mesin Pencari (Web UI Modern & CLI Interaktif)
       │
       ▼
[7] Evaluasi Performa (10 Query Uji, Ground Truth, P@k, R@k, F1@k)
```

---

## 📂 Struktur Direktori

```
Tugas 2/
├── data/
│   ├── dataset_crawling_ir.csv       # Dataset mentah hasil Tugas 1 (1.182 dokumen)
│   └── dataset_preprocessed.csv      # Dataset hasil prapemrosesan
├── index/
│   └── inverted_index.json           # Inverted index dalam format JSON
├── src/
│   ├── preprocessing.py              # Pipeline prapemrosesan teks & penanganan duplikasi
│   ├── indexing.py                   # Pembangunan dan serialisasi Inverted Index
│   ├── boolean_retrieval.py          # Mesin pencari Boolean (AND / OR / NOT)
│   ├── tfidf_retrieval.py            # Pembobotan TF-IDF & Cosine Similarity
│   ├── bm25.py                       # Implementasi Okapi BM25 murni
│   ├── search_engine.py              # Controller mesin pencari & CLI interaktif
│   └── evaluation.py                 # Modul evaluasi komparatif (P@k, R@k, F1@k)
├── evaluation/
│   ├── queries.csv                   # 10 query pengujian & ground truth
│   └── evaluation_results.csv        # Tabel hasil metrik per query & perbandingan
├── app.py                            # Antarmuka Web Mesin Pencari (Flask)
├── requirements.txt                  # Daftar pustaka & dependensi Python
├── Tugas2_Retrieval_Data_Mesin_Pencari.ipynb  # Notebook Jupyter end-to-end
└── README.md                         # Panduan instalasi dan penggunaan
```

---

## ⚙️ Instalasi dan Prasyarat

Pastikan telah menginstal **Python 3.8+**. Instal seluruh dependensi yang dibutuhkan secara langsung melalui `requirements.txt`:

```bash
pip install -r requirements.txt
```

Atau instal paket-paket utama secara manual:
```bash
pip install Flask pandas Sastrawi matplotlib python-docx Pillow
```

---

## 🚀 Panduan Menjalankan Program

### 1. Prapemrosesan Dataset
Menjalankan pembersihan teks, stopword removal, dan stemming Bahasa Indonesia menggunakan Sastrawi dengan *vocabulary caching/memoization* untuk efisiensi tinggi:

```bash
python src/preprocessing.py
```
*Output: `data/dataset_preprocessed.csv`*

### 2. Membangun Inverted Index
Membangun struktur data pemetaan term ke postings list beserta frekuensi term (TF), document frequency (DF), dan panjang dokumen:

```bash
python src/indexing.py
```
*Output: `index/inverted_index.json`*

### 3. Menguji Model Retrieval Mandiri
- **Boolean Retrieval:**
  ```bash
  python src/boolean_retrieval.py
  ```
- **TF-IDF + Cosine Similarity:**
  ```bash
  python src/tfidf_retrieval.py
  ```
- **Okapi BM25 Ranking:**
  ```bash
  python src/bm25.py
  ```

### 4. Menjalankan Evaluasi Model (10 Query Uji)
Menghitung Precision@5, Precision@10, Recall@5, Recall@10, F1@5, dan F1@10 secara komparatif:

```bash
python src/evaluation.py
```
*Output: `evaluation/queries.csv` dan `evaluation/evaluation_results.csv`*

### 5. Menjalankan Mesin Pencari Berbasis CLI (Terminal)
Interaktif langsung di terminal dengan fitur penggantian metode (`:method bm25`/`tfidf`/`boolean`) dan jumlah hasil (`:top 5`/`10`):

```bash
python src/search_engine.py
```

### 6. Menjalankan Antarmuka Web Mesin Pencari (Flask)
Jalankan server web lokal:

```bash
python app.py
```
Buka browser dan akses: **`http://127.0.0.1:5000`**

Fitur Web:
- Input query pencarian bebas.
- Pilihan metode retrieval dinamis: **Okapi BM25**, **TF-IDF + Cosine**, atau **Boolean**.
- Pilihan jumlah dokumen: **Top-5**, **Top-10**, **Top-20**.
- Tampilan kartu hasil informatif: peringkat, judul (link sumber), skor relevansi, badge sumber (detik.com/bola.com), dan cuplikan teks (*snippet*) dengan **keyword highlighting**.
- Tombol uji cepat untuk 10 query pengujian.

---

## 📐 Dasar Teori & Formula Matematis

### 1. TF-IDF (Term Frequency - Inverse Document Frequency)
- **Term Frequency Logaritmik:**
  $$\text{TF}(t, d) = \begin{cases} 1 + \log_{10}(\text{count}(t, d)), & \text{jika } \text{count}(t, d) > 0 \\ 0, & \text{lainnya} \end{cases}$$
- **Inverse Document Frequency:**
  $$\text{IDF}(t) = \log_{10}\left(\frac{N}{\text{DF}(t)}\right)$$
- **Bobot Term:**
  $$w(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

### 2. Vector Space Model & Cosine Similarity
$$\text{Cosine Sim}(q, d) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|} = \frac{\sum_{t \in q \cap d} w(t, q) \cdot w(t, d)}{\sqrt{\sum_{t \in q} w(t, q)^2} \sqrt{\sum_{t \in d} w(t, d)^2}}$$

### 3. Okapi BM25
$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}_{\text{BM25}}(t) \cdot \frac{\text{TF}(t, d) \cdot (k_1 + 1)}{\text{TF}(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{\text{len}(d)}{\text{avgdl}}\right)}$$
Dengan parameter default:
- $k_1 = 1.2$ (mengontrol laju saturasi TF)
- $b = 0.75$ (mengontrol penalti panjang dokumen)
- $\text{IDF}_{\text{BM25}}(t) = \ln\left(\frac{N - \text{DF}(t) + 0.5}{\text{DF}(t) + 0.5} + 1\right)$

---

## 🔍 Analisis Perbandingan: TF-IDF vs BM25

1. **TF Saturation (Saturasi Frekuensi Term):**
   - Pada TF-IDF logaritmik, peningkatan skor terhadap kemunculan term berulang masih bertambah secara perlahan tanpa batas atas tegas.
   - Pada BM25, skor memiliki asimtot $k_1 + 1$. Pengulangan term yang sama berkali-kali pada dokumen spam/panjang tidak akan melipatgandakan skor secara tidak wajar.
2. **Panjang Dokumen (Document Length Normalization):**
   - VSM Cosine menormalisasi vektor berdasarkan panjang euclidean vektor bobot, yang cenderung memberikan penalti berlebih pada dokumen panjang yang membahas beberapa sub-topik.
   - BM25 menormalisasi secara fleksibel melalui parameter $b \times \frac{\text{len}(d)}{\text{avgdl}}$, membedakan apakah dokumen panjang karena banyak informasi atau karena repetisi kata.
3. **Hasil Evaluasi:**
   - BM25 secara konsisten menghasilkan **Precision@5** dan **Recall@5** yang lebih stabil pada query natural/informational, terutama pada artikel berita yang bervariasi panjangnya.

