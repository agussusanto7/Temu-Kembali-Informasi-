"""
=============================================================================
 Modul Prapemrosesan (Preprocessing) — Tugas 2 IR
=============================================================================
 Pipeline:
   1. Penanganan nilai kosong & duplikat
   2. Case folding (lowercase)
   3. Normalisasi teks (hapus URL, angka, tanda baca, Gambas tag, dsb.)
   4. Tokenisasi (split berdasarkan spasi)
   5. Stopword removal (Sastrawi + custom domain berita)
   6. Stemming (Sastrawi — Nazief-Adriani) dengan memoization / vocabulary cache
=============================================================================
"""

import re
import os
import time
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ---------------------------------------------------------------------------
# Inisialisasi Sastrawi
# ---------------------------------------------------------------------------
_stemmer_factory = StemmerFactory()
STEMMER = _stemmer_factory.create_stemmer()

_sw_factory = StopWordRemoverFactory()
STOPWORDS_SASTRAWI = set(_sw_factory.get_stop_words())

# Stopword tambahan khusus domain berita online & noise crawler
CUSTOM_STOPWORDS = {
    "bola", "com", "jakarta", "detik", "foto", "video",
    "baca", "juga", "selengkapnya", "lihat", "artikel",
    "sumber", "halaman", "klik", "undefined", "advertisement",
    "https", "http", "www", "html", "quot", "amp", "simak",
    "wib", "senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu"
}

STOPWORDS = STOPWORDS_SASTRAWI | CUSTOM_STOPWORDS

# Cache kamus stemming untuk kecepatan tinggi (memoization)
STEM_CACHE = {}


# ---------------------------------------------------------------------------
# Fungsi-fungsi preprocessing
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalisasi teks mentah:
    - Lowercase (case folding)
    - Hapus URL
    - Hapus mention (@...) dan hashtag (#...)
    - Hapus tag [Gambas:Video ...]
    - Hapus angka
    - Hapus karakter non-alfabet
    - Rapikan spasi
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()                                         # case folding
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)          # hapus URL
    text = re.sub(r"[@#]\w+", " ", text)                        # mention/hashtag
    text = re.sub(r"\[gambas:[^\]]*\]", " ", text, flags=re.IGNORECASE)  # Gambas tag
    text = re.sub(r"\d+", " ", text)                            # hapus angka
    text = re.sub(r"[^a-z\s]", " ", text)                       # hanya huruf & spasi
    text = re.sub(r"\s+", " ", text).strip()                    # rapikan spasi
    return text


def tokenize(text: str) -> list:
    """Tokenisasi sederhana: split berdasarkan spasi."""
    return text.split()


def remove_stopwords(tokens: list) -> list:
    """Hapus stopwords (Sastrawi + custom)."""
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def stem_token(token: str) -> str:
    """Stem satu token dengan memoization / caching."""
    if token not in STEM_CACHE:
        STEM_CACHE[token] = STEMMER.stem(token)
    return STEM_CACHE[token]


def stem_tokens(tokens: list) -> list:
    """Stemming setiap token menggunakan Sastrawi (Nazief-Adriani) ter-cache."""
    return [stem_token(t) for t in tokens]


def preprocess_text(text: str) -> str:
    """
    Pipeline lengkap: clean → tokenize → stopword removal → stemming.
    Mengembalikan string token yang sudah diproses (dipisah spasi).
    """
    cleaned  = clean_text(text)
    tokens   = tokenize(cleaned)
    filtered = remove_stopwords(tokens)
    stemmed  = stem_tokens(filtered)
    # Buang token kosong atau karakter tunggal
    result   = [t for t in stemmed if t and len(t) > 1]
    return " ".join(result)


# ---------------------------------------------------------------------------
# Fungsi utama: proses seluruh dataset
# ---------------------------------------------------------------------------

def preprocess_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Membaca CSV hasil Tugas 1, melakukan prapemrosesan, dan menyimpan hasilnya.

    Kolom yang ditambahkan:
        - content_preprocessed : isi dokumen setelah preprocessing
        - title_preprocessed   : judul setelah preprocessing
    """
    print("=" * 60)
    print("PRAPEMROSESAN DATASET")
    print("=" * 60)

    # --- Baca dataset ---
    df = pd.read_csv(input_path)
    print(f"Dokumen awal          : {len(df)}")

    # --- Penanganan nilai kosong ---
    null_before = df.isnull().sum().to_dict()
    df = df.dropna(subset=["content", "title"])
    df["content"] = df["content"].fillna("")
    df["title"]   = df["title"].fillna("")
    print(f"Pengecekan null       : {null_before}")

    # --- Penanganan duplikat ---
    dup_url = df.duplicated(subset=["url"]).sum()
    df = df.drop_duplicates(subset=["url"])
    print(f"Duplikat URL dibuang  : {dup_url}")

    dup_title = df.duplicated(subset=["title"]).sum()
    df = df.drop_duplicates(subset=["title"])
    print(f"Duplikat judul dibuang: {dup_title}")

    # --- Reset index & doc_id teratur ---
    df = df.reset_index(drop=True)
    df["doc_id"] = [f"D{i+1}" for i in range(len(df))]

    print(f"Dokumen siap diproses : {len(df)}")
    print("Memulai preprocessing teks dengan caching kata unik...")

    t0 = time.time()
    
    # Preprocessing konten dan judul
    contents_prep = []
    titles_prep = []
    total = len(df)
    
    for i, row in enumerate(df.itertuples(), 1):
        titles_prep.append(preprocess_text(row.title))
        contents_prep.append(preprocess_text(row.content))
        if i % 100 == 0 or i == total:
            elapsed = time.time() - t0
            print(f"  Progres: {i}/{total} dokumen ({i/total*100:.1f}%) - Waktu: {elapsed:.1f}s - Kata unik di-cache: {len(STEM_CACHE)}")

    df["title_preprocessed"] = titles_prep
    df["content_preprocessed"] = contents_prep

    print(f"Preprocessing selesai dalam {time.time() - t0:.2f} detik! Total kata unik di-stem: {len(STEM_CACHE)}")

    # --- Contoh sebelum & sesudah ---
    print("\n" + "=" * 60)
    print("CONTOH SEBELUM DAN SESUDAH PREPROCESSING")
    print("=" * 60)
    sample = df.iloc[0]
    print(f"\n[{sample['doc_id']}] {sample['title']}")
    print(f"\nSEBELUM (150 karakter pertama):")
    print(f"  {sample['content'][:150]}...")
    print(f"\nSESUDAH (150 karakter pertama):")
    print(f"  {sample['content_preprocessed'][:150]}...")

    # --- Simpan ---
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nDataset preprocessed disimpan ke: {output_path}")
    print(f"Bentuk akhir: {df.shape}")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv  = os.path.join(BASE_DIR, "data", "dataset_crawling_ir.csv")
    output_csv = os.path.join(BASE_DIR, "data", "dataset_preprocessed.csv")
    preprocess_dataset(input_csv, output_csv)
