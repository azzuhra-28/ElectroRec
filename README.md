# ⚡ ElectroRec

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Sistem Rekomendasi Produk Elektronik berbasis **Content-Based Filtering** dan **Hybrid Scoring**, menggunakan dataset Amazon Marketplace.

🔗 **Live Demo:** [electrorec-mqnd9hsgkznu95s8wumfn3.streamlit.app](https://electrorec-mqnd9hsgkznu95s8wumfn3.streamlit.app/)

---

## 📌 Deskripsi Project

Project ini dibuat untuk tugas Mata Kuliah **Sistem Rekomendasi**, Semester Genap TA. 2025/2026.

- **Topik:** Marketplace
- **Judul:** Sistem Rekomendasi Produk Elektronik Menggunakan Content-Based Filtering pada Amazon Marketplace

---

## 🗃️ Dataset

[Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset) dari Kaggle, difilter pada kategori **produk elektronik** (480 produk, 10 sub-kategori).

Variabel yang digunakan:
- `product_name` — nama produk
- `category` — kategori produk
- `about_product` — deskripsi produk
- `rating` — rating produk
- `discounted_price` — harga produk
- `img_link` — gambar produk

---

## ⚙️ Metodologi

1. **Pengumpulan & Seleksi Data** — filter kategori elektronik dari dataset
2. **Preprocessing Data** — bersihkan data, handle missing value, gabungkan fitur teks
3. **Ekstraksi Fitur (TF-IDF)** — ubah teks produk menjadi representasi numerik
4. **Cosine Similarity** — hitung kemiripan antar produk
5. **Content-Based Recommendation System** — bangun fungsi rekomendasi Top-N
6. **Hybrid Scoring** — gabungkan similarity score dengan rating produk
7. **Implementasi Streamlit** — aplikasi web interaktif
8. **Evaluasi & Interpretasi Hasil** — Precision@K berbasis kategori & analisis kualitatif

---

## ✨ Fitur Aplikasi

- 🗂️ **Cari berdasarkan Kategori** — pilih kategori & produk, dapatkan rekomendasi produk serupa
- 💬 **Cari berdasarkan Kata Kunci** — cari produk dengan kata kunci bebas
- ⭐ Filter rating minimum & kemiripan minimum
- 📊 Hybrid score (kombinasi similarity + rating)
- ☁️ Word cloud deskripsi produk
- 📈 Grafik perbandingan rating hasil rekomendasi
- 💰 Harga ditampilkan dalam Rupiah (estimasi)

---

## 🛠️ Cara Menjalankan Lokal

```bash
git clone https://github.com/azzuhra-28/ElectroRec.git
cd ElectroRec
pip install -r requirements.txt
streamlit run app.py
```

---

## 📚 Teknologi yang Digunakan

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn (TF-IDF, Cosine Similarity)
- WordCloud, Matplotlib

---

## 📄 Referensi

- Amazon Sales Dataset — Kaggle
- Dokumentasi Scikit-learn (TfidfVectorizer, cosine_similarity)
- Dokumentasi Streamlit
