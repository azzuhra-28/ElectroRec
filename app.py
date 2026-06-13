# ============================================================
# FASE 4 - APLIKASI WEB STREAMLIT
# Sistem Rekomendasi Produk Elektronik
# Nama  : Intan Azzuhra Permadani
# NRP   : 3324600028
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="ElectroRec | Rekomendasi Produk Elektronik",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — Light Mode friendly
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        color: #111111;
        line-height: 1.1;
    }
    .hero-accent { color: #d4a000; }
    .hero-sub {
        font-size: 1rem;
        color: #555;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    /* ===== Dashboard Landing ===== */
    .dash-greeting {
        background: linear-gradient(135deg, #f0c040 0%, #ffd76a 100%);
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        color: #111111;
        margin-bottom: 1rem;
    }
    .dash-greeting h2 {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        color: #111111 !important;
        margin: 0;
        font-size: 1.6rem;
    }
    .dash-greeting p {
        color: #5a4a00;
        margin: 0.3rem 0 0 0;
        font-size: 0.85rem;
    }
    .dash-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 1rem;
        height: 100%;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .dash-card-accent {
        background: #fffbea;
        border: 1px solid #f0c04055;
    }
    .dash-stat-label {
        font-size: 0.75rem;
        color: #999;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .dash-stat-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.9rem;
        font-weight: 800;
        color: #111111;
    }
    .dash-stat-sub {
        font-size: 0.75rem;
        color: #aaa;
        margin-top: 0.2rem;
    }
    .dash-bar-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.6rem;
    }
    .dash-bar-label {
        font-size: 0.78rem;
        color: #555;
        width: 150px;
        flex-shrink: 0;
    }
    .dash-bar-track {
        flex: 1;
        background: #f5f5f5;
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
    }
    .dash-bar-fill {
        background: #f0c040;
        height: 100%;
        border-radius: 6px;
    }
    .dash-bar-count {
        font-size: 0.75rem;
        color: #999;
        width: 32px;
        text-align: right;
        flex-shrink: 0;
    }
    .dash-product-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.55rem 0;
        border-bottom: 1px solid #f0f0f0;
        font-size: 0.8rem;
    }
    .dash-product-row:last-child { border-bottom: none; }
    .dash-product-name {
        color: #333;
        max-width: 70%;
    }
    .dash-product-meta {
        color: #d4a000;
        font-weight: 700;
        white-space: nowrap;
    }
    .card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .card:hover {
        border-color: #d4a000;
        box-shadow: 0 2px 10px rgba(212,160,0,0.15);
    }
    .card-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        color: #111111;
        margin-bottom: 0.3rem;
        line-height: 1.3;
    }
    .card-category {
        font-size: 0.72rem;
        color: #d4a000;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .card-meta { font-size: 0.78rem; color: #555; }
    .badge-sim {
        display: inline-block;
        background: #fff8e1;
        color: #b8860b;
        border: 1px solid #f0c04088;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .badge-hybrid {
        display: inline-block;
        background: #e3f6fd;
        color: #0077aa;
        border: 1px solid #40c0f088;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .selected-product {
        background: #fffbea;
        border: 2px solid #d4a000;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }
    .selected-label {
        font-size: 0.72rem;
        color: #d4a000;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }
    .selected-name {
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #111111;
        margin-top: 0.3rem;
    }
    .divider {
        border: none;
        border-top: 1px solid #e0e0e0;
        margin: 1.5rem 0;
    }
    .metric-box {
        background: #fffbea;
        border: 1px solid #f0c040;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .metric-val {
        font-family: 'Syne', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        color: #d4a000;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #777;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #111111;
        margin: 1.5rem 0 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA & MODEL
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv('amazon_elektronik_clean.csv')
    return df

@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    scaler = MinMaxScaler()
    df['rating_normalized'] = scaler.fit_transform(df[['rating']])
    indices = pd.Series(df.index, index=df['product_name']).drop_duplicates()
    return tfidf, tfidf_matrix, cosine_sim, indices

df = load_data()
tfidf, tfidf_matrix, cosine_sim, indices = build_model(df)

# ============================================================
# KONFIGURASI TAMBAHAN
# ============================================================
# Kurs estimasi INR -> IDR (ilustratif, untuk keperluan demo/poster)
KURS_INR_TO_IDR = 190

def format_rupiah(harga_inr):
    """Konversi harga dari INR ke estimasi Rupiah dan format dengan Rp."""
    harga_idr = harga_inr * KURS_INR_TO_IDR
    return f"Rp{harga_idr:,.0f}".replace(",", ".")

# Mapping kategori ke nama yang lebih familiar (Bahasa Indonesia)
KATEGORI_MAP = {
    'HomeTheater,TV&Video'              : 'TV & Home Theater',
    'Headphones,Earbuds&Accessories'    : 'Headphone & Earphone',
    'HomeAudio'                         : 'Audio Rumah',
    'Mobiles&Accessories'               : 'HP & Aksesoris',
    'Accessories'                       : 'Aksesoris',
    'Cameras'                           : 'Kamera',
    'Cameras&Photography'               : 'Kamera & Fotografi',
    'WearableTechnology'                : 'Perangkat Wearable',
    'Laptops'                           : 'Laptop',
    'Computers&Accessories'             : 'Komputer & Aksesoris',
    'NetworkingDevices'                 : 'Perangkat Jaringan',
    'ExternalDevices&DataStorage'       : 'Penyimpanan Data',
    'PowerAccessories'                  : 'Aksesoris Daya',
    'HomeImprovement'                   : 'Perlengkapan Rumah',
    'Kitchen&HomeAppliances'            : 'Peralatan Dapur',
    'GeneralPurposeBatteries&BatteryChargers' : 'Baterai & Charger',
    'OfficeElectronics'                 : 'Elektronik Kantor',
}

def kategori_familiar(kategori):
    """Ubah nama kategori jadi lebih familiar (Bahasa Indonesia)."""
    if pd.isnull(kategori):
        return kategori
    return KATEGORI_MAP.get(kategori, kategori)

# Daftar kata yang dibuang dari word cloud (kode model, satuan, dll yang
# membuat tampilan kurang rapi)
STOPWORDS_WC = set([
    'cm', 'inches', 'inch', 'hz', 'mp', 'gb', 'tb', 'mm', 'ft', 'feet',
    'pack', 'pcs', 'set'
])

def bersihkan_teks_wordcloud(teks):
    """
    Bersihkan teks sebelum dibuat word cloud:
    - Hapus kode model produk (kombinasi huruf+angka, misal AR55AR2851UDPRO)
    - Hapus angka tunggal & satuan ukuran yang tidak informatif
    """
    if pd.isnull(teks):
        return ''
    kata_kata = str(teks).split()
    hasil = []
    for kata in kata_kata:
        kata_bersih = kata.strip('.,()|')
        lower = kata_bersih.lower()

        # Lewati kata yang kosong
        if not kata_bersih:
            continue

        # Lewati stopword satuan
        if lower in STOPWORDS_WC:
            continue

        # Lewati kode model: kombinasi huruf besar & angka, panjang >= 5
        if (len(kata_bersih) >= 5
                and any(c.isdigit() for c in kata_bersih)
                and any(c.isalpha() for c in kata_bersih)
                and kata_bersih == kata_bersih.upper()):
            continue

        # Lewati angka murni
        if kata_bersih.replace('.', '').isdigit():
            continue

        hasil.append(kata_bersih)
    return ' '.join(hasil)


# ============================================================
# LANDING PAGE
# ============================================================
if 'masuk_app' not in st.session_state:
    st.session_state.masuk_app = False

if not st.session_state.masuk_app:
    # --- Greeting Header ---
    st.markdown("""
    <div class="dash-greeting">
        <h2>⚡ Halo, Selamat Datang di ElectroRec!</h2>
        <p>Sistem Rekomendasi Produk Elektronik berbasis Content-Based Filtering &amp; Hybrid Scoring, menggunakan dataset Amazon Marketplace.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Stat Cards ---
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"""
        <div class="dash-card">
            <div class="dash-stat-label">Total Produk</div>
            <div class="dash-stat-value">{len(df)}</div>
            <div class="dash-stat-sub">Produk elektronik</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="dash-card">
            <div class="dash-stat-label">Kategori</div>
            <div class="dash-stat-value">{df['category_clean'].nunique()}</div>
            <div class="dash-stat-sub">Jenis kategori</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="dash-card">
            <div class="dash-stat-label">Rating Rata-rata</div>
            <div class="dash-stat-value">{df['rating'].mean():.1f} ⭐</div>
            <div class="dash-stat-sub">dari skala 5.0</div>
        </div>
        """, unsafe_allow_html=True)
    with s4:
        harga_rata = format_rupiah(df['discounted_price'].mean())
        st.markdown(f"""
        <div class="dash-card">
            <div class="dash-stat-label">Harga Rata-rata</div>
            <div class="dash-stat-value" style="font-size:1.3rem;">{harga_rata}</div>
            <div class="dash-stat-sub">per produk</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Category Distribution & Top Products ---
    c1, c2 = st.columns([1.3, 1])
    with c1:
        cat_counts = df['category_clean'].value_counts()
        max_count = cat_counts.max()
        bars_html = '<div class="dash-card"><div class="card-title" style="margin-bottom:0.8rem;">📦 Distribusi Kategori Produk</div>'
        for kat, jumlah in cat_counts.items():
            pct = int((jumlah / max_count) * 100)
            bars_html += (
                '<div class="dash-bar-row">'
                f'<div class="dash-bar-label">{kategori_familiar(kat)}</div>'
                f'<div class="dash-bar-track"><div class="dash-bar-fill" style="width:{pct}%;"></div></div>'
                f'<div class="dash-bar-count">{jumlah}</div>'
                '</div>'
            )
        bars_html += '</div>'
        st.markdown(bars_html, unsafe_allow_html=True)

    with c2:
        top_produk = df.sort_values('rating', ascending=False).head(5)
        rows_html = '<div class="dash-card dash-card-accent"><div class="card-title" style="margin-bottom:0.8rem;">⭐ Produk Rating Tertinggi</div>'
        for _, row in top_produk.iterrows():
            nama = row['product_name']
            nama_singkat = nama if len(nama) <= 38 else nama[:38] + '...'
            rows_html += (
                '<div class="dash-product-row">'
                f'<div class="dash-product-name">{nama_singkat}</div>'
                f'<div class="dash-product-meta">⭐ {row["rating"]}</div>'
                '</div>'
            )
        rows_html += '</div>'
        st.markdown(rows_html, unsafe_allow_html=True)

    # --- Feature Cards ---
    st.markdown('<div class="section-title">✨ Fitur Pencarian</div>', unsafe_allow_html=True)
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        st.markdown("""
        <div class="card">
            <div class="card-title">🗂️ Cari berdasarkan Kategori</div>
            <div class="card-meta">
                Pilih kategori produk, lalu pilih satu produk untuk melihat
                rekomendasi produk lain yang paling mirip secara konten.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with fcol2:
        st.markdown("""
        <div class="card">
            <div class="card-title">💬 Cari berdasarkan Kata Kunci</div>
            <div class="card-meta">
                Ketik kata kunci bebas (misal "wireless headphone" atau
                "4K smart TV") untuk menemukan produk yang relevan.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- CTA Button ---
    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_b:
        if st.button("🚀 Mulai Sekarang", use_container_width=True):
            st.session_state.masuk_app = True
            st.rerun()

    st.stop()

# ============================================================
# FUNGSI REKOMENDASI
# ============================================================
def rekomendasikan_produk(nama_produk, top_n=5, min_rating=4.0,
                           bobot_sim=0.7, bobot_rating=0.3, min_similarity=0.0):
    if nama_produk not in indices:
        return None
    idx = indices[nama_produk]
    sim_scores = [(i, cosine_sim[idx][i]) for i in range(len(df)) if i != idx]
    hasil = []
    for i, sim in sim_scores:
        # Lewati produk yang similarity-nya di bawah ambang batas
        if sim <= min_similarity:
            continue
        rating_asli = df['rating'].iloc[i]
        if rating_asli < min_rating:
            continue
        rating_norm = df['rating_normalized'].iloc[i]
        hybrid = (bobot_sim * sim) + (bobot_rating * rating_norm)
        hasil.append((i, sim, hybrid))
    hasil = sorted(hasil, key=lambda x: x[2], reverse=True)[:top_n]
    output = []
    for i, sim, hybrid in hasil:
        output.append({
            'product_name'     : df['product_name'].iloc[i],
            'category'         : df['category_clean'].iloc[i],
            'rating'           : df['rating'].iloc[i],
            'discounted_price' : df['discounted_price'].iloc[i],
            'similarity_score' : round(sim * 100, 2),
            'hybrid_score'     : round(hybrid * 100, 2),
            'img_link'         : df['img_link'].iloc[i]
        })
    return pd.DataFrame(output)

def cari_by_keyword(keyword, top_n=5, min_rating=4.0, min_similarity=0.0):
    keyword_clean = keyword.lower()
    keyword_clean = re.sub(r'[^a-z0-9\s]', ' ', keyword_clean)
    keyword_clean = re.sub(r'\s+', ' ', keyword_clean).strip()
    keyword_vec = tfidf.transform([keyword_clean])
    keyword_sim = cosine_similarity(keyword_vec, tfidf_matrix).flatten()
    hasil = []
    for i, sim in enumerate(keyword_sim):
        # Lewati produk yang similarity-nya di bawah ambang batas
        if sim <= min_similarity:
            continue
        rating_asli = df['rating'].iloc[i]
        if rating_asli < min_rating:
            continue
        rating_norm = df['rating_normalized'].iloc[i]
        hybrid = (0.7 * sim) + (0.3 * rating_norm)
        hasil.append((i, sim, hybrid))
    hasil = sorted(hasil, key=lambda x: x[2], reverse=True)[:top_n]
    output = []
    for i, sim, hybrid in hasil:
        output.append({
            'product_name'     : df['product_name'].iloc[i],
            'category'         : df['category_clean'].iloc[i],
            'rating'           : df['rating'].iloc[i],
            'discounted_price' : df['discounted_price'].iloc[i],
            'similarity_score' : round(sim * 100, 2),
            'hybrid_score'     : round(hybrid * 100, 2),
            'img_link'         : df['img_link'].iloc[i]
        })
    return pd.DataFrame(output)


# ============================================================
# FUNGSI TAMPILAN KARTU PRODUK
# ============================================================
def tampilkan_kartu(row, rank):
    img_html = f'<img src="{row["img_link"]}" width="80" style="border-radius:8px; object-fit:cover;" onerror="this.style.display=\'none\'">' if pd.notna(row['img_link']) else ''
    kategori_tampil = kategori_familiar(row['category'])
    harga_tampil = format_rupiah(row['discounted_price'])
    st.markdown(f"""
    <div class="card">
        <div style="display:flex; gap:1rem; align-items:flex-start;">
            <div style="min-width:80px;">{img_html}</div>
            <div style="flex:1;">
                <div class="card-category">#{rank} &nbsp;·&nbsp; {kategori_tampil}</div>
                <div class="card-title">{row['product_name']}</div>
                <div class="card-meta" style="margin-bottom:0.5rem;">
                    ⭐ {row['rating']} &nbsp;|&nbsp; {harga_tampil}
                </div>
                <span class="badge-sim">Kemiripan {row['similarity_score']}%</span>
                <span class="badge-hybrid">Skor Hybrid {row['hybrid_score']}%</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:#d4a000;">⚡ ElectroRec</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.85rem;color:#555;margin-bottom:1rem;">Sistem Rekomendasi Produk Elektronik</div>', unsafe_allow_html=True)

    if st.button("⬅️ Kembali ke Halaman Utama"):
        st.session_state.masuk_app = False
        st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("**Mode Pencarian**")
    mode = st.radio("", ["🗂️ Cari berdasarkan Kategori", "💬 Cari berdasarkan Kata Kunci"], label_visibility="collapsed")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("**Filter & Pengaturan**")
    top_n        = st.slider("Jumlah Rekomendasi", 3, len(df), 5)
    if top_n == len(df):
        st.caption(f"✅ Menampilkan SEMUA produk ({len(df)} produk)")
    min_rating     = st.slider("Rating Minimum ⭐", 1.0, 5.0, 4.0, 0.1)
    min_similarity = st.slider("Kemiripan Minimum (%)", 0, 50, 15, 1)
    st.caption("Produk dengan kemiripan di bawah angka ini tidak akan ditampilkan, meski masih ada kaitan tipis (misal TV yang sekadar punya HDMI).")
    bobot_sim    = st.slider("Bobot Kemiripan", 0.0, 1.0, 0.7, 0.1)
    bobot_rating = round(1.0 - bobot_sim, 1)
    st.caption(f"Bobot Rating otomatis: {bobot_rating}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val">{len(df)}</div>
        <div class="metric-label">Total Produk Elektronik</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN CONTENT
# ============================================================
st.markdown("""
<div class="hero-title">
    Temukan Produk <span class="hero-accent">Elektronik</span><br>yang Kamu Cari
</div>
<div class="hero-sub">Content-Based Filtering · Hybrid Scoring · Dataset Amazon</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ============================================================
# MODE 1: CARI BY NAMA PRODUK
# ============================================================
if mode == "🗂️ Cari berdasarkan Kategori":
    # Buat daftar kategori dengan nama familiar, urut alfabet
    kategori_unik = sorted(df['category_clean'].dropna().unique().tolist(),
                            key=lambda x: kategori_familiar(x))
    kategori_options = [kategori_familiar(k) for k in kategori_unik]
    kategori_map_balik = dict(zip(kategori_options, kategori_unik))

    kategori_dipilih_familiar = st.selectbox("Pilih Kategori:", kategori_options)
    kategori_dipilih_asli = kategori_map_balik[kategori_dipilih_familiar]

    df_kategori = df[df['category_clean'] == kategori_dipilih_asli]
    st.caption(f"Ditemukan {len(df_kategori)} produk dalam kategori \"{kategori_dipilih_familiar}\".")

    produk_list    = df_kategori['product_name'].tolist()
    produk_dipilih = st.selectbox("Pilih Produk:", produk_list)

    if produk_dipilih:
        produk_info = df[df['product_name'] == produk_dipilih].iloc[0]
        kategori_terpilih = kategori_familiar(produk_info['category_clean'])
        harga_terpilih = format_rupiah(produk_info['discounted_price'])
        st.markdown(f"""
        <div class="selected-product">
            <div class="selected-label">Produk yang Dipilih</div>
            <div class="selected-name">{produk_dipilih}</div>
            <div class="card-meta" style="margin-top:0.4rem;">
                ⭐ {produk_info['rating']} &nbsp;·&nbsp;
                {kategori_terpilih} &nbsp;·&nbsp;
                {harga_terpilih}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔎 Tampilkan Rekomendasi"):
            hasil = rekomendasikan_produk(
                produk_dipilih, top_n=top_n,
                min_rating=min_rating,
                bobot_sim=bobot_sim,
                bobot_rating=bobot_rating,
                min_similarity=min_similarity / 100
            )
            if hasil is not None and len(hasil) > 0:
                st.markdown('<div class="section-title">🎯 Rekomendasi Teratas untuk Kamu</div>', unsafe_allow_html=True)
                if len(hasil) < top_n:
                    st.caption(f"Ditemukan {len(hasil)} produk relevan (dari {top_n} yang diminta). Produk lain tidak cukup mirip secara konten.")
                col1, col2 = st.columns(2)
                for i, row in hasil.iterrows():
                    with (col1 if i % 2 == 0 else col2):
                        tampilkan_kartu(row, i + 1)

                # Word Cloud
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">☁️ Kata Kunci Produk Terpilih</div>', unsafe_allow_html=True)
                teks_produk = produk_info['combined_features'] if 'combined_features' in produk_info else produk_info['about_product_clean']
                teks_produk_bersih = bersihkan_teks_wordcloud(teks_produk)
                wc = WordCloud(
                    width=800, height=300,
                    background_color='#ffffff',
                    colormap='YlOrBr',
                    max_words=80
                ).generate(teks_produk_bersih)
                fig, ax = plt.subplots(figsize=(10, 3))
                fig.patch.set_facecolor('#ffffff')
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

                # Grafik rating
                st.markdown('<div class="section-title">📊 Perbandingan Rating Rekomendasi</div>', unsafe_allow_html=True)
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                fig2.patch.set_facecolor('#ffffff')
                ax2.set_facecolor('#ffffff')
                ax2.bar(
                    [f"#{i+1}" for i in range(len(hasil))],
                    hasil['rating'],
                    color='#f0c040', edgecolor='#ffffff'
                )
                ax2.set_ylabel('Rating', color='#111111')
                ax2.set_xlabel('Peringkat Produk', color='#111111')
                ax2.tick_params(colors='#111111', labelsize=9)
                ax2.set_ylim(0, 5)
                for spine in ax2.spines.values():
                    spine.set_edgecolor('#e0e0e0')
                st.pyplot(fig2)
            else:
                st.warning("Tidak ada rekomendasi yang memenuhi filter. Coba turunkan rating minimum.")

# ============================================================
# MODE 2: CARI BY KEYWORD
# ============================================================
else:
    keyword = st.text_input(
        "Ketik kata kunci produk yang kamu cari:",
        placeholder="contoh: wireless bluetooth headphone, 4K smart TV, fast charger..."
    )

    if st.button("🔎 Cari Produk"):
        if keyword.strip() == "":
            st.warning("Kata kunci tidak boleh kosong.")
        else:
            hasil = cari_by_keyword(keyword, top_n=top_n, min_rating=min_rating, min_similarity=min_similarity / 100)
            if len(hasil) > 0:
                st.markdown(f'<div class="section-title">🔎 Hasil Pencarian: "{keyword}"</div>', unsafe_allow_html=True)
                if len(hasil) < top_n:
                    st.caption(f"Ditemukan {len(hasil)} produk relevan (dari {top_n} yang diminta). Produk lain tidak cukup mirip dengan kata kunci.")
                col1, col2 = st.columns(2)
                for i, row in hasil.iterrows():
                    with (col1 if i % 2 == 0 else col2):
                        tampilkan_kartu(row, i + 1)

                # Word Cloud
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">☁️ Kata Kunci Hasil Pencarian</div>', unsafe_allow_html=True)
                gabung = ' '.join(hasil['product_name'].tolist())
                gabung_bersih = bersihkan_teks_wordcloud(gabung)
                wc = WordCloud(
                    width=800, height=300,
                    background_color='#ffffff',
                    colormap='cool',
                    max_words=60
                ).generate(gabung_bersih)
                fig, ax = plt.subplots(figsize=(10, 3))
                fig.patch.set_facecolor('#ffffff')
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.warning("Tidak ada produk yang cocok. Coba kata kunci lain atau turunkan rating minimum.")