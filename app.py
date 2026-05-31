import streamlit as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Konfigurasi Halaman Utama
st.set_page_config(page_title="Dashboard ML UMKM Keramik", layout="wide")

# =========================================================
# STYLE DASHBOARD (HEADER)
# =========================================================
st.markdown("""
    <div style="text-align:center; color:white; background:linear-gradient(to right,#4e54c8,#8f94fb); padding:20px; border-radius:15px; margin-bottom: 25px;">
        <h1 style="margin:0;">DASHBOARD MACHINE LEARNING UMKM KERAMIK</h1>
    </div>
""", unsafe_allow_html=True)

# =========================================================
# LOAD & CLEANING DATA (Cached agar loading cepat)
# =========================================================
@st.cache_data
def load_data():
    file_excel = 'Data Pemasukan dan pengeluaran UMKM Keramik.xlsx'
    xls = pd.ExcelFile(file_excel)
    all_data = []
    
    for sheet in xls.sheet_names:
        df = pd.read_excel(file_excel, sheet_name=sheet)
        df['Bulan'] = sheet
        all_data.append(df)
        
    data = pd.concat(all_data, ignore_index=True)
    data.columns = data.columns.str.strip()
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data = data.dropna(how='all')
    
    numeric_cols = ['Unit Terjual', 'Harga Per Unit', 'Total', 'Total Penegluaran']
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
    penjualan = data[data['Produk Terjual'].notna()].copy()
    pengeluaran = data[data['Jenis Pengeluaran'].notna()].copy()
    
    return penjualan, pengeluaran

try:
    penjualan, pengeluaran = load_data()
except Exception as e:
    st.error(f"Gagal membaca file Excel. Pastikan file 'Data Pemasukan dan pengeluaran UMKM Keramik.xlsx' sudah diunggah ke GitHub. Error: {e}")
    st.stop()

# =========================================================
# METRICS DASHBOARD
# =========================================================
total_penjualan = penjualan['Total'].sum()
total_pengeluaran = pengeluaran['Total Penegluaran'].sum()
keuntungan = total_penjualan - total_pengeluaran
jumlah_produk = penjualan['Produk Terjual'].nunique()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div style="background:#4e54c8; color:white; padding:20px; border-radius:15px; text-align:center; box-shadow:2px 2px 10px lightgray;"><h3>Total Penjualan</h3><h2>Rp {total_penjualan:,.0f}</h2></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div style="background:#ff6b6b; color:white; padding:20px; border-radius:15px; text-align:center; box-shadow:2px 2px 10px lightgray;"><h3>Total Pengeluaran</h3><h2>Rp {total_pengeluaran:,.0f}</h2></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div style="background:#1dd1a1; color:white; padding:20px; border-radius:15px; text-align:center; box-shadow:2px 2px 10px lightgray;"><h3>Keuntungan</h3><h2>Rp {keuntungan:,.0f}</h2></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div style="background:#feca57; color:white; padding:20px; border-radius:15px; text-align:center; box-shadow:2px 2px 10px lightgray;"><h3>Jumlah Produk</h3><h2>{jumlah_produk}</h2></div>', unsafe_allow_html=True)

st.write("---")

# =========================================================
# TABS UNTUK MENU NAVIGATION (Agar rapi tidak memanjang kebawah)
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Analisis Produk & Pengeluaran", 
    "🤖 Prediksi ML (Linear Regression)", 
    "📊 Forecasting (MA & ES)", 
    "🎯 Clustering Produk",
    "📋 Data Mentah"
])

# ---------------------------------------------------------
# TAB 1: ANALISIS PRODUK & PENGELUARAN
# ---------------------------------------------------------
with tab1:
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("<h3 style='color:#4e54c8'>TOP 10 PRODUK PALING LAKU</h3>", unsafe_allow_html=True)
        produk_laris = penjualan.groupby('Produk Terjual')['Unit Terjual'].sum().sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(produk_laris.index, produk_laris.values, color='#8f94fb')
        ax.set_title('10 Produk Paling Laku', fontsize=14)
        ax.set_ylabel('Unit Terjual')
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom')
        st.pyplot(fig)
        
    with col_right:
        st.markdown("<h3 style='color:#ff6b6b'>ANALISIS PENGELUARAN</h3>", unsafe_allow_html=True)
        pengeluaran_chart = pengeluaran.groupby('Jenis Pengeluaran')['Total Penegluaran'].sum()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(pengeluaran_chart, labels=pengeluaran_chart.index, autopct='%1.1f%%', startangle=90, colors=['#ff6b6b','#feca57','#1dd1a1','#4e54c8'])
        ax.set_title('Persentase Pengeluaran', fontsize=14)
        st.pyplot(fig)

# ---------------------------------------------------------
# PRE-PROCESSING UNTUK DERET WAKTU (TIME SERIES)
# ---------------------------------------------------------
penjualan_harian = penjualan.groupby('Tanggal penjualan')['Total'].sum().reset_index().dropna()
penjualan_harian['Hari'] = np.arange(len(penjualan_harian))

# ---------------------------------------------------------
# TAB 2: LINEAR REGRESSION
# ---------------------------------------------------------
with tab2:
    st.markdown("<h3 style='color:#1dd1a1'>PREDIKSI PENJUALAN MACHINE LEARNING</h3>", unsafe_allow_html=True)
    
    X = penjualan_harian[['Hari']]
    y = penjualan_harian['Total']
    model = LinearRegression().fit(X, y)
    prediksi = model.predict(X)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(penjualan_harian['Tanggal penjualan'], y, linewidth=3, label='Data Asli', color='#4e54c8')
    ax.plot(penjualan_harian['Tanggal penjualan'], prediksi, linewidth=3, linestyle='--', label='Prediksi ML', color='#1dd1a1')
    ax.set_title('Prediksi Tren Penjualan (Linear Regression)', fontsize=16)
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Total Penjualan')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ---------------------------------------------------------
# TAB 3: MOVING AVERAGE & EXPONENTIAL SMOOTHING
# ---------------------------------------------------------
with tab3:
    st.markdown("<h3 style='color:#feca57'>FORECASTING MOVING AVERAGE & EXPONENTIAL SMOOTHING</h3>", unsafe_allow_html=True)
    
    # Hitung MA dan ES
    penjualan_harian['Moving Average'] = penjualan_harian['Total'].rolling(window=7).mean()
    penjualan_harian['Exponential Smoothing'] = penjualan_harian['Total'].ewm(alpha=0.3).mean()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(penjualan_harian['Tanggal penjualan'], penjualan_harian['Total'], linewidth=1.5, label='Data Asli', color='gray', alpha=0.6)
    ax.plot(penjualan_harian['Tanggal penjualan'], penjualan_harian['Moving Average'], linewidth=3, label='Moving Average (7 Hari)', color='#feca57')
    ax.plot(penjualan_harian['Tanggal penjualan'], penjualan_harian['Exponential Smoothing'], linewidth=3, label='Exponential Smoothing (α=0.3)', color='#5f27cd')
    ax.set_title('Perbandingan Analisis Tren Penjualan', fontsize=16)
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Penjualan')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ---------------------------------------------------------
# TAB 4: CLUSTERING K-MEANS
# ---------------------------------------------------------
with tab4:
    st.markdown("<h3 style='color:#ee5253'>CLUSTERING PRODUK K-MEANS</h3>", unsafe_allow_html=True)
    
    cluster_data = penjualan[['Unit Terjual', 'Harga Per Unit']].dropna()
    scaled_data = StandardScaler().fit_transform(cluster_data)
    
    # Input interaktif untuk menentukan jumlah cluster langsung di web!
    n_clusters = st.slider("Pilih Jumlah Cluster (K):", min_value=2, max_value=6, value=3)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_data['Cluster'] = kmeans.fit_predict(scaled_data)
    
    col_c1, col_c2 = st.columns([2, 1])
    
    with col_c1:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(cluster_data['Unit Terjual'], cluster_data['Harga Per Unit'], c=cluster_data['Cluster'], cmap='rainbow', s=100, alpha=0.8)
        ax.set_title('Visualisasi Cluster Produk Keramik', fontsize=14)
        ax.set_xlabel('Unit Terjual')
        ax.set_ylabel('Harga Per Unit')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
    with col_c2:
        st.markdown("<h4>Jumlah Produk per Cluster</h4>", unsafe_allow_html=True)
        counts = cluster_data['Cluster'].value_counts().sort_index()
        st.dataframe(counts.rename("Jumlah Produk"))

# ---------------------------------------------------------
# TAB 5: TABEL DATA MENTAH
# ---------------------------------------------------------
with tab5:
    st.markdown("<h3 style='color:#222f3e'>DATA PENJUALAN (20 BARIS PERTAMA)</h3>", unsafe_allow_html=True)
    st.dataframe(penjualan.head(20), use_container_width=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
    <div style="margin-top:30px; padding:15px; background:#4e54c8; color:white; border-radius:15px; text-align:center;">
        <h4 style="margin:0;">Dashboard Selesai Dibuat</h4>
        <p style="margin:5px 0 0 0;">Analisis Machine Learning, Forecasting, dan Clustering UMKM Keramik</p>
    </div>
""", unsafe_allow_html=True)
