"""
Script: Peta Validasi Multi-Split (Training, Val, Test)
Fungsi: Memvisualisasikan 3 dataset cleaned secara bersamaan untuk memverifikasi 
        kualitas map snapping, filtering anomali, dan pemisahan trip.
Author: Faizal Adi Purwoko
"""
import folium
import pandas as pd
import numpy as np
import re
import os
from datetime import datetime


# 1. KONFIGURASI PATH
BASE_DIR = #masukan path lokasi

# 3 File Cleaned yang akan divisualisasikan
SPLIT_FILES = {
    'Training': 'Dataset_Training_Cleaned_Ultimated.csv',
    'Validation': 'Dataset_Val_Cleaned_Ultimated.csv',
    'Test': 'Dataset_Test_Cleaned_Ultimated.csv'
}

DATAROUTE_PATH = os.path.join(BASE_DIR, "dataroute.txt")
ANOMALIROUTE_PATH = os.path.join(BASE_DIR, "anomaliroute.txt")


# 2. FUNGSI PARSER FILE TEKS (Dipanggil 1x saja)
def parse_dataroute(path):
    route = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line: continue
            try:
                parts = line.split(',')
                route.append([float(parts[0]), float(parts[1])])
            except: continue
    return route

def parse_anomaliroute(path):
    polygons = []
    current_poly = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r'^(-?\d+\.\d+),\s*(-?\d+\.\d+)', line.strip())
            if match:
                current_poly.append([float(match.group(1)), float(match.group(2))])
            else:
                if len(current_poly) >= 3: polygons.append(current_poly)
                current_poly = []
    if len(current_poly) >= 3: polygons.append(current_poly)
    return polygons


# 3. KOORDINAT HALTE (Global)
HALTE_LOCATIONS = {
    "h01": [-7.054518, 110.444139], "h03": [-7.055610, 110.439100], "h04": [-7.053615, 110.439196],
    "h05": [-7.052103, 110.438083], "h06": [-7.050873, 110.437184], "h07": [-7.050370, 110.436099],
    "h08": [-7.049832, 110.438496], "h10": [-7.048677, 110.440215], "h11": [-7.047137, 110.438692],
    "h13": [-7.047569, 110.441010], "h14": [-7.048907, 110.442522], "h15": [-7.050684, 110.442049],
    "h16": [-7.053006, 110.441307], "h17": [-7.055236, 110.439457], "h18": [-7.055973, 110.439395]
}


# 4. FUNGSI UTAMA: BUAT PETA PER SPLIT
def build_validation_map(split_name, csv_path, dataroute, anomalies):
    """Membuat peta folium untuk satu split dataset."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Memproses split: {split_name.upper()}")
    
    # A. Inisialisasi Peta
    m = folium.Map(
        location=[-7.052000, 110.440000], 
        zoom_start=16,
        tiles='OpenStreetMap'
    )
    
    # B. Tambah Legend/Keterangan Warna
    legend_html = f'''
    <div style="position: fixed; bottom: 30px; left: 30px; width: 220px; height: 180px; 
                border:2px solid grey; z-index:9999; font-size:13px;
                background-color:white; padding: 10px 15px; border-radius: 8px;
                box-shadow: 2px 2px 6px #999;">
        <b>Legenda Peta - {split_name}</b><br><br>
        <i style="background:green; width:20px; height:3px; display:inline-block;"></i> Rute Resmi (dataroute)<br>
        <i style="background:blue; width:20px; height:3px; display:inline-block;"></i> GPS Bersih (Map Snapped)<br>
        <i style="background:red; width:12px; height:12px; display:inline-block; border-radius:50%;"></i> Area Anomali<br>
        <i style="background:yellow; width:12px; height:12px; display:inline-block; border-radius:50%; border:1px solid black;"></i> Lokasi Halte<br>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # C. Gambar Rute Resmi (HIJAU)
    folium.PolyLine(
        dataroute, color="green", weight=5, opacity=0.8, 
        popup="Rute Resmi (dataroute.txt)"
    ).add_to(m)
    
    # D. Gambar Area Anomali (MERAH)
    for i, poly in enumerate(anomalies):
        folium.Polygon(
            locations=poly, color="red", weight=2, fill=True, 
            fill_color="red", fill_opacity=0.2, 
            popup=f"Area Anomali {i+1} (Harus Kosong dari GPS)"
        ).add_to(m)
    
    # E. Gambar Lokasi Halte (KUNING)
    for h_id, coords in HALTE_LOCATIONS.items():
        folium.CircleMarker(
            location=coords, radius=7, color="black", fill=True, 
            fill_color="yellow", fill_opacity=0.9, 
            popup=f"Halte {h_id}"
        ).add_to(m)
    
    # F. Muat & Gambar Data GPS Bersih (BIRU)
    df_clean = pd.read_csv(csv_path)
    time_col = 'Waktu Titik' if 'Waktu Titik' in df_clean.columns else 'Point Time'
    
    total_trips = df_clean['Trip_ID'].nunique()
    total_points = len(df_clean)
    
    print(f"    Total Trip: {total_trips} | Total Titik GPS: {total_points:,}")
    
    # Sampling: Ambil max 5 trip biar ga lag
    max_sample = 5
    sample_trips = df_clean['Trip_ID'].unique()[:max_sample]
    df_sample = df_clean[df_clean['Trip_ID'].isin(sample_trips)]
    
    print(f"    Menggambar sampel {len(sample_trips)} trip pertama...")
    
    for trip_id, group in df_sample.groupby('Trip_ID'):
        group = group.sort_values(time_col)
        rute_bersih = list(zip(group['Latitude'], group['Longitude']))
        
        # Popup informatif
        popup_text = (
            f"<b>Trip ID:</b> {trip_id}<br>"
            f"<b>Jumlah Titik:</b> {len(group)}<br>"
            f"<b>Waktu:</b> {group[time_col].iloc[0]} s/d {group[time_col].iloc[-1]}"
        )
        
        folium.PolyLine(
            rute_bersih, color="blue", weight=4, opacity=0.75, 
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(m)
    
    # G. Tambah Info Statistik di Peta
    info_html = f'''
    <div style="position: fixed; top: 10px; right: 10px; width: 200px; 
                border:2px solid #333; z-index:9999; font-size:12px;
                background-color:#f8f9fa; padding: 10px; border-radius: 6px;">
        <b>Statistik {split_name}</b><br>
        Total Trip: <b>{total_trips}</b><br>
        Total Titik: <b>{total_points:,}</b><br>
        Sampel Digambar: <b>{len(sample_trips)}</b> trip
    </div>
    '''
    m.get_root().html.add_child(folium.Element(info_html))
    
    return m


# 5. EKSEKUSI UTAMA
if __name__ == "__main__":
    t0 = datetime.now()
    print("=" * 70)
    print(f"[{t0.strftime('%Y-%m-%d %H:%M:%S')}] MEMULAI PEMBUATAN PETA VALIDASI (3 SPLIT)")
    print("=" * 70)
    
    # Parse rute & anomali 
    print("\n[1/4] Parsing dataroute.txt & anomaliroute.txt...")
    dataroute = parse_dataroute(DATAROUTE_PATH)
    anomalies = parse_anomaliroute(ANOMALIROUTE_PATH)
    print(f"    Rute resmi: {len(dataroute)} titik")
    print(f"    Area anomali: {len(anomalies)} poligon")
    
    print("\n[2/4] Memuat data cleaned & membuat peta...")
    generated_files = []
    
    for split_name, csv_filename in SPLIT_FILES.items():
        csv_path = os.path.join(BASE_DIR, csv_filename)
        
        if not os.path.exists(csv_path):
            print(f"    File tidak ditemukan: {csv_path} (Dilewati)")
            continue
        
        # Buat peta
        peta = build_validation_map(split_name, csv_path, dataroute, anomalies)
        
        # Simpan ke HTML
        output_html = os.path.join(BASE_DIR, f"Peta_Validasi_{split_name}.html")
        peta.save(output_html)
        generated_files.append(output_html)
        print(f"    Tersimpan: {output_html}")
    
   
    print("\n" + ":" * 70)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] SELESAI! {len(generated_files)} peta berhasil dibuat.")
    print(":" * 70)
    print("\n File Output:")
    for f in generated_files:
        print(f"   • {f}")
    print("\n Bcek live server file html")
    print("   ( garis biru untuk lihat detail trip)")