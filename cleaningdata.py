"""
Script: Spatial Preprocessing Pipeline V3.1 (Session-Safe & Memory-Optimized)
Fungsi: Membaca file RAW COMBINED V2, melakukan ekstraksi fitur spasial, filtering 
        anomali (Ray-Casting), resample temporal, dan map snapping.
        Sinkron dengan Aggregation V2.0 
Author: Faizal Adi Purwoko
"""
import pandas as pd
import numpy as np
import os
import time
import re
from datetime import datetime


# 1. KONFIGURASI PATH & PARAMETER
BASE_INPUT_DIR  = # masukan path
BASE_OUTPUT_DIR = # masukan path lokasinya juga

DATAROUTE_PATH    = os.path.join(BASE_OUTPUT_DIR, "dataroute.txt")
ANOMALIROUTE_PATH = os.path.join(BASE_OUTPUT_DIR, "anomaliroute.txt")

# read file gabungan V2 dari Tahap 1
SPLITS = {
    'Dataset_Training_Raw_CombinedV2.csv': 'Dataset_Training_Cleaned_Ultimated.csv',
    'Dataset_Val_Raw_CombinedV2.csv'     : 'Dataset_Val_Cleaned_Ultimated.csv',
    'Dataset_Test_Raw_CombinedV2.csv'    : 'Dataset_Test_Cleaned_Ultimated.csv'
}

CENTER_LAT, CENTER_LNG = -7.052000, 110.440000
MAX_RADIUS_METERS   = 1200.0  
MAX_SPEED_KMH       = 30.0     
STUCK_THRESHOLD_SEC = 900.0    
GSM_BLINDSPOT_SEC   = 60.0     
BLINDSPOT_DIST_M    = 50.0     
GARASI_BOUNDS = dict(lat_min=-7.058000, lat_max=-7.055000, lng_min=110.442000, lng_max=110.446000)
RESAMPLE_FREQ = '5s'
MAX_GAP_POINTS = 12

#FIX 1: pakai session_id 
GROUP_KEY = 'session_id'


# 2. FUNGSI UTILITAS
def haversine_dist(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * np.arcsin(np.sqrt(a)) * 6_367_000.0

def point_in_polygon(lat, lng, polygon):
    n, inside = len(polygon), False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if lat > min(p1y, p2y) and lat <= max(p1y, p2y) and lng <= max(p1x, p2x):
            if p1y != p2y: xinters = (lat - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or lng <= xinters: inside = not inside
        p1x, p1y = p2x, p2y
    return inside

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
    return np.array(route)

def parse_anomaliroute(path):
    polygons, current_poly = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r'^(-?\d+\.\d+),\s*(-?\d+\.\d+)', line.strip())
            if match: current_poly.append((float(match.group(1)), float(match.group(2))))
            else:
                if len(current_poly) >= 3: polygons.append(current_poly)
                current_poly = []
    if len(current_poly) >= 3: polygons.append(current_poly)
    return polygons


# 3. FUNGSI UTAMA PROSES PER FILE GABUNGAN
def process_file(input_filename, output_filename, DATAROUTE, ANOMALY_POLYS):
    curr_time = datetime.now().strftime('%H:%M:%S')
    print(f"[{curr_time}] INFO: Initiating spatial processing for file: '{input_filename}'")
    
    file_path = os.path.join(BASE_INPUT_DIR, input_filename)
    if not os.path.exists(file_path):
        print(f"[{curr_time}] WARNING: File not found at {file_path}. Skipping.")
        return

    # A. LOAD DATA
    df_raw = pd.read_csv(file_path)
    
    # Fallback defensif jika ada file lama
    time_col = 'Waktu Titik' if 'Waktu Titik' in df_raw.columns else 'Point Time'
    pax_col  = 'Penumpang per Titik' if 'Penumpang per Titik' in df_raw.columns else ('Passengers per Point' if 'Passengers per Point' in df_raw.columns else None)
    
    if time_col not in df_raw.columns and 'Point Time' in df_raw.columns:
        df_raw.rename(columns={'Point Time': time_col}, inplace=True)
    if pax_col not in df_raw.columns and 'Passengers per Point' in df_raw.columns:
        df_raw.rename(columns={'Passengers per Point': pax_col}, inplace=True)

    df_raw[time_col] = pd.to_datetime(df_raw[time_col], utc=True, errors='coerce')
    df_raw = df_raw.dropna(subset=[time_col, 'Latitude', 'Longitude'])
    
    # FIX 2: Sortir berdasarkan session_id, bukan Source_File
    df_raw = df_raw.sort_values([GROUP_KEY, time_col]).reset_index(drop=True)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] INFO: Raw GPS loaded: {len(df_raw):,} rows | Sessions: {df_raw[GROUP_KEY].nunique()}")

    # B. GEOFENCE & GARASI
    df_raw['dist_center'] = haversine_dist(df_raw['Longitude'], df_raw['Latitude'], CENTER_LNG, CENTER_LAT)
    df_clean = df_raw[df_raw['dist_center'] <= MAX_RADIUS_METERS].copy()
    
    g = GARASI_BOUNDS
    in_garage = (df_clean['Latitude'].between(g['lat_min'], g['lat_max']) & 
                 df_clean['Longitude'].between(g['lng_min'], g['lng_max']))
    df_clean = df_clean[~in_garage].copy()

    # C. FILTER ANOMALI (RAY-CASTING POLYGON)
    is_anomaly = np.zeros(len(df_clean), dtype=bool)
    lats, lngs = df_clean['Latitude'].values, df_clean['Longitude'].values
    for poly in ANOMALY_POLYS:
        min_lat, max_lat = min(p[0] for p in poly), max(p[0] for p in poly)
        min_lng, max_lng = min(p[1] for p in poly), max(p[1] for p in poly)
        in_bbox = (lats >= min_lat) & (lats <= max_lat) & (lngs >= min_lng) & (lngs <= max_lng)
        for idx in np.where(in_bbox)[0]:
            if point_in_polygon(lats[idx], lngs[idx], poly): is_anomaly[idx] = True
    df_clean = df_clean[~is_anomaly].copy()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] INFO: Spatial anomaly filter applied. Valid rows: {len(df_clean):,}.")

    # D. TRIP SEGMENTATION & SPEED FILTER
    # FIX 3: Groupby menggunakan session_id
    df_clean['time_diff'] = df_clean.groupby(GROUP_KEY)[time_col].diff().dt.total_seconds().fillna(0)
    df_clean['prev_lat'] = df_clean.groupby(GROUP_KEY)['Latitude'].shift(1)
    df_clean['prev_lng'] = df_clean.groupby(GROUP_KEY)['Longitude'].shift(1)
    df_clean['dist_diff'] = haversine_dist(
        df_clean['Longitude'], df_clean['Latitude'],
        df_clean['prev_lng'].fillna(df_clean['Longitude']),
        df_clean['prev_lat'].fillna(df_clean['Latitude'])
    ).fillna(0)
    
    df_clean['is_new_trip'] = (
        (df_clean['time_diff'] > STUCK_THRESHOLD_SEC) | 
        ((df_clean['time_diff'] > GSM_BLINDSPOT_SEC) & (df_clean['dist_diff'] > BLINDSPOT_DIST_M))
    ).astype(int)
    
    df_clean['Trip_ID'] = (df_clean[GROUP_KEY].astype(str) + "_Part_" + 
                           df_clean.groupby(GROUP_KEY)['is_new_trip'].cumsum().astype(str))
    
    df_clean['calculated_speed'] = np.where(
        df_clean['time_diff'] > 0, 
        (df_clean['dist_diff'] / 1000.0) / (df_clean['time_diff'] / 3600.0), 
        0.0
    )
    df_clean = df_clean[df_clean['calculated_speed'] <= MAX_SPEED_KMH].copy()

    # E. INTERPOLASI 5 DETIK
    df_indexed = df_clean.set_index(time_col)
    frames_clean = []
    
    for trip_id, group in df_indexed.groupby('Trip_ID'):
        if len(group) < 2: continue
        
        resampled = group.resample(RESAMPLE_FREQ).mean(numeric_only=True)
        resampled['Latitude'] = resampled['Latitude'].interpolate(method='linear', limit=MAX_GAP_POINTS)
        resampled['Longitude'] = resampled['Longitude'].interpolate(method='linear', limit=MAX_GAP_POINTS)
        
        if pax_col:
            resampled[pax_col] = group[pax_col].resample(RESAMPLE_FREQ).max()
            resampled[pax_col] = resampled[pax_col].ffill(limit=MAX_GAP_POINTS).fillna(0)
        
        # fix kan kolom YANG HILANG KARENA mean()
        cols_to_keep = [GROUP_KEY, 'Kode Armada', 'Sesi', 'Tanggal', 'Tipe Baris', 'Source_File']
        for col in cols_to_keep:
            if col in group.columns:
                resampled[col] = group[col].iloc[0]
                
        resampled['Trip_ID'] = trip_id
        resampled = resampled.dropna(subset=['Latitude', 'Longitude'])
        frames_clean.append(resampled)

    df_final = pd.concat(frames_clean).reset_index()
    
    if time_col not in df_final.columns and 'index' in df_final.columns:
        df_final.rename(columns={'index': time_col}, inplace=True)

    # F. MAP SNAPPING
    route_lats = np.radians(DATAROUTE[:, 0])
    route_lngs = np.radians(DATAROUTE[:, 1])
    row_lats = np.radians(df_final['Latitude'].values)
    row_lngs = np.radians(df_final['Longitude'].values)
    
    nearest_idx = np.zeros(len(df_final), dtype=int)
    batch_size = 10000  # Proses per 10.000 baris agar RAM tidak meledak
    
    for i in range(0, len(df_final), batch_size):
        batch_row_lats = row_lats[i:i+batch_size, None]
        batch_row_lngs = row_lngs[i:i+batch_size, None]
        
        dlat = route_lats[None, :] - batch_row_lats
        dlon = route_lngs[None, :] - batch_row_lngs
        a = np.sin(dlat / 2) ** 2 + np.cos(batch_row_lats) * np.cos(route_lats[None, :]) * np.sin(dlon / 2) ** 2
        
        # CLIP: Mencegah floating-point error yang menghasilkan NaN
        dist_matrix = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) * 6_367_000.0 
        nearest_idx[i:i+batch_size] = np.argmin(dist_matrix, axis=1)

    df_final['Latitude'] = DATAROUTE[nearest_idx, 0]
    df_final['Longitude'] = DATAROUTE[nearest_idx, 1]

    # G. EKSPOR
    output_path = os.path.join(BASE_OUTPUT_DIR, output_filename)
    df_final.to_csv(output_path, index=False)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] SUCCESS: Exported to {output_path}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] DETAILS: Output shape: {df_final.shape}\n")


# 4. EKSEKUSI UTAMA
if __name__ == "__main__":
    t0 = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SYSTEM: Starting Spatial Preprocessing Pipeline V3.1")
    print(":" * 75)
    
    DATAROUTE = parse_dataroute(DATAROUTE_PATH)
    ANOMALY_POLYS = parse_anomaliroute(ANOMALIROUTE_PATH)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] SYSTEM: Route: {len(DATAROUTE)} pts | Anomaly zones: {len(ANOMALY_POLYS)}\n")
    
    for in_file, out_file in SPLITS.items():
        process_file(in_file, out_file, DATAROUTE, ANOMALY_POLYS)
        
    print(":" * 75)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SYSTEM: Pipeline finished in {time.time() - t0:.2f} seconds.")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SYSTEM: Output: {BASE_OUTPUT_DIR}")