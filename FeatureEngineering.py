"""
Script: Feature Engineering, Route Distance & Outlier Separation (FINAL v3.2)
Fungsi: Mengekstrak segmen rute, menghitung jarak aspal, memperbaiki bug timezone, 
        memfilter mangkir, menambahkan flag 'is_rusun' dan 'is_departure_from_rusun',
        dan mencegah Data Leakage.
Author: Faizal Adi Purwoko
"""
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

BASE_DIR = #masukan path nya bang

INPUT_FILES = {
    'training': 'Dataset_Training_Cleaned_Ultimated.csv',
    'val': 'Dataset_Val_Cleaned_Ultimated.csv',
    'test': 'Dataset_Test_Cleaned_Ultimated.csv'
}

OUTPUT_FILES = {
    'training': 'Segments_Training.csv',
    'val': 'Segments_Val.csv',
    'test': 'Segments_Test.csv'
}

OUTLIER_FILES = {
    'training': 'Outliers_Log_Training.csv',
    'val': 'Outliers_Log_Val.csv',
    'test': 'Outliers_Log_Test.csv'
}

HALTE_LOCATIONS = {
    'h01': {'lat': -7.054518537168431, 'lng': 110.44413919120406},
    'h03': {'lat': -7.0556100, 'lng': 110.4391},
    'h04': {'lat': -7.053615652182, 'lng': 110.43919618890992},
    'h05': {'lat': -7.052103865215362, 'lng': 110.43808378253539},
    'h06': {'lat': -7.050873236387066, 'lng': 110.43718416935035},
    'h07': {'lat': -7.050370746018777, 'lng': 110.43609972172088},
    'h08': {'lat': -7.049832325888632, 'lng': 110.43849680096805},
    'h10': {'lat': -7.048677336244937, 'lng': 110.44021522652281},
    'h11': {'lat': -7.04713778936035, 'lng': 110.43869200447789},
    'h13': {'lat': -7.047569654368096, 'lng': 110.44101030995277},
    'h14': {'lat': -7.048907407951046, 'lng': 110.44252222022146},
    'h15': {'lat': -7.050684864323637, 'lng': 110.4420491664416},
    'h16': {'lat': -7.053006517891536, 'lng': 110.44130798104808},
    'h17': {'lat': -7.0552369, 'lng': 110.4394576},
    'h18': {'lat': -7.055973568692425, 'lng': 110.43939589722012},
}

STOP_RADIUS_M = 25.0  
TIME_COL = 'Waktu Titik'
ROUTE_FILE = os.path.join(BASE_DIR, "dataroute.txt")

def log_print(msg, level="INFO"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {level}: {msg}")

def haversine_dist(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 2 * np.arcsin(np.sqrt(a)) * 6_367_000.0

def load_route_distance():
    route_points = []
    with open(ROUTE_FILE, 'r') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 2:
                    route_points.append((float(parts[0]), float(parts[1])))
    route_arr = np.array(route_points)
    cum_dist = [0.0]
    for i in range(1, len(route_arr)):
        d = haversine_dist(route_arr[i-1, 1], route_arr[i-1, 0], route_arr[i, 1], route_arr[i, 0])
        cum_dist.append(cum_dist[-1] + d)
    return route_arr, np.array(cum_dist)

def get_halte_route_index(route_arr):
    halte_idx = {}
    for h_id, h_data in HALTE_LOCATIONS.items():
        dists = haversine_dist(route_arr[:, 1], route_arr[:, 0], h_data['lng'], h_data['lat'])
        halte_idx[h_id] = np.argmin(dists)
    return halte_idx

def calculate_route_distance(from_halte, to_halte, halte_idx, cum_dist):
    idx_a = halte_idx[from_halte]
    idx_b = halte_idx[to_halte]
    total_dist = cum_dist[-1]
    if idx_b >= idx_a:
        return cum_dist[idx_b] - cum_dist[idx_a]
    else:
        return (total_dist - cum_dist[idx_a]) + cum_dist[idx_b]

def detect_nearest_halte(lat, lng):
    min_dist = float('inf')
    nearest_halte = None
    for halte_id, data in HALTE_LOCATIONS.items():
        dist = haversine_dist(lng, lat, data['lng'], data['lat'])
        if dist <= STOP_RADIUS_M and dist < min_dist:
            min_dist = dist
            nearest_halte = halte_id
    return nearest_halte

def extract_segments(df, halte_idx, cum_dist):
    segments = []
    outliers = []
    
    group_col = 'session_id' if 'session_id' in df.columns else None
    
    if group_col:
        df = df.sort_values([group_col, TIME_COL]).reset_index(drop=True)
        grouped = df.groupby(group_col)
    else:
        df = df.sort_values(TIME_COL).reset_index(drop=True)
        grouped = [(None, df)]

    for _, group_df in grouped:
        current_halte, departure_time, max_passengers = None, None, 0
        
        for idx, row in group_df.iterrows():
            lat, lng = row['Latitude'], row['Longitude']
            time_val = row[TIME_COL]
            pax = row.get('Penumpang per Titik', 0)
            if pd.isna(pax): pax = 0
            
            halte = detect_nearest_halte(lat, lng)
            
            if halte:
                if current_halte is None:
                    current_halte, departure_time = halte, time_val
                    max_passengers = pax
                elif halte != current_halte:
                    travel_time = (time_val - departure_time).total_seconds()
                    jarak_aspal = calculate_route_distance(current_halte, halte, halte_idx, cum_dist)
                    speed_kmh = (jarak_aspal / 1000.0) / (travel_time / 3600.0) if travel_time > 0 else 0
                    
                    record = {
                        'from_halte': current_halte, 'to_halte': halte,
                        'departure_time': departure_time, 'arrival_time': time_val,
                        'travel_time_sec': travel_time, 'max_passengers': max_passengers,
                        'route_distance_m': jarak_aspal, 'avg_speed_kmh': round(speed_kmh, 2)
                    }
                    
                    # ATURAN GANDA (CONDITIONAL FILTER)
                    if current_halte == 'h01' or halte == 'h01':
                        max_time = 1800  # 30 menit
                        min_speed = 0.0  # Izinkan 0 karena ngetem
                    else:
                        max_time = 420   # 7 menit
                        min_speed = 3.0  # Kecepatan normal

                    if 15 <= travel_time <= max_time and speed_kmh >= min_speed:
                        segments.append(record)
                    else:
                        outliers.append(record)
                        
                    current_halte, departure_time = halte, time_val
                    max_passengers = pax
                else:
                    if pax > max_passengers: 
                        max_passengers = pax
                    
    return pd.DataFrame(segments), pd.DataFrame(outliers)

def add_features(df_segments):
    if len(df_segments) == 0: 
        return df_segments
    
    df = df_segments.copy()
    
    if df['departure_time'].dt.tz is not None:
        df['departure_time'] = df['departure_time'].dt.tz_convert('Asia/Jakarta')
    else:
        df['departure_time'] = df['departure_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Jakarta')
        
    df['hour_of_day'] = df['departure_time'].dt.hour
    df['day_of_week'] = df['departure_time'].dt.dayofweek 
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_peak_hour'] = df['hour_of_day'].isin([7, 8, 9, 10, 11, 13, 14, 15, 16]).astype(int)
    
    halte_list = sorted(HALTE_LOCATIONS.keys())
    halte_to_idx = {h: i for i, h in enumerate(halte_list)}
    df['from_halte_idx'] = df['from_halte'].map(halte_to_idx)
    df['to_halte_idx'] = df['to_halte'].map(halte_to_idx)
    df['segment_id'] = df['from_halte'] + '_to_' + df['to_halte']
    
    df['is_rusun'] = ((df['from_halte'] == 'h01') | (df['to_halte'] == 'h01')).astype(int)
    
    # FITUR BARU: DEPARTURE DARI RUSUNAWA
    df['is_departure_from_rusun'] = (df['from_halte'] == 'h01').astype(int)
    
    return df

if __name__ == "__main__":
    t0 = time.time()
    print(":" * 75)
    log_print("SYSTEM: Initiating Feature Engineering Pipeline (v3.2 Final)")
    route_arr, cum_dist = load_route_distance()
    halte_idx = get_halte_route_index(route_arr)
    log_print(f"Route loaded. Total distance: {cum_dist[-1]:.2f} meters")
    
    for split_name, input_file in INPUT_FILES.items():
        log_print(f"Processing batch: {split_name.upper()}")
        df = pd.read_csv(os.path.join(BASE_DIR, input_file))
        
        if 'Point Time' in df.columns: df.rename(columns={'Point Time': TIME_COL}, inplace=True)
        elif 'index' in df.columns: df.rename(columns={'index': TIME_COL}, inplace=True)
        
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors='coerce')
        df = df.dropna(subset=[TIME_COL])
        df[TIME_COL] = df[TIME_COL].dt.tz_convert('Asia/Jakarta')
        
        df_segments, df_outliers = extract_segments(df, halte_idx, cum_dist)
        df_segments = add_features(df_segments)
        
        out_path = os.path.join(BASE_DIR, OUTPUT_FILES[split_name])
        df_segments.to_csv(out_path, index=False)
        
        if len(df_outliers) > 0:
            outlier_path = os.path.join(BASE_DIR, OUTLIER_FILES[split_name])
            df_outliers.to_csv(outlier_path, index=False)
            log_print(f"Isolated {len(df_outliers)} anomalies", "WARNING")
            
        log_print(f"Valid segments saved. Shape: {df_segments.shape} | Avg Time: {df_segments['travel_time_sec'].mean():.1f}s", "SUCCESS")
        
    print(":" * 75)
    log_print(f"SYSTEM: Pipeline Completed in {time.time() - t0:.2f} seconds.", "SUCCESS")