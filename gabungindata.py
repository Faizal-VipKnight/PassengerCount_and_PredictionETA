"""
Script: Data Aggregation Module (Combine CSV) - V2.0 (Multi-Language & Session Safe)
Fungsi: Menggabungkan file CSV, menyeragamkan nama kolom (ID/EN), dan membuat 
        session_id unik dari Buggy ID + Sesi + Tanggal agar tidak ada segmen hantu.
Author: Faizal Adi Purwoko
"""
import pandas as pd
import glob
import os
import time
from datetime import datetime

# 1. KONFIGURASI PATH 
BASE_DIR = #input folder path data


FOLDERS_TO_COMBINE = {
    'dtraining': 'Dataset_Training_Raw_CombinedV2.csv',
    'dvalidasi'  : 'Dataset_Val_Raw_CombinedV2.csv',
    'dtest'      : 'Dataset_Test_Raw_CombinedV2.csv'
}

def combine_folder_data(folder_name, output_filename):
    curr_time = datetime.now().strftime('%H:%M:%S')
    print(f"[{curr_time}] INFO: Initiating data aggregation for directory: '{folder_name}'")
    
    search_path = os.path.join(BASE_DIR, folder_name, "*.csv")
    csv_files = glob.glob(search_path)
    
    if not csv_files:
        print(f"[{curr_time}] WARNING: No CSV files found in {search_path}")
        return

    print(f"[{curr_time}] INFO: Found {len(csv_files)} files. Extracting GPS coordinates...")
    
    frames = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        
        # A. Ekstraksi GPS (Handle Bahasa Indonesia & Inggris)
        if 'Tipe Baris' in df.columns:
            gps_data = df[df['Tipe Baris'] == 'Titik GPS'].copy()
        else:
            gps_data = df[df['Row Type'] == 'GPS Point'].copy()
            
        # B. STANDARDISASI NAMA KOLOM 
        rename_map = {
            'Row Type': 'Tipe Baris',
            'Fleet Code': 'Kode Armada',       # Buggy ID
            'Session': 'Sesi',                 # Session ID
            'Date': 'Tanggal',
            'Point Time': 'Waktu Titik',
            'Passengers per Point': 'Penumpang per Titik'
        }
        gps_data.rename(columns=rename_map, inplace=True)
        
        # C. BUAT SESSION ID UNIK (Mencegah segmen hantu antar-sesi)
        # Format: KodeArmada_Sesi_Tanggal (Cth: B01_Session 1_2026-06-19)
        if 'Kode Armada' in gps_data.columns and 'Sesi' in gps_data.columns and 'Tanggal' in gps_data.columns:
            gps_data['session_id'] = (gps_data['Kode Armada'].astype(str) + '_' + 
                                      gps_data['Sesi'].astype(str) + '_' + 
                                      gps_data['Tanggal'].astype(str))
        else:
            # Fallback jika kolom tidak ada (menggunakan nama file)
            gps_data['session_id'] = os.path.basename(file_path)
            
        gps_data['Source_File'] = os.path.basename(file_path)
        frames.append(gps_data)
        
    df_combined = pd.concat(frames, ignore_index=True)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] INFO: Aggregation complete. Total raw rows: {len(df_combined):,}")

    # D. PRE-PROCESSING (Waktu & Sortir)
    time_col = 'Waktu Titik' 
    
    # Konversi waktu ke datetime UTC
    df_combined[time_col] = pd.to_datetime(df_combined[time_col], utc=True, errors='coerce')
    
    # Hapus baris yang waktu atau koordinatnya kosong/rusak
    df_combined = df_combined.dropna(subset=[time_col, 'Latitude', 'Longitude'])
    
    # FIX KRITIS: Sortir berdasarkan session_id, lalu waktu. 
    # Ini memastikan data per sesi tidak tercampur dan urutan waktunya kronologis.
    df_combined = df_combined.sort_values(by=['session_id', time_col]).reset_index(drop=True)
    
    # E. EKSPOR
    output_path = os.path.join(BASE_DIR, output_filename)
    df_combined.to_csv(output_path, index=False)
    
    # Format waktu untuk log
    min_time = df_combined[time_col].min().strftime('%Y-%m-%d %H:%M:%S')
    max_time = df_combined[time_col].max().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] SUCCESS: Exported to {output_filename}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] DETAILS: Temporal range [{min_time} to {max_time}]")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] DETAILS: Unique sessions detected: {df_combined['session_id'].nunique()}\n")


if __name__ == "__main__":
    start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SYSTEM: Starting Data Aggregation Pipeline V2.0")
    print(":" * 70)
    
    for folder, out_file in FOLDERS_TO_COMBINE.items():
        combine_folder_data(folder, out_file)
        
    print(":" * 70)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SYSTEM: Pipeline execution finished in {round(time.time() - start_time, 2)} seconds.")