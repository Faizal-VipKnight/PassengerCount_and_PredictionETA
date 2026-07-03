"""
Script: XGBoost ETA Prediction Model Training (FINAL v3.2)
Fungsi: Melatih model prediksi waktu tempuh dengan hyperparameter optimal.
        Menggunakan reg:squarederror/absoluteerror untuk memprediksi rata-rata waktu tempuh
        termasuk pola ngetem di Rusunawa.
Author: Faizal Adi Purwoko
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
import joblib

# 1. KONFIGURASI PATH
BASE_DIR = #masukan path datanya
PATH_TRAIN = os.path.join(BASE_DIR, "Segments_Training.csv")
PATH_VAL   = os.path.join(BASE_DIR, "Segments_Val.csv")
PATH_TEST  = os.path.join(BASE_DIR, "Segments_Test.csv")
MODEL_SAVE = os.path.join(BASE_DIR, "xgboost_eta_model_ultimate.json")

# 10 feature tambahan depature for rusun
FEATURES = [
    'route_distance_m', 'max_passengers', 'hour_of_day', 'day_of_week',
    'is_weekend', 'is_peak_hour', 'from_halte_idx', 'to_halte_idx', 'is_rusun',
    'is_departure_from_rusun'
]
TARGET = 'travel_time_sec'

def log_print(msg, level="INFO"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {level}: {msg}")

def validate_data(df, name):
    print(f"\n{'='*60}")
    print(f"Validasi Dataset: {name}")
    print('='*60)
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols: 
        raise ValueError(f"[{name}] Kolom hilang: {missing_cols}")
    
    nan_count = df[FEATURES + [TARGET]].isna().sum().sum()
    if nan_count > 0:
        log_print(f"[{name}] Ditemukan {nan_count} nilai NaN, akan di-drop", "WARNING")
        df = df.dropna(subset=FEATURES + [TARGET])
    
    print(f"  Total records: {len(df)}")
    print(f"  Rata-rata travel time: {df[TARGET].mean():.1f} detik")
    return df

if __name__ == "__main__":
    t0_global = time.time()
    print("="*80)
    print(" SYSTEM: Initiating XGBoost ETA Model Training (FINAL v3.2)")
    print("="*80)
    
    # Cek file ada atau tidak
    for path, name in [(PATH_TRAIN, 'Training'), (PATH_VAL, 'Validation'), (PATH_TEST, 'Test')]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {name} tidak ditemukan: {path}\nJalankan Feature Engineering terlebih dahulu!")
        log_print(f"File {name} ditemukan: {os.path.basename(path)}")
    
    log_print("Loading segmented routing datasets...")
    df_train = validate_data(pd.read_csv(PATH_TRAIN), 'Training')
    df_val   = validate_data(pd.read_csv(PATH_VAL), 'Validation')
    df_test  = validate_data(pd.read_csv(PATH_TEST), 'Test')

    X_train, y_train = df_train[FEATURES], df_train[TARGET]
    X_val,   y_val   = df_val[FEATURES],   df_val[TARGET]
    X_test,  y_test  = df_test[FEATURES],  df_test[TARGET]

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=False)
    dval   = xgb.DMatrix(X_val,  label=y_val,  enable_categorical=False)
    dtest  = xgb.DMatrix(X_test, label=y_test, enable_categorical=False)

    log_print("Starting GPU-Accelerated Training (CUDA)...")
    t0_train = time.time()

    xgb_version = tuple(map(int, xgb.__version__.split('.')[:2]))
    log_print(f"XGBoost version: {xgb.__version__}")
    
    # PARAMETER FINAL v3.2
    params = {
        # 'objective': 'reg:squarederror',  #  (prediksi mean/rata-rata)
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'learning_rate': 0.005,
        'max_depth': 8,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.2,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'seed': 42 
    }
    
    if xgb_version >= (2, 0):
        params['tree_method'] = 'hist'
        params['device'] = 'cuda'
        log_print("Using XGBoost 2.0+ syntax: tree_method='hist', device='cuda'", "SUCCESS")
    else:
        params['tree_method'] = 'gpu_hist'
        log_print(f"Using XGBoost {xgb.__version__} syntax: tree_method='gpu_hist'", "SUCCESS")

    evallist = [(dtrain, 'train'), (dval, 'validation')]
    
    log_print("Memulai training 10.000 putaran (dengan Early Stopping)...")
    
    model = xgb.train(
        params, dtrain, num_boost_round=10000, evals=evallist,
        early_stopping_rounds=150, verbose_eval=200
    )

    elapsed_train = time.time() - t0_train
    log_print(f"Training concluded. Duration: {elapsed_train/60:.2f} minutes.", "SUCCESS")
    log_print(f"Best iteration: {model.best_iteration} (score: {model.best_score:.2f})", "SUCCESS")

    log_print("Executing model inference on independent Test Set...")
    y_pred = model.predict(dtest)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    r2 = r2_score(y_test, y_pred)
    
    train_mae = mean_absolute_error(y_train, model.predict(dtrain))
    val_mae = mean_absolute_error(y_val, model.predict(dval))

    print("\n" + "="*80)
    print(" MODEL EVALUATION METRICS")
    print("="*80)
    print(f" Train MAE : {train_mae:.2f} detik")
    print(f" Val MAE   : {val_mae:.2f} detik")
    print(f" Test MAE  : {mae:.2f} detik")
    print(f" Test RMSE : {rmse:.2f} detik")
    print(f" Test MAPE : {mape:.2f} %")
    print(f" Test R²   : {r2:.4f}")
    print("="*80 + "\n")

    log_print("Saving model architecture and weights...")
    model.save_model(MODEL_SAVE)
    log_print(f"Model saved to: {MODEL_SAVE}", "SUCCESS")
    joblib.dump(model, os.path.join(BASE_DIR, "xgboost_model.pkl"))
    
    # Plot 1: Feature Importance
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(model, importance_type='gain', max_num_features=len(FEATURES), height=0.6,
                        title='Feature Importance (Gain)', xlabel='Gain (Peningkatan Akurasi)', ylabel='Fitur')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'new_Feature_Importance_Ultimate.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Komparasi Aktual vs Prediksi
    plt.figure(figsize=(14, 6))
    subset_size = min(200, len(y_test))
    indices = np.arange(subset_size)
    plt.plot(indices, y_test.values[:subset_size], label='Waktu Aktual (Ground Truth)', color='blue', marker='o', alpha=0.6, markersize=4)
    plt.plot(indices, y_pred[:subset_size], label='Prediksi XGBoost (Ultimate)', color='red', linestyle='--', marker='x', alpha=0.7, markersize=4)
    plt.title('Komparasi Waktu Tempuh Aktual vs Prediksi ETA (Data Uji)', fontsize=14, fontweight='bold')
    plt.xlabel('Indeks Segmen Perjalanan (Test Set)', fontsize=12)
    plt.ylabel('Waktu Tempuh (Detik)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'new_Kurva_Komparasi_Ultimate.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Scatter Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Waktu Aktual (Detik)', fontsize=12)
    plt.ylabel('Waktu Prediksi (Detik)', fontsize=12)
    plt.title('Scatter Plot: Aktual vs Prediksi', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'new_ScatterPlot_Ultimate.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: Error Distribution
    errors = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    plt.xlabel('Error (Aktual - Prediksi) [Detik]', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribusi Error Prediksi', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'new_Error_Distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "="*80)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SYSTEM: Pipeline terminated successfully.")
    print("="*80)
    print(f"\n Output files:")
    print(f"  - Model: {MODEL_SAVE}")
    print(f"  - Plots: new_Feature_Importance_Ultimate.png")
    print(f"           new_Kurva_Komparasi_Ultimate.png")
    print(f"           new_ScatterPlot_Ultimate.png")
    print(f"           new_Error_Distribution.png")
    print("="*80)