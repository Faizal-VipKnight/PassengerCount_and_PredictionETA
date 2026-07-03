"""
Script: Model Comparison for ETA Prediction (MLR vs Random Forest vs XGBoost FINAL v3.2)
Fungsi: Melatih 3 model berbeda dengan dataset yang sama, membandingkan metrik 
        evaluasi (MAE, RMSE, MAPE, R²), dan men-generate grafik komparasi untuk Bab 4.
        FIX: Sinkron dengan Feature Engineering v3.2 (10 Fitur).
Author: Faizal Adi Purwoko
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from datetime import datetime

# Import Model dari Scikit-Learn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. KONFIGURASI PATH & PARAMETER
BASE_DIR = #path masukan disini

PATH_TRAIN = os.path.join(BASE_DIR, "Segments_Training.csv")
PATH_VAL   = os.path.join(BASE_DIR, "Segments_Val.csv")
PATH_TEST  = os.path.join(BASE_DIR, "Segments_Test.csv")

# UPDATE: 10 Fitur (Tambah 'is_departure_from_rusun')
FEATURES = [
    'route_distance_m', 'max_passengers', 'hour_of_day', 'day_of_week', 
    'is_weekend', 'is_peak_hour', 'from_halte_idx', 'to_halte_idx', 'is_rusun',
    'is_departure_from_rusun' 
]
TARGET = 'travel_time_sec'

def log_print(msg, level="INFO"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {level}: {msg}")

def calculate_metrics(y_true, y_pred):
    """Fungsi helper untuk menghitung MAE, RMSE, MAPE, dan R²"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    epsilon = 1e-8 # Anti pembagian dengan nol
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    r2 = r2_score(y_true, y_pred)  # BARU: Koefisien Determinasi
    return mae, rmse, mape, r2

# 2. LOAD DATA
if __name__ == "__main__":
    t0_global = time.time()
    print("="*75)
    print("SYSTEM: Initiating Multi-Model Comparison Pipeline (MLR vs RF vs XGB FINAL v3.2)")
    print("="*75)
    
    log_print("Loading datasets...")
    df_train = pd.read_csv(PATH_TRAIN)
    df_val   = pd.read_csv(PATH_VAL)
    df_test  = pd.read_csv(PATH_TEST)
    
    X_train, y_train = df_train[FEATURES], df_train[TARGET]
    X_val,   y_val   = df_val[FEATURES],   df_val[TARGET]
    X_test,  y_test  = df_test[FEATURES],  df_test[TARGET]
    
    log_print(f"Data loaded. Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 3. TRAINING MODEL 1: MULTIPLE LINEAR REGRESSION (MLR)
    log_print("Training Model 1: Multiple Linear Regression (MLR)")
    t0_mlr = time.time()
    model_mlr = LinearRegression()
    model_mlr.fit(X_train, y_train)
    log_print(f"MLR trained in {time.time() - t0_mlr:.2f}s")

    # 4. TRAINING MODEL 2: RANDOM FOREST REGRESSOR (RF)
    log_print("Training Model 2: Random Forest Regressor (RF)")
    t0_rf = time.time()
    model_rf = RandomForestRegressor(
        n_estimators=100, 
        max_depth=12, 
        random_state=42, 
        n_jobs=-1 # Pakai semua core CPU
    )
    model_rf.fit(X_train, y_train)
    log_print(f"Random Forest trained in {time.time() - t0_rf:.2f}s")

    # 5. TRAINING MODEL 3: XGBOOST (NATIVE API)
    log_print("Training Model 3: XGBoost (Native API)...")
    t0_xgb = time.time()
    
    # Konversi ke DMatrix (Wajib untuk Native API)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val, label=y_val)
    dtest  = xgb.DMatrix(X_test, label=y_test)
    
    # Deteksi versi XGBoost untuk GPU
    xgb_version = tuple(map(int, xgb.__version__.split('.')[:2]))
    
    # UPDATE: Gunakan absoluteerror agar MAPE konsisten dengan model utama (39%)
    params = {
        'objective': 'reg:absoluteerror',  # Anti outlier (ngetem ekstrem)
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
    else:
        params['tree_method'] = 'gpu_hist'
        
    evallist = [(dtrain, 'train'), (dval, 'validation')]
    
    # Training dengan Early Stopping
    model_xgb = xgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        evals=evallist,
        early_stopping_rounds=150,
        verbose_eval=False # Matikan print agar tidak spam di terminal
    )
    log_print(f"XGBoost trained in {time.time() - t0_xgb:.2f}s")

    # 6. EVALUASI & INFERENCE (TEST SET)
    log_print("Executing inference on independent Test Set for all models...")
    
    y_pred_mlr = model_mlr.predict(X_test)
    y_pred_rf  = model_rf.predict(X_test)
    y_pred_xgb = model_xgb.predict(dtest) # Pakai dtest karena Native API
    
    # UPDATE: Sekarang return 4 nilai (mae, rmse, mape, r2)
    mae_mlr, rmse_mlr, mape_mlr, r2_mlr = calculate_metrics(y_test, y_pred_mlr)
    mae_rf,  rmse_rf,  mape_rf,  r2_rf  = calculate_metrics(y_test, y_pred_rf)
    mae_xgb, rmse_xgb, mape_xgb, r2_xgb = calculate_metrics(y_test, y_pred_xgb)

    # 7. TABEL PERBANDINGAN METRIK
    results = {
        'Model': ['Multiple Linear Regression', 'Random Forest', 'XGBoost (Proposed)'],
        'MAE (seconds)': [round(mae_mlr, 2), round(mae_rf, 2), round(mae_xgb, 2)],
        'RMSE (seconds)': [round(rmse_mlr, 2), round(rmse_rf, 2), round(rmse_xgb, 2)],
        'MAPE (%)': [round(mape_mlr, 2), round(mape_rf, 2), round(mape_xgb, 2)],
        'R² Score': [round(r2_mlr, 4), round(r2_rf, 4), round(r2_xgb, 4)],  # BARU
        'Training Time (s)': [
            round(time.time() - t0_mlr, 2), 
            round(time.time() - t0_rf, 2), 
            round(time.time() - t0_xgb, 2)
        ]
    }
    df_results = pd.DataFrame(results)
    
    print("\n" + ":"*75)
    print("MODEL EVALUATION COMPARISON (INDEPENDENT TEST SET)")
    print("="*75)
    print(df_results.to_string(index=False))
    print(":"*75 + "\n")
    
    # BARU: Print detail R² per model agar lebih jelas di terminal
    print("DETAIL R² (Koefisien Determinasi)")
    print(f"  MLR     : {r2_mlr:.4f}  → Model menjelaskan {r2_mlr*100:.2f}% variansi data")
    print(f"  RF      : {r2_rf:.4f}  → Model menjelaskan {r2_rf*100:.2f}% variansi data")
    print(f"  XGBoost : {r2_xgb:.4f}  → Model menjelaskan {r2_xgb*100:.2f}% variansi data")
    print("\n")
    
    # Save tabel ke CSV
    csv_path = os.path.join(BASE_DIR, 'modelbaru1_Tabel_Perbandingan_Model.csv')
    df_results.to_csv(csv_path, index=False)
    log_print(f"Comparison table saved to: {csv_path}", "SUCCESS")

    # 8. VISUALISASI 1: BAR CHART PERBANDINGAN METRIK
    log_print("Generating comparison bar charts...")
    sns.set_theme(style="whitegrid")
    
    # UPDATE: 4 subplot (tambah R²)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    colors = ['#3498db', '#2ecc71', '#e74c3c'] # Biru, Hijau, Merah
    
    # Fix warning seaborn dengan menambahkan hue
    sns.barplot(x='MAE (seconds)', y='Model', data=df_results, palette=colors, ax=axes[0], hue='Model', legend=False)
    axes[0].set_title('Perbandingan Mean Absolute Error (MAE)', fontweight='bold')
    axes[0].set_xlabel('Error (Detik) - Semakin kecil semakin baik')
    
    sns.barplot(x='RMSE (seconds)', y='Model', data=df_results, palette=colors, ax=axes[1], hue='Model', legend=False)
    axes[1].set_title('Perbandingan Root Mean Squared Error (RMSE)', fontweight='bold')
    axes[1].set_xlabel('Error (Detik) - Semakin kecil semakin baik')
    
    sns.barplot(x='MAPE (%)', y='Model', data=df_results, palette=colors, ax=axes[2], hue='Model', legend=False)
    axes[2].set_title('Perbandingan Mean Absolute Percentage Error (MAPE)', fontweight='bold')
    axes[2].set_xlabel('Error (%) - Semakin kecil semakin baik')
    
    # UPDATE: Bar chart R² Score
    sns.barplot(x='R² Score', y='Model', data=df_results, palette=colors, ax=axes[3], hue='Model', legend=False)
    axes[3].set_title('Perbandingan R² Score (Koefisien Determinasi)', fontweight='bold')
    axes[3].set_xlabel('R² - Semakin dekat 1 semakin baik')
    # Tambahkan garis referensi R² = 0
    axes[3].axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='R² = 0 (baseline)')
    axes[3].legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'modelbaru1_BarChart_Perbandingan_Model.png'), dpi=300)
    plt.close()

    # 9. VISUALISASI 2: SCATTER PLOT ACTUAL vs PREDICTED
    log_print("Generating Actual vs Predicted scatter plots")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    def plot_scatter(ax, y_true, y_pred, model_name, color, r2_val):
        ax.scatter(y_true, y_pred, alpha=0.5, color=color, edgecolor='k', s=40)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
        
        mae_val = mean_absolute_error(y_true, y_pred)
        # UPDATE: Tampilkan R² di subtitle
        ax.set_title(f'{model_name}\n(MAE: {mae_val:.1f}s | R²: {r2_val:.4f})', fontweight='bold')
        ax.set_xlabel('Waktu Aktual (Detik)')
        ax.set_ylabel('Waktu Prediksi (Detik)')
        ax.legend(loc='upper left')
        ax.grid(True, linestyle=':', alpha=0.6)

    plot_scatter(axes[0], y_test, y_pred_mlr, 'Multiple Linear Regression', '#3498db', r2_mlr)
    plot_scatter(axes[1], y_test, y_pred_rf, 'Random Forest', '#2ecc71', r2_rf)
    plot_scatter(axes[2], y_test, y_pred_xgb, 'XGBoost (Proposed)', '#e74c3c', r2_xgb)
    
    plt.suptitle('Komparasi Scatter Plot: Waktu Aktual vs Prediksi (Test Set)', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'modelbaru1_ScatterPlot_Perbandingan_Model.png'), dpi=300, bbox_inches='tight')
    plt.close()

    log_print("All analytical plots generated successfully!", "SUCCESS")
    print(":" * 75)
    print(f"SYSTEM: Pipeline Completed in {time.time() - t0_global:.2f} seconds.")