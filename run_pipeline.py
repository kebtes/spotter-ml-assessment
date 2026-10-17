"""
End-to-end execution pipeline for Spotter Freight Rate Assessment.
Trains models, evaluates on Out-of-Time validation split, predicts validation set,
generates December chart, and runs score.py validation.
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_PATH,
    VAL_PATH,
    DECEMBER_PATH,
    TEMPLATE_PATH,
    OUTPUT_DIR,
    PREDICTIONS_OUTPUT_PATH,
    ALL_MODEL_FEATURES,
)
from src.data import (
    load_raw_data,
    extract_city_coordinates,
    prepare_december_inputs,
    get_time_based_split,
)
from src.features import engineer_features, prepare_feature_matrices
from src.models import (
    DomainBaselineModel,
    LightGBMRateModel,
    CatBoostRateModel,
    FreightRateEnsemble,
)
from src.evaluate import calculate_metrics, format_metrics_table


def main():
    print("=" * 70)
    print("SPOTTER FREIGHT RATE PREDICTION - END-TO-END PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print("\n[Step 1/6] Loading datasets...")
    df_train_raw, df_val_raw, df_dec_raw = load_raw_data()
    print(f"  Training set loads:    {len(df_train_raw):,} (dates: {df_train_raw['date'].min()} to {df_train_raw['date'].max()})")
    print(f"  Validation set loads:  {len(df_val_raw):,} (dates: {df_val_raw['date'].min()} to {df_val_raw['date'].max()})")
    print(f"  December test inputs:  {len(df_dec_raw):,} (dates: {df_dec_raw['date'].min()} to {df_dec_raw['date'].max()})")

    # 2. Coordinate Extraction & Feature Engineering
    print("\n[Step 2/6] Engineering geospatial, temporal, and market features...")
    city_coords = extract_city_coordinates(df_train_raw)
    
    df_train_fe = engineer_features(df_train_raw, city_coords)
    df_val_fe = engineer_features(df_val_raw, city_coords)
    
    # Enrich December chart inputs with coordinates and daily December signals
    df_dec_enriched = prepare_december_inputs(df_dec_raw, df_val_raw)
    df_dec_fe = engineer_features(df_dec_enriched, city_coords)

    # 3. Out-Of-Time (OOT) Validation Split Benchmarking
    print("\n[Step 3/6] Running Out-Of-Time (OOT) Validation Benchmark (Train: Jan-Aug | Val: Sep-Oct)...")
    train_split_raw, val_split_raw = get_time_based_split(df_train_fe, split_date="2025-09-01")
    
    X_oot_train, y_oot_train = prepare_feature_matrices(train_split_raw)
    X_oot_val, y_oot_val = prepare_feature_matrices(val_split_raw)
    
    print(f"  OOT Training split:    {len(X_oot_train):,} loads")
    print(f"  OOT Validation split:  {len(X_oot_val):,} loads")

    benchmark_results = {}

    # Model 1: Baseline
    print("  Training Domain Baseline...")
    baseline = DomainBaselineModel()
    baseline.fit(X_oot_train, y_oot_train)
    preds_base = baseline.predict(X_oot_val)
    benchmark_results["Domain Baseline (Distance x Quote)"] = calculate_metrics(y_oot_val, preds_base)

    # Model 2: LightGBM
    print("  Training LightGBM Regressor (MAE objective)...")
    lgb_oot = LightGBMRateModel(n_estimators=1000, learning_rate=0.03)
    lgb_oot.fit(X_oot_train, y_oot_train)
    preds_lgb = lgb_oot.predict(X_oot_val)
    benchmark_results["LightGBM Regressor"] = calculate_metrics(y_oot_val, preds_lgb)

    # Model 3: CatBoost
    print("  Training CatBoost Regressor (MAE objective)...")
    cb_oot = CatBoostRateModel(iterations=800, learning_rate=0.06)
    cb_oot.fit(X_oot_train, y_oot_train)
    preds_cb = cb_oot.predict(X_oot_val)
    benchmark_results["CatBoost Regressor"] = calculate_metrics(y_oot_val, preds_cb)

    # Model 4: Ensemble
    print("  Evaluating Ensemble (0.55 LightGBM + 0.45 CatBoost)...")
    preds_ens = 0.55 * preds_lgb + 0.45 * preds_cb
    benchmark_results["Ensemble (LightGBM + CatBoost)"] = calculate_metrics(y_oot_val, preds_ens)

    metrics_df = format_metrics_table(benchmark_results)
    print("\n" + "-" * 70)
    print("OUT-OF-TIME VALIDATION BENCHMARK RESULTS (Sep-Oct 2025):")
    print("-" * 70)
    print(metrics_df.to_string())
    print("-" * 70)

    # 4. Train Champion Ensemble on Full Training Set (Jan-Oct 2025)
    print("\n[Step 4/6] Retraining Champion Ensemble on Full Development Dataset (48,000 loads)...")
    X_full_train, y_full_train = prepare_feature_matrices(df_train_fe)
    
    champion_ensemble = FreightRateEnsemble(lgb_weight=0.55, cb_weight=0.45)
    champion_ensemble.fit(X_full_train, y_full_train)
    print("  Champion ensemble successfully trained on full data.")

    # 5. Predict on Validation Set & December Stress Test
    print("\n[Step 5/6] Generating final predictions...")
    X_val, _ = prepare_feature_matrices(df_val_fe)
    val_predictions = champion_ensemble.predict(X_val)

    # Format validation_predictions.csv
    template = pd.read_csv(TEMPLATE_PATH)
    template["predicted_rate"] = np.round(val_predictions, 2)
    
    # Save validation_predictions.csv to root and data/
    template.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    template.to_csv(PROJECT_ROOT / "data" / "validation_predictions.csv", index=False)
    print(f"  Saved {len(template):,} predictions to: {PREDICTIONS_OUTPUT_PATH}")

    # Predict December inputs
    X_dec, _ = prepare_feature_matrices(df_dec_fe)
    dec_predictions = champion_ensemble.predict(X_dec)
    
    # Update december-chart-inputs.csv with predicted_rate
    df_dec_raw["predicted_rate"] = np.round(dec_predictions, 2)
    df_dec_raw.to_csv(DECEMBER_PATH, index=False)
    df_dec_raw.to_csv(PROJECT_ROOT / "data" / "december_chart_inputs.csv", index=False)
    df_dec_raw.to_csv(PROJECT_ROOT / "december-chart-inputs.csv", index=False)
    print(f"  Saved 31 daily December predictions to: {DECEMBER_PATH}")

    # 6. Run score.py Scorer Validation & Generate Chart
    print("\n[Step 6/6] Executing official score.py validation...")
    score_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "score.py"),
        "--predictions", str(PREDICTIONS_OUTPUT_PATH),
        "--december-predictions", str(DECEMBER_PATH),
        "--output-dir", str(OUTPUT_DIR),
    ]
    result = subprocess.run(score_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR running score.py:")
        print(result.stderr)
        sys.exit(1)

    print("\n" + "=" * 70)
    print("ALL DELIVERABLES SUCCESSFULLY GENERATED AND VERIFIED!")
    print(f"  1. Validation Predictions: {PREDICTIONS_OUTPUT_PATH}")
    print(f"  2. December Chart:         {OUTPUT_DIR / 'candidate_december.png'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
