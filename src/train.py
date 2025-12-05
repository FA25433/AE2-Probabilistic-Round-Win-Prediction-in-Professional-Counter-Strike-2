#!/usr/bin/env python3
"""
train.py - Model Training for CS2 Round Prediction
===================================================

This script trains all models: Baseline A, Baseline B (with/without Elo), 
and the main LightGBM model with isotonic calibration.

Usage:
    python train.py --config config.yaml

Output:
    - All trained models (.pkl, .txt files)
    - Training metrics and results (JSON)
"""

import pandas as pd
import numpy as np
import yaml
import argparse
import sys
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
import lightgbm as lgb

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_data(config):
    """Load and prepare data for training"""
    output_dir = Path(config['data']['output_dir'])
    rounds_file = output_dir / config['output_files']['rounds_processed']
    
    df = pd.read_csv(rounds_file)
    
    # Create event groups for CV
    unique_events = sorted(df['event_name'].unique())
    event_to_id = {event: i for i, event in enumerate(unique_events)}
    df['event_group'] = df['event_name'].map(event_to_id)
    
    # Train/test split
    n_groups = df['event_group'].nunique()
    n_test = max(1, int(n_groups * config['split']['test_size']))
    test_groups = list(range(n_groups - n_test, n_groups))
    
    train_mask = ~df['event_group'].isin(test_groups)
    df_train = df[train_mask].copy()
    df_test = df[~train_mask].copy()
    
    return df_train, df_test, df

def encode_features(df_train, df_test, categorical_features):
    """One-hot encode categorical features"""
    df_train_encoded = df_train.copy()
    df_test_encoded = df_test.copy()
    
    for cat_feat in categorical_features:
        train_dummies = pd.get_dummies(df_train[cat_feat], prefix=cat_feat, drop_first=True)
        test_dummies = pd.get_dummies(df_test[cat_feat], prefix=cat_feat, drop_first=True)
        
        # Ensure same columns
        for col in train_dummies.columns:
            if col not in test_dummies.columns:
                test_dummies[col] = 0
        
        df_train_encoded = pd.concat([df_train_encoded, train_dummies], axis=1)
        df_test_encoded = pd.concat([df_test_encoded, test_dummies], axis=1)
    
    return df_train_encoded, df_test_encoded

def train_baseline_a(df_train, df_test, config):
    """Train Baseline A: Map + Side win rates"""
    print("\n[Baseline A] Training Map + Side model...")
    
    target = config['features']['target']
    
    # Calculate win rates
    stats = df_train.groupby(['map', 'side'])[target].agg(['mean', 'count']).reset_index()
    stats.columns = ['map', 'side', 'win_rate', 'count']
    
    # Make predictions
    def predict(df, stats_table):
        predictions = []
        for _, row in df.iterrows():
            match = stats_table[(stats_table['map'] == row['map']) & 
                               (stats_table['side'] == row['side'])]
            predictions.append(match['win_rate'].values[0] if len(match) > 0 else 0.5)
        return np.array(predictions)
    
    y_pred_train = predict(df_train, stats)
    y_pred_test = predict(df_test, stats)
    
    # Evaluate
    results = {
        'train': {
            'log_loss': log_loss(df_train[target], y_pred_train),
            'brier': brier_score_loss(df_train[target], y_pred_train),
            'auc': roc_auc_score(df_train[target], y_pred_train),
            'accuracy': accuracy_score(df_train[target], (y_pred_train > 0.5).astype(int))
        },
        'test': {
            'log_loss': log_loss(df_test[target], y_pred_test),
            'brier': brier_score_loss(df_test[target], y_pred_test),
            'auc': roc_auc_score(df_test[target], y_pred_test),
            'accuracy': accuracy_score(df_test[target], (y_pred_test > 0.5).astype(int))
        }
    }
    
    print(f"  Test Log Loss: {results['test']['log_loss']:.4f}")
    print(f"  Test AUC: {results['test']['auc']:.4f}")
    
    return stats, y_pred_train, y_pred_test, results

def train_baseline_b(df_train, df_test, feature_cols, config, use_elo=False):
    """Train Baseline B: Logistic Regression"""
    model_name = "Baseline B+" if use_elo else "Baseline B"
    print(f"\n[{model_name}] Training Logistic Regression...")
    
    target = config['features']['target']
    
    # Select features
    if not use_elo:
        elo_features = config['features']['elo']
        feature_cols = [f for f in feature_cols if f not in elo_features]
    
    X_train = df_train[feature_cols].values
    X_test = df_test[feature_cols].values
    y_train = df_train[target].values
    y_test = df_test[target].values
    
    # Train model
    lr_config = config['models']['baseline_b']
    model = LogisticRegression(
        penalty=lr_config['penalty'],
        C=lr_config['C'],
        max_iter=lr_config['max_iter'],
        solver=lr_config['solver'],
        random_state=config['random_state']['seed']
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    results = {
        'train': {
            'log_loss': log_loss(y_train, y_pred_train),
            'brier': brier_score_loss(y_train, y_pred_train),
            'auc': roc_auc_score(y_train, y_pred_train),
            'accuracy': accuracy_score(y_train, (y_pred_train > 0.5).astype(int))
        },
        'test': {
            'log_loss': log_loss(y_test, y_pred_test),
            'brier': brier_score_loss(y_test, y_pred_test),
            'auc': roc_auc_score(y_test, y_pred_test),
            'accuracy': accuracy_score(y_test, (y_pred_test > 0.5).astype(int))
        }
    }
    
    print(f"  Test Log Loss: {results['test']['log_loss']:.4f}")
    print(f"  Test AUC: {results['test']['auc']:.4f}")
    
    return model, y_pred_train, y_pred_test, results

def train_lightgbm(df_train, df_test, feature_cols, config):
    """Train LightGBM with isotonic calibration"""
    print("\n[LightGBM] Training main model...")
    
    target = config['features']['target']
    
    X_train = df_train[feature_cols].values
    X_test = df_test[feature_cols].values
    y_train = df_train[target].values
    y_test = df_test[target].values
    
    # Create validation split
    val_group = df_train['event_group'].max()
    val_mask = df_train['event_group'] == val_group
    
    X_train_lgb = X_train[~val_mask]
    y_train_lgb = y_train[~val_mask]
    X_val = X_train[val_mask]
    y_val = y_train[val_mask]
    
    print(f"  Training: {len(X_train_lgb):,} samples")
    print(f"  Validation: {len(X_val):,} samples")
    
    # Train LightGBM
    lgb_config = config['models']['lightgbm']
    train_data = lgb.Dataset(X_train_lgb, label=y_train_lgb, feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_data)
    
    evals_result = {}
    lgb_model = lgb.train(
        lgb_config,
        train_data,
        num_boost_round=lgb_config['n_estimators'],
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=lgb_config['early_stopping_rounds'], verbose=False),
            lgb.record_evaluation(evals_result)
        ]
    )
    
    print(f"  Best iteration: {lgb_model.best_iteration}")
    print(f"  Validation log loss: {evals_result['valid']['binary_logloss'][lgb_model.best_iteration-1]:.4f}")
    
    # Predictions (uncalibrated)
    y_pred_train_uncal = lgb_model.predict(X_train, num_iteration=lgb_model.best_iteration)
    y_pred_test_uncal = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    
    # Isotonic calibration
    print("  Applying isotonic calibration...")
    y_pred_val = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    isotonic = IsotonicRegression(out_of_bounds='clip')
    isotonic.fit(y_pred_val, y_val)
    
    # Calibrated predictions
    y_pred_train_cal = isotonic.transform(y_pred_train_uncal)
    y_pred_test_cal = isotonic.transform(y_pred_test_uncal)
    
    # Evaluate
    results_uncal = {
        'train': {
            'log_loss': log_loss(y_train, y_pred_train_uncal),
            'brier': brier_score_loss(y_train, y_pred_train_uncal),
            'auc': roc_auc_score(y_train, y_pred_train_uncal),
            'accuracy': accuracy_score(y_train, (y_pred_train_uncal > 0.5).astype(int))
        },
        'test': {
            'log_loss': log_loss(y_test, y_pred_test_uncal),
            'brier': brier_score_loss(y_test, y_pred_test_uncal),
            'auc': roc_auc_score(y_test, y_pred_test_uncal),
            'accuracy': accuracy_score(y_test, (y_pred_test_uncal > 0.5).astype(int))
        }
    }
    
    results_cal = {
        'train': {
            'log_loss': log_loss(y_train, y_pred_train_cal),
            'brier': brier_score_loss(y_train, y_pred_train_cal),
            'auc': roc_auc_score(y_train, y_pred_train_cal),
            'accuracy': accuracy_score(y_train, (y_pred_train_cal > 0.5).astype(int))
        },
        'test': {
            'log_loss': log_loss(y_test, y_pred_test_cal),
            'brier': brier_score_loss(y_test, y_pred_test_cal),
            'auc': roc_auc_score(y_test, y_pred_test_cal),
            'accuracy': accuracy_score(y_test, (y_pred_test_cal > 0.5).astype(int))
        }
    }
    
    print(f"  Test Log Loss (uncalibrated): {results_uncal['test']['log_loss']:.4f}")
    print(f"  Test Log Loss (calibrated): {results_cal['test']['log_loss']:.4f}")
    print(f"  Test AUC: {results_cal['test']['auc']:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': lgb_model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    return lgb_model, isotonic, results_uncal, results_cal, feature_importance

def main(config_path='config.yaml'):
    """Main training pipeline"""
    
    print("=" * 80)
    print("CS2 MODEL TRAINING - train.py")
    print("=" * 80)
    
    # Load config
    print("\n[1/7] Loading configuration...")
    config = load_config(config_path)
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Set random seed
    np.random.seed(config['random_state']['seed'])
    print(f" Random seed: {config['random_state']['seed']}")
    
    # Prepare data
    print("\n[2/7] Preparing data...")
    df_train, df_test, df_all = prepare_data(config)
    print(f" Train: {len(df_train):,} rounds")
    print(f" Test: {len(df_test):,} rounds")
    
    # Encode categorical features
    print("\n[3/7] Encoding features...")
    categorical_features = config['features']['categorical']
    df_train_enc, df_test_enc = encode_features(df_train, df_test, categorical_features)
    
    # Get feature columns
    numeric_features = [f for f in config['features']['freeze_time'] 
                       if f not in categorical_features]
    encoded_cat = [col for col in df_train_enc.columns 
                   if any(col.startswith(f"{cat}_") for cat in categorical_features)]
    feature_cols_freeze = numeric_features + encoded_cat
    feature_cols_full = feature_cols_freeze + config['features']['elo']
    
    print(f" Features (freeze-only): {len(feature_cols_freeze)}")
    print(f" Features (with Elo): {len(feature_cols_full)}")
    
    # Train models
    print("\n[4/7] Training Baseline A...")
    baseline_a_stats, _, _, baseline_a_results = train_baseline_a(
        df_train, df_test, config)
    
    print("\n[5/7] Training Baseline B (No Elo)...")
    baseline_b_model, _, _, baseline_b_results = train_baseline_b(
        df_train_enc, df_test_enc, feature_cols_freeze, config, use_elo=False)
    
    print("\n[6/7] Training Baseline B+ (With Elo)...")
    baseline_b_plus_model, _, _, baseline_b_plus_results = train_baseline_b(
        df_train_enc, df_test_enc, feature_cols_full, config, use_elo=True)
    
    print("\n[7/7] Training LightGBM...")
    lgb_model, isotonic, lgb_uncal_results, lgb_cal_results, feature_importance = train_lightgbm(
        df_train_enc, df_test_enc, feature_cols_full, config)
    
    # Save models
    print("\n" + "=" * 80)
    print("SAVING MODELS")
    print("=" * 80)
    
    baseline_a_stats.to_csv(output_dir / config['output_files']['baseline_a_stats'], index=False)
    print(f" Saved: {config['output_files']['baseline_a_stats']}")
    
    joblib.dump(baseline_b_model, output_dir / config['output_files']['baseline_b_model'])
    print(f" Saved: {config['output_files']['baseline_b_model']}")
    
    joblib.dump(baseline_b_plus_model, output_dir / config['output_files']['baseline_b_plus_model'])
    print(f" Saved: {config['output_files']['baseline_b_plus_model']}")
    
    lgb_model.save_model(str(output_dir / config['output_files']['lightgbm_model']))
    print(f" Saved: {config['output_files']['lightgbm_model']}")
    
    joblib.dump(isotonic, output_dir / config['output_files']['isotonic_calibrator'])
    print(f" Saved: {config['output_files']['isotonic_calibrator']}")
    
    feature_importance.to_csv(output_dir / config['output_files']['feature_importance'], index=False)
    print(f" Saved: {config['output_files']['feature_importance']}")
    
    # Save results
    import json
    all_results = {
        'baseline_a': baseline_a_results,
        'baseline_b': baseline_b_results,
        'baseline_b_plus': baseline_b_plus_results,
        'lightgbm_uncalibrated': lgb_uncal_results,
        'lightgbm_calibrated': lgb_cal_results
    }
    
    with open(output_dir / config['output_files']['results_json'], 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f" Saved: {config['output_files']['results_json']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print("\nTest Set Performance:")
    print(f"  Baseline A:    Log Loss = {baseline_a_results['test']['log_loss']:.4f}")
    print(f"  Baseline B:    Log Loss = {baseline_b_results['test']['log_loss']:.4f}")
    print(f"  Baseline B+:   Log Loss = {baseline_b_plus_results['test']['log_loss']:.4f}")
    print(f"  LightGBM:      Log Loss = {lgb_uncal_results['test']['log_loss']:.4f}")
    print(f"  LightGBM Cal:  Log Loss = {lgb_cal_results['test']['log_loss']:.4f}")
    
    print("\n Ready for evaluation (evaluate.py)")
    print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CS2 round prediction models')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    
    args = parser.parse_args()
    
    try:
        main(args.config)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
