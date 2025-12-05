# ============================================================================
# PROJECT: Probabilistic Round-Win Prediction in CS2 (AE2)
# AUTHOR: FA25433
# TYPE: Implementation Pipeline (Python Script)
# ============================================================================

import os
import sys
import subprocess
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import lightgbm as lgb
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
from sklearn.model_selection import GroupKFold

# Ignore standard warnings to keep the output clean for presentation
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & PATH SETUP (CRITICAL FOR GITHUB)
# ============================================================================
print("=" * 80)
print("STEP 0: INITIALISATION AND PATH SETUP")
print("=" * 80)

# Define directories using relative paths
# 'data' folder for inputs, 'progression' folder for outputs
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'progression')

# Create the output directory if it does not exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f" Created directory: {OUTPUT_DIR}")

# Define file paths
# Note: Ensure 'matches.xlsx' and 'freeze_time_features.csv' are in the 'data' folder
FREEZE_TIME_FILE = os.path.join(DATA_DIR, 'freeze_time_features.csv')
MATCHES_FILE = os.path.join(DATA_DIR, 'matches.xlsx')

print(f" Working Directory: {BASE_DIR}")
print(f" Input Data:        {DATA_DIR}")
print(f" Output Progression:{OUTPUT_DIR}")

# ============================================================================
# STEP 1: LOADING DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING RAW DATASETS")
print("=" * 80)

# Load the freeze-time features (round-by-round data)
print(f"Loading rounds data from: {FREEZE_TIME_FILE}")
if os.path.exists(FREEZE_TIME_FILE):
    df_rounds = pd.read_csv(FREEZE_TIME_FILE)
    print(f" Loaded {len(df_rounds):,} rows from freeze_time_features.csv")
else:
    print(f" ERROR: Could not find {FREEZE_TIME_FILE}")
    sys.exit(1) # Stop the programme if data is missing

# Load the match metadata (for Elo calculation)
print(f"\nLoading matches metadata from: {MATCHES_FILE}")
if os.path.exists(MATCHES_FILE):
    df_matches = pd.read_excel(MATCHES_FILE)
    print(f" Loaded {len(df_matches):,} rows from matches.xlsx")
    # Ensure time column is datetime format for chronological sorting
    df_matches['Time'] = pd.to_datetime(df_matches['Time'])
else:
    print(f" ERROR: Could not find {MATCHES_FILE}")
    sys.exit(1)

# ============================================================================
# STEP 2: ELO RATING CONSTRUCTION (INTEGRITY STEP)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: FEATURE ENGINEERING & ELO RATING CONSTRUCTION")
print("=" * 80)

# 2.1 Extract Series Type
def extract_series_type(maps_str):
    """Extract if match is Best-of-1, Best-of-3, etc."""
    if pd.isna(maps_str): return 1
    maps_str = str(maps_str).lower().strip()
    if 'bo5' in maps_str: return 5
    elif 'bo3' in maps_str: return 3
    elif 'bo2' in maps_str: return 2
    elif 'bo1' in maps_str: return 1
    else: return 3 # Default assumption

df_matches['series_type'] = df_matches['Maps'].apply(extract_series_type)

# 2.2 Prepare Data for Elo
# We create a clean, sorted dataframe for the Elo calculation loop
df_matches_elo = df_matches[['Match ID', 'Time', 'Event Name', 'Team 1', 'Team 2', 'Result 1', 'Result 2', 'series_type']].copy()
df_matches_elo.columns = ['match_id', 'match_time', 'event_name', 'team1', 'team2', 'score1', 'score2', 'series_type']
df_matches_elo = df_matches_elo.sort_values('match_time').reset_index(drop=True)

# 2.3 Elo Class Implementation
class EloRatingSystem:
    """
    Custom Elo system that supports 'Freezing' ratings before an event.
    This prevents data leakage by ensuring predictions only use past knowledge.
    """
    def __init__(self, k_factor=32, default_rating=1500):
        self.k_factor = k_factor
        self.default_rating = default_rating
        self.ratings = {}  # Live ratings
        self.event_ratings = {}  # Frozen ratings (The Integrity Lock)
        self.rating_history = []

    def get_rating(self, team):
        return self.ratings.get(team, self.default_rating)

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def freeze_rating_for_event(self, team, event):
        """Snapshots the rating at the start of the tournament."""
        key = (team, event)
        if key not in self.event_ratings:
            self.event_ratings[key] = self.get_rating(team)

    def get_pre_event_rating(self, team, event):
        """Retrieves the frozen rating to use for prediction."""
        key = (team, event)
        # Fallback logic if event not tracked yet
        if key not in self.event_ratings:
            self.event_ratings[key] = self.get_rating(team)
        
        rating = self.event_ratings[key]
        # Check if team is new (has history?)
        has_history = any(h['team_a'] == team or h['team_b'] == team for h in self.rating_history)
        return rating, 0 if has_history else 1

    def update_ratings(self, team_a, team_b, score_a, score_b, match_id, match_time):
        """Updates live ratings after a match concludes."""
        ra, rb = self.get_rating(team_a), self.get_rating(team_b)
        total = score_a + score_b
        actual_a = score_a / total if total > 0 else 0.5
        expected_a = self.expected_score(ra, rb)
        
        new_ra = ra + self.k_factor * (actual_a - expected_a)
        new_rb = rb + self.k_factor * ((1 - actual_a) - (1 - expected_a))
        
        self.ratings[team_a] = new_ra
        self.ratings[team_b] = new_rb
        
        self.rating_history.append({
            'match_id': match_id, 'team_a': team_a, 'team_b': team_b,
            'rating_a_after': new_ra, 'rating_b_after': new_rb
        })

# Initialise the system
elo_system = EloRatingSystem(k_factor=32, default_rating=1500)
processed_events = set()
match_elo_data = []

print(" Calculating chronological Elo ratings...")

# 2.4 Run the Chronological Loop
for idx, row in df_matches_elo.iterrows():
    event = row['event_name']
    
    # FREEZE LOGIC: If this is a new event, snapshot everyone's ratings
    if event not in processed_events:
        event_matches = df_matches_elo[df_matches_elo['event_name'] == event]
        event_teams = set(event_matches['team1'].tolist() + event_matches['team2'].tolist())
        for team in event_teams:
            elo_system.freeze_rating_for_event(team, event)
        processed_events.add(event)

    # Get the SAFE ratings for this match
    t1_elo, t1_miss = elo_system.get_pre_event_rating(row['team1'], event)
    t2_elo, t2_miss = elo_system.get_pre_event_rating(row['team2'], event)

    match_elo_data.append({
        'match_id': row['match_id'],
        'team1': row['team1'],
        'team2': row['team2'],
        'team1_elo_pre_event': t1_elo,
        'team2_elo_pre_event': t2_elo,
        'elo_diff_team1': t1_elo - t2_elo,
        'team1_elo_missing': t1_miss,
        'team2_elo_missing': t2_miss
    })

    # Update live ratings (for the future)
    elo_system.update_ratings(row['team1'], row['team2'], row['score1'], row['score2'], row['match_id'], row['match_time'])

df_match_elo = pd.DataFrame(match_elo_data)
print(f" Ratings calculated for {len(df_match_elo)} matches.")

# 2.5 Merge Elo into Round Data
print(" Merging Elo ratings into round data...")
# Optimised mapping using dictionary for speed
elo_lookup = {row['match_id']: row for _, row in df_match_elo.iterrows()}

# Lists to construct new columns
team_elos, opp_elos, elo_diffs, elo_missings = [], [], [], []

for _, row in df_rounds.iterrows():
    m_data = elo_lookup.get(row['match_id'])
    if m_data is None:
        # Default if match missing (shouldn't happen)
        team_elos.append(1500); opp_elos.append(1500); elo_diffs.append(0); elo_missings.append(1)
        continue
        
    if row['team_name'] == m_data['team1']:
        team_elos.append(m_data['team1_elo_pre_event'])
        opp_elos.append(m_data['team2_elo_pre_event'])
        elo_diffs.append(m_data['elo_diff_team1'])
        elo_missings.append(m_data['team1_elo_missing'])
    else:
        team_elos.append(m_data['team2_elo_pre_event'])
        opp_elos.append(m_data['team1_elo_pre_event'])
        elo_diffs.append(-m_data['elo_diff_team1']) # Flip the difference
        elo_missings.append(m_data['team2_elo_missing'])

df_rounds['team_elo_pre_event'] = team_elos
df_rounds['opp_elo_pre_event'] = opp_elos
df_rounds['elo_diff'] = elo_diffs
df_rounds['elo_missing'] = elo_missings

# Merge Event Name for grouping
df_rounds = df_rounds.merge(df_matches[['Match ID', 'Event Name']].rename(columns={'Match ID':'match_id', 'Event Name':'event_name'}), on='match_id', how='left')

# Save intermediate file to progression
df_rounds.to_csv(os.path.join(OUTPUT_DIR, 'rounds_with_elo.csv'), index=False)
print(" Elo Integration Complete. Saved 'rounds_with_elo.csv'.")

# ============================================================================
# STEP 3: DATA PREPARATION & SPLITTING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: PREPARING MATRICES FOR TRAINING")
print("=" * 80)

# Feature Definitions
FEATURES_NUMERIC = [
    'score_diff', 'start_cash', 'loss_bonus', 'consec_losses', 'equip_value',
    'rifle_cnt', 'smg_cnt', 'shotgun_cnt', 'awp_cnt',
    'helmets', 'kevlar', 'kits',
    'flash_cnt', 'smoke_cnt', 'he_cnt', 'molotov_cnt',
    'opp_rifle_cnt', 'opp_smg_cnt', 'opp_awp_cnt', # Key opponent gears
    'timeout_flag'
]
FEATURES_ELO = ['team_elo_pre_event', 'opp_elo_pre_event', 'elo_diff', 'elo_missing']
FEATURES_CAT = ['map', 'side']
TARGET = 'round_win'

# One-Hot Encoding for Categorical Features
print(" Encoding categorical features...")
df_encoded = pd.get_dummies(df_rounds, columns=FEATURES_CAT, drop_first=True)

# Define feature lists dynamically based on encoded columns
encoded_cols = [c for c in df_encoded.columns if c.startswith('map_') or c.startswith('side_')]
features_freeze = FEATURES_NUMERIC + encoded_cols
features_full = features_freeze + FEATURES_ELO

# Time-Aware Splitting
# We group by EVENT to ensure no data leakage between training and testing
# The last 2 events are held out as the Test Set.
unique_events = df_encoded['event_name'].unique()
event_to_id = {evt: i for i, evt in enumerate(sorted(unique_events))}
df_encoded['event_group'] = df_encoded['event_name'].map(event_to_id)

test_groups = [len(unique_events)-1, len(unique_events)-2] # Last 2 events
train_mask = ~df_encoded['event_group'].isin(test_groups)

# Create matrices
X_train_full = df_encoded.loc[train_mask, features_full]
y_train = df_encoded.loc[train_mask, TARGET]
X_test_full = df_encoded.loc[~train_mask, features_full]
y_test = df_encoded.loc[~train_mask, TARGET]

# Groups for CV
groups_train = df_encoded.loc[train_mask, 'event_group']

print(f" Train Shape: {X_train_full.shape}")
print(f" Test Shape:  {X_test_full.shape}")

# ============================================================================
# STEP 4: BASELINE MODELS (ABLATION STUDY)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: TRAINING BASELINE MODELS")
print("=" * 80)

# Baseline B: Logistic Regression (No Elo) - The "Economy Only" Model
print(" Training Baseline B (No Elo)...")
# We filter X_train_full to only include freeze-time features (exclude Elo)
X_train_freeze = X_train_full[features_freeze]
X_test_freeze = X_test_full[features_freeze]

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_freeze, y_train)
y_pred_baseline = lr_model.predict_proba(X_test_freeze)[:, 1]

loss_base = log_loss(y_test, y_pred_baseline)
print(f" Baseline B Log Loss: {loss_base:.4f}")

# Baseline B+: Logistic Regression (With Elo) - To quantify Elo value
print(" Training Baseline B+ (With Elo)...")
lr_elo_model = LogisticRegression(max_iter=1000)
lr_elo_model.fit(X_train_full, y_train)
y_pred_base_elo = lr_elo_model.predict_proba(X_test_full)[:, 1]

loss_base_elo = log_loss(y_test, y_pred_base_elo)
print(f" Baseline B+ Log Loss: {loss_base_elo:.4f}")
print(f" Improvement from Elo: {loss_base - loss_base_elo:.4f}")

# ============================================================================
# STEP 5: MAIN MODEL (LightGBM + CALIBRATION)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: TRAINING MAIN MODEL (LightGBM)")
print("=" * 80)

# Create a validation split for Early Stopping & Calibration
# We take one event group out of the training set to be the validation set
val_group = groups_train.unique()[-1] 
train_idx = groups_train != val_group
val_idx = groups_train == val_group

X_tr_lgb, y_tr_lgb = X_train_full[train_idx], y_train[train_idx]
X_val_lgb, y_val_lgb = X_train_full[val_idx], y_train[val_idx]

# LightGBM Parameters (Optimised for Log Loss)
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

train_data = lgb.Dataset(X_tr_lgb, label=y_tr_lgb)
val_data = lgb.Dataset(X_val_lgb, label=y_val_lgb, reference=train_data)

print(" Training LightGBM...")
bst = lgb.train(
    params, 
    train_data, 
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

# Predictions (Uncalibrated)
y_pred_uncal = bst.predict(X_test_full, num_iteration=bst.best_iteration)
loss_uncal = log_loss(y_test, y_pred_uncal)
print(f" Uncalibrated Log Loss: {loss_uncal:.4f}")

# Calibration Step (Isotonic Regression)
print(" Applying Post-Hoc Calibration (Isotonic)...")
# We calibrate on the validation set predictions
val_preds = bst.predict(X_val_lgb, num_iteration=bst.best_iteration)
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(val_preds, y_val_lgb)

# Final Calibrated Predictions
y_pred_cal = iso.transform(y_pred_uncal)
loss_cal = log_loss(y_test, y_pred_cal)
print(f" Calibrated Log Loss:   {loss_cal:.4f}")

# ============================================================================
# STEP 6: EVALUATION & REPORTING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: FINAL EVALUATION")
print("=" * 80)

# Calculate ECE (Expected Calibration Error)
def calculate_ece(y_true, y_pred, bins=15):
    bin_edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i+1])
        if mask.sum() > 0:
            conf = y_pred[mask].mean()
            acc = y_true[mask].mean()
            ece += (mask.sum() / len(y_true)) * np.abs(conf - acc)
    return ece

ece_final = calculate_ece(y_test, y_pred_cal)
brier_final = brier_score_loss(y_test, y_pred_cal)

# Save final report to 'progression' folder
report_path = os.path.join(OUTPUT_DIR, 'FINAL_REPORT.txt')
with open(report_path, 'w') as f:
    f.write("CS2 PREDICTION MODEL - FINAL RESULTS\n")
    f.write("====================================\n")
    f.write(f"Baseline B (Economy):  {loss_base:.4f}\n")
    f.write(f"Main Model (Calibrated): {loss_cal:.4f}\n")
    f.write(f"Improvement:           {(loss_base - loss_cal)/loss_base*100:.2f}%\n\n")
    f.write(f"Brier Score:           {brier_final:.4f}\n")
    f.write(f"ECE (Calibration):     {ece_final:.4f}\n")

print(f" Final ECE: {ece_final:.4f}")
print(f" Report saved to: {report_path}")

# Plot Calibration Curve
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
prob_true, prob_pred = sns.calibration.calibration_curve(y_test, y_pred_cal, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', label='Main Model')
plt.title(f'Reliability Diagram (ECE: {ece_final:.4f})')
plt.xlabel('Predicted Probability')
plt.ylabel('True Win Rate')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, 'calibration_curve.png'))
print(f" Calibration plot saved to: {os.path.join(OUTPUT_DIR, 'calibration_curve.png')}")

print("\n" + "=" * 80)
print(" PIPELINE COMPLETE SUCCESSFULLY")
print("=" * 80)
