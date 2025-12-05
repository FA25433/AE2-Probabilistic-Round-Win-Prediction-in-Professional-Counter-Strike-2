#!/usr/bin/env python3
"""
make_rounds.py - Feature Engineering for CS2 Round Prediction
==============================================================

This script processes freeze-time features from CS2 demo files and match metadata.
Extracts all required features and ensures only freeze-time information is used.

Usage:
    python make_rounds.py --config config.yaml

Output:
    - rounds_with_features.csv: Processed round-level features
    - feature_dictionary.txt: One-line description of each feature
"""

import pandas as pd
import numpy as np
import yaml
import argparse
import sys
from pathlib import Path

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_freeze_time_features(df, required_features):
    """
    Validate that all required freeze-time features are present
    """
    missing = [f for f in required_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    
    print(f" All {len(required_features)} required features present")
    return True

def extract_series_type(maps_str):
    """Extract series type from Maps column"""
    if pd.isna(maps_str):
        return 1
    
    maps_str = str(maps_str).lower().strip()
    
    if 'bo5' in maps_str:
        return 5
    elif 'bo3' in maps_str:
        return 3
    elif 'bo2' in maps_str:
        return 2
    elif 'bo1' in maps_str or 'def' in maps_str:
        return 1
    else:
        import re
        match = re.search(r'bo(\d+)', maps_str)
        if match:
            return int(match.group(1))
        return 3  # Default to bo3

def create_feature_dictionary(features_dict, output_path):
    """
    Create a one-line data dictionary for each feature
    Format: feature_name | description | unit | example
    """
    
    feature_descriptions = {
        # Match Context
        'map': 'Counter-Strike 2 map name | categorical | de_mirage',
        'side': 'Team side (CT or T) | categorical | CT',
        'series_type': 'Best-of series format | integer | 3',
        'is_pistol': 'Pistol round flag (first round) | binary | 0',
        'is_ot': 'Overtime round flag | binary | 0',
        
        # Score & Economy
        'score_diff': 'Round score difference (team - opponent) | integer | -2',
        'start_cash': 'Team starting cash | dollars | 16000',
        'loss_bonus': 'Loss bonus amount | dollars | 1400',
        'consec_losses': 'Consecutive losses for team | integer | 0',
        'equip_value': 'Total equipment value | dollars | 23500',
        
        # Weapons
        'rifle_cnt': 'Number of rifles (AK/M4) | count | 4',
        'smg_cnt': 'Number of SMGs | count | 0',
        'shotgun_cnt': 'Number of shotguns | count | 0',
        'awp_cnt': 'Number of AWP sniper rifles | count | 1',
        
        # Armor & Utility
        'helmets': 'Players with helmets | count | 5',
        'kevlar': 'Players with kevlar | count | 5',
        'kits': 'Defuse kits (CT only) | count | 4',
        
        # Grenades
        'flash_cnt': 'Flashbang grenades | count | 8',
        'smoke_cnt': 'Smoke grenades | count | 5',
        'he_cnt': 'HE grenades | count | 4',
        'molotov_cnt': 'Molotov/Incendiary grenades | count | 3',
        
        # Other
        'timeout_flag': 'Tactical timeout called | binary | 0',
        
        # Target
        'round_win': 'Round won by team | binary | 1'
    }
    
    with open(output_path, 'w') as f:
        f.write("FEATURE DICTIONARY - CS2 Round Prediction\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Feature':<20} | {'Description':<40} | {'Unit':<15} | {'Example':<10}\n")
        f.write("-" * 80 + "\n")
        
        for feature, desc in feature_descriptions.items():
            f.write(f"{feature:<20} | {desc}\n")
    
    print(f"✓ Feature dictionary saved to: {output_path}")

def main(config_path='config.yaml'):
    """Main processing pipeline"""
    
    print("=" * 80)
    print("CS2 ROUND FEATURE EXTRACTION - make_rounds.py")
    print("=" * 80)
    
    # Load configuration
    print("\n[1/6] Loading configuration...")
    config = load_config(config_path)
    print(f" Configuration loaded from {config_path}")
    
    # Set random seed
    np.random.seed(config['random_state']['seed'])
    print(f" Random seed set to {config['random_state']['seed']}")
    
    # Load data
    print("\n[2/6] Loading data files...")
    freeze_time_file = config['data']['freeze_time_features']
    matches_file = config['data']['matches_metadata']
    
    df_rounds = pd.read_csv(freeze_time_file)
    df_matches = pd.read_excel(matches_file)
    
    print(f" Loaded {len(df_rounds):,} rounds from {freeze_time_file}")
    print(f" Loaded {len(df_matches):,} matches from {matches_file}")
    
    # Validate features
    print("\n[3/6] Validating freeze-time features...")
    required_features = config['features']['freeze_time']
    validate_freeze_time_features(df_rounds, required_features)
    
    # Add series_type from matches
    print("\n[4/6] Engineering additional features...")
    df_matches['series_type'] = df_matches['Maps'].apply(extract_series_type)
    
    # Merge series_type into rounds
    df_matches_meta = df_matches[['Match ID', 'series_type', 'Event Name', 'Time']].copy()
    df_matches_meta.columns = ['match_id', 'series_type', 'event_name', 'match_time']
    
    df_rounds_processed = df_rounds.merge(df_matches_meta, on='match_id', how='left')
    
    # Fill missing series_type with default (bo3)
    df_rounds_processed['series_type'].fillna(3, inplace=True)
    df_rounds_processed['series_type'] = df_rounds_processed['series_type'].astype(int)
    
    print(f" Added series_type feature")
    print(f" Total features: {len(df_rounds_processed.columns)}")
    
    # Save processed rounds
    print("\n[5/6] Saving processed data...")
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'rounds_with_features.csv'
    df_rounds_processed.to_csv(output_file, index=False)
    print(f" Saved processed rounds to: {output_file}")
    
    # Create feature dictionary
    print("\n[6/6] Creating feature dictionary...")
    dict_file = output_dir / 'feature_dictionary.txt'
    create_feature_dictionary(config['features'], dict_file)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nDataset Summary:")
    print(f"  Total rounds: {len(df_rounds_processed):,}")
    print(f"  Matches: {df_rounds_processed['match_id'].nunique()}")
    print(f"  Maps: {df_rounds_processed['map'].nunique()}")
    print(f"  Teams: {df_rounds_processed['team_name'].nunique()}")
    
    print(f"\nFeature Summary:")
    print(f"  Freeze-time features: {len(required_features)}")
    print(f"  Total columns: {len(df_rounds_processed.columns)}")
    
    print(f"\nOutput Files:")
    print(f"   {output_file}")
    print(f"   {dict_file}")
    
    print("\n Ready for Elo rating construction (make_elo.py)")
    print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and process CS2 round features')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    
    args = parser.parse_args()
    
    try:
        main(args.config)
    except Exception as e:
        print(f"\n Error: {e}", file=sys.stderr)
        sys.exit(1)
