#!/usr/bin/env python3
"""
make_elo.py - Elo Rating Construction for CS2 Teams
====================================================

This script constructs Elo ratings for CS2 teams with event-based freezing.
Processes matches chronologically and freezes ratings at the start of each event.

Usage:
    python make_elo.py --config config.yaml

Output:
    - rounds_with_elo.csv: Rounds with Elo ratings added
    - match_elo.csv: Match-level Elo ratings
    - elo_history.csv: Evolution of ratings over time
"""

import pandas as pd
import numpy as np
import yaml
import argparse
import sys
from pathlib import Path

class EloRatingSystem:
    """
    Elo rating system for CS2 teams with event-based freezing
    """
    def __init__(self, k_factor=32, default_rating=1500):
        self.k_factor = k_factor
        self.default_rating = default_rating
        self.ratings = {}  # team -> current rating
        self.event_ratings = {}  # (team, event) -> pre-event rating
        self.rating_history = []
        
    def get_rating(self, team):
        """Get current rating for a team"""
        if team not in self.ratings:
            self.ratings[team] = self.default_rating
        return self.ratings[team]
    
    def expected_score(self, rating_a, rating_b):
        """Calculate expected score for team A"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, team_a, team_b, score_a, score_b, match_id, match_time):
        """
        Update ratings based on match result
        Score is rounds won (e.g., 27-23)
        """
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        
        # Calculate actual score (normalize to 0-1)
        total_rounds = score_a + score_b
        if total_rounds == 0:
            actual_a = 0.5
        else:
            actual_a = score_a / total_rounds
        
        # Calculate expected scores
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1 - expected_a
        
        # Update ratings
        new_rating_a = rating_a + self.k_factor * (actual_a - expected_a)
        new_rating_b = rating_b + self.k_factor * ((1 - actual_a) - expected_b)
        
        # Store history
        self.rating_history.append({
            'match_id': match_id,
            'match_time': match_time,
            'team_a': team_a,
            'team_b': team_b,
            'rating_a_before': rating_a,
            'rating_b_before': rating_b,
            'rating_a_after': new_rating_a,
            'rating_b_after': new_rating_b,
            'score_a': score_a,
            'score_b': score_b
        })
        
        # Update current ratings
        self.ratings[team_a] = new_rating_a
        self.ratings[team_b] = new_rating_b
        
        return new_rating_a, new_rating_b
    
    def freeze_rating_for_event(self, team, event):
        """Freeze team's rating at the start of an event"""
        key = (team, event)
        if key not in self.event_ratings:
            self.event_ratings[key] = self.get_rating(team)
        return self.event_ratings[key]
    
    def get_pre_event_rating(self, team, event):
        """
        Get the rating a team had before an event started
        Returns (rating, is_missing_flag)
        """
        key = (team, event)
        
        # Check if we have a frozen rating for this team-event combo
        if key in self.event_ratings:
            has_history = len([h for h in self.rating_history 
                             if h['team_a'] == team or h['team_b'] == team]) > 0
            return self.event_ratings[key], 0 if has_history else 1
        
        # If not frozen yet, freeze it now
        rating = self.get_rating(team)
        has_history = len([h for h in self.rating_history 
                          if h['team_a'] == team or h['team_b'] == team]) > 0
        
        self.event_ratings[key] = rating
        return rating, 0 if has_history else 1

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def extract_match_results(df_rounds):
    """
    Extract match results from round data
    Returns DataFrame with match-level information
    """
    matches = []
    
    for match_id in df_rounds['match_id'].unique():
        match_data = df_rounds[df_rounds['match_id'] == match_id]
        
        # Get team names
        teams = match_data['team_name'].unique()
        if len(teams) != 2:
            continue
        
        team1, team2 = teams[0], teams[1]
        
        # Calculate match score (rounds won)
        team1_score = match_data[match_data['team_name'] == team1]['round_win'].sum()
        team2_score = match_data[match_data['team_name'] == team2]['round_win'].sum()
        
        # Get metadata
        event = match_data['event_name'].iloc[0] if 'event_name' in match_data.columns else 'Unknown'
        match_time = match_data['match_time'].iloc[0] if 'match_time' in match_data.columns else pd.Timestamp('2022-01-01')
        
        matches.append({
            'match_id': match_id,
            'team1': team1,
            'team2': team2,
            'score1': int(team1_score),
            'score2': int(team2_score),
            'event_name': event,
            'match_time': pd.to_datetime(match_time)
        })
    
    return pd.DataFrame(matches).sort_values('match_time').reset_index(drop=True)

def merge_elo_to_rounds(df_rounds, df_match_elo):
    """Merge Elo ratings into rounds dataframe"""
    df_rounds_with_elo = df_rounds.copy()
    
    # Initialize Elo columns
    df_rounds_with_elo['team_elo_pre_event'] = np.nan
    df_rounds_with_elo['opp_elo_pre_event'] = np.nan
    df_rounds_with_elo['elo_diff'] = np.nan
    df_rounds_with_elo['elo_missing'] = 0
    
    # Create lookup dictionary
    elo_lookup = {row['match_id']: row for _, row in df_match_elo.iterrows()}
    
    merge_count = 0
    for idx in df_rounds_with_elo.index:
        match_id = df_rounds_with_elo.loc[idx, 'match_id']
        team_name = df_rounds_with_elo.loc[idx, 'team_name']
        
        if match_id not in elo_lookup:
            continue
        
        match_elo = elo_lookup[match_id]
        
        # Determine if this team is team1 or team2
        if team_name == match_elo['team1']:
            df_rounds_with_elo.loc[idx, 'team_elo_pre_event'] = match_elo['team1_elo_pre_event']
            df_rounds_with_elo.loc[idx, 'opp_elo_pre_event'] = match_elo['team2_elo_pre_event']
            df_rounds_with_elo.loc[idx, 'elo_diff'] = match_elo['elo_diff_team1']
            df_rounds_with_elo.loc[idx, 'elo_missing'] = match_elo['team1_elo_missing']
            merge_count += 1
        elif team_name == match_elo['team2']:
            df_rounds_with_elo.loc[idx, 'team_elo_pre_event'] = match_elo['team2_elo_pre_event']
            df_rounds_with_elo.loc[idx, 'opp_elo_pre_event'] = match_elo['team1_elo_pre_event']
            df_rounds_with_elo.loc[idx, 'elo_diff'] = -match_elo['elo_diff_team1']
            df_rounds_with_elo.loc[idx, 'elo_missing'] = match_elo['team2_elo_missing']
            merge_count += 1
    
    return df_rounds_with_elo, merge_count

def main(config_path='config.yaml'):
    """Main Elo construction pipeline"""
    
    print("=" * 80)
    print("CS2 ELO RATING CONSTRUCTION - make_elo.py")
    print("=" * 80)
    
    # Load configuration
    print("\n[1/7] Loading configuration...")
    config = load_config(config_path)
    k_factor = config['elo']['k_factor']
    default_rating = config['elo']['default_rating']
    print(f"✓ K-factor: {k_factor}")
    print(f"✓ Default rating: {default_rating}")
    
    # Load processed rounds
    print("\n[2/7] Loading processed rounds...")
    output_dir = Path(config['data']['output_dir'])
    rounds_file = output_dir / 'rounds_with_features.csv'
    
    df_rounds = pd.read_csv(rounds_file)
    print(f" Loaded {len(df_rounds):,} rounds from {rounds_file}")
    
    # Extract match results
    print("\n[3/7] Extracting match results...")
    df_matches = extract_match_results(df_rounds)
    print(f" Extracted {len(df_matches)} matches")
    print(f" Date range: {df_matches['match_time'].min()} to {df_matches['match_time'].max()}")
    
    # Initialize Elo system
    print("\n[4/7] Calculating Elo ratings...")
    elo_system = EloRatingSystem(k_factor=k_factor, default_rating=default_rating)
    
    processed_events = set()
    match_elo_data = []
    
    for _, row in df_matches.iterrows():
        match_id = row['match_id']
        event = row['event_name']
        team1 = row['team1']
        team2 = row['team2']
        score1 = row['score1']
        score2 = row['score2']
        match_time = row['match_time']
        
        # Freeze ratings for this event if first time seeing it
        if event not in processed_events:
            event_matches = df_matches[df_matches['event_name'] == event]
            event_teams = set(event_matches['team1'].tolist() + event_matches['team2'].tolist())
            
            for team in event_teams:
                elo_system.freeze_rating_for_event(team, event)
            
            processed_events.add(event)
            print(f"  Event: '{event}' - Froze ratings for {len(event_teams)} teams")
        
        # Get pre-event ratings
        team1_elo, team1_missing = elo_system.get_pre_event_rating(team1, event)
        team2_elo, team2_missing = elo_system.get_pre_event_rating(team2, event)
        
        # Store match-level Elo
        match_elo_data.append({
            'match_id': match_id,
            'event_name': event,
            'team1': team1,
            'team2': team2,
            'team1_elo_pre_event': team1_elo,
            'team2_elo_pre_event': team2_elo,
            'team1_elo_missing': team1_missing,
            'team2_elo_missing': team2_missing,
            'elo_diff_team1': team1_elo - team2_elo
        })
        
        # Update ratings after match
        elo_system.update_ratings(team1, team2, score1, score2, match_id, match_time)
    
    df_match_elo = pd.DataFrame(match_elo_data)
    
    print(f"\n Processed {len(df_matches)} matches")
    print(f" Tracked {len(processed_events)} events")
    print(f" Rated {len(elo_system.ratings)} unique teams")
    
    # Merge Elo into rounds
    print("\n[5/7] Merging Elo ratings into rounds...")
    df_rounds_with_elo, merge_count = merge_elo_to_rounds(df_rounds, df_match_elo)
    print(f" Merged Elo for {merge_count:,}/{len(df_rounds):,} rounds ({merge_count/len(df_rounds)*100:.1f}%)")
    
    # Fill any missing Elo with defaults
    missing = df_rounds_with_elo['team_elo_pre_event'].isna().sum()
    if missing > 0:
        print(f"  Filling {missing} missing values with default Elo ({default_rating})")
        df_rounds_with_elo['team_elo_pre_event'].fillna(default_rating, inplace=True)
        df_rounds_with_elo['opp_elo_pre_event'].fillna(default_rating, inplace=True)
        df_rounds_with_elo['elo_diff'].fillna(0, inplace=True)
        df_rounds_with_elo['elo_missing'] = df_rounds_with_elo['elo_missing'].fillna(1)
    
    # Save outputs
    print("\n[6/7] Saving outputs...")
    
    # Rounds with Elo
    rounds_elo_file = output_dir / config['output_files']['rounds_processed']
    df_rounds_with_elo.to_csv(rounds_elo_file, index=False)
    print(f" Saved rounds with Elo to: {rounds_elo_file}")
    
    # Match-level Elo
    match_elo_file = output_dir / config['output_files']['match_elo']
    df_match_elo.to_csv(match_elo_file, index=False)
    print(f" Saved match Elo to: {match_elo_file}")
    
    # Elo history
    elo_history_file = output_dir / config['output_files']['elo_history']
    df_elo_history = pd.DataFrame(elo_system.rating_history)
    df_elo_history.to_csv(elo_history_file, index=False)
    print(f" Saved Elo history to: {elo_history_file}")
    
    # Summary statistics
    print("\n[7/7] Summary statistics...")
    ratings_list = list(elo_system.ratings.values())
    print(f"\nElo Rating Statistics:")
    print(f"  Mean: {np.mean(ratings_list):.1f}")
    print(f"  Std: {np.std(ratings_list):.1f}")
    print(f"  Min: {np.min(ratings_list):.1f}")
    print(f"  Max: {np.max(ratings_list):.1f}")
    
    print(f"\nTop 10 Teams by Elo:")
    top_teams = sorted(elo_system.ratings.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (team, rating) in enumerate(top_teams, 1):
        print(f"  {i:2d}. {team:30s} {rating:.1f}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("ELO CONSTRUCTION COMPLETE")
    print("=" * 80)
    print(f"\nOutput Files:")
    print(f"   {rounds_elo_file}")
    print(f"   {match_elo_file}")
    print(f"   {elo_history_file}")
    
    print("\n Ready for model training (train.py)")
    print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct Elo ratings for CS2 teams')
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
