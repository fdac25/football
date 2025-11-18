import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import sklearn

path = kagglehub.dataset_download("philiphyde1/nfl-stats-1999-2022") # To access just use path variable

offensePlayerDF = pd.read_csv(path + "/yearly_player_stats_offense.csv")

df_yearly_team_defense_stats = pd.read_csv(path + "/yearly_team_stats_defense.csv")

# Start of function
def preprocess(year, offensePlayerDF, defenseDF):

   # Filter the dataframe to keep only the columns we need (Only year 2024, those not POST season, and things to calculate our score)
   df_yearly_player_offense_stats_filtered = offensePlayerDF.loc[(offensePlayerDF['season'] == year) & (offensePlayerDF['season_type'] != 'POST'),
                                                               ['player_name', 'position', 'season_complete_pass', 'season_pass_attempts', 'season_passing_yards',
                                                               'season_receiving_yards', 'season_rush_attempts', 'season_rushing_yards', 'season_fantasy_points_ppr', 'games_played_season',
                                                               'season_rush_touchdown', 'season_pass_touchdown', 'season_interception', 'season_fumble_lost',
                                                               'season_receptions', 'season_targets', 'season_receiving_touchdown'
                                                               ]]

   # Explore the positions we have in our dataset
   # print("Unique positions in dataset:")
   unique_positions = df_yearly_player_offense_stats_filtered['position'].unique()
   # print(unique_positions)

   # print("\nPlayers per position:")
   position_counts = df_yearly_player_offense_stats_filtered['position'].value_counts()
   # print(position_counts)

   # Clean data - removing players with missing data

   # Remove any rows that don't have a player name, position, or games played
   df_yearly_player_offense_stats_filtered = df_yearly_player_offense_stats_filtered.dropna(subset=(['player_name', 'position', 'games_played_season']))

   # Replace any missing numbers with 0 (means they didn't do that stat)
   numeric_columns = df_yearly_player_offense_stats_filtered.select_dtypes(include=[np.number]).columns
   df_yearly_player_offense_stats_filtered[numeric_columns] = df_yearly_player_offense_stats_filtered[numeric_columns].fillna(0)

   # Reset the index after dropping rows
   df_yearly_player_offense_stats_filtered = df_yearly_player_offense_stats_filtered.reset_index(drop=True)

   # After cleaning Players per position
   position_counts = df_yearly_player_offense_stats_filtered['position'].value_counts()
   # print(position_counts)

   # Only focusing on the main offensive positions, combining FB with RB
   offensive_positions = ['QB', 'RB', 'WR', 'TE', 'FB']

   df_yearly_player_offense_stats_filtered = df_yearly_player_offense_stats_filtered[df_yearly_player_offense_stats_filtered['position'].isin(offensive_positions)]

   # combine the fb positions into the rb column
   df_yearly_player_offense_stats_filtered['position'] = df_yearly_player_offense_stats_filtered['position'].replace('FB', 'RB')

   # Filter to players who played at least half the season (8 games)
   df_yearly_player_offense_stats_filtered = df_yearly_player_offense_stats_filtered[df_yearly_player_offense_stats_filtered['games_played_season'] >= 8]

   # print(f"Total offensive position players: {len(df_yearly_player_offense_stats_filtered)}")
   # print(df_yearly_player_offense_stats_filtered['position'].value_counts())

   # Setting up the scoring for positions
   allQBs = (df_yearly_player_offense_stats_filtered['position'] == 'QB')
   qbsAboveThresh = (df_yearly_player_offense_stats_filtered['season_passing_yards'] > 1500)


   # Filter so we only include quarter backs that have at least 1500 passing yards
   qbs = df_yearly_player_offense_stats_filtered[allQBs & qbsAboveThresh].copy()

   # Shows that we got rid of 4 QBs
   # print(qbs.shape)

   # Passing stats for a QB
   qbs['completion_percent'] = (qbs['season_complete_pass'] / qbs['season_pass_attempts']) * 100
   qbs['yards_per_attempt'] = qbs['season_passing_yards'] / qbs['season_pass_attempts']
   qbs['touchdown_percent'] = (qbs['season_pass_touchdown'] / qbs['season_pass_attempts']) * 100
   qbs['interception_percent'] = (qbs['season_interception'] / qbs['season_pass_attempts']) * 100
   qbs['yards_per_carry'] = qbs['season_rushing_yards'] / qbs['season_rush_attempts']
   # will use fumbles stats too but already exists so dont need to make column
   # print(qbs[['player_name', 'completion_percent', 'yards_per_attempt', 'touchdown_percent', 'interception_percent', 'season_fumble_lost']].head())

   # we decided to use z score because it standardizes the data making it easier to compare to one another
   qbs['completion_zScore'] = (qbs['completion_percent'] - qbs['completion_percent'].mean()) / qbs['completion_percent'].std()
   qbs['yards_per_attempt_zScore'] = (qbs['yards_per_attempt'] - qbs['yards_per_attempt'].mean()) / qbs['yards_per_attempt'].std()
   qbs['touchdown_zScore'] = (qbs['touchdown_percent'] - qbs['touchdown_percent'].mean()) / qbs['touchdown_percent'].std()
   qbs['interception_zScore'] = -1 * ((qbs['interception_percent'] - qbs['interception_percent'].mean()) / qbs['interception_percent'].std())
   qbs['yards_per_carry_zScore'] = (qbs['yards_per_carry'] - qbs['yards_per_carry'].mean()) / qbs['yards_per_carry'].std()
   qbs['fantasy_points_zScore'] = (qbs['season_fantasy_points_ppr'] - qbs['season_fantasy_points_ppr'].mean()) / qbs['season_fantasy_points_ppr'].std()

   # Fumbles as raw count (flip sign because fewer is better)
   qbs['fumbles_zScore'] = -1 * ((qbs['season_fumble_lost'] - qbs['season_fumble_lost'].mean()) / qbs['season_fumble_lost'].std())

   # Average the z-scores for overall passing rating
   qbs['overall_QB_rating'] = (qbs['completion_zScore'] + qbs['yards_per_attempt_zScore'] + qbs['touchdown_zScore'] + qbs['interception_zScore'] + qbs['fumbles_zScore'] + qbs['yards_per_carry_zScore'] + qbs['fantasy_points_zScore']) / 7

   # print("\nTop 10 QBs:")
   # print(qbs.nlargest(10, 'overall_QB_rating')[['player_name', 'completion_percent', 'yards_per_attempt', 'touchdown_percent', 'interception_percent', 'overall_QB_rating']])

   # same thing for rb

   # Setting up the scoring for positions
   allRBs = (df_yearly_player_offense_stats_filtered['position'] == 'RB')
   rbsAboveThresh = (df_yearly_player_offense_stats_filtered['season_rushing_yards'] > 300)

   # Filter so we only include running backs that have at least 500 rushing yards
   rbs = df_yearly_player_offense_stats_filtered[allRBs & rbsAboveThresh].copy()

   # Shows that we got rid of 
   # print(rbs.shape)

   # Calculate rushing stats
   rbs['yards_per_carry'] = rbs['season_rushing_yards'] / rbs['season_rush_attempts']
   rbs['rush_touchdown_rate'] = (rbs['season_rush_touchdown'] / rbs['season_rush_attempts']) * 100
   rbs['fumble_rate'] = (rbs['season_fumble_lost'] / rbs['season_rush_attempts']) * 100
   rbs['receiving_yards_per_game'] = rbs['season_receiving_yards'] / rbs['games_played_season']

   # print(rbs[['player_name', 'yards_per_carry', 'rush_touchdown_rate', 'fumble_rate', 'receiving_yards_per_game']].head())

   # Calculate z-scores for rushing
   rbs['yards_per_carry_zScore'] = (rbs['yards_per_carry'] - rbs['yards_per_carry'].mean()) / rbs['yards_per_carry'].std()
   rbs['touchdown_rate_zScore'] = (rbs['rush_touchdown_rate'] - rbs['rush_touchdown_rate'].mean()) / rbs['rush_touchdown_rate'].std()
   rbs['total_yards_zScore'] = (rbs['season_rushing_yards'] - rbs['season_rushing_yards'].mean()) / rbs['season_rushing_yards'].std()
   rbs['fumble_zScore'] = -1 * ((rbs['fumble_rate'] - rbs['fumble_rate'].mean()) / rbs['fumble_rate'].std())
   rbs['receiving_yards_zScore'] = (rbs['receiving_yards_per_game'] - rbs['receiving_yards_per_game'].mean()) / rbs['receiving_yards_per_game'].std()
   rbs['fantasy_points_zScore'] = (rbs['season_fantasy_points_ppr'] - rbs['season_fantasy_points_ppr'].mean()) / rbs['season_fantasy_points_ppr'].std()

   # Average the z-scores for overall rushing rating
   rbs['overall_RB_rating'] = (rbs['yards_per_carry_zScore'] + rbs['touchdown_rate_zScore'] + rbs['total_yards_zScore'] + rbs['fumble_zScore'] + rbs['receiving_yards_zScore'] + rbs['fantasy_points_zScore']) / 6

   # print("\nTop 10 RBs:")
   # print(rbs.nlargest(10, 'overall_RB_rating')[['player_name', 'season_rushing_yards', 'yards_per_carry', 'rush_touchdown_rate', 'receiving_yards_per_game', 'overall_RB_rating']])

   # add them back to main df
   df_yearly_player_offense_stats_filtered['overall_QB_rating'] = 0.0
   df_yearly_player_offense_stats_filtered['overall_RB_rating'] = 0.0

   # Update with calculated ratings
   for idx in qbs.index:
      df_yearly_player_offense_stats_filtered.loc[idx, 'overall_QB_rating'] = qbs.loc[idx, 'overall_QB_rating']

   for idx in rbs.index:
      df_yearly_player_offense_stats_filtered.loc[idx, 'overall_RB_rating'] = rbs.loc[idx, 'overall_RB_rating']

   df_yearly_player_offense_stats_filtered.head()

   # Receiving rating for WRs
   # Setting up the scoring for positions
   allWRs = (df_yearly_player_offense_stats_filtered['position'] == 'WR')
   wrsAboveThresh = (df_yearly_player_offense_stats_filtered['season_receiving_yards'] > 500)
   wrsAboveRecThresh = (df_yearly_player_offense_stats_filtered['season_receptions'] > 48)

   # Filter so we only include running backs that have at least 500 receiving yards
   wideRec = df_yearly_player_offense_stats_filtered[allWRs & wrsAboveThresh & wrsAboveRecThresh].copy()

   # Shows how many we got rid of
   # print(wideRec.shape)

   # Calculate receiving stats
   wideRec['yards_per_reception'] = wideRec['season_receiving_yards'] / wideRec['season_receptions']
   wideRec['catch_rate'] = (wideRec['season_receptions'] / wideRec['season_targets']) * 100
   wideRec['touchdown_rate'] = (wideRec['season_receiving_touchdown'] / wideRec['season_receptions']) * 100
   wideRec['yards_per_game'] = wideRec['season_receiving_yards'] / wideRec['games_played_season']

   # print(wideRec[['player_name', 'yards_per_reception', 'catch_rate', 'touchdown_rate', 'yards_per_game']].head())

   # Calculate z-scores for receiving
   wideRec['yards_per_reception_zScore'] = (wideRec['yards_per_reception'] - wideRec['yards_per_reception'].mean()) / wideRec['yards_per_reception'].std()
   wideRec['catch_rate_zScore'] = (wideRec['catch_rate'] - wideRec['catch_rate'].mean()) / wideRec['catch_rate'].std()
   wideRec['touchdown_rate_zScore'] = (wideRec['touchdown_rate'] - wideRec['touchdown_rate'].mean()) / wideRec['touchdown_rate'].std()
   wideRec['yards_per_game_zScore'] = (wideRec['yards_per_game'] - wideRec['yards_per_game'].mean()) / wideRec['yards_per_game'].std()
   wideRec['fumbles_zScore'] = -1 * ((wideRec['season_fumble_lost'] - wideRec['season_fumble_lost'].mean()) / wideRec['season_fumble_lost'].std())
   wideRec['fantasy_points_zScore'] = (wideRec['season_fantasy_points_ppr'] - wideRec['season_fantasy_points_ppr'].mean()) / wideRec['season_fantasy_points_ppr'].std()

   # Average the z-scores for overall receiving rating
   wideRec['overall_WR_rating'] = (wideRec['yards_per_reception_zScore'] + wideRec['catch_rate_zScore'] + wideRec['touchdown_rate_zScore'] + wideRec['yards_per_game_zScore'] + wideRec['fumbles_zScore'] + wideRec['fantasy_points_zScore']) / 6

   # print("\nTop 10 WRs:")
   # print(wideRec.nlargest(10, 'overall_WR_rating')[['player_name', 'season_receptions', 'yards_per_reception', 'catch_rate', 'touchdown_rate', 'overall_WR_rating']])

   # Receiving rating for TEs (same calculations as WRs)
   # Setting up the scoring for positions
   allTEs = (df_yearly_player_offense_stats_filtered['position'] == 'TE')
   tesAboveThresh = (df_yearly_player_offense_stats_filtered['season_receiving_yards'] > 300)
   tesAboveRecThresh = (df_yearly_player_offense_stats_filtered['season_receptions'] > 40)

   # Filter so we only include running backs that have at least 500 receiving yards
   tightEnds = df_yearly_player_offense_stats_filtered[allTEs & tesAboveThresh & tesAboveRecThresh].copy()

   # Shows how many we got rid of
   # print(tightEnds.shape)

   # Calculate receiving stats
   tightEnds['yards_per_reception'] = tightEnds['season_receiving_yards'] / tightEnds['season_receptions']
   tightEnds['catch_rate'] = (tightEnds['season_receptions'] / tightEnds['season_targets']) * 100
   tightEnds['touchdown_rate'] = (tightEnds['season_receiving_touchdown'] / tightEnds['season_receptions']) * 100
   tightEnds['yards_per_game'] = tightEnds['season_receiving_yards'] / tightEnds['games_played_season']

   # print(tightEnds[['player_name', 'yards_per_reception', 'catch_rate', 'touchdown_rate', 'yards_per_game']].head())

   # Calculate z-scores for TEs
   tightEnds['yards_per_reception_zScore'] = (tightEnds['yards_per_reception'] - tightEnds['yards_per_reception'].mean()) / tightEnds['yards_per_reception'].std()
   tightEnds['catch_rate_zScore'] = (tightEnds['catch_rate'] - tightEnds['catch_rate'].mean()) / tightEnds['catch_rate'].std()
   tightEnds['touchdown_rate_zScore'] = (tightEnds['touchdown_rate'] - tightEnds['touchdown_rate'].mean()) / tightEnds['touchdown_rate'].std()
   tightEnds['yards_per_game_zScore'] = (tightEnds['yards_per_game'] - tightEnds['yards_per_game'].mean()) / tightEnds['yards_per_game'].std()
   tightEnds['fumbles_zScore'] = -1 * ((tightEnds['season_fumble_lost'] - tightEnds['season_fumble_lost'].mean()) / tightEnds['season_fumble_lost'].std())
   tightEnds['fantasy_points_zScore'] = (tightEnds['season_fantasy_points_ppr'] - tightEnds['season_fantasy_points_ppr'].mean()) / tightEnds['season_fantasy_points_ppr'].std()

   # Average the z-scores for overall receiving rating
   tightEnds['overall_TE_rating'] = (tightEnds['yards_per_reception_zScore'] + tightEnds['catch_rate_zScore'] + tightEnds['touchdown_rate_zScore'] + tightEnds['yards_per_game_zScore'] + tightEnds['fumbles_zScore'] + tightEnds['fantasy_points_zScore']) / 6

   # print("\nTop 10 TEs:")
   # print(tightEnds.nlargest(10, 'overall_TE_rating')[['player_name', 'season_receptions', 'yards_per_reception', 'catch_rate', 'touchdown_rate', 'overall_TE_rating']])

   # add them back to main df
   df_yearly_player_offense_stats_filtered['overall_WR_rating'] = 0.0
   df_yearly_player_offense_stats_filtered['overall_TE_rating'] = 0.0

   # Update with calculated ratings
   for idx in wideRec.index:
      df_yearly_player_offense_stats_filtered.loc[idx, 'overall_WR_rating'] = wideRec.loc[idx, 'overall_WR_rating']

   for idx in tightEnds.index:
      df_yearly_player_offense_stats_filtered.loc[idx, 'overall_TE_rating'] = tightEnds.loc[idx, 'overall_TE_rating']

   # df_yearly_player_offense_stats_filtered.head()

   # Do the same for defensive teams
   df_yearly_team_defense_stats_filtered = defenseDF.loc[(defenseDF['season'] == year) & (defenseDF['season_type'] != 'POST'),
                                                       ['team', 'safety', 'interception', 'fumble_forced', 'sack', 'def_touchdown', 'win']
                                                       ]
   
   # all columns are filled with values so no dropping needed
   
   # start calculating z scores for defense stats
   df_yearly_team_defense_stats_filtered['interception_zScore'] = (df_yearly_team_defense_stats_filtered['interception'] - df_yearly_team_defense_stats_filtered['interception'].mean()) / df_yearly_team_defense_stats_filtered['interception'].std()
   df_yearly_team_defense_stats_filtered['fumble_forced_zScore'] = (df_yearly_team_defense_stats_filtered['fumble_forced'] - df_yearly_team_defense_stats_filtered['fumble_forced'].mean()) / df_yearly_team_defense_stats_filtered['fumble_forced'].std()
   df_yearly_team_defense_stats_filtered['sack_zScore'] = (df_yearly_team_defense_stats_filtered['sack'] - df_yearly_team_defense_stats_filtered['sack'].mean()) / df_yearly_team_defense_stats_filtered['sack'].std()
   df_yearly_team_defense_stats_filtered['safety_zScore'] = (df_yearly_team_defense_stats_filtered['safety'] - df_yearly_team_defense_stats_filtered['safety'].mean()) / df_yearly_team_defense_stats_filtered['safety'].std()
   df_yearly_team_defense_stats_filtered['def_touchdown_zScore'] = (df_yearly_team_defense_stats_filtered['def_touchdown'] - df_yearly_team_defense_stats_filtered['def_touchdown'].mean()) / df_yearly_team_defense_stats_filtered['def_touchdown'].std()
   df_yearly_team_defense_stats_filtered['win_zScore'] = (df_yearly_team_defense_stats_filtered['win'] - df_yearly_team_defense_stats_filtered['win'].mean()) / df_yearly_team_defense_stats_filtered['win'].std()

   # Overall defensive rating
   df_yearly_team_defense_stats_filtered['overall_DEF_rating'] = (df_yearly_team_defense_stats_filtered['interception_zScore'] +
                                                                       df_yearly_team_defense_stats_filtered['fumble_forced_zScore'] +
                                                                       df_yearly_team_defense_stats_filtered['sack_zScore'] +
                                                                       df_yearly_team_defense_stats_filtered['safety_zScore'] +
                                                                       df_yearly_team_defense_stats_filtered['def_touchdown_zScore'] +
                                                                       df_yearly_team_defense_stats_filtered['win_zScore']
                                                                       ) / 6
   
   # print("\nTop 10 Defensive Teams:")
   # print(df_yearly_team_defense_stats_filtered.nlargest(10, 'overall_defense_rating')[['team', 'interception', 'fumble_forced', 'sack', 'safety', 'def_touchdown', 'win', 'overall_defense_rating']])

   # Add DEF rating to players
   df_yearly_player_offense_stats_filtered['overall_DEF_rating'] = 0.0

   # Add player_name and position to defense (keep team for identification)
   df_yearly_team_defense_stats_filtered['player_name'] = df_yearly_team_defense_stats_filtered['team'] + ' Defense'
   df_yearly_team_defense_stats_filtered['position'] = 'DEF'
   df_yearly_team_defense_stats_filtered['games_played_season'] = 17
   df_yearly_team_defense_stats_filtered['season_fantasy_points_ppr'] = 0  # calculate if you have data
   df_yearly_team_defense_stats_filtered['overall_QB_rating'] = 0.0
   df_yearly_team_defense_stats_filtered['overall_RB_rating'] = 0.0
   df_yearly_team_defense_stats_filtered['overall_WR_rating'] = 0.0
   df_yearly_team_defense_stats_filtered['overall_TE_rating'] = 0.0

   # Concat everything
   df_all = pd.concat([df_yearly_player_offense_stats_filtered, df_yearly_team_defense_stats_filtered], ignore_index=True)

   # Drop all intermediate columns - keep only what's needed for ML
   columns_to_keep = ['player_name', 'position', 'games_played_season', 
                     'season_fantasy_points_ppr',
                     'overall_QB_rating', 'overall_RB_rating', 
                     'overall_WR_rating', 'overall_TE_rating', 'overall_DEF_rating']
   
   # final df with unique player names
   df_final = df_all[columns_to_keep].drop_duplicates(subset=['player_name'], keep='first')
   
   # Remove players who didn't meet any position threshold (all ratings are 0)
   # Keep row if it has non-zero rating or if it's a defense
   df_final = df_final[
      (df_final['overall_QB_rating'] != 0.0) |
      (df_final['overall_RB_rating'] != 0.0) |
      (df_final['overall_WR_rating'] != 0.0) |
      (df_final['overall_TE_rating'] != 0.0) |
      (df_final['position'] == 'DEF')  # Keep all defenses
   ]

   return df_final
   
df_processed_2021 = preprocess(2021, offensePlayerDF, df_yearly_team_defense_stats)
df_processed_2022 = preprocess(2022, offensePlayerDF, df_yearly_team_defense_stats)
df_processed_2023 = preprocess(2023, offensePlayerDF, df_yearly_team_defense_stats)
df_processed_2024 = preprocess(2024, offensePlayerDF, df_yearly_team_defense_stats)

# print(df_processed_2021.head(50))
# print(df_processed_2021.tail(50))


# one hot encode player name and position
def one_hot_encode(df):
   df_encoded = pd.get_dummies(df, columns=['position'], prefix='pos')
   return df_encoded

df_encoded_2021 = one_hot_encode(df_processed_2021)
df_encoded_2022 = one_hot_encode(df_processed_2022)
df_encoded_2023 = one_hot_encode(df_processed_2023)
df_encoded_2024 = one_hot_encode(df_processed_2024)

# print(df_encoded_2021.head(50))

# normalize the games played and fantasy points
def normalize_features(df):
   games_played_season_min = df['games_played_season'].min()
   games_played_season_max = df['games_played_season'].max()
   season_fantasy_points_ppr_min = df['season_fantasy_points_ppr'].min()
   season_fantasy_points_ppr_max = df['season_fantasy_points_ppr'].max()
   df['games_played_season'] = (df['games_played_season'] - games_played_season_min) / (games_played_season_max - games_played_season_min)
   df['season_fantasy_points_ppr'] = (df['season_fantasy_points_ppr'] - season_fantasy_points_ppr_min) / (season_fantasy_points_ppr_max - season_fantasy_points_ppr_min)
   return df


# Save player names before doing anything
names_2021 = df_processed_2021['player_name'].copy()
names_2022 = df_processed_2022['player_name'].copy()
names_2023 = df_processed_2023['player_name'].copy()
names_2024 = df_processed_2024['player_name'].copy()

# Match players: find who played in both year n and year n+1
def match_players(df1, df2):
    # Only keep players who appear in both dataframes
    common_players = set(df1['player_name']) & set(df2['player_name'])
    
    df1_matched = df1[df1['player_name'].isin(common_players)].sort_values('player_name').reset_index(drop=True)
    df2_matched = df2[df2['player_name'].isin(common_players)].sort_values('player_name').reset_index(drop=True)
    
    return df1_matched, df2_matched

# Match players for each year pair
df_2021_matched, df_2022_matched = match_players(df_processed_2021, df_processed_2022)
df_2022_matched2, df_2023_matched = match_players(df_processed_2022, df_processed_2023)
df_2023_matched2, df_2024_matched = match_players(df_processed_2023, df_processed_2024)

# Drop player_name and one-hot encode
df_2021_encoded = one_hot_encode(df_2021_matched.drop(columns=['player_name']))
df_2022_encoded = one_hot_encode(df_2022_matched.drop(columns=['player_name']))
df_2022_encoded_v2 = one_hot_encode(df_2022_matched2.drop(columns=['player_name']))
df_2023_encoded = one_hot_encode(df_2023_matched.drop(columns=['player_name']))
df_2023_encoded_v2 = one_hot_encode(df_2023_matched2.drop(columns=['player_name']))
df_2024_encoded = one_hot_encode(df_2024_matched.drop(columns=['player_name']))

# Normalize
df_normalized_2021 = normalize_features(df_2021_encoded)
df_normalized_2022 = normalize_features(df_2022_encoded)
df_normalized_2022_v2 = normalize_features(df_2022_encoded_v2)
df_normalized_2023 = normalize_features(df_2023_encoded)
df_normalized_2023_v2 = normalize_features(df_2023_encoded_v2)
df_normalized_2024 = normalize_features(df_2024_encoded)

# Create training data
X_train = pd.concat([df_normalized_2021, df_normalized_2022_v2, df_normalized_2023_v2])
y_train = pd.concat([
    df_normalized_2022['season_fantasy_points_ppr'],
    df_normalized_2023['season_fantasy_points_ppr'],
    df_normalized_2024['season_fantasy_points_ppr']
])

X_train = X_train.drop(columns=['season_fantasy_points_ppr'])

# Train
model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict 2025
df_2025_encoded = one_hot_encode(df_processed_2024.drop(columns=['player_name']))
df_2025_normalized = normalize_features(df_2025_encoded)
data_2025 = df_2025_normalized.drop(columns=['season_fantasy_points_ppr'])
predictions_2025 = model.predict(data_2025)

# Add names back
results = pd.DataFrame({
    'player_name': names_2024,
    'predicted_2025_points': predictions_2025
})
print(results.nlargest(40, 'predicted_2025_points'))