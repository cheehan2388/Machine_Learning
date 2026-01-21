import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import shap
import matplotlib.pyplot as plt
from data_cleaning import drop_sparse_columns



train_file = 'train_data.csv'
df_whole = pd.read_csv(train_file)

 
df_whole['date'] = pd.to_datetime(df_whole['date'], errors='coerce')
df_whole['season'] = df_whole['date'].dt.year 

 
df, test_df1 = train_test_split(df_whole, test_size=0.2, random_state=42)

print(f"訓練集筆數: {len(df)}")
df = pd.read_csv(train_file)
df = drop_sparse_columns(df, threshold=0.5)   
 
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.sort_values('date')
df['year'] = df['date'].dt.year 

test_df1['date'] = pd.to_datetime(test_df1['date'], errors='coerce')
test_df1 = test_df1.sort_values('date')

 
selected_cols = [
    'home_batting_batting_avg_10RA',
    'away_batting_batting_avg_10RA',
    'date',
    'home_pitching_earned_run_avg_10RA',
    'away_pitching_earned_run_avg_10RA',
    'home_batting_onbase_plus_slugging_10RA',
    'away_batting_onbase_plus_slugging_10RA',
    'home_batting_RBI_10RA',
    'away_batting_RBI_10RA',
    'home_team_wins_mean',
    'away_team_wins_mean',
    'home_team_abbr',
    'away_team_abbr',
    'home_team_win',
    'home_pitching_SO_batters_faced_mean',
    'away_pitching_SO_batters_faced_mean',
    'home_pitching_wpa_def_mean',
    'away_pitching_wpa_def_mean',
    'home_batting_wpa_bat_mean',
    'away_batting_wpa_bat_mean',
    'home_team_spread_mean',
    'away_team_spread_mean'
]
valid_cols = [c for c in selected_cols if c in df.columns]
df = df[valid_cols] 

df.drop(columns=['date'], inplace=True, errors='ignore')

 
df['diff_batting_avg_10RA'] = df['home_batting_batting_avg_10RA'] - df['away_batting_batting_avg_10RA']
df['diff_ERA_10RA'] = df['home_pitching_earned_run_avg_10RA'] - df['away_pitching_earned_run_avg_10RA']
df['diff_batting_onbase_plus_slugging_10RA'] = df['home_batting_onbase_plus_slugging_10RA'] - df['away_batting_onbase_plus_slugging_10RA']
df['diff_batting_RBI_10RA'] = df['home_batting_RBI_10RA'] - df['away_batting_RBI_10RA']
df['diff_home_pitching_SO_batters_faced_mean'] = df['home_pitching_SO_batters_faced_mean'] - df['away_pitching_SO_batters_faced_mean']
df['plus_home_pitching_wpa_def_batted_mean'] = df['home_batting_wpa_bat_mean'] + df['home_pitching_wpa_def_mean']
df['plus_away_pitching_wpa_def_batted_mean'] = df['away_batting_wpa_bat_mean'] + df['away_pitching_wpa_def_mean']
df['diff_plus_pitching_wpa_def_batted_mean'] = df['plus_home_pitching_wpa_def_batted_mean'] - df['plus_away_pitching_wpa_def_batted_mean']
df['spread_team_mean'] = df['home_team_spread_mean'] / (df['away_team_spread_mean'] + 1)

# 步驟 2: 特徵工程 - 測試資料
test_selected_cols = [
    'home_batting_batting_avg_10RA', 'id',
    'away_batting_batting_avg_10RA',
    'home_pitching_earned_run_avg_10RA',
    'away_pitching_earned_run_avg_10RA',
    'home_batting_onbase_plus_slugging_10RA',
    'away_batting_onbase_plus_slugging_10RA',
    'home_batting_RBI_10RA',
    'away_batting_RBI_10RA',
    'home_team_wins_mean',
    'away_team_wins_mean',
    'home_team_abbr',
    'away_team_abbr',
    'home_pitching_SO_batters_faced_mean',
    'away_pitching_SO_batters_faced_mean',
    'home_pitching_wpa_def_mean',
    'away_pitching_wpa_def_mean',
    'home_team_spread_mean',
    'away_team_spread_mean',
    'home_batting_wpa_bat_mean',
    'away_batting_wpa_bat_mean'
]
valid_test_cols = [c for c in test_selected_cols if c in test_df1.columns]
test_df = test_df1[valid_test_cols]

# test_df['year'] = test_df1['season']

test_df['diff_batting_avg_10RA'] = test_df['home_batting_batting_avg_10RA'] - test_df['away_batting_batting_avg_10RA']
test_df['diff_ERA_10RA'] = test_df['home_pitching_earned_run_avg_10RA'] - test_df['away_pitching_earned_run_avg_10RA']
test_df['diff_batting_onbase_plus_slugging_10RA'] = test_df['home_batting_onbase_plus_slugging_10RA'] - test_df['away_batting_onbase_plus_slugging_10RA']
test_df['diff_batting_RBI_10RA'] = test_df['home_batting_RBI_10RA'] - test_df['away_batting_RBI_10RA']
test_df['diff_home_pitching_SO_batters_faced_mean'] = test_df['home_pitching_SO_batters_faced_mean'] - test_df['away_pitching_SO_batters_faced_mean']
test_df['plus_home_pitching_wpa_def_batted_mean'] = test_df['home_batting_wpa_bat_mean'] + test_df['home_pitching_wpa_def_mean']
test_df['plus_away_pitching_wpa_def_batted_mean'] = test_df['away_batting_wpa_bat_mean'] + test_df['away_pitching_wpa_def_mean']
test_df['diff_plus_pitching_wpa_def_batted_mean'] = test_df['plus_home_pitching_wpa_def_batted_mean'] - test_df['plus_away_pitching_wpa_def_batted_mean']
test_df['spread_team_mean'] = test_df['home_team_spread_mean'] / (test_df['away_team_spread_mean'] + 1)


 
df['clutch_factor'] = df['home_batting_wpa_bat_mean'] * df['home_batting_leverage_index_avg_mean']
test_df['clutch_factor'] = test_df['home_batting_wpa_bat_mean'] * test_df['home_batting_leverage_index_avg_mean']

 
df['wpa_diff'] = df['home_batting_wpa_bat_mean'] - df['away_pitching_wpa_def_mean']
test_df['wpa_diff'] = test_df['home_batting_wpa_bat_mean'] - test_df['away_pitching_wpa_def_mean']

# 2.3 為其他 home/away 配對建立差分特徵 (擴展 base_features)
base_features = [
    'pitcher_earned_run_avg_10RA',
    'batting_onbase_plus_slugging_10RA',
    'batting_batting_avg_10RA',
    'batting_RBI_10RA',
    'team_wins_mean',
    'pitching_SO_batters_faced_mean',
    'pitching_wpa_def_mean',
    'batting_wpa_bat_mean',
    'team_spread_mean'
     
]

for feat in base_features:
    home_col = f'home_{feat}'
    away_col = f'away_{feat}'
    if home_col in df.columns and away_col in df.columns:
        df[f'diff_{feat}'] = df[home_col] - df[away_col]
    if home_col in test_df.columns and away_col in test_df.columns:
        test_df[f'diff_{feat}'] = test_df[home_col] - test_df[away_col]


df['rest_opponent_batting_interaction'] = df['home_pitcher_rest'] * df['away_batting_avg_mean']
test_df['rest_opponent_batting_interaction'] = test_df['home_pitcher_rest'] * test_df['away_batting_avg_mean']


columns_to_drop = [
    'home_batting_batting_avg_10RA', 'away_batting_batting_avg_10RA',
    'home_batting_onbase_perc_10RA', 'away_batting_onbase_perc_10RA',
    'home_batting_onbase_plus_slugging_10RA', 'away_batting_onbase_plus_slugging_10RA',
    'home_batting_RBI_10RA', 'away_batting_RBI_10RA',
    'home_pitching_earned_run_avg_10RA', 'away_pitching_earned_run_avg_10RA',
    'home_pitching_wpa_def_mean', 'away_pitching_wpa_def_mean',
    'home_batting_wpa_bat_mean', 'away_batting_wpa_bat_mean',
    'home_pitching_SO_batters_faced_mean', 'away_pitching_SO_batters_faced_mean',
    'plus_home_pitching_wpa_def_batted_mean', 'plus_away_pitching_wpa_def_batted_mean',
    'home_team_spread_mean', 'away_team_spread_mean'
]

df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
test_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')


drop_cols = [ 'home_team_abbr', 'away_team_abbr', 'home_team_win', 'id'] 
X = df.drop(columns=drop_cols, errors='ignore')
y = df['home_team_win']


X_test_final = test_df.drop(columns=drop_cols, errors='ignore')

 
unique_years = sorted( df['year'].dropna().unique())  
results = []

for i in range(1, len(unique_years)):
    train_years = unique_years[:i]
    test_year = unique_years[i]
    
    train_idx = df['year'].isin(train_years)
    test_idx = df['year'] == test_year
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
     
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        colsample_bytree=0.6,   
        min_child_weight=7,     
        reg_lambda=3,           
        use_label_encoder=False,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
 
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
  
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_pred_prob)
    results.append({'test_year': test_year, 'accuracy': acc, 'log_loss': ll})
    print(f"測試年份: {test_year}, 準確率: {acc:.4f}, Log Loss: {ll:.4f}")


avg_acc = np.mean([r['accuracy'] for r in results])
avg_ll = np.mean([r['log_loss'] for r in results])
print(f"平均準確率: {avg_acc:.4f}, 平均 Log Loss: {avg_ll:.4f}")

final_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    colsample_bytree=0.6,
    min_child_weight=7,
    reg_lambda=3,
    use_label_encoder=False,
    random_state=42
)
final_model.fit(X, y)

 
if 'X_test_final' in locals():
    test_pred_prob = final_model.predict_proba(X_test_final)[:, 1]
    test_df['predicted_home_win_prob'] = test_pred_prob
    test_df[['id', 'predicted_home_win_prob']].to_csv('predictions.csv', index=False)
    print(" 'predictions.csv'")

 
explainer = shap.TreeExplainer(model)   
shap_values = explainer.shap_values(X_test)


shap.summary_plot(shap_values, X_test, show=False)
plt.savefig('shap_summary.png')
plt.show()


instance_idx = 0
shap.initjs()
force_plot = shap.force_plot(explainer.expected_value, shap_values[instance_idx,:], X_test.iloc[instance_idx,:])
