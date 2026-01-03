import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import joblib
df = pd.read_csv("miss_smaler_7per.csv")

df = df[['home_batting_batting_avg_10RA',
         'away_batting_batting_avg_10RA',
        'date',
        "home_pitching_earned_run_avg_10RA",
        'away_pitching_earned_run_avg_10RA',
        'home_batting_onbase_plus_slugging_10RA',
        'away_batting_onbase_plus_slugging_10RA',
        'home_batting_RBI_10RA',
        'away_batting_RBI_10RA',
        'home_team_wins_mean' ,
        'away_team_wins_mean' ,
        'home_team_abbr' , 
        'away_team_abbr',
        'home_team_win' ,
        'home_pitching_SO_batters_faced_mean',
        'away_pitching_SO_batters_faced_mean',
        'home_pitching_wpa_def_mean',
         'away_pitching_wpa_def_mean',
        'home_batting_wpa_bat_mean',
        'away_batting_wpa_bat_mean' ,
        'home_team_spread_mean' ,
        'away_team_spread_mean' 




        ]]
 
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year


df.drop(columns=['date'], inplace=True, errors='ignore')
df

df['diff_batting_avg_10RA'] = df['home_batting_batting_avg_10RA'] - df['away_batting_batting_avg_10RA']
df['diff_ERA_10RA'] = df['home_pitching_earned_run_avg_10RA'] - df['away_pitching_earned_run_avg_10RA']
df['diff_batting_onbase_plus_slugging_10RA'] = df['home_batting_onbase_plus_slugging_10RA'] - df['away_batting_onbase_plus_slugging_10RA']
df['diff_batting_RBI_10RA'] = df['home_batting_RBI_10RA'] - df['away_batting_RBI_10RA']
df['diff_home_pitching_SO_batters_faced_mean'] = df['home_pitching_SO_batters_faced_mean'] - df['away_pitching_SO_batters_faced_mean']
df ['plus_home_pitching_wpa_def_batted_mean']  = df['home_batting_wpa_bat_mean'] +  df['home_pitching_wpa_def_mean']
df ['plus_away_pitching_wpa_def_batted_mean']  = df['away_batting_wpa_bat_mean'] + df['away_pitching_wpa_def_mean']
df['diff_plus_pitching_wpa_def_batted_mean'] = df['plus_home_pitching_wpa_def_batted_mean']-df['plus_away_pitching_wpa_def_batted_mean']
df['spread_team_mean']  = df['home_team_spread_mean'] / (df['away_team_spread_mean'] + 1)



columns_to_drop = [
    'home_batting_batting_avg_10RA', 'away_batting_batting_avg_10RA', 
    'home_batting_onbase_perc_10RA', 'away_batting_onbase_perc_10RA',
    'home_batting_onbase_plus_slugging_10RA','away_batting_onbase_plus_slugging_10RA',
    'home_batting_RBI_10RA','away_batting_RBI_10RA',   
    'home_pitching_earned_run_avg_10RA','away_pitching_earned_run_avg_10RA',
    'home_pitching_wpa_def_mean' , 'away_pitching_wpa_def_mean',
    'home_batting_wpa_bat_mean' , 'away_batting_wpa_bat_mean',
    "home_pitching_SO_batters_faced_mean","away_pitching_SO_batters_faced_mean",
    "plus_home_pitching_wpa_def_batted_mean","plus_away_pitching_wpa_def_batted_mean",
    'home_team_spread_mean','away_team_spread_mean'
 ]

df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
df

# 分離目標變量
y = df['home_team_win'].astype(bool)
X = df.drop(columns=['home_team_win'])

# 編碼分類特徵（隊伍和日期相關）
cat_features = ['home_team_abbr', 'away_team_abbr']
num_features = [col for col in X.columns if col not in cat_features + ['year']]

# 定義預處理器
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', 'passthrough', num_features)
    ]
)

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('imputer', SimpleImputer(strategy='mean')),
    ('clf', SVC(probability=True, random_state=42))  # 使用 SVM
])
 
param_grid = {
    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],  
    'clf__gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1],  
    'clf__class_weight': [None, 'balanced']  
}
grid = list(ParameterGrid(param_grid))

available_years = sorted(df['year'].unique())

models = {}

def split_train_test_per_year(df, year, test_size=0.2):
    df_year = df[df['year'] == year]
    n_total = len(df_year)
    if n_total == 0:
        
        return pd.DataFrame(), pd.DataFrame()
    n_test = max(1, int(n_total * test_size))   
    df_train = df_year.iloc[:-n_test]
    df_test = df_year.iloc[-n_test:]
    return df_train, df_test


for year in available_years:
   
    train_df, test_df = split_train_test_per_year(df, year, test_size=0.2)
    
  
    
    X_train = train_df.drop(columns=['home_team_win', 'year'])
    y_train = train_df['home_team_win']
    
    X_test = test_df.drop(columns=['home_team_win', 'year'])
    y_test = test_df['home_team_win']
    
    best_score = -np.inf
    best_params = None
    
    for params in grid:
        pipeline.set_params(**params)
        
        try:
            pipeline.fit(X_train, y_train)
        except Exception as e:
            print(f"{e}")
            continue
        
        try:
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
        except Exception as e:
            print(f"{e}")
            continue
        
        if acc > best_score:
            best_score = acc
            best_params = params
    
    print(f"bestparam: {best_params}")
    print(f"bestscore: {best_score:.4f}")
    
    pipeline.set_params(**best_params)
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        print(f"Error:{year} {best_params} {e}")
        continue
    
    models[year] = pipeline




joblib.dump(models, 'trained_models_3.pkl')
print(" 'trained_models.pkl'。")


#test
test_df1 = pd.read_csv("test_2.csv")

test_df = test_df1[['home_batting_batting_avg_10RA','id',
                    
         'away_batting_batting_avg_10RA',
        "home_pitching_earned_run_avg_10RA",
        'away_pitching_earned_run_avg_10RA',
        'home_batting_onbase_plus_slugging_10RA',
        'away_batting_onbase_plus_slugging_10RA',
        'home_batting_RBI_10RA',
        'away_batting_RBI_10RA',
        'home_team_wins_mean' ,
        'away_team_wins_mean' ,
        #'home_team_errors_mean',
        #'away_team_errors_mean',
        'home_team_abbr' , 
        'away_team_abbr',
        'home_pitching_SO_batters_faced_mean',
        'away_pitching_SO_batters_faced_mean',
        'home_pitching_wpa_def_mean',
        'away_pitching_wpa_def_mean',
        'home_team_spread_mean',
        'away_team_spread_mean',
        'home_batting_wpa_bat_mean',
        'away_batting_wpa_bat_mean' ,





        ]]

test_df['year'] = test_df1['season']

test_df['diff_batting_avg_10RA'] = test_df['home_batting_batting_avg_10RA'] - test_df['away_batting_batting_avg_10RA']
test_df['diff_ERA_10RA'] = test_df['home_pitching_earned_run_avg_10RA'] - test_df['away_pitching_earned_run_avg_10RA']
test_df['diff_batting_onbase_plus_slugging_10RA'] = test_df['home_batting_onbase_plus_slugging_10RA'] - test_df['away_batting_onbase_plus_slugging_10RA']
test_df['diff_batting_RBI_10RA'] = test_df['home_batting_RBI_10RA'] - test_df['away_batting_RBI_10RA']
test_df['diff_home_pitching_SO_batters_faced_mean'] = test_df['home_pitching_SO_batters_faced_mean'] - test_df['away_pitching_SO_batters_faced_mean']
test_df ['plus_home_pitching_wpa_def_batted_mean']  = test_df['home_batting_wpa_bat_mean'] +  test_df['home_pitching_wpa_def_mean']
test_df ['plus_away_pitching_wpa_def_batted_mean']  = test_df['away_batting_wpa_bat_mean'] + test_df['away_pitching_wpa_def_mean']
test_df['diff_plus_pitching_wpa_def_batted_mean'] = test_df['plus_home_pitching_wpa_def_batted_mean']-test_df['plus_away_pitching_wpa_def_batted_mean']
test_df['spread_team_mean']  = test_df['home_team_spread_mean'] / (test_df['away_team_spread_mean'] + 1)



test_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
test_df

unique_years = test_df['year'].unique()

for yr in unique_years:
 

    df_year = test_df[test_df['year'] == yr].copy()
 
    df_year.drop(columns=['year', 'date'], errors='ignore', inplace=True)
    
    y_pred = models[yr].predict(df_year)
    
    test_df.loc[df_year.index, 'home_team_win'] = y_pred


submission = test_df[['id', 'home_team_win']].copy()
submission.to_csv("submission_3_1.5.csv", index=False)

