import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
from xgboost.callback import EarlyStopping
import shap
warnings.filterwarnings('ignore')
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest

def analyze_night_game_effect(df):
    print("\n=== is_night_game 影響分析 ===")
    
    # 先確保 target 是 0/1
    df['home_win_int'] = df['home_team_win'].astype(int)
    
    # 分組
    day_games = df[df['is_night_game'] == False]
    night_games = df[df['is_night_game'] == True]
    unknown_games = df[df['is_night_game'].isna()]
    
    # 計算勝率與樣本數
    def win_rate_stats(group):
        if len(group) == 0:
            return 0.0, 0, 1.0
        wins = group['home_win_int'].sum()
        n = len(group)
        rate = wins / n
        return rate, n, wins
    
    day_rate, day_n, day_wins = win_rate_stats(day_games)
    night_rate, night_n, night_wins = win_rate_stats(night_games)
    unk_rate, unk_n, unk_wins = win_rate_stats(unknown_games)
    
    print(f"日間比賽：{day_n} 場，勝率 {day_rate:.4f} ({day_wins}/{day_n})")
    print(f"夜間比賽：{night_n} 場，勝率 {night_rate:.4f} ({night_wins}/{night_n})")
    print(f"未知 (NaN)：{unk_n} 場，勝率 {unk_rate:.4f} ({unk_wins}/{unk_n})")
    
    # 1. 單樣本檢定：每組 vs 0.5
    def binom_pvalue(n, wins):
        if n == 0:
            return 1.0
        return stats.binomtest(wins, n, p=0.5).pvalue
    
    day_p = binom_pvalue(day_n, day_wins)
    night_p = binom_pvalue(night_n, night_wins)
    unk_p = binom_pvalue(unk_n, unk_wins)
    
    print(f"\n單樣本檢定 (vs 隨機 0.5):")
    print(f"日間 p-value: {day_p:.4f} {'← 無顯著影響 (隨機)' if day_p > 0.05 else '← 有影響！'}")
    print(f"夜間 p-value: {night_p:.4f} {'← 有特別影響！' if night_p < 0.05 else '← 無顯著差異'}")
    print(f"未知 p-value: {unk_p:.4f}")
    
    # 2. 雙樣本檢定：日間 vs 夜間
    if day_n > 0 and night_n > 0:
        count = np.array([day_wins, night_wins])
        nobs = np.array([day_n, night_n])
        stat, p2 = proportions_ztest(count, nobs)
        print(f"\n日間 vs 夜間 差異檢定 p-value: {p2:.4f} {'← 顯著不同！' if p2 < 0.05 else '← 無顯著差異'}")
    else:
        print("\n樣本不足，無法比較日夜")
    
    # 3. 自動建議填值
    print("\n=== 填值建議 ===")
    if day_p > 0.05 and night_p < 0.05 and (day_n > 30 and night_n > 30):
        print("✅ 結論：日間勝率接近隨機，夜間有特別影響 → 建議把 NaN 填成 0（日間）")
        return 0  # 返回建議填值
    elif unk_rate > 0.6 or unk_rate < 0.4:  # 未知組本身有偏
        print("⚠️ 未知組勝率偏離 0.5，建議加 missing flag 不要強制填")
        return None
    else:
        print("❌ 不符合條件，建議保留 NaN + 加 missing flag")
        return None
# ==========================================
# 1. CONFIG
# ==========================================
CONFIG = {
    'SEED': 42,
    'TEST_START_MONTH': 8,
    'TARGET': 'home_team_win',
    'DROP_COLS': ['id', 'home_team_win', 'home_team_abbr', 'away_team_abbr', 
                  'home_pitcher', 'away_pitcher', 'home_team_season', 'away_team_season', 'date'],
    'XGB_PARAMS': {
        'n_estimators': 5000,
        'learning_rate': 0.01,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_jobs': -1,
        'random_state': 42,
        'tree_method': 'hist'
    },
    'EARLY_STOPPING_ROUNDS': 100,
    'CALIBRATION_METHOD': 'sigmoid'
}

# ==========================================
# 2. LOAD & PREPROCESS
# ==========================================
def load_and_preprocess(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Date
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # is_night_game: 安全處理 NaN → 0
    df['is_night_game'] = (df['is_night_game'] == True).astype(int)
    
    # Target to int
    df[CONFIG['TARGET']] = (df[CONFIG['TARGET']] == True).astype(int)
    
    # Missing handling (numeric only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[f'{col}_missing'] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(df.groupby('season')[col].transform('median'))
    suggested_fill = analyze_night_game_effect(df)
    if suggested_fill == 0:
        df['is_night_game'] = df['is_night_game'].fillna(False)  # 填 False → 後續轉 0
        print("已自動將 NaN 填成日間 (0)")
    elif suggested_fill is None:
        # 保守處理：加 flag + 填 0（或保留 NaN 讓 XGBoost 處理）
        df['is_night_game_missing'] = df['is_night_game'].isna().astype(int)
        df['is_night_game'] = df['is_night_game'].fillna(False)  # 還是填 0 但有 flag
        print("未通過檢測，已加 missing flag")
    
    # 最後轉 int
    df['is_night_game'] = (df['is_night_game'] == True).astype(int)
    return df

# ==========================================
# 3. SABERMETRICS ENGINEERING
# ==========================================
def engineer_sabermetrics(df):
    print("Engineering Sabermetric features...")
    
    for side in ['home', 'away']:
        df[f'{side}_batting_SLG_10RA'] = df[f'{side}_batting_onbase_plus_slugging_10RA'] - df[f'{side}_batting_onbase_perc_10RA']
        df[f'{side}_batting_ISO_10RA'] = df[f'{side}_batting_SLG_10RA'] - df[f'{side}_batting_batting_avg_10RA']
        df[f'{side}_batting_wOPS_10RA'] = (1.8 * df[f'{side}_batting_onbase_perc_10RA']) + df[f'{side}_batting_SLG_10RA']
        df[f'{side}_pitching_K_minus_BB_10RA'] = df[f'{side}_pitching_SO_batters_faced_10RA'] - df[f'{side}_pitching_BB_batters_faced_10RA']
        df[f'{side}_pitcher_K_minus_BB_10RA'] = df[f'{side}_pitcher_SO_batters_faced_10RA'] - df[f'{side}_pitcher_BB_batters_faced_10RA']
        df[f'{side}_starter_superiority'] = df[f'{side}_pitching_earned_run_avg_10RA'] - df[f'{side}_pitcher_earned_run_avg_10RA']
    
    # Interactions
    df['matchup_power_vs_pitcher'] = df['home_batting_ISO_10RA'] * df['away_pitcher_earned_run_avg_10RA']
    df['matchup_control_vs_patience'] = df['away_pitcher_K_minus_BB_10RA'] - df['home_batting_onbase_perc_10RA']
    df['rest_diff'] = df['home_team_rest'] - df['away_team_rest']
    df['pitcher_rest_diff'] = df['home_pitcher_rest'] - df['away_pitcher_rest']
    df['momentum_diff'] = df['home_team_wins_mean'] - df['away_team_wins_mean']
    df['home_advantage'] = df['momentum_diff'] + (df['rest_diff'] * 0.1)
    
    return df

# ==========================================
# 4. ENCODING (Team + Pitcher LabelEncoder)
# ==========================================
def encode_features(df):
    print("Encoding categorical features...")
    le_team = LabelEncoder()
    df['home_team_code'] = le_team.fit_transform(df['home_team_abbr'])
    df['away_team_code'] = le_team.fit_transform(df['away_team_abbr'])
    
    # Pitcher: fill missing + single encoder
    df['home_pitcher'] = df['home_pitcher'].fillna('unknown')
    df['away_pitcher'] = df['away_pitcher'].fillna('unknown')
    
    all_pitchers = pd.unique(pd.concat([df['home_pitcher'], df['away_pitcher']]))
    le_pitcher = LabelEncoder()
    le_pitcher.fit(all_pitchers)
    
    df['home_pitcher_code'] = le_pitcher.transform(df['home_pitcher'])
    df['away_pitcher_code'] = le_pitcher.transform(df['away_pitcher'])
    
    return df

# ==========================================
# 5. MAIN
# ==========================================
def main():
    try:
        df = load_and_preprocess('train_data.csv')
    except FileNotFoundError:
        print("Error: 'train_data.csv' not found.")
        return

    df = engineer_sabermetrics(df)
    df = encode_features(df)
    
    # Feature list: 包含所有滾動、歷史 mean/std/skew 等（你的數據很多這些）
    feature_cols = [c for c in df.columns if c not in CONFIG['DROP_COLS']]
    
    # Walk-forward split
    df = df.sort_values('date')
    years = sorted(df['year'].unique())
    
    if len(years) > 1:
        test_year = years[-1]
        train_mask = df['year'] < test_year
        test_mask = df['year'] == test_year
        print(f"Multi-season: Train before {test_year}, Test {test_year}")
    else:
        train_mask = df['month'] < CONFIG['TEST_START_MONTH']
        test_mask = df['month'] >= CONFIG['TEST_START_MONTH']
        print("Single-season: Train Jan-Jul, Test Aug-Dec")
    
    X_train_full = df.loc[train_mask, feature_cols]
    y_train_full = df.loc[train_mask, CONFIG['TARGET']]
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, CONFIG['TARGET']]
  
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=CONFIG['SEED'])
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    early_stop = EarlyStopping(
        rounds=CONFIG['EARLY_STOPPING_ROUNDS'], 
        save_best=True,
        metric_name='logloss',
        data_name='validation_0'
    )
    
    base_model = xgb.XGBClassifier(**CONFIG['XGB_PARAMS'])
    calibrated_model = CalibratedClassifierCV(base_model, method=CONFIG['CALIBRATION_METHOD'], cv='prefit')
    
    print("\nTraining with Early Stopping + Calibration...")

    base_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stop],
        verbose=100
    )
        
    calibrated_model.fit(X_train, y_train)  # 用訓練集校準
    
    # Predict on test
    preds_proba = calibrated_model.predict_proba(X_test)[:, 1]
    preds_class = (preds_proba > 0.5).astype(int)
    # 在 main() 最後，替換 feature importance 區塊
    import shap  # 加這行 import

    # ... 訓練完 base_model 和 calibrated_model 後 ...

    print("\n=== SHAP 分析 (TreeExplainer for XGBoost) ===")

    # SHAP TreeExplainer (最快，專為 XGBoost 優化)
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X_test)  # shape: (n_samples, n_features)

    # 1. 全局特徵重要性 (SHAP mean abs value > XGBoost gain)
    shap_importance = pd.DataFrame({
        'Feature': feature_cols,
        'SHAP_Mean_Abs': np.abs(shap_values).mean(axis=0)
    }).sort_values('SHAP_Mean_Abs', ascending=False)

    print("Top 15 SHAP Features (Mean |SHAP|):")
    print(shap_importance.head(15))

    # 2. 樣本級解釋 (前 10 場比賽)
    sample_idx = 0  # 或隨機選錯預測的樣本
    print(f"\nSample {sample_idx} Explanation (True: {y_test.iloc[sample_idx]}, Pred: {preds_proba[sample_idx]:.3f}):")
    single_shap = shap_values[sample_idx]
    shap.force_plot(explainer.expected_value, single_shap, X_test.iloc[sample_idx], matplotlib=True, show=False)
    plt.title(f"SHAP Force Plot - Match {sample_idx}")
    plt.show()

    # 3. Summary Plot (最重要視覺化)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, max_display=20, show=False)
    plt.title("SHAP Summary Plot (Red=推高勝率, Blue=壓低勝率)")
    plt.tight_layout()
    plt.show()

    # 4. Dependence Plot (找非線性關係，例如 pitcher_rest_diff)
    top_feature = shap_importance.iloc[0]['Feature']
    shap.dependence_plot(top_feature, shap_values, X_test, feature_names=feature_cols, show=False)
    plt.title(f"SHAP Dependence: {top_feature}")
    plt.show()

    # 5. 保存 SHAP 值 (供後續分析)
    shap_df = pd.DataFrame(shap_values, columns=[f'SHAP_{c}' for c in feature_cols])
    shap_df['true_label'] = y_test.values
    shap_df['pred_proba'] = preds_proba
    shap_df.to_csv('shap_explanations.csv', index=False)
    print("SHAP values saved to 'shap_explanations.csv'")
    # Metrics
    acc = accuracy_score(y_test, preds_class)
    ll = log_loss(y_test, preds_proba)
    bs = brier_score_loss(y_test, preds_proba)
    
    print("-" * 40)
    print("FINAL METRICS")
    print("-" * 40)
    print(f"Accuracy:    {acc:.4f}")
    print(f"Log Loss:    {ll:.4f}  ← 應該 < 0.65")
    print(f"Brier Score: {bs:.4f}")
    
    # Importance
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': base_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 15 Features:")
    print(importance.head(15))
    
    plt.figure(figsize=(10, 8))
    importance.head(15).plot(kind='barh', x='Feature', y='Importance', legend=False)
    plt.title('Top 15 Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":
    main()