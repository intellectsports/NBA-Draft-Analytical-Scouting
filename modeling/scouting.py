# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 19:40:42 2025

@author: natha
"""

import pandas as pd
data = pd.read_csv("C:/Users/natha/Downloads/nbaprospectnbacareersummary - Sheet1.csv")
collegeonly = data[data['College'].notna()]
noncollege = data[data['College'].isna()]
collegeonly.to_csv('C:/Users/natha/Downloads/collegeonly.csv',index=False)
noncollege.to_csv('C:/Users/natha/Downloads/noncollege.csv',index=False)
cbbdata = pd.read_csv("C:/Users/natha/Downloads/cbb_player_data.csv")
teaminfo = pd.read_csv("C:/Users/natha/Downloads/teaminfo.csv")
combinedata = pd.read_csv("C:/Users/natha/Downloads/nbadraft25/Draft Combine - Kaggle.csv")
print(collegeonly.info())
print(teaminfo.info())
print(combinedata.info())
id_cols = {'Year', 'Rk', 'Pk', 'Tm', 'Player', 'College', 'Yrs', 'player'}
per_game_map = {
    'MP.1': 'mp/gm_nba',
    'PTS.1': 'pts/gm_nba',
    'TRB.1': 'trb/gm_nba',
    'AST.1': 'ast/gm_nba'
}
advanced_cols = {'WS', 'WS/48', 'BPM', 'VORP', 'VORP/Gm', 'VORP/Yr'}

def rename_col(col):
    if col in id_cols:
        return col
    elif col in per_game_map:
        return per_game_map[col]
    elif col in advanced_cols:
        return col
    else:
        return f"{col}_nba"

collegeonly.columns = [rename_col(col) for col in collegeonly.columns]
#-------------------------
cbbdata['player'] = cbbdata['player'].str.strip().str.lower()
import re

def clean_name(name):
    if pd.isna(name):
        return name
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name

cbbdata['player'] = cbbdata['player'].apply(clean_name)
collegeonly['player'] = collegeonly['Player'].apply(clean_name)
merged = pd.merge(cbbdata, collegeonly, on='player', how='inner')
#-------------------------
def convert_exp_to_year(exp):
    exp_map = {
        'fr': 1,
        'so': 0,
        'jr': 0,
        'sr': 0
    }
    if pd.isna(exp):
        return None
    return exp_map.get(exp.strip().lower(), None) 

merged['fr'] = merged['exp'].apply(convert_exp_to_year)
print(merged['pos'].unique())
def group_position(pos):
    pos = pos.lower()
    if pos in ['pure pg', 'scoring pg','combo g']:
        return 'g'
    elif pos in ['wing f', 'wing g']:
        return 'wing'
    elif pos in ['pf/c', 'c']:
        return 'big'
    elif pos == 'stretch 4':
        return 'stretch'
    else:
        return 'other'

merged['pos_grouped'] = merged['pos'].apply(group_position)
#---------------------------------------------------------------
#---------------- Team Related Stuff ---------------------------
#---------------------------------------------------------------
"""team_info = teaminfo.rename(columns={
    'Mapped ESPN Team Name':'team',
    'Season':'year'})
team_info['team'] = team_info['team'].str.strip().str.lower()
merged['team'] = merged['team'].str.strip().str.lower()
team_info['team'] = team_info['team'].apply(clean_name)
merged['team'] = merged['team'].apply(clean_name)
merged.to_csv("C:/Users/natha/Downloads/nbadraft25/merged.csv", index=False)
merged = pd.read_csv("C:/Users/natha/Downloads/nbadraft25/merged.csv")
merged = pd.merge(merged, team_info, on=['team','year'], how='left')"""
print(team_info['Short Conference Name'].unique())
power_confs = ['ACC', 'SEC', 'B10', 'B12', 'P10', 'P12', 'BE']

team_info['power_conf'] = team_info['Short Conference Name'].apply(
    lambda x: 1 if x in power_confs else 0
)
teampower = team_info[['team','year','power_conf']]
merged = pd.merge(merged, teampower, on=['team','year'], how='left')
merged['power_conf'] = merged['conf'].apply(
    lambda x: 1 if x in power_confs else 0
)

#-----------------------------------------------------------
#----------------- Adding in Combine Data ------------------
#-----------------------------------------------------------
combinedata = combinedata.rename(columns={
    'YEAR':'year',
    'PLAYER':'player'})
def clean_name(name):
    if pd.isna(name):
        return ''
    
    parts = name.split(',')
    if len(parts) != 2:
        return name.lower().strip()
    
    last = parts[0].strip()
    first = parts[1].strip()

    suffixes = ['jr', 'sr', 'ii', 'iii', 'iv']
    first_parts = first.lower().replace('.', '').split()
    last_parts = last.lower().replace('.', '').split()

    suffix = ''
    if first_parts and first_parts[-1] in suffixes:
        suffix = first_parts.pop(-1)

    full = ' '.join(first_parts + last_parts)
    if suffix:
        full += f' {suffix}'
    
    return full.strip()

combinedata['player'] = combinedata['player'].apply(clean_name)
def clean_name_extended(name):
    cleaned = clean_name(name)
    cleaned = re.sub(r'[^a-z0-9\s]', '', cleaned) 
    cleaned = re.sub(r'\s+', ' ', cleaned)       
    return cleaned.strip().lower()

combinedata['player'] = combinedata['player'].apply(clean_name_extended)
combinedata.to_csv("C:/Users/natha/Downloads/nbadraft25/combinedata.csv")
merged = pd.merge(merged, combinedata, on=['player','year'], how='left')
combine_stat_cols = ['HGT', 'WGT', 'BMI', 'BF', 'WNGSPN', 'STNDRCH', 'HANDL', 'HANDW',
                     'STNDVERT', 'LPVERT', 'LANE', 'SHUTTLE', 'SPRINT', 'BENCH', 'PAN', 'PBHGT', 'PDHGT']
def compute_combine_percentiles(group):
    return group[combine_stat_cols].rank(pct=True) * 100

combine_percentiles = merged.groupby('pos_grouped', group_keys=False).apply(compute_combine_percentiles)
combine_percentiles.columns = [f"{col}_pct" for col in combine_percentiles.columns]
merged = pd.concat([merged.reset_index(drop=True), combine_percentiles.reset_index(drop=True)], axis=1)
#-----------------------------------------------------------
#----------------- Modeling and such -----------------------
#-----------------------------------------------------------
print(merged.columns.tolist())
print(merged.dtypes.astype(str).to_string())
def pick_to_tier(Pk):
    try:
        pick = int(Pk)
    except:
        return 6
    if pick == 1:
        return 0
    elif pick <= 5:
        return 1
    elif pick <= 14:
        return 2
    elif pick <= 30:
        return 3
    elif pick <= 45:
        return 4
    elif pick <= 60:
        return 5
    else:
        return 6

merged['picktier'] = merged['Pk'].apply(pick_to_tier)

stat_cols = [
    'ppg', 'rpg', 'apg', 'spg', 'bpg', 'tov', 'usg', 'ortg', 'efg', 'ts',
    'obpm', 'dbpm', 'bpm', 'porpag', 'dporpag', 'adj_oe', 'drtg', 'adj_de'
]
scouting_df = merged[merged['fr'].notna() & merged['pos_grouped'].notna()].copy()
scouting_df['pos_fr_group'] = scouting_df['pos_grouped'] + '_Y' + scouting_df['fr'].astype(str)

def compute_percentiles(group):
    return group[stat_cols].rank(pct=True) * 100

percentiles = scouting_df.groupby('pos_fr_group', group_keys=False).apply(compute_percentiles)
percentiles.columns = [f"{col}_pct" for col in percentiles.columns]

combine_cols = [col for col in merged.columns if col.endswith('_pct') and col not in percentiles.columns]
combine_percentiles = scouting_df[combine_cols].reset_index(drop=True)

scouting_df_final = pd.concat([
    scouting_df[['player', 'pos_grouped', 'fr', 'VORP', 'VORP/Gm', 'VORP/Yr', 'power_conf', 'picktier']].reset_index(drop=True),
    percentiles.reset_index(drop=True),
    combine_percentiles
], axis=1)
print(scouting_df_final.info())
combine_pct_cols = [col for col in scouting_df_final.columns if col.endswith('_pct') and col not in ['ppg_pct', 'rpg_pct', 'apg_pct', 'spg_pct', 'bpg_pct', 'tov_pct', 'usg_pct', 'ortg_pct', 'efg_pct', 'ts_pct', 'obpm_pct', 'dbpm_pct', 'bpm_pct', 'porpag_pct', 'dporpag_pct', 'adj_oe_pct', 'drtg_pct', 'adj_de_pct']]
for col in combine_pct_cols:
    scouting_df_final[col] = scouting_df_final.groupby('pos_grouped')[col].transform(lambda x: x.fillna(x.median()))
scouting_df_final.to_csv("C:/Users/natha/Downloads/nbadraft25/scoutingdf.csv")
#-------------------
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import numpy as np

features = [
    'ppg_pct','rpg_pct','apg_pct','spg_pct','bpg_pct','tov_pct','usg_pct',
    'ortg_pct','efg_pct','ts_pct','obpm_pct','dbpm_pct','bpm_pct',
    'porpag_pct','dporpag_pct','adj_oe_pct','drtg_pct','adj_de_pct',
    'HGT_pct','WGT_pct','BMI_pct','BF_pct','WNGSPN_pct','STNDRCH_pct',
    'HANDL_pct','HANDW_pct','STNDVERT_pct','LPVERT_pct','LANE_pct','SHUTTLE_pct',
    'SPRINT_pct','BENCH_pct','PAN_pct','PBHGT_pct','PDHGT_pct',
    'fr','picktier','power_conf','pos_grouped'
]

scouting_df_final['VORP/Yr'] = scouting_df_final['VORP/Yr'].replace('#DIV/0!', np.nan)
scouting_df_final['VORP/Yr'] = pd.to_numeric(scouting_df_final['VORP/Yr'], errors='coerce')
scouting_df_final = scouting_df_final.dropna(subset=['VORP/Yr'])

X = scouting_df_final[features].copy()
X = pd.get_dummies(X, columns=['pos_grouped'], drop_first=True)

y = scouting_df_final['VORP/Yr']
X = X.fillna(X.median())
y = y.fillna(y.median())

models = {
    'Random Forest': RandomForestRegressor(n_estimators=1000, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=1000, random_state=42),
    'Linear Regression': LinearRegression(),
    'Elastic Net': make_pipeline(StandardScaler(), ElasticNet(random_state=42)),
    'SVR': make_pipeline(StandardScaler(), SVR(kernel='rbf')),
    'MLP Regressor': make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(50,50), max_iter=10000, random_state=42)),
    'XGBoost': xgb.XGBRegressor(n_estimators=1000, random_state=42, eval_metric='rmse', use_label_encoder=False),
    'LightGBM': lgb.LGBMRegressor(n_estimators=1000, random_state=42),
    'CatBoost': CatBoostRegressor(iterations=1000, random_seed=42, verbose=False)
}

kf = KFold(n_splits=10, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    pipeline = model if name in ['Elastic Net', 'SVR', 'MLP Regressor'] else make_pipeline(StandardScaler(), model)
    neg_mse_scores = cross_val_score(pipeline, X, y, scoring='neg_mean_squared_error', cv=kf)
    rmse_scores = np.sqrt(-neg_mse_scores)
    r2_scores = cross_val_score(pipeline, X, y, scoring='r2', cv=kf)
    results[name] = {'RMSE Mean': rmse_scores.mean(), 'RMSE Std': rmse_scores.std(),
                     'R2 Mean': r2_scores.mean(), 'R2 Std': r2_scores.std()}

results_df = pd.DataFrame(results).T.sort_values('RMSE Mean')
print(results_df)

#-------------------------------------------------------------------------------
rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

top_models = {
    'CatBoost': make_pipeline(StandardScaler(), CatBoostRegressor(iterations=1000, random_seed=42, verbose=False)),
    'SVR': make_pipeline(StandardScaler(), SVR(kernel='rbf')),
    'Linear Regression': make_pipeline(StandardScaler(), LinearRegression())
}

kf = KFold(n_splits=10, shuffle=True, random_state=42)

rmse_scores = []
r2_scores = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    preds = []
    for model in top_models.values():
        model.fit(X_train, y_train)
        preds.append(model.predict(X_test))
    
    ensemble_preds = np.mean(preds, axis=0)
    
    rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds))
    r2 = r2_score(y_test, ensemble_preds)
    
    rmse_scores.append(rmse)
    r2_scores.append(r2)

print("Ensemble RMSE Mean:", np.mean(rmse_scores))
print("Ensemble RMSE Std:", np.std(rmse_scores))
print("Ensemble R2 Mean:", np.mean(r2_scores))
print("Ensemble R2 Std:", np.std(r2_scores))
combine_features = [col for col in X.columns if col.endswith('_pct') and col.split('_')[0] in [
    'HGT','WGT','BMI','BF','WNGSPN','STNDRCH','HANDL','HANDW','STNDVERT','LPVERT','LANE','SHUTTLE','SPRINT','BENCH','PAN','PBHGT','PDHGT'
]]
non_combine_features = [col for col in X.columns if col not in combine_features]

X = X[non_combine_features]

cat = CatBoostRegressor(iterations=1000, random_seed=42, verbose=False)
cat.fit(X, y)
importances = cat.get_feature_importance(prettified=True)
print(importances.sort_values(by='Importances', ascending=False))

cat.fit(X, y)
preds = cat.predict(X)
residuals = y - preds
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,6))
sns.histplot(residuals, bins=30, kde=True)
plt.axvline(0, color='red', linestyle='--')
plt.title("Residuals: Actual VORP/Yr - Predicted VORP/Yr")
plt.xlabel("Residual")
plt.ylabel("Count")
plt.show()

# -------------------------------
# Random Forest Hyperparameter Tuning
# -------------------------------
"""rf_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('rf', RandomForestRegressor(random_state=42))
])

rf_param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2]
}

rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring=rmse_scorer, n_jobs=-1, verbose=1)
rf_grid.fit(X, y)

print("Best Random Forest RMSE:", -rf_grid.best_score_)
print("Best RF Params:", rf_grid.best_params_)

# -------------------------------
# SVR Hyperparameter Tuning
# -------------------------------
svr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR())
])

svr_param_grid = {
    'svr__C': [0.1, 1, 10],
    'svr__epsilon': [0.01, 0.1, 0.2],
    'svr__kernel': ['rbf', 'linear']
}

svr_grid = GridSearchCV(svr_pipeline, svr_param_grid, cv=5, scoring=rmse_scorer, n_jobs=-1, verbose=1)
svr_grid.fit(X, y)

print("Best SVR RMSE:", -svr_grid.best_score_)
print("Best SVR Params:", svr_grid.best_params_)
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# -------------------------------
# Ridge Regression Tuning
# -------------------------------
ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

ridge_param_grid = {
    'ridge__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
}

ridge_grid = GridSearchCV(ridge_pipeline, ridge_param_grid, cv=5, scoring=rmse_scorer, n_jobs=-1, verbose=1)
ridge_grid.fit(X, y)

print("Best Ridge RMSE:", -ridge_grid.best_score_)
print("Best Ridge Params:", ridge_grid.best_params_)

# -------------------------------
# Lasso Regression Tuning
# -------------------------------
lasso_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(max_iter=10000))
])

lasso_param_grid = {
    'lasso__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]
}

lasso_grid = GridSearchCV(lasso_pipeline, lasso_param_grid, cv=5, scoring=rmse_scorer, n_jobs=-1, verbose=1)
lasso_grid.fit(X, y)

print("Best Lasso RMSE:", -lasso_grid.best_score_)
print("Best Lasso Params:", lasso_grid.best_params_)

# -------------------------------
# ElasticNet Tuning
# -------------------------------
enet_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('enet', ElasticNet(max_iter=10000))
])

enet_param_grid = {
    'enet__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'enet__l1_ratio': [0.1, 0.5, 0.7, 0.9]
}

enet_grid = GridSearchCV(enet_pipeline, enet_param_grid, cv=5, scoring=rmse_scorer, n_jobs=-1, verbose=1)
enet_grid.fit(X, y)

print("Best ElasticNet RMSE:", -enet_grid.best_score_)
print("Best ElasticNet Params:", enet_grid.best_params_)
"elasticnet, svr, random forest the best"
#----------------------------------------------------------
#------------------ Ensembling Models ---------------------
#----------------------------------------------------------
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

# Set up models with best known hyperparameters
enet = make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000))
lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.001, max_iter=10000))
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Voting Regressor Ensemble
ensemble = VotingRegressor([('enet', enet), ('lasso', lasso), ('rf', rf)])

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = -cross_val_score(ensemble, X, y, scoring='neg_root_mean_squared_error', cv=kf)
r2_scores = cross_val_score(ensemble, X, y, scoring='r2', cv=kf)

print("Ensemble RMSE Mean:", rmse_scores.mean())
print("Ensemble RMSE Std:", rmse_scores.std())
print("Ensemble R2 Mean:", r2_scores.mean())
print("Ensemble R2 Std:", r2_scores.std())
#------------------------------------------------------------------------------
import joblib
import os
ensemble.fit(X, y)
save_dir = r"C:\Users\natha\Downloads\nbadraft25"
ensemble.fit(X, y)
joblib.dump(ensemble, os.path.join(save_dir, 'vorp_ensemble_model.pkl'))
joblib.dump(X.columns.tolist(), os.path.join(save_dir, 'vorp_model_features.pkl'))

load_dir = r"C:\Users\natha\Downloads\nbadraft25"
ensemble = joblib.load(os.path.join(load_dir, 'vorp_ensemble_model.pkl'))
feature_cols = joblib.load(os.path.join(load_dir, 'vorp_model_features.pkl'))
#------------------------------------------------------------------------------

"""