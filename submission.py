import os
os.system('pip install -q catboost lightgbm xgboost geohash2 scikit-learn')

import pandas as pd
import numpy as np
import geohash2
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from google.colab import drive

try:
    drive.mount('/content/drive')
except:
    pass

DATA_DIR = '/content/drive/MyDrive/dataset/'

train = pd.read_csv(DATA_DIR + 'train.csv')
test = pd.read_csv(DATA_DIR + 'test.csv')
sample_sub = pd.read_csv(DATA_DIR + 'sample_submission.csv')


def parse_time_features(df):
    parts = df['timestamp'].str.split(':', expand=True).astype(int)
    df['hour'] = parts[0]
    df['minute'] = parts[1]
    df['time_in_minutes'] = df['hour'] * 60 + df['minute']
    df['time_idx'] = df['hour'] * 4 + df['minute'] // 15
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['time_in_minutes'] / 1440)
    df['minute_cos'] = np.cos(2 * np.pi * df['time_in_minutes'] / 1440)
    df['time_period'] = pd.cut(df['hour'], bins=[-1, 4, 8, 12, 16, 20, 24],
        labels=['late_night', 'early_morning', 'morning', 'afternoon', 'evening', 'night']).astype(str)
    df['day_of_week'] = df['day'] % 7
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    return df


def decode_geohash(df):
    coords = df['geohash'].apply(lambda g: geohash2.decode(g))
    df['latitude'] = coords.apply(lambda x: float(x[0]))
    df['longitude'] = coords.apply(lambda x: float(x[1]))
    df['geohash_4'] = df['geohash'].str[:4]
    df['geohash_5'] = df['geohash'].str[:5]
    return df


def smart_imputation(train_df, test_df):
    combined = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    temp_by_gh_hour = combined.groupby(['geohash', 'hour'])['Temperature'].transform('mean')
    temp_by_hour = combined.groupby('hour')['Temperature'].transform('mean')
    temp_global = combined['Temperature'].mean()
    combined['Temperature'] = combined['Temperature'].fillna(temp_by_gh_hour).fillna(temp_by_hour).fillna(temp_global)
    roadtype_map = combined.dropna(subset=['RoadType']).drop_duplicates('geohash').set_index('geohash')['RoadType']
    mask = combined['RoadType'].isna()
    combined.loc[mask, 'RoadType'] = combined.loc[mask, 'geohash'].map(roadtype_map)
    combined['RoadType'] = combined['RoadType'].fillna('Unknown')
    weather_map = combined.dropna(subset=['Weather']).drop_duplicates(subset=['geohash', 'hour']).set_index(['geohash', 'hour'])['Weather']
    mask = combined['Weather'].isna()
    for idx in combined[mask].index:
        key = (combined.loc[idx, 'geohash'], combined.loc[idx, 'hour'])
        if key in weather_map.index:
            combined.loc[idx, 'Weather'] = weather_map[key]
    combined['Weather'] = combined['Weather'].fillna('Unknown')
    n_train = len(train_df)
    return combined.iloc[:n_train].copy(), combined.iloc[n_train:].copy()


train = parse_time_features(train)
test = parse_time_features(test)

train = decode_geohash(train)
test = decode_geohash(test)

train, test = smart_imputation(train, test)


def build_historical_features(train_df, test_df):
    gh_stats = train_df.groupby('geohash')['demand'].agg(
        geohash_mean_demand='mean', geohash_std_demand='std', geohash_median_demand='median',
        geohash_max_demand='max', geohash_min_demand='min', geohash_count='count'
    ).reset_index()
    gh_stats['geohash_std_demand'] = gh_stats['geohash_std_demand'].fillna(0)
    train_df = train_df.merge(gh_stats, on='geohash', how='left')
    test_df = test_df.merge(gh_stats, on='geohash', how='left')
    global_mean = train_df['demand'].mean()
    for col in ['geohash_mean_demand', 'geohash_median_demand']:
        test_df[col] = test_df[col].fillna(global_mean)
    for col in ['geohash_std_demand', 'geohash_max_demand', 'geohash_min_demand']:
        test_df[col] = test_df[col].fillna(0)
    test_df['geohash_count'] = test_df['geohash_count'].fillna(0)
    
    gh_hour_stats = train_df.groupby(['geohash', 'hour'])['demand'].agg(gh_hour_mean='mean', gh_hour_std='std').reset_index()
    gh_hour_stats['gh_hour_std'] = gh_hour_stats['gh_hour_std'].fillna(0)
    train_df = train_df.merge(gh_hour_stats, on=['geohash', 'hour'], how='left')
    test_df = test_df.merge(gh_hour_stats, on=['geohash', 'hour'], how='left')
    test_df['gh_hour_mean'] = test_df['gh_hour_mean'].fillna(test_df['geohash_mean_demand'])
    test_df['gh_hour_std'] = test_df['gh_hour_std'].fillna(0)
    train_df['gh_hour_mean'] = train_df['gh_hour_mean'].fillna(train_df['geohash_mean_demand'])
    train_df['gh_hour_std'] = train_df['gh_hour_std'].fillna(0)
    
    day48 = train_df[train_df.day == 48][['geohash', 'timestamp', 'demand']].copy()
    day48_lag = day48.rename(columns={'demand': 'exact_prev_day_demand'})
    train_df = train_df.merge(day48_lag, on=['geohash', 'timestamp'], how='left')
    test_df = test_df.merge(day48_lag, on=['geohash', 'timestamp'], how='left')
    train_df['exact_prev_day_demand'] = train_df['exact_prev_day_demand'].fillna(train_df['geohash_mean_demand'])
    test_df['exact_prev_day_demand'] = test_df['exact_prev_day_demand'].fillna(test_df['geohash_mean_demand'])
    
    day48_by_gh = day48.copy()
    parts = day48_by_gh['timestamp'].str.split(':', expand=True).astype(int)
    day48_by_gh['time_idx'] = parts[0] * 4 + parts[1] // 15
    for offset, name in [(1, 'prev_day_lag_15min'), (2, 'prev_day_lag_30min'),
                          (-1, 'prev_day_lead_15min'), (-2, 'prev_day_lead_30min')]:
        shifted = day48_by_gh[['geohash', 'time_idx', 'demand']].copy()
        shifted['time_idx'] = shifted['time_idx'] - offset
        shifted = shifted.rename(columns={'demand': name})
        train_df = train_df.merge(shifted, on=['geohash', 'time_idx'], how='left')
        test_df = test_df.merge(shifted, on=['geohash', 'time_idx'], how='left')
        train_df[name] = train_df[name].fillna(train_df['geohash_mean_demand'])
        test_df[name] = test_df[name].fillna(test_df['geohash_mean_demand'])
        
    available_days = sorted(train_df['day'].unique())
    for ref_day in [44, 45, 46, 47]:
        if ref_day not in available_days:
            continue
        day_data = train_df[train_df.day == ref_day][['geohash', 'timestamp', 'demand']].copy()
        col_name = f'day{ref_day}_demand'
        day_data = day_data.rename(columns={'demand': col_name})
        day_data = day_data.drop_duplicates(subset=['geohash', 'timestamp'])
        train_df = train_df.merge(day_data, on=['geohash', 'timestamp'], how='left')
        test_df = test_df.merge(day_data, on=['geohash', 'timestamp'], how='left')
        train_df[col_name] = train_df[col_name].fillna(train_df['geohash_mean_demand'])
        test_df[col_name] = test_df[col_name].fillna(test_df['geohash_mean_demand'])
        
    multi_day_cols = [c for c in ['day44_demand', 'day45_demand', 'day46_demand', 'day47_demand'] if c in train_df.columns]
    if len(multi_day_cols) > 0:
        all_day_cols = ['exact_prev_day_demand'] + multi_day_cols
        for df in [train_df, test_df]:
            df['multi_day_mean'] = df[all_day_cols].mean(axis=1)
            df['multi_day_std'] = df[all_day_cols].std(axis=1).fillna(0)
            df['multi_day_trend'] = df['exact_prev_day_demand'] - df['multi_day_mean']
            if 'day47_demand' in df.columns:
                df['d48_d47_ratio'] = df['exact_prev_day_demand'] / (df['day47_demand'] + 1e-8)
            if 'day46_demand' in df.columns:
                df['d48_d46_ratio'] = df['exact_prev_day_demand'] / (df['day46_demand'] + 1e-8)
                
    train_sorted = train_df.sort_values(['geohash', 'day', 'time_idx'])
    last_known = train_sorted.groupby('geohash').tail(1)[['geohash', 'demand', 'time_idx', 'day']].copy()
    last_known = last_known.rename(columns={'demand': 'last_known_demand', 'time_idx': 'last_known_time_idx', 'day': 'last_known_day'})
    test_df = test_df.merge(last_known, on='geohash', how='left')
    test_df['last_known_demand'] = test_df['last_known_demand'].fillna(global_mean)
    test_df['last_known_time_idx'] = test_df['last_known_time_idx'].fillna(0)
    test_df['last_known_day'] = test_df['last_known_day'].fillna(48)
    test_df['time_gap_from_last'] = (test_df['day'] - test_df['last_known_day']) * 96 + test_df['time_idx'] - test_df['last_known_time_idx']
    
    train_sorted['last_known_demand'] = train_sorted.groupby('geohash')['demand'].shift(1).fillna(global_mean)
    train_sorted['prev_time_idx'] = train_sorted.groupby('geohash')['time_idx'].shift(1).fillna(0)
    train_sorted['prev_day'] = train_sorted.groupby('geohash')['day'].shift(1)
    train_sorted['prev_day'] = train_sorted['prev_day'].fillna(train_sorted['day'])
    train_sorted['time_gap_from_last'] = (train_sorted['day'] - train_sorted['prev_day']) * 96 + train_sorted['time_idx'] - train_sorted['prev_time_idx']
    train_df = train_sorted.copy()
    
    for lag in [2, 3, 4]:
        col = f'lag_{lag}_demand'
        train_df[col] = train_df.groupby('geohash')['demand'].shift(lag).fillna(global_mean)
        lag_data = train_sorted.groupby('geohash').apply(
            lambda g: g.iloc[-lag]['demand'] if len(g) >= lag else global_mean
        ).reset_index(name=col)
        test_df = test_df.merge(lag_data, on='geohash', how='left')
        test_df[col] = test_df[col].fillna(global_mean)
        
    for window in [3, 6, 12]:
        col = f'rolling_mean_{window}'
        train_df[col] = train_df.groupby('geohash')['demand'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        ).fillna(global_mean)
        roll_data = train_sorted.groupby('geohash').apply(
            lambda g: g['demand'].tail(window).mean() if len(g) > 0 else global_mean
        ).reset_index(name=col)
        test_df = test_df.merge(roll_data, on='geohash', how='left')
        test_df[col] = test_df[col].fillna(global_mean)
        
    train_df = train_df.drop(columns=['prev_time_idx', 'prev_day'])
    return train_df, test_df


train, test = build_historical_features(train, test)


def build_spatial_features(train_df, test_df):
    for prefix_len, prefix_col in [(4, 'geohash_4'), (5, 'geohash_5')]:
        agg_col = f'spatial_mean_{prefix_len}'
        spatial_stats = train_df.groupby(prefix_col)['demand'].mean().reset_index(name=agg_col)
        train_df = train_df.merge(spatial_stats, on=prefix_col, how='left')
        test_df = test_df.merge(spatial_stats, on=prefix_col, how='left')
        gm = train_df['demand'].mean()
        train_df[agg_col] = train_df[agg_col].fillna(gm)
        test_df[agg_col] = test_df[agg_col].fillna(gm)
        
        agg_col = f'spatial_hour_mean_{prefix_len}'
        spatial_hour = train_df.groupby([prefix_col, 'hour'])['demand'].mean().reset_index(name=agg_col)
        train_df = train_df.merge(spatial_hour, on=[prefix_col, 'hour'], how='left')
        test_df = test_df.merge(spatial_hour, on=[prefix_col, 'hour'], how='left')
        train_df[agg_col] = train_df[agg_col].fillna(train_df[f'spatial_mean_{prefix_len}'])
        test_df[agg_col] = test_df[agg_col].fillna(test_df[f'spatial_mean_{prefix_len}'])
        
    roadtype_enc = train_df.groupby('RoadType')['demand'].mean().to_dict()
    train_df['roadtype_target_enc'] = train_df['RoadType'].map(roadtype_enc)
    test_df['roadtype_target_enc'] = test_df['RoadType'].map(roadtype_enc).fillna(train_df['demand'].mean())
    lv_enc = train_df.groupby('LargeVehicles')['demand'].mean().to_dict()
    train_df['lv_target_enc'] = train_df['LargeVehicles'].map(lv_enc)
    test_df['lv_target_enc'] = test_df['LargeVehicles'].map(lv_enc).fillna(train_df['demand'].mean())
    
    for df in [train_df, test_df]:
        df['temp_x_lanes'] = df['Temperature'] * df['NumberofLanes']
        df['lanes_squared'] = df['NumberofLanes'] ** 2
        df['is_highway'] = (df['RoadType'] == 'Highway').astype(int)
        df['is_street'] = (df['RoadType'] == 'Street').astype(int)
        df['high_capacity'] = (df['NumberofLanes'] >= 4).astype(int)
        df['highway_x_hour'] = df['is_highway'] * df['hour']
        df['high_cap_x_hour'] = df['high_capacity'] * df['hour']
        df['demand_momentum'] = df['exact_prev_day_demand'] - df['geohash_mean_demand']
        df['prev_day_to_mean_ratio'] = df['exact_prev_day_demand'] / (df['geohash_mean_demand'] + 1e-8)
        
    d49_early = train_df[(train_df['day'] == 49) & (train_df['time_idx'] <= 8)]
    d48_early = train_df[(train_df['day'] == 48) & (train_df['time_idx'] <= 8)]
    d49_em = d49_early.groupby('geohash')['demand'].mean().rename('d49_early_mean')
    d48_em = d48_early.groupby('geohash')['demand'].mean().rename('d48_early_mean')
    early_cal = pd.concat([d49_em, d48_em], axis=1).reset_index()
    early_cal['d49_early_mean'] = early_cal['d49_early_mean'].fillna(0)
    early_cal['d48_early_mean'] = early_cal['d48_early_mean'].fillna(0)
    early_cal['early_morning_ratio'] = (early_cal['d49_early_mean'] + 0.01) / (early_cal['d48_early_mean'] + 0.01)
    early_cal['early_morning_ratio'] = early_cal['early_morning_ratio'].clip(0.5, 2.0)
    
    d49_residuals = d49_early.copy()
    d48_early_lookup = d48_early.set_index(['geohash', 'time_idx'])['demand']
    d49_residuals['d48_demand'] = d49_residuals.set_index(['geohash', 'time_idx']).index.map(
        lambda x: d48_early_lookup.get(x, np.nan)
    ).values
    d49_residuals['residual'] = d49_residuals['demand'] - d49_residuals['d48_demand']
    d49_residual_mean = d49_residuals.groupby('geohash')['residual'].mean().rename('d49_residual_mean').reset_index()
    d49_residual_mean['d49_residual_mean'] = d49_residual_mean['d49_residual_mean'].fillna(0)
    
    def compute_slope(group):
        if len(group) < 2: return 0.0
        x = group['time_idx'].values.astype(float)
        y = group['demand'].values
        x_mean, y_mean = x.mean(), y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0: return 0.0
        return ((x - x_mean) * (y - y_mean)).sum() / denom
        
    d49_trend = d49_early.groupby('geohash').apply(compute_slope).rename('d49_early_trend').reset_index()
    d49_vol = d49_early.groupby('geohash')['demand'].std().fillna(0).rename('d49_early_std').reset_index()
    
    for cal_df in [early_cal[['geohash', 'd49_early_mean', 'd48_early_mean', 'early_morning_ratio']],
                   d49_residual_mean, d49_trend, d49_vol]:
        train_df = train_df.merge(cal_df, on='geohash', how='left')
        test_df = test_df.merge(cal_df, on='geohash', how='left')
        
    for col in ['d49_early_mean', 'd48_early_mean', 'd49_residual_mean', 'd49_early_trend', 'd49_early_std']:
        train_df[col] = train_df[col].fillna(0)
        test_df[col] = test_df[col].fillna(0)
        
    train_df['early_morning_ratio'] = train_df['early_morning_ratio'].fillna(1.0)
    test_df['early_morning_ratio'] = test_df['early_morning_ratio'].fillna(1.0)
    train_df['exact_prev_day_adjusted'] = train_df['exact_prev_day_demand'] * train_df['early_morning_ratio']
    test_df['exact_prev_day_adjusted'] = test_df['exact_prev_day_demand'] * test_df['early_morning_ratio']
    
    for df in [train_df, test_df]:
        df['demand_velocity_15'] = df['exact_prev_day_demand'] - df['prev_day_lag_15min']
        df['demand_velocity_30'] = df['exact_prev_day_demand'] - df['prev_day_lag_30min']
        df['demand_acceleration'] = df['demand_velocity_15'] - (df['prev_day_lag_15min'] - df['prev_day_lag_30min'])
        df['demand_range_30'] = df[['exact_prev_day_demand', 'prev_day_lag_15min', 'prev_day_lag_30min']].max(axis=1) - \
                                df[['exact_prev_day_demand', 'prev_day_lag_15min', 'prev_day_lag_30min']].min(axis=1)
        lag_cols = ['exact_prev_day_demand', 'prev_day_lag_15min', 'prev_day_lag_30min']
        df['demand_lag_cv'] = df[lag_cols].std(axis=1) / (df[lag_cols].mean(axis=1) + 1e-8)
        
    return train_df, test_df


train, test = build_spatial_features(train, test)

num_features_v14 = [
    'hour', 'minute', 'time_in_minutes', 'time_idx',
    'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
    'latitude', 'longitude', 'NumberofLanes', 'Temperature',
    'geohash_mean_demand', 'geohash_std_demand', 'geohash_median_demand',
    'geohash_max_demand', 'geohash_min_demand', 'geohash_count',
    'gh_hour_mean', 'gh_hour_std',
    'exact_prev_day_demand', 'prev_day_lag_15min', 'prev_day_lag_30min',
    'prev_day_lead_15min', 'prev_day_lead_30min',
    'last_known_demand', 'time_gap_from_last',
    'lag_2_demand', 'lag_3_demand', 'lag_4_demand',
    'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
    'spatial_mean_4', 'spatial_mean_5', 'spatial_hour_mean_4', 'spatial_hour_mean_5',
    'roadtype_target_enc', 'lv_target_enc',
    'temp_x_lanes', 'lanes_squared', 'is_highway', 'is_street', 'high_capacity',
    'highway_x_hour', 'high_cap_x_hour', 'demand_momentum', 'prev_day_to_mean_ratio',
    'd49_early_mean', 'd48_early_mean', 'early_morning_ratio',
    'd49_residual_mean', 'd49_early_trend', 'd49_early_std',
    'exact_prev_day_adjusted',
    'day_of_week', 'day_of_week_sin', 'day_of_week_cos',
    'demand_velocity_15', 'demand_velocity_30', 'demand_acceleration',
    'demand_range_30', 'demand_lag_cv',
]

v22_feature_candidates = [
    'day44_demand', 'day45_demand', 'day46_demand', 'day47_demand',
    'd48_d47_ratio', 'd48_d46_ratio', 'multi_day_mean', 'multi_day_std', 'multi_day_trend',
]

num_features_v22_extra = [f for f in v22_feature_candidates if f in train.columns]

num_features = num_features_v14 + num_features_v22_extra

cat_features = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather', 'time_period', 'geohash_4', 'geohash_5']
all_features = num_features + cat_features

for col in cat_features:
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)
    
for col in num_features:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)
    
X_train = train[all_features].copy()
y_train = train['demand'].values
X_test = test[all_features].copy()

y_train_log = np.log1p(y_train)
y_train_sqrt = np.sqrt(y_train)

import subprocess
try:
    subprocess.check_output('nvidia-smi')
    TASK_TYPE = 'GPU'
except:
    TASK_TYPE = 'CPU'

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

model_names_v14 = ['cb1', 'lgb1', 'xgb', 'cb2', 'lgb_dart', 'et', 'cb_log', 'lgb_sqrt']
model_names_v21 = ['cb_mae', 'lgb_huber', 'xgb_mae']
model_names_v22 = ['cb_quantile', 'cb_shallow', 'lgb_goss']
model_names_v23 = ['hist_gb', 'cb_mape']
model_names_all = model_names_v14 + model_names_v21 + model_names_v22 + model_names_v23

oof_preds = {k: np.zeros(len(X_train)) for k in model_names_all}
test_preds = {k: np.zeros(len(X_test)) for k in model_names_all}

cat_feature_indices = [all_features.index(c) for c in cat_features]
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()

for col in cat_features:
    X_train_lgb[col] = X_train_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')
    
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    cb1 = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=8, l2_leaf_reg=3,
        cat_features=cat_feature_indices, verbose=0, random_seed=42+fold, task_type=TASK_TYPE, early_stopping_rounds=150)
    cb1.fit(X_train.iloc[tr_idx], y_train[tr_idx], eval_set=(X_train.iloc[val_idx], y_train[val_idx]))
    oof_preds['cb1'][val_idx] = cb1.predict(X_train.iloc[val_idx])
    test_preds['cb1'] += cb1.predict(X_test) / n_folds
    
    lgb1 = LGBMRegressor(n_estimators=2000, learning_rate=0.03, max_depth=9, num_leaves=127,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=42+fold, verbose=-1, n_jobs=-1)
    lgb1.fit(X_train_lgb.iloc[tr_idx], y_train[tr_idx],
        eval_set=[(X_train_lgb.iloc[val_idx], y_train[val_idx])], callbacks=[__import__('lightgbm').early_stopping(150), __import__('lightgbm').log_evaluation(0)])
    oof_preds['lgb1'][val_idx] = lgb1.predict(X_train_lgb.iloc[val_idx])
    test_preds['lgb1'] += lgb1.predict(X_test_lgb) / n_folds
    
    xgb_m = XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=8,
        min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
        random_state=42+fold, verbosity=0, n_jobs=-1, tree_method='hist',
        enable_categorical=False, early_stopping_rounds=150)
    xgb_m.fit(X_train[num_features].iloc[tr_idx], y_train[tr_idx],
        eval_set=[(X_train[num_features].iloc[val_idx], y_train[val_idx])], verbose=0)
    oof_preds['xgb'][val_idx] = xgb_m.predict(X_train[num_features].iloc[val_idx])
    test_preds['xgb'] += xgb_m.predict(X_test[num_features]) / n_folds
    
    cb2 = CatBoostRegressor(iterations=1500, learning_rate=0.05, depth=10, l2_leaf_reg=10,
        cat_features=cat_feature_indices, verbose=0, random_seed=123+fold, task_type=TASK_TYPE, early_stopping_rounds=100)
    cb2.fit(X_train.iloc[tr_idx], y_train[tr_idx], eval_set=(X_train.iloc[val_idx], y_train[val_idx]))
    oof_preds['cb2'][val_idx] = cb2.predict(X_train.iloc[val_idx])
    test_preds['cb2'] += cb2.predict(X_test) / n_folds
    
    lgb_dart = LGBMRegressor(boosting_type='dart', n_estimators=500, learning_rate=0.1,
        max_depth=8, random_state=123+fold, verbose=-1, n_jobs=-1)
    lgb_dart.fit(X_train_lgb.iloc[tr_idx], y_train[tr_idx])
    oof_preds['lgb_dart'][val_idx] = lgb_dart.predict(X_train_lgb.iloc[val_idx])
    test_preds['lgb_dart'] += lgb_dart.predict(X_test_lgb) / n_folds
    
    et = ExtraTreesRegressor(n_estimators=200, max_depth=20, min_samples_leaf=5,
        random_state=42+fold, n_jobs=-1)
    et.fit(X_train[num_features].iloc[tr_idx], y_train[tr_idx])
    oof_preds['et'][val_idx] = et.predict(X_train[num_features].iloc[val_idx])
    test_preds['et'] += et.predict(X_test[num_features]) / n_folds
    
    cb_log = CatBoostRegressor(iterations=1500, learning_rate=0.04, depth=8,
        cat_features=cat_feature_indices, verbose=0, random_seed=77+fold, task_type=TASK_TYPE, early_stopping_rounds=100)
    cb_log.fit(X_train.iloc[tr_idx], y_train_log[tr_idx], eval_set=(X_train.iloc[val_idx], y_train_log[val_idx]))
    oof_preds['cb_log'][val_idx] = np.expm1(cb_log.predict(X_train.iloc[val_idx]))
    test_preds['cb_log'] += np.expm1(cb_log.predict(X_test)) / n_folds
    
    lgb_sqrt = LGBMRegressor(n_estimators=1500, learning_rate=0.04, max_depth=9,
        random_state=77+fold, verbose=-1, n_jobs=-1)
    lgb_sqrt.fit(X_train_lgb.iloc[tr_idx], y_train_sqrt[tr_idx],
        eval_set=[(X_train_lgb.iloc[val_idx], y_train_sqrt[val_idx])], callbacks=[__import__('lightgbm').early_stopping(100), __import__('lightgbm').log_evaluation(0)])
    oof_preds['lgb_sqrt'][val_idx] = lgb_sqrt.predict(X_train_lgb.iloc[val_idx]) ** 2
    test_preds['lgb_sqrt'] += (lgb_sqrt.predict(X_test_lgb) ** 2) / n_folds
    
    cb_mae = CatBoostRegressor(iterations=1500, learning_rate=0.04, depth=8, l2_leaf_reg=3, loss_function='MAE',
        cat_features=cat_feature_indices, verbose=0, random_seed=300+fold, task_type=TASK_TYPE, early_stopping_rounds=100)
    cb_mae.fit(X_train.iloc[tr_idx], y_train[tr_idx], eval_set=(X_train.iloc[val_idx], y_train[val_idx]))
    oof_preds['cb_mae'][val_idx] = cb_mae.predict(X_train.iloc[val_idx])
    test_preds['cb_mae'] += cb_mae.predict(X_test) / n_folds
    
    lgb_huber = LGBMRegressor(objective='huber', n_estimators=1500, learning_rate=0.04, max_depth=9, num_leaves=127,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=300+fold, verbose=-1, n_jobs=-1)
    lgb_huber.fit(X_train_lgb.iloc[tr_idx], y_train[tr_idx], eval_set=[(X_train_lgb.iloc[val_idx], y_train[val_idx])], callbacks=[__import__('lightgbm').early_stopping(100), __import__('lightgbm').log_evaluation(0)])
    oof_preds['lgb_huber'][val_idx] = lgb_huber.predict(X_train_lgb.iloc[val_idx])
    test_preds['lgb_huber'] += lgb_huber.predict(X_test_lgb) / n_folds
    
    xgb_mae = XGBRegressor(n_estimators=1500, learning_rate=0.04, max_depth=8, objective='reg:absoluteerror',
        min_child_weight=20, subsample=0.8, colsample_bytree=0.8, random_state=300+fold, verbosity=0, n_jobs=-1, tree_method='hist', enable_categorical=False, early_stopping_rounds=100)
    xgb_mae.fit(X_train[num_features].iloc[tr_idx], y_train[tr_idx], eval_set=[(X_train[num_features].iloc[val_idx], y_train[val_idx])], verbose=0)
    oof_preds['xgb_mae'][val_idx] = xgb_mae.predict(X_train[num_features].iloc[val_idx])
    test_preds['xgb_mae'] += xgb_mae.predict(X_test[num_features]) / n_folds
    
    cb_quantile = CatBoostRegressor(iterations=1500, learning_rate=0.04, depth=8, loss_function='Quantile:alpha=0.5',
        cat_features=cat_feature_indices, verbose=0, random_seed=500+fold, task_type=TASK_TYPE, early_stopping_rounds=100)
    cb_quantile.fit(X_train.iloc[tr_idx], y_train[tr_idx], eval_set=(X_train.iloc[val_idx], y_train[val_idx]))
    oof_preds['cb_quantile'][val_idx] = cb_quantile.predict(X_train.iloc[val_idx])
    test_preds['cb_quantile'] += cb_quantile.predict(X_test) / n_folds
    
    cb_shallow = CatBoostRegressor(iterations=2500, learning_rate=0.02, depth=6, l2_leaf_reg=5,
        cat_features=cat_feature_indices, verbose=0, random_seed=500+fold, task_type=TASK_TYPE, early_stopping_rounds=200)
    cb_shallow.fit(X_train.iloc[tr_idx], y_train[tr_idx], eval_set=(X_train.iloc[val_idx], y_train[val_idx]))
    oof_preds['cb_shallow'][val_idx] = cb_shallow.predict(X_train.iloc[val_idx])
    test_preds['cb_shallow'] += cb_shallow.predict(X_test) / n_folds
    
    lgb_goss = LGBMRegressor(boosting_type='goss', n_estimators=1500, learning_rate=0.04, max_depth=8, num_leaves=63,
        random_state=500+fold, verbose=-1, n_jobs=-1)
    lgb_goss.fit(X_train_lgb.iloc[tr_idx], y_train[tr_idx], eval_set=[(X_train_lgb.iloc[val_idx], y_train[val_idx])], callbacks=[__import__('lightgbm').early_stopping(100), __import__('lightgbm').log_evaluation(0)])
    oof_preds['lgb_goss'][val_idx] = lgb_goss.predict(X_train_lgb.iloc[val_idx])
    test_preds['lgb_goss'] += lgb_goss.predict(X_test_lgb) / n_folds
    
    hist_gb = HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05, max_depth=9,
        min_samples_leaf=20, random_state=800+fold, early_stopping=True, validation_fraction=0.1)
    hist_gb.fit(X_train[num_features].iloc[tr_idx], y_train[tr_idx])
    oof_preds['hist_gb'][val_idx] = hist_gb.predict(X_train[num_features].iloc[val_idx])
    test_preds['hist_gb'] += hist_gb.predict(X_test[num_features]) / n_folds
    
    cb_mape = CatBoostRegressor(iterations=1500, learning_rate=0.04, depth=8, loss_function='MAPE',
        cat_features=cat_feature_indices, verbose=0, random_seed=800+fold, task_type=TASK_TYPE, early_stopping_rounds=100)
    cb_mape.fit(X_train.iloc[tr_idx], y_train[tr_idx]+1e-5, eval_set=(X_train.iloc[val_idx], y_train[val_idx]+1e-5))
    oof_preds['cb_mape'][val_idx] = np.clip(cb_mape.predict(X_train.iloc[val_idx]), 0, None)
    test_preds['cb_mape'] += np.clip(cb_mape.predict(X_test), 0, None) / n_folds


stack_a_train = np.column_stack([oof_preds[k] for k in model_names_v14])
stack_a_test = np.column_stack([test_preds[k] for k in model_names_v14])

meta_a = BayesianRidge()
meta_a.fit(stack_a_train, y_train)
pred_a = np.clip(meta_a.predict(stack_a_test), 0, 1)

stack_c_train = np.column_stack([oof_preds[k] for k in model_names_all])
stack_c_test = np.column_stack([test_preds[k] for k in model_names_all])

meta_c = BayesianRidge()
meta_c.fit(stack_c_train, y_train)
pred_c = np.clip(meta_c.predict(stack_c_test), 0, 1)

pd.DataFrame({'Index': test['Index'], 'demand': pred_c}).to_csv('submission_v23_16models.csv', index=False)

for w_a in [0.1, 0.2, 0.3]:
    pred_blend = np.clip(w_a * pred_a + (1 - w_a) * pred_c, 0, 1)
    fname = f'submission_v23_blend_{int(w_a*100)}a_{int((1-w_a)*100)}c.csv'
    pd.DataFrame({'Index': test['Index'], 'demand': pred_blend}).to_csv(fname, index=False)
