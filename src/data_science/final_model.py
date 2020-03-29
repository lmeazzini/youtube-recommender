import pandas as pd
import numpy as np
from scipy.sparse import hstack
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib as jb
from sklearn.metrics import roc_auc_score, average_precision_score
from skopt import forest_minimize
import matplotlib.pyplot as plt
import scikitplot as skplt


def tune_lgbm(params):
    lr = params[0]
    max_depth = params[1]
    min_child_samples = params[2]
    subsample = params[3]
    colsample_bytree = params[4]
    n_estimators = params[5]

    min_df = params[6]
    ngram_range = (1, params[7])

    title_vec = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range)
    title_bow_train = title_vec.fit_transform(train_titles)
    title_bow_val = title_vec.transform(val_titles)

    X_train_title = hstack([X_train, title_bow_train])
    X_val_title = hstack([X_val, title_bow_val])

    mdl = LGBMClassifier(learning_rate=lr, num_leaves=2 ** max_depth,
                         max_depth=max_depth,
                         min_child_samples=min_child_samples,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         bagging_freq=1, n_estimators=n_estimators,
                         random_state=42, class_weight="balanced", n_jobs=8)
    mdl.fit(X_train_title, y_train)

    p = mdl.predict_proba(X_val_title)[:, 1]

    print(roc_auc_score(y_val, p))

    return -average_precision_score(y_val, p)


df = pd.read_csv('../../data/raw_data_labeled.csv')
df = df[df['y'].notnull()]

# Clean date attribute
clean_date = df['watch-time-text'].str.extract(r"(\d+) de ([a-z]+)\. de (\d+)")
clean_date[0] = clean_date[0].dropna().astype(str)
clean_date[2] = clean_date[2].dropna().astype(str)

month_map = {"jan": "Jan",
             "fev": "Feb",
             "mar": "Mar",
             "abr": "Apr",
             "mai": "May",
             "jun": "Jun",
             "jul": "Jul",
             "ago": "Aug",
             "set": "Sep",
             "out": "Oct",
             "nov": "Nov",
             "dez": "Dec"}

clean_date[1] = clean_date[1].map(month_map)
clean_date = clean_date.dropna().apply(lambda x: " ".join(x), axis=1)
clean_date = pd.to_datetime(clean_date, format="%d %b %Y")

# Clean view number
views = df['watch-view-count'].str.extract(r"(\d+\.?\d*)", expand=False)
views = views.str.replace(".", "").fillna(0).astype(int)

features = pd.DataFrame()
y = df['y'].copy()

features['time_since_pub'] = (pd.to_datetime("2020-03-29") -  # HARDCODED
                              clean_date) / np.timedelta64(1, 'D')

# Extracting n of view feature
features['views'] = views

# Extracting n of view/day feature
features['views_per_day'] = features['views'] / features['time_since_pub']

# Droping time_since_pub to prevent bias
features = features.drop(['time_since_pub'], axis=1)

# Dropping problematic features
y = y[features.index]
df = df.loc[features.index]

resolutions = []
for height, width in zip(df['og:video:height'], df['og:video:width']):
    try:
        height = float(height)
        width = float(width)
    except ValueError:
        resolutions.append(np.nan)
        continue

    resolutions.append(height*width)

features['resolution'] = resolutions

# Around 75% train and 25% to validation
split_date = '2019-09-01'
mask_train = (clean_date < split_date) & (clean_date.notnull())
mask_val = (clean_date >= split_date) & (clean_date.notnull())

X_train, X_val = features[mask_train.values], features[mask_val.values]
y_train, y_val = y[mask_train.values], y[mask_val.values]

# Filling NaNs
X_train['resolution'] = X_train['resolution'].fillna(X_train['resolution'].mean())
X_val['resolution'] = X_val['resolution'].fillna(X_train['resolution'].mean())

space = [(1e-3, 1e-1, 'log-uniform'),  # lr
         (1, 10),  # max_depth
         (1, 20),  # min_child_samples
         (0.05, 1.),  # subsample
         (0.05, 1.),  # colsample_bytree
         (100, 1000),  # n_estimators
         (1, 5),  # min_df
         (1, 5)]  # ngram_range

train_titles = df[mask_train]['watch-title']
val_titles = df[mask_val]['watch-title']

res = forest_minimize(tune_lgbm, space, random_state=42,
                      n_random_starts=100, n_calls=150, verbose=1)

lr, max_depth, min_child_samples, subsample, colsample_bytree, n_estimators, min_df, ngram_range = res.x

ngram_range = (1, ngram_range)
title_vec = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range)
title_bow_train = title_vec.fit_transform(train_titles)
title_bow_val = title_vec.transform(val_titles)

X_train_title = hstack([X_train, title_bow_train])
X_val_title = hstack([X_val, title_bow_val])

lgbm = LGBMClassifier(learning_rate=lr, num_leaves=2 ** max_depth,
                      max_depth=max_depth,
                      min_child_samples=min_child_samples,
                      subsample=subsample,
                      colsample_bytree=colsample_bytree,
                      bagging_freq=1, n_estimators=n_estimators,
                      random_state=42, class_weight="balanced", n_jobs=8)
lgbm.fit(X_train_title, y_train)

lgbm_train_proba = lgbm.predict_proba(X_train_title)
lgbm_val_proba = lgbm.predict_proba(X_val_title)

# LGBM - Getting the metrics
print('LGBM TRAIN METRICS:')
print('avg_precision_score: ', average_precision_score(y_train,
                                                       lgbm_train_proba[:, 1]))
print('roc_auc: ', roc_auc_score(y_train, lgbm_train_proba[:, 1]))

print('\nLGBM VALIDATION METRICS:')
print('avg_precision_score: ', average_precision_score(y_val,
                                                       lgbm_val_proba[:, 1]))
print('roc_auc: ', roc_auc_score(y_val, lgbm_val_proba[:, 1]))

# Random Forest Model
rfc = RandomForestClassifier(n_estimators=200, random_state=42,
                             class_weight="balanced", n_jobs=8)
rfc.fit(X_train_title, y_train)

# Predicting
rf_train_proba = rfc.predict_proba(X_train_title)
rf_val_proba = rfc.predict_proba(X_val_title)

# RFC - Getting the metrics
print('\nTRAIN METRICS:')
print('avg_precision_score: ', average_precision_score(y_train,
                                                       rf_train_proba[:, 1]))
print('roc_auc: ', roc_auc_score(y_train, rf_train_proba[:, 1]))

print('\nVALIDATION METRICS:')
print('avg_precision_score: ', average_precision_score(y_val,
                                                       rf_val_proba[:, 1]))
print('roc_auc: ', roc_auc_score(y_val, rf_val_proba[:, 1]))

ensamble_proba = 0.21*rf_val_proba + 0.79*lgbm_val_proba

print('\nEnsamble Model:')
print('avg_precision_score: ', average_precision_score(y_val,
                                                       ensamble_proba[:, 1]))
print('roc_auc: ', roc_auc_score(y_val, ensamble_proba[:, 1]))

# Saving model pkls
jb.dump(lgbm, "../deploy/pkls/lgbm_20200324.pkl.z")
jb.dump(rfc, "../deploy/pkls/rf_20200324.pkl.z")
jb.dump(title_vec, "../deploy/pkls/titlebow_20200324.pkl.z")

fig, ax = plt.subplots(1, 1)
roc = skplt.metrics.plot_roc(y_val, ensamble_proba, figsize=(8, 7), ax=ax)
fig.savefig('../../figures/ensamble_roc.png')

fig, ax = plt.subplots(1, 1)
roc = skplt.estimators.plot_learning_curve(lgbm, features, y, ax=ax, cv=5)
fig.savefig('../../figures/LCA_lgbm.png')
