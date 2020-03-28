import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss


# Reading data
df1 = pd.read_csv("../../data/raw_data.csv")
df1 = df1[df1['y'].notnull()]

df2 = pd.read_csv("../../data/active_label_done.csv", index_col=0)
df2 = df2[df2['y'].notnull()]
df2['new_data'] = 1

# Concating original data + active learning data
df = pd.concat([df1, df2.drop("p", axis=1)])

df_clean = pd.DataFrame(index=df.index)
df_clean['new_data'] = df2['new_data']
df_clean['new_data'].fillna(0, inplace=True)
df_clean['title'] = df['watch-title']

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

df_clean['date'] = pd.to_datetime(clean_date, format="%d %b %Y")

# Clean view number
views = df['watch-view-count'].str.extract(r"(\d+\.?\d*)", expand=False)
df_clean['views'] = views.str.replace(".", "").fillna(0).astype(int)

# Makaing features DataFrame
features = pd.DataFrame(index=df_clean.index)
y = df['y'].copy()

# Extracting time since publication feature
features['time_since_pub'] = (pd.to_datetime("2020-03-24") -  # HARDCODED
                              df_clean['date']) / np.timedelta64(1, 'D')

# Extracting n of view feature
features['views'] = df_clean['views']

# Extracting n of view/day feature
features['views_per_day'] = features['views'] / features['time_since_pub']

# Droping time_since_pub to prevent bias
features = features.drop(['time_since_pub'], axis=1)

# Around 75% train and 25% to validation
split_date = '2020-02-27'
mask_train = (df_clean['date'] < split_date) & (df_clean['date'].notnull())
mask_val = (df_clean['date'] >= split_date) & (df_clean['date'].notnull())

X_train, X_val = features[mask_train.values], features[mask_val.values]
y_train, y_val = y[mask_train.values], y[mask_val.values]

# Extracting features from title
train_titles = df_clean[mask_train]['title']
val_titles = df_clean[mask_val]['title']

title_vec = TfidfVectorizer(min_df=2)
title_bow_train = title_vec.fit_transform(train_titles)
title_bow_val = title_vec.transform(val_titles)

# Concat the BoW into features df
X_train_title = hstack([X_train, title_bow_train])
X_val_title = hstack([X_val, title_bow_val])

# Random Forest Model
mdl = RandomForestClassifier(n_estimators=1000, random_state=42,
                             class_weight="balanced", n_jobs=8)
mdl.fit(X_train_title, y_train)

# Predicting
train_proba = mdl.predict_proba(X_train_title)
train_preds = mdl.predict(X_train_title)
val_proba = mdl.predict_proba(X_val_title)
val_preds = mdl.predict(X_val_title)

# Getting the metrics
print('TRAIN METRICS:')
print('log_loss: ', log_loss(y_train, train_preds))
print('avg_precision_score: ', average_precision_score(y_train, train_proba[:, 1]))
print('roc_auc: ', roc_auc_score(y_train, train_proba[:, 1]))

print('\nVALIDATION METRICS:')
print('log_loss: ', log_loss(y_val, val_preds))
print('avg_precision_score: ', average_precision_score(y_val, val_proba[:, 1]))
print('roc_auc: ', roc_auc_score(y_val, val_proba[:, 1]))
