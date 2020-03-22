import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import scikitplot as skplt
from sklearn.tree import plot_tree


df = pd.read_csv('../data/raw_data.csv')
df = df[df['y'].notnull()]

df_clean = pd.DataFrame(index=df.index)

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
features['time_since_pub'] = (pd.to_datetime("2020-03-15") -  # HARDCODED
                              df_clean['date']) / np.timedelta64(1, 'D')

# Extracting n of view feature
features['views'] = df_clean['views']

# Extracting n of view/day feature
features['views_per_day'] = features['views'] / features['time_since_pub']

# Droping time_since_pub to prevent bias
features = features.drop(['time_since_pub'], axis=1)

# How is the date of publication distributed?
df_clean['date'].value_counts().plot(figsize=(20, 10))
plt.title('Videos Dates', fontsize=20)
plt.xlabel('Dates', fontsize=15)
plt.ylabel('Video Count', fontsize=15)
plt.savefig('../figures/video_dates.png')
# plt.show()

# Configuring to make a temporal train_val split
features['date'] = df_clean['date']
features['index'] = features.index
features = features.set_index('date').sort_index().dropna()

y = pd.DataFrame(y)
y['date'] = df_clean['date']
y['index'] = y.index
y = y.set_index('date').sort_index()
y = y[y.index.notna()]

# Splitting the data set - 60% train 40% validation
n = len(features)
n_train = np.ceil(n * 0.6) - 1
n_val = n - n_train

X_train, X_val = (features.reset_index().loc[:n_train],
                  features.reset_index().loc[n_train+1:])
y_train, y_val = y.reset_index().loc[:n_train], y.reset_index().loc[n_train+1:]

# Dropping features that can not be used
X_train = X_train.drop(['date', 'index'], axis=1)
X_val = X_val.drop(['date', 'index'], axis=1)
y_train = y_train['y']
y_val = y_val['y']

# Training a Decision Tree
mdl = DecisionTreeClassifier(random_state=0, max_depth=3,
                             class_weight="balanced")
mdl = mdl.fit(X_train, y_train)

# Predicting on the validation set
val_proba = mdl.predict_proba(X_val)
preds = mdl.predict(X_val)

# Getting the metrics
print('log_loss: ', log_loss(y_val, preds))
print('avg_precision_score: ', average_precision_score(y_val, val_proba[:, 1]))
print('roc_auc: ', roc_auc_score(y_val, val_proba[:, 1]))

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
roc = skplt.metrics.plot_roc(y_val, val_proba, figsize=(8, 7), ax=ax)
fig.savefig('../figures/dt_roc.png')

# Drawing the tree
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plot_tree(mdl, ax=ax, feature_names=X_train.columns)
fig.savefig('../figures/dt_fig.png')
