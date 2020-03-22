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
views = df['watch-view-count'].str.extract(r"(\d+\.?\d*)",
           expand=False).str.replace(".", "").fillna(0).astype(int)
df_clean['views'] = views

# Makaing features DataFrame
features = pd.DataFrame(index=df_clean.index)
y = df['y'].copy()

# Extracting time since publication feature
features['time_since_pub'] = (pd.to_datetime("2020-03-14") -  # HARDCODED
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

# roc.savefig('../figures/dt_roc.png')
