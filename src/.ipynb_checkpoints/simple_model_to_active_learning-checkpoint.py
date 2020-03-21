import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('../data/raw_data.csv')
df = df[df['y'].notnull()]

df_clean = pd.DataFrame(index=df.index)

# Clean date attribute
clean_date = df['watch-time-text'].str.extract(r"(\d+) de ([a-z]+)\. de (\d+)")
clean_date[0] = clean_date[0].map(lambda x: "0"+x[0] if len(x) == 1 else x)

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
clean_date = clean_date.apply(lambda x: " ".join(x), axis=1)

df_clean['date'] = pd.to_datetime(clean_date, format="%d %b %Y")

# Clean view number
views = df['watch-view-count'].str.extract(r"(\d+\.?\d*)",
           expand=False).str.replace(".", "").fillna(0).astype(int)
df_clean['views'] = views

# Makaing features DataFrame
features = pd.DataFrame(index=df_clean.index)
y = df['y'].copy()

features['time_since_pub'] = (pd.to_datetime("2020-19-20") -  # HARDCODED
                              df_clean['date']) / np.timedelta64(1, 'D')
features['views'] = df_clean['views']
features['views_per_day'] = features['views'] / features['time_since_pub']
features = features.drop(['tempo_desde_pub'], axis=1)

df_clean['date'].value_counts().plot(figsize=(20, 10))
plt.show()
