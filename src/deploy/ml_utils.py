import pandas as pd
import re
import joblib as jb
from scipy.sparse import hstack, csr_matrix
import numpy as np


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


mdl_rf = jb.load("pkls/rf_20200324.pkl.z")
mdl_lgbm = jb.load("pkls/lgbm_20200324.pkl.z")
title_vec = jb.load("pkls/titlebow_20200324.pkl.z")


def clean_date(data):
    if re.search(r"(\d+) de ([a-z]+)\. de (\d+)", data['watch-time-text']) is None:
        return None

    raw_date_str_list = list(re.search(r"(\d+) de ([a-z]+)\. de (\d+)", data['watch-time-text']).groups())

    if len(raw_date_str_list[0]) == 1:
        raw_date_str_list[0] = "0"+raw_date_str_list[0]

    raw_date_str_list[1] = month_map[raw_date_str_list[1]]

    clean_date_str = " ".join(raw_date_str_list)

    return pd.to_datetime(clean_date_str, format="%d %b %Y")


def clean_views(data):
    raw_views_str = re.match(r"(\d+\.?\d*)", data['watch-view-count'])
    if raw_views_str is None:
        return 0
    raw_views_str = raw_views_str.group(1).replace(".", "")

    return int(raw_views_str)


def get_resolution(data):
    height, width = data['og:video:height'], data['og:video:width']

    try:
        resolution = float(height) * float(width)
    except ValueError:
        resolution = np.nan
        return None

    return resolution


def compute_features(data):

    if 'og:video:height' not in data:
        return None
    if 'og:video:width' not in data:
        return None
    if 'watch-view-count' not in data:
        return None

    publish_date = clean_date(data)
    if publish_date is None:
        return None

    views = clean_views(data)
    title = data['watch-title']

    features = dict()

    features['tempo_desde_pub'] = (pd.Timestamp.today() - publish_date) / np.timedelta64(1, 'D')
    features['views'] = views
    features['views_por_dia'] = features['views'] / features['tempo_desde_pub']
    del features['tempo_desde_pub']

    features['resolution'] = get_resolution(data)

    vectorized_title = title_vec.transform([title])

    num_features = csr_matrix(np.array([features['views'],
                                        features['views_por_dia'],
                                        features['resolution']]))

    feature_array = hstack([num_features, vectorized_title])

    print(feature_array.shape)
    return feature_array


def compute_prediction(data):
    feature_array = compute_features(data)

    if feature_array is None:
        return 0

    p_rf = mdl_rf.predict_proba(feature_array)[0][1]
    p_lgbm = mdl_lgbm.predict_proba(feature_array)[0][1]

    p = 0.21*p_rf + 0.79*p_lgbm

    return p


def log_data(data, feature_array, p):

    data['prediction'] = p
    data['feature_array'] = feature_array.todense().tolist()
