from get_data import (download_search_page, parse_search_page,
                      download_video_page, parse_video_page)
from ml_utils import compute_prediction
import time
import json
import sqlite3 as sql

queries = ["crise+bolsa", "fundos+imobiliarios",
           "investimento+bolsa", "ibovespa", "ITSA4"]
db_name = 'videos.db'


def update_db():
    with sql.connect(db_name) as conn:
        for query in queries:
            for page in range(1, 4):
                print(query, page)
                search_page = download_search_page(query, page)
                video_list = parse_search_page(search_page)

                for video in video_list:
                    video_page = download_video_page(video['link'])
                    video_json_data = parse_video_page(video_page)

                    if 'watch-time-text' not in video_json_data:
                        continue

                    p = compute_prediction(video_json_data)

                    video_id = video_json_data.get('og:video:url', '')
                    watch_title = video_json_data['watch-title'].replace("'",
                                                                         "")
                    data_front = {"title": watch_title,
                                  "score": float(p),
                                  "video_id": video_id}
                    data_front['update_time'] = time.time_ns()

                    print(video_id, json.dumps(data_front))
                    c = conn.cursor()
                    c.execute("INSERT INTO videos VALUES ('{title}', '{video_id}', {score}, {update_time})".format(**data_front))
                    conn.commit()
    return True
