from get_data import (download_search_page, parse_search_page,
                      download_video_page, parse_video_page)
from ml_utils import compute_prediction
import time
import json

queries = ["crise+bolsa", "fundos+imobiliarios", "investimento-bolsa"]


def update_db():
    with open("new_videos.json", 'w+') as output:
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
                    data_front = {"title": video_json_data['watch-title'],
                                  "score": float(p), "video_id": video_id}
                    data_front['update_time'] = time.time_ns()

                    print(video_id, json.dumps(data_front))
                    output.write("{}\n".format(json.dumps(data_front)))
    return True
