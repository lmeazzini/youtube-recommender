import os.path
from flask import Flask
import os
import json
import run_backend as run_backend
import time


app = Flask(__name__)


def get_predictions():

    videos = []

    new_videos_json = "new_videos.json"
    if not os.path.exists(new_videos_json):
        run_backend.update_db()

    last_update = os.path.getmtime(new_videos_json) * 1e9

    # if time.time_ns() - last_update > (720*3600*1e9):  # aprox. 1 mes
    #    run_backend.update_db()

    with open("new_videos.json", 'r') as data_file:
        for line in data_file:
            line_json = json.loads(line)
            videos.append(line_json)

    predictions = []
    for video in videos:
        predictions.append((video['video_id'], video['title'],
                            float(video['score'])))

    predictions = sorted(predictions, key=lambda x: x[2], reverse=True)[:20]

    predictions_formatted = []
    for e in predictions:
        predictions_formatted.append("<tr><th><a href=\"{link}\">{title}</a></th><th>{score}</th></tr>".format(title=e[1], link=e[0], score=e[2]))

    return '\n'.join(predictions_formatted), last_update


@app.route('/')
def main_page():
    preds, last_update = get_predictions()
    return """<head><h1>Recomendador de Vídeos do Youtube</h1></head>
    <body>
    Segundos desde a última atualização: {}
    <table>
             {}
    </table>
    </body>""".format((time.time_ns() - last_update) / 1e9, preds)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
