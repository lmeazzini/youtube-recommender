from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap
import run_backend as run_backend
import get_data
import ml_utils
import sqlite3 as sql
import json
import bs4 as bs

app = Flask(__name__)
bootstrap = Bootstrap(app)


def get_predictions():

    videos = []

    with sql.connect(run_backend.db_name) as conn:
        c = conn.cursor()
        for line in c.execute("SELECT * FROM videos"):
            line_json = {"title": line[0],
                         "video_id": line[1],
                         "score": line[2],
                         "update_time": line[3]}
            videos.append(line_json)

    predictions = []
    for video in videos:
        predictions.append((video['video_id'], video['title'],
                            float(video['score'])))

    return sorted(predictions,
                  key=lambda x: x[2],
                  reverse=False)[:30]


@app.route('/')
def main_page():
    preds = get_predictions()

    for pred in preds:

        with open('templates/index.html') as page_html:

            soup = bs.BeautifulSoup(page_html, features='html.parser')
            link, title, score = pred

            if len(soup.tbody) > 59:
                break

            new_tr = soup.new_tag("tr", **{'class': "row100 body"})
            soup.tbody.insert(0, new_tr)
            new_td1 = soup.new_tag("td", **{'class': "cell100 column1"})
            new_td2 = soup.new_tag("td", **{'class': "cell100 column2"})
            soup.tbody.tr.insert(0, new_td1)
            soup.tbody.tr.insert(1, new_td2)
            new_td2.string = str(score)
            new_link = soup.new_tag("a", href=link)
            soup.td.append(new_link)
            new_link.string = title

        with open("templates/index.html", "w") as outf:
            outf.write(soup.prettify())

    return render_template("index.html")


@app.route('/predict')
def predict_api():
    yt_video_id = request.args.get("yt_video_id", default='')
    video_page = get_data.download_video_page("/watch?v={}".format(yt_video_id))
    video_json_data = get_data.parse_video_page(video_page)

    if 'watch-time-text' not in video_json_data:
        return "not found"

    p = ml_utils.compute_prediction(video_json_data)
    output = {"title": video_json_data['watch-title'], "score": p}

    return json.dumps(output)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
