import requests
import pandas as pd
import time
import glob
import bs4 as bs
import re
import json


df = pd.read_json('data/parsed_videos.json', lines=True)
url = 'https://www.youtube.com{link}'

links = df['link'].unique()

# Collecting data from videos
for link in links:
    page_url = url.format(link=link)

    response = requests.get(page_url)
    link_name = re.search('v=(.*)', link).group(1)

    with open('pages/video_{}.html'.format(link_name), 'w+') as output:
        output.write(response.text)

    time.sleep(1)

# Processing video data
with open('data/parsed_video_data.json', 'w+') as output:
    video_paths = glob.glob('pages/video*')
    for video_file in sorted(video_paths):
        with open(video_file, 'r+') as inp:
            html_page = inp.read()
            parsed = bs.BeautifulSoup(html_page, 'html.parser')

            watch_class = parsed.find_all(attrs={
                          'class': re.compile(r'watch')})
            watch_id = parsed.find_all(attrs={
                          'id': re.compile(r'watch')})
            channel = parsed.find_all('a', attrs={
                           'href': re.compile(r'channel')})
            meta = parsed.find_all('meta')

            data = {}
            for clas in watch_class:
                col = '_'.join(clas['class'])
                if 'clearfix' in col:
                    continue
                data[col] = clas.text.strip()

            for idd in watch_id:
                col = '_'.join(idd['id'])
                data[col] = idd.text.strip()

            for i, c in enumerate(channel):
                data['channel_link_{}'.format(i)] = c['href']

            for m in meta:
                col = m.get('property')
                if col is not None:
                    data[col] = m['content']

            output.write('{}\n'.format(json.dumps(data)))

df = pd.read_json('data/parsed_video_data.json', lines=True)

# Selecting just some columns
selected_cols = ['watch-title', 'watch-view-count', 'watch-time-text',
                 'watch-extras-section', 'content_watch-info-tag-list',
                 'og:image', 'og:image:width', 'og:image:height',
                 'og:description', 'og:video:width', 'og:video:height',
                 'og:video:tag', 'channel_link_0']

# 'watch7-headline', 'watch7-user-header', 'watch8-sentiment-actions'

df[selected_cols].to_feather('data/raw_data.feather')
df[selected_cols].to_csv('data/raw_data.csv', index=False)
