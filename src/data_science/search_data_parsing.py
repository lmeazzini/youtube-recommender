import bs4 as bs
import json
import yaml


with open('../../parameters.yaml') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

queries = params['queries']
urls = params['urls']
number_of_pages = params['number_of_pages']

for query in queries:
    for page in range(1, number_of_pages+1):
        with open("../../pages/{}_{}.html".format(query, page), 'r+') as inp:

            html_page = inp.read()
            parsed = bs.BeautifulSoup(html_page, features='html.parser')
            tags = parsed.findAll("a")

            for tag in tags:
                if tag.has_attr('aria-describedby'):
                    link = tag['href']
                    title = tag['title']
                    with open('../../data/parsed_videos.json', 'a+') as output:
                        data = {'link': link,
                                'title': title,
                                'query': query}
                        output.write("{}\n".format(json.dumps(data)))
