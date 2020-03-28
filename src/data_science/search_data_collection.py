import time
import requests
import yaml


with open('../../parameters.yaml') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

queries = params['queries']
urls = params['urls']
number_of_pages = params['number_of_pages']

# Collecting data from youtube pages
for query in queries:
    for page in range(1, number_of_pages+1):
        url = urls.format(query=query, page=page)
        response = requests.get(url)

        with open('../../pages/{}_{}.html'.format(query,
                                                  page), 'w+') as output:
            output.write(response.text)

        time.sleep(1)
