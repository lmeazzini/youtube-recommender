import requests as rq
import bs4 as bs
import re


def download_search_page(query, page):
    url = "https://www.youtube.com/results?search_query={query}&sp=CAI%253D&p={page}"
    urll = url.format(query=query, page=page)
    response = rq.get(urll, headers={"Accept-Language": "pt-BR,pt;q=0.8,en-US;q=0.5,en;q=0.3"})

    return response.text


def download_video_page(link):
    url = "https://www.youtube.com{link}"
    urll = url.format(link=link)
    response = rq.get(urll, headers={"Accept-Language": "pt-BR,pt;q=0.8,en-US;q=0.5,en;q=0.3"})

    return response.text


def parse_search_page(page_html):
    parsed = bs.BeautifulSoup(page_html, features='html.parser')

    tags = parsed.findAll("a")

    video_list = []

    for e in tags:
        if e.has_attr("aria-describedby"):
            link = e['href']
            title = e['title']
            data = {"link": link, "title": title}
            video_list.append(data)
    return video_list


def parse_video_page(page_html):
    parsed = bs.BeautifulSoup(page_html, 'html.parser')

    class_watch = parsed.find_all(attrs={"class": re.compile(r"watch")})
    id_watch = parsed.find_all(attrs={"id": re.compile(r"watch")})
    channel = parsed.find_all("a", attrs={"href": re.compile(r"channel")})
    meta = parsed.find_all("meta")

    data = dict()

    for e in class_watch:
        colname = "_".join(e['class'])
        if "clearfix" in colname:
            continue
        data[colname] = e.text.strip()

    for e in id_watch:
        colname = e['id']
        data[colname] = e.text.strip()

    for e in meta:
        colname = e.get('property')
        if colname is not None:
            data[colname] = e['content']

    for link_num, e in enumerate(channel):
        data["channel_link_{}".format(link_num)] = e['href']

    return data
