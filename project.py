from bs4 import BeautifulSoup
from requests import get

import common


def analyze_str(s: str) -> None:
    if ',' in s:
        l = s.strip('\n').strip(' ').replace(',', '').split(' ')
        for x in l:
            if x not in genres:
                genres[x] = 0
            genres[x] = genres[x] + 1


# Create URL
base_url = 'https://www.imdb.com'
url = base_url + '/search/title?title_type=feature&countries=us&languages=en&count=250'

contains_next = True
n = 0

genres = {}

while contains_next:
    ans = []
    print('Inside loop')

    # Throw a GET request
    response = get(url)

    # Get data.
    html_soup = BeautifulSoup(response.text, 'html.parser')
    temp_c = html_soup.find_all('div', class_='lister-item mode-advanced')
    movie_containers = temp_c

    for mc in movie_containers:
        try:
            genre_text = mc.find('span', attrs={'class', 'genre'}).text
            analyze_str(genre_text)
        except Exception:
            continue

    next_link = html_soup.find('a', class_='lister-page-next next-page')
    print(next_link)
    contains_next = (next_link is not None)

    if contains_next:
        url = base_url + next_link['href']

    n += 1

    if n == 25:
        break


print(genres)
common.draw_barplot(values=genres,
                    svg_file='images/project.html',
                    title='Number of movies per genre',
                    x_label='Genre',
                    y_label='Number of movies')
