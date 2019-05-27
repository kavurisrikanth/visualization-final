import collections

import pandas as pd
from bs4 import BeautifulSoup
from requests import get

import common


def str_to_list(s: str) -> list:
    return s.strip('\n').strip(' ').replace(',', '').split(' ')


# Create URL
base_url = 'https://www.imdb.com'
url = base_url + '/search/title?title_type=feature&countries=us&languages=en&count=250'

contains_next = True
genres = {}
n = 0

# Scrape IMDB data
while contains_next and n < 5:

    # Throw a GET request
    response = get(url)

    # Get data.
    html_soup = BeautifulSoup(response.text, 'html.parser')
    temp_c = html_soup.find_all('div', class_='lister-item mode-advanced')
    movie_containers = temp_c

    for mc in movie_containers:
        try:
            # Get the genres and year
            genre_text = mc.find('span', attrs={'class', 'genre'}).text
            y_str = mc.h3.find('span', class_='lister-item-year text-muted unbold').text[1:5]
            year = int(y_str.replace(',', ''))

            # Using collections because we care about counts
            if year not in genres:
                genres[year] = collections.Counter()
            genres[year].update(collections.Counter(str_to_list(genre_text)))
        except Exception as e:
            continue

    if False:
        print(genres)
        imdb_df = pd.DataFrame(genres)
        print(imdb_df)

        # Logic to add a header.
        if n == 0:
            imdb_df.to_csv('data/project_imdb.csv', index=False, mode='a', header=True)
        else:
            imdb_df.to_csv('data/project_imdb.csv', index=False, mode='a', header=False)

    # Go to the next page
    next_link = html_soup.find('a', class_='lister-page-next next-page')
    contains_next = (next_link is not None)

    if contains_next:
        url = base_url + next_link['href']

    n += 1

# Sort the genres
sg = sorted(genres)
x_labels = list(map(str, sg))

data = [('Action', []), ('Adventure', []), ('Comedy', []), ('Drama', []), ('Fantasy', []), ('Horror', []), ('Thriller', [])]

# Separate the dictionary into lists
for year in sg:
    for gt in data:
        gt[1].append(genres[year][gt[0]])

# Plot a line chart
common.pygal_line(data, title='Genres per year', x_labels=x_labels, filename='images/project_line.html')


bar_data = collections.Counter()
for g in genres:
    bar_data.update(genres[g])

common.pygal_bar(data_dict=bar_data, y_label='Genre', file_name='images/project_bar.html')
# common.pygal_pie(data=genres, title='Number of movies per genre', filename='images/project.html')

for g in genres:
    title = 'Genre count for year ' + str(g)
    filename = 'images/project_pie_' + str(g) + '.html'
    common.pygal_pie(data=genres[g], title=title, filename=filename)