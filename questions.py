# Import statements
import threading

from bs4 import BeautifulSoup
from pathlib import Path
from requests import get
from selenium import webdriver
from sklearn import metrics, linear_model
from tornado import template

import json
import math
import datetime
import numpy as np
import pandas as pd
import pygal as pygal
import pdfkit
import time


# Helper methods
import common


def is_pal(s):
    '''
    Check if a string is a palindrome.
    :param s: The string.
    :return: True if palindrome, False if not.
    '''

    # String of length 1 or less are palindromes.
    if len(s) <= 1:
        return True

    # Check if the first and last chars are the same.
    if s[0] != s[-1]:
        return False

    # If they are, then lose those chars and check the rest of the string recursively.
    return is_pal(s[1:-1])


def analyze_line(l):
    '''
    Analyzes a string, counting the number of characters and words in it.
    :param l: The string.
    :return: A tuple - (number of chars, number of words)
    '''

    words = l.split(' ')
    num_words = len(words)
    num_chars = len(''.join(words))

    return num_chars, num_words


def str_to_float(s):
    '''
    Converts a number in string format to float. This is a bit of a specialized function. It converts
    a number like 1,234 to 1234.
    :param s: A variable, presumably a string.
    :return: Floating point representation of s.
    '''

    # Do some basic validation. Required for the data we are handling.
    if not isinstance(s, str):
        return s

    # Remove commas and convert the rest to float.
    s = s.replace(',', '')
    return float(s)


# The first set of questions.
class PartOnePython:
    def q1(self):
        # Print all multiples of 3 under 20
        n = 3
        i = 3
        while n < 20:
            print(n)
            n += i

    def q2(self, s):
        # Check if s is a palindrome
        ans = ''
        ans += '\'' + s + '\' is '
        if not is_pal(s):
            ans += 'not '
        ans += 'palindromic.'

        print(ans)

    def q3(self, path):
        # Print number of chars, words, and lines in a file
        chars, words, lines = 0, 0, 0
        try:
            with open(path) as f:
                for line in f:
                    c, w = analyze_line(line)
                    chars += c
                    words += w
                    lines += 1
            print('lines: ' + str(lines) + ', words: ' + str(words) + ', chars: ' + str(chars))
        except FileNotFoundError:
            print('File not found: \'' + path + '\'')

    def q4(self):
        salaries = pd.read_csv('data/salaries.csv', header=0)

        # Get cities sorted in descending order of lawyer salary
        lawyers = salaries.loc[salaries['Job'] == 'Lawyers']
        lawyers.reset_index(inplace=True, drop=True)
        lawyers.sort_values(by=['Salary'], ascending=False, inplace=True)

        # Get the city names
        print(lawyers['City'])

        # Get median salary of each job.
        salaries.drop(['City'], axis=1, inplace=True)
        print(salaries.groupby(['Job']).median())


# Second set of questions.
class PartTwoHandlingBigData:
    # Basic URLs
    base_url = 'https://www.imdb.com/search/title?'
    type_url_feature = 'title_type=feature'
    rating_gt_8_url = 'user_rating=8.0,'
    num_votes_gt_url = 'num_votes=40000,'
    year_gt_url = 'release_date=2000-01-01,'

    def scrape_imdb(self):
        '''
        This function scrapes IMDB and extracts a whole lot of movies. This data is written to a file.
        :return: Nothing
        '''

        # Create URL
        base_url = 'https://www.imdb.com'
        url = base_url + '/search/title?title_type=feature&countries=us&languages=en&count=250'

        contains_next = True
        n = 0

        while contains_next:
            ans = []
            print('Inside loop')

            # Throw a GET request
            response = get(url)

            # Get data.
            html_soup = BeautifulSoup(response.text, 'html.parser')
            temp_c = html_soup.find_all('div', class_='lister-item mode-advanced')
            movie_containers = temp_c

            print(type(movie_containers))

            for mc in movie_containers:
                # Go through the movie containers and arrange data.
                try:
                    title = mc.h3.a.text
                    # print(title)
                    rating = float(mc.strong.text)

                    votes_str = mc.find('span', attrs={'name': 'nv'}).text
                    num_votes = int(votes_str.replace(',', ''))
                    y_str = mc.h3.find('span', class_='lister-item-year text-muted unbold').text[1:5]
                    year = int(y_str.replace(',', ''))

                    ans.append([title, rating, num_votes, year])
                except Exception:
                    # Ignore any errors.
                    continue

            print('ans len: ' + str(len(ans)))
            imdb_df = pd.DataFrame(ans, columns=['title', 'rating', 'num_votes', 'year'])
            print('saving to csv')

            # Logic to add a header.
            if n == 0:
                imdb_df.to_csv('data/imdb.csv', index=False, mode='a', header=True)
            else:
                imdb_df.to_csv('data/imdb.csv', index=False, mode='a', header=False)

            # Go to next link
            next_link = html_soup.find('a', class_='lister-page-next next-page')
            print(next_link)
            contains_next = (next_link is not None)

            if contains_next:
                url = base_url + next_link['href']

            n += 1

            # Stop at 300 links.
            if n == 300:
                break

    def analyze_imdb(self):
        '''
        This method performs the needed analysis on the IMDB data.
        :return: Nothing
        '''

        # Check if the IMDB CSV file exists.
        f = Path('data/imdb.csv')

        # If it doesn't, then scrape IMDB.
        if not f.exists():
            self.scrape_imdb()

        imdb_df = pd.read_csv('data/imdb.csv', header=0, converters={'rating':float, 'num_votes':int, 'year':int})
        print(imdb_df.head())

        # Part one
        # find the 20 most popular movies with a rank more than 8.0
        rating_over_8 = imdb_df.loc[imdb_df['rating'] > 8.0]
        print(type(rating_over_8))
        print(rating_over_8.head(20))

        # Part two
        # find the 20 best rated movies with over 40,000 votes in the 2000s (year >= 2000)
        votes_over_40k = imdb_df.loc[(imdb_df['num_votes'] > 40000) & (imdb_df['year'] >= 2000)]
        print(votes_over_40k.head(20))

    """
    def scrape_bse(self):
        # This method is not used.
        equities = pd.read_csv('data/EQ180119.CSV')

        print(equities.head())
        codes = equities['SC_CODE']
        print(type(codes))

        # This method can be used to iterate.
        # for c in codes:
        #     print(c)
    """

    def is_dividend(self, line):
        '''
        This method checks if a stock line is a dividend line or a normal line.
        :param line: Stock line
        :return: Dividend line or not
        '''

        pieces = line.split(' ')
        return True if len(pieces) < 7 else False

    def process_stock_line(self, line):
        '''
        This method processes a stock line
        :param line: Stock line
        :return: [date, closing price]
        '''
        date = line[:12]
        line = line[13:]

        return [date, line.split(' ')[3]]

    def scrape_snp(self):
        '''
        This method scrapes S&P data.
        To parallelize the downloads, this method uses threads.
        :return: Nothing
        '''

        # Get the Wikipedia page and read the symbols.
        data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        table = data[0]
        tickers = table['Symbol'].tolist()

        slices = [tickers[x: x+10] for x in range(0, len(tickers), 10)]

        for piece in slices:
            threading.Thread(target=self.scrape_snp_mini, args=[piece]).start()

    def scrape_snp_mini(self, tickers):
        '''
        This method contains the code for ONE thread to scrape S&P data.
        :param tickers: List of company symbols.
        :return: Nothing.
        '''

        # Get the timestamp for today and one month ago.
        today_ms = math.floor(time.time())
        one_month_ago = datetime.date.today() - datetime.timedelta(days=100)
        oma_ms = int(time.mktime(one_month_ago.timetuple()))

        # Open the web driver and get the list of symbols for S&P companies
        option = webdriver.ChromeOptions()
        option.add_argument("--incognito")
        driver_path = "chromedriver.exe"

        # Iterate through the symbols and get closing prices.
        for ele in tickers:

            # Get the name of the current CSV file and...
            cur_file_name = 'data/prices/' + ele + '.csv'
            f = Path(cur_file_name)

            # If it does not exist, then read the data and create it.
            if not f.exists():
                browser = webdriver.Chrome(driver_path)

                # Get values from Yahoo!
                browser.get(
                    "https://finance.yahoo.com/quote/" + ele + "/history?period1=" + str(oma_ms) + '&period2=' + str(
                        today_ms) + '&interval=1d&filter=history&frequency=1d')

                # Scroll down to let the page display all values.
                j = 0
                while j < 2:
                    browser.execute_script("window.scrollTo(0, 108000);")
                    time.sleep(0.25)
                    j = j + 1

                # Get the price values.
                titles_element = browser.find_elements_by_xpath(".//tbody[@data-reactid='50']")
                one_big_text = titles_element[0].text
                lines = one_big_text.split('\n')

                # Get the closing prices along with dates for 30 days.
                formatted_data = []
                cnt = 0
                for x in lines:
                    if not self.is_dividend(x):
                        formatted_data.append(self.process_stock_line(x))
                        cnt += 1
                    if cnt == 30:
                        break

                # Read them into a DataFrame and save to a CSV file.
                temp_df = pd.DataFrame(formatted_data, columns=['Date', 'Close'])
                print(temp_df.head())
                temp_df.to_csv(cur_file_name)

                browser.quit()

    def analyze_snp(self):
        '''
        This method runs analysis on S&P data.
        :return: Nothing.
        '''

        # First, scrape the S&P data.
        self.scrape_snp()

        # For each CSV file (one per company)
        data_dir = 'data/prices/'
        pathlist = Path(data_dir).glob('*.csv')
        gain_list = []

        # Perform the required analysis
        for p in pathlist:
            name = str(p).split('\\')[-1].split('.')[0]
            temp_df = pd.read_csv(p)
            first_close = str_to_float(temp_df.iloc[0]['Close'])
            last_close = str_to_float(temp_df.iloc[-1]['Close'])
            pgain = (last_close - first_close)/first_close * 100
            gain_list.append([name, pgain])

        gain_df = pd.DataFrame(gain_list, columns=['Name', '% gain'])
        gain_df.sort_values(by=['% gain'], ascending=False, inplace=True)
        print('Top 10 companies by % gain:')
        print(gain_df.head(10))


class PartThreeAnalysis:
    def read_imdb(self):
        '''
        This method reads the IMDB data and returns a DataFrame.
        :return: IDMB DataFrame.
        '''

        # Check if the IMDB CSV file exists.
        f = Path('data/imdb.csv')

        # If it doesn't, then scrape IMDB and create the file.
        if not f.exists():
            p2 = PartTwoHandlingBigData()
            p2.scrape_imdb()

        # Read the IMDB CSV file into a DataFrame
        imdb_df = pd.read_csv('data/imdb.csv', header=0, converters={'rating': float, 'num_votes': int, 'year': int})

        return imdb_df

    def analyze_imdb(self):
        '''
        Performs the IMDB analysis.
        :return:
        '''

        imdb_df = self.read_imdb()

        '''
        Q1 find the average rank of the 10 most popular movies between 2000-2009 (inclusive)
        '''
        # Create a separate DF for the movies that were made in the 2000s
        # and sort it by number of votes. This makes sense because the most
        # popular movies are the ones with the greatest ratings.
        movies_2000s = imdb_df.loc[(imdb_df['year'] >= 2000) & (imdb_df['year'] <= 2009)]
        movies_2000s.reset_index(drop=True, inplace=True)
        movies_2000s = movies_2000s.sort_values(by=['num_votes'], ascending=False)

        # This next option keeps PyCharm from truncating the DFs while printing them.
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            ans = movies_2000s.iloc[:10]['rating'].mean()
            print('average rating: ' + str(ans))

        '''
        Q2
        find the year in the 1900s when the average rank increased the most, compared to the previous year.
        (Ignore movies with votes < 1000)
        '''
        # Create a DF for movies that had more than 1000 votes.
        votes_over_1000 = imdb_df.loc[imdb_df['num_votes'] > 1000]

        # Out of those movies, pick the ones that were made in the 1900s
        movies_1900s = votes_over_1000.loc[(votes_over_1000['year'] >= 1900) & (votes_over_1000['year'] < 2000)]
        movies_1900s.reset_index(drop=True, inplace=True)

        # Group these movies by year and get the average
        gb = movies_1900s.groupby(['year'])['rating']

        # Store the results in a DF and clean up a bit.
        mean_per_year = gb.mean()
        mean_per_year = pd.DataFrame(mean_per_year).reset_index()
        mean_per_year.columns = ['year', 'avg rating']

        # Get the differences between rows.
        mpy_diff = mean_per_year['avg rating'].diff()

        # And get the result.
        print('Year with greatest increase: ' +  str(int(mean_per_year.iloc[mpy_diff.idxmax()]['year'])))

        '''
        Q3 
        find the expected average rank for 2013 using linear regression. How good is this regression?
        (Ignore movies with votes < 1000.)
        '''

        # Group all the votes over 1000 to get average rating per year, as before.
        gb = votes_over_1000.groupby(['year'])['rating']
        mean_per_year = gb.mean()
        mean_per_year = pd.DataFrame(mean_per_year).reset_index()
        mean_per_year.columns = ['year', 'avg rating']

        # Get x and y values for the regression
        x_vals = np.array(mean_per_year['year'])
        y_vals = np.array(mean_per_year['avg rating'])
        x_vals = x_vals.reshape(-1, 1)

        # Perform the regression
        regr = linear_model.LinearRegression()
        regr.fit(x_vals, y_vals)

        # Predict the rating for the year 2013, and calculate the testing metrics
        x_test = np.array(2013)
        pred_2013 = regr.predict(x_test.reshape(-1, 1))
        act_2013  = list(mean_per_year.loc[mean_per_year['year'] == 2013]['avg rating'])
        r2_value  = metrics.r2_score(act_2013, pred_2013)
        mean_sq_err = metrics.mean_squared_error(act_2013, pred_2013)

        print()
        print('Predicted rating for 2013: ' + str(pred_2013))
        print('Actual rating for 2013: ' + str(act_2013))
        print('R2 score: ' + str(r2_value))
        print('Mean squared error: ' + str(mean_sq_err))

        '''
        find the correlation between rank and votes for each year in the 1900s.
        '''
        print()
        print(movies_1900s.head())
        ratings_and_votes = movies_1900s.drop(columns=['title'], axis=1)
        print(ratings_and_votes.head())
        corr = ratings_and_votes.groupby('year')[['rating', 'num_votes']].corr()
        print()
        corr = corr['rating'][:, 'num_votes']
        print(corr)
        print()
        diffs = corr.diff()
        diffs.dropna(inplace=True)
        print(diffs)

    def read_snp(self):
        '''
        Reads S&P data into a DataFrame and returns the DataFrame.
        :return: S&P DataFrame.
        '''

        # Scrape data if required. The checks are done in the scrape_snp() method.
        p2 = PartTwoHandlingBigData()
        p2.scrape_snp()

        # Read stocks data into DataFrame and return it.
        stocks = pd.DataFrame()

        data_dir = 'data/prices/'
        pathlist = Path(data_dir).glob('*.csv')
        for p in pathlist:
            name = str(p).split('\\')[-1].split('.')[0]
            temp_df = pd.read_csv(p, index_col=0, header=0)
            closing_price = np.array(temp_df['Close'])
            # print('name: ' + name)
            stocks[name] = pd.Series(closing_price)

        return stocks

    def analyze_snp(self):
        '''
        identify the stock most correlated with ICICI Bank's stockprice
        
        Since BSE data is not available, doing similar analysis with S&P data.
        Find the stock most correlated with CERN's stockprice
        '''
        stocks = self.read_snp()

        stock_name = 'CERN'
        corr = stocks.corr()
        print(corr.head())
        print()
        cern_corr = corr[stock_name]
        cern_corr.drop(index=[stock_name], inplace=True)

        print('Max. correlation with ' + stock_name + ' -> ' + cern_corr.idxmax() + ': ' + str(cern_corr.max()))


class PartFourPlotting:
    def bar(self, data_list):
        # Set up a chart object
        chart = pygal.Bar()

        # Set up x and y data
        x_data = map(str, range(len(data_list)))
        # y_data = data_list

        # Plot
        chart.add('Data', data_list)
        chart.x_labels = x_data

        # Finally, save the file.
        chart.render_to_file('images/bar-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.svg')

    def line(self, data_list):
        # Set up chart object
        chart = pygal.Line()

        # Set up x and y data
        x_data = map(str, range(len(data_list)))
        # y_data = data_list

        # Plot
        chart.add('Data', data_list)
        chart.x_labels = x_data

        # Finally, save the file.
        chart.render_to_file('images/line-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.svg')

    def scatter(self, data_list, title='Title', label='Data', x_label='X', y_label='Y', file_name=None):
        # Set up chart object
        chart = pygal.XY(stroke=False, title=title)

        # Plot. We assume that data_list is a list of tuples.
        chart.add(label, data_list)

        # Add labels
        chart.x_title = x_label
        chart.y_title = y_label

        # Save file
        # If no file name is provided, then we generate one.
        # If one is provided, we assume that it includes a file extension, but doesn't include a location.
        # We put the image in the images/ directory.
        if file_name is None:
            file_name = 'scatter-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.svg'
        chart.render_to_file('images/' + file_name)

    def get_imdb_scatterplot(self):
        '''
        Generates a scatterplot for IMDB data.
        :return: Nothing.
        '''

        # Read IMDB data.
        p3 = PartThreeAnalysis()
        imdb_df = p3.read_imdb()

        # Get all movies with more than 10000 votes.
        votes_over_10000 = imdb_df.loc[imdb_df['num_votes'] >= 10000]
        votes_over_10000.reset_index()

        # Get the average # votes
        mean_votes = votes_over_10000['num_votes'].mean()

        # Extract #votes and rating from the DF, and convert them to a list.
        r_and_nv = votes_over_10000[['num_votes', 'rating']]
        r_and_nv = list(map(tuple, r_and_nv.values))

        # Generate a scatterplot.
        self.scatter(r_and_nv[:150], title='IMDB #votes vs. Rating', label='',
                     x_label='Number of votes', y_label='Rating', file_name='scatter_pre_norm.svg')

        # Now normalize the data.
        r_and_nv = list(map(lambda x: (x[0]/mean_votes, x[1]), r_and_nv))

        # And create another scatterplot.
        self.scatter(r_and_nv[:150], title='IMDB #votes vs. Normalized Rating', label='',
                     x_label='Number of votes', y_label='Rating', file_name='scatter_post_norm.svg')

    def get_corr_matrix(self):
        '''
        Generates a correlation matrix for S&P data.
        :return: Nothing
        '''

        # Get data.
        p3 = PartThreeAnalysis()
        stocks = p3.read_snp()

        # Pick 30 random stocks
        stocks = stocks.sample(30, axis=1)

        # Create a heatmap.
        import seaborn as sns

        print(stocks.head())
        corr = stocks.corr()
        print()
        print(corr.head())

        colormap = sns.diverging_palette(10, 150, as_cmap=True)
        corr_plot = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=colormap)
        fig = corr_plot.get_figure()
        fig.savefig('images/corr.svg')


class PartFiveTemplates:
    def bar_template(self):
        '''
        Tornado template to generate a bar plot for the IMDB data.
        :return: Nothing.
        '''

        # Get IMDB data.
        p3 = PartThreeAnalysis()
        imdb_df = p3.read_imdb()

        # Drop unnecessary columns.
        imdb_df.drop(['title', 'rating', 'num_votes'], axis=1, inplace=True)

        # Group by year, get count, and convert to dictionary.
        gb = imdb_df.groupby(['year'])['year'].count()
        movies = gb.to_dict()

        # Now create a bar plot.
        common.draw_barplot(values=movies,
                          svg_file='images/bar-template.html',
                          title='Number of movies per year.',
                          x_label='Year',
                          y_label='Number of movies')

        if False:
            path_pdf = 'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'
            config = pdfkit.configuration(wkhtmltopdf=path_pdf)
            pdfkit.from_file('images/bar-template.svg', 'pdfs/bar-template.pdf', configuration=config)

    def spark_template(self):
        '''
        Tornado template for generating a sparkline for each company in the S&P data.
        :return: Nothing
        '''

        # Get S&P data.
        p3 = PartThreeAnalysis()
        stocks = p3.read_snp()

        print(stocks.head())

        for col in stocks:
            # Check if a sparkline already exists. If it does, then move on.
            # f = Path('images/sparkline-' + col + '.html')
            # if f.exists():
            #     continue

            # Extract stocks for one company.
            print('company: ' + col)
            ser = stocks[col]
            ser = ser.apply(str_to_float)

            # Convert to dict, since it's easier to process.
            sd = ser.to_dict()

            # Required data for making the plot.
            title = 'Sparkline for stock of company: ' + col
            smax = ser.max()
            min_x, min_y = 0,  -1
            width, height = len(ser), 0.2 * smax
            translate_y = -0.3 * smax

            # Create a data string to make the sparkline with.
            ss = ''
            for day in sd:
                ss += str(day)
                ss += ','
                ss += str((sd[day])/4)
                ss += ' '
            ss.strip(' ')

            # Make a sparkline with the data.
            loader = template.Loader('.')
            html = loader.load('spark.html').generate(sd=json.dumps(sd), points_str=ss, min_x=min_x, min_y=min_y, width=width, height=height, translate_y=translate_y, title=title)

            f = open('images/sparkline-' + col + '.html', 'w')
            f.write(html.decode('utf-8'))
            f.close()

            # Convert to PDF also.
            path_pdf = 'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'
            config = pdfkit.configuration(wkhtmltopdf=path_pdf)
            pdfkit.from_file('images/sparkline-' + col + '.html', 'pdfs/sparkline-' + col + '.pdf', configuration=config)

            # break


if __name__ == '__main__':
    if False:
        p1 = PartOnePython()
        p1.q1()
        print()
        p1.q2('apple')
        p1.q2('a')
        p1.q2('aplpa')
        print()
        p1.q3('')
        print()
        p1.q4()

    if False:
        p2 = PartTwoHandlingBigData()
        # p2.scrape_snp()
        # p2.analyze_imdb()
        p2.analyze_snp()

    if False:
        p3 = PartThreeAnalysis()
        # p3.analyze_imdb()
        p3.analyze_snp()

    if False:
        p4 = PartFourPlotting()
        sample_data = [23, 80, 92, 62, 98, 7, 9, 56, 19, 68]
        p4.bar(sample_data)
        p4.line(sample_data)
        p4.get_imdb_scatterplot()
        p4.get_corr_matrix()

    if True:
        p5 = PartFiveTemplates()
        p5.bar_template()
        # p5.spark_template()