import requests
import pandas as pd 
from textblob import TextBlob
from bs4 import BeautifulSoup


class Fetch:

    def __init__(self):

        self.base_url = 'https://www.imdb.com/search/title/?title_type=feature,tv_movie&release_date=2000-01-01,2019-12-31&count=250'
        self.url_header = 'https://www.imdb.com'
        self.url_end = 'reviews?ref_=tt_urv'

    def generate_dataset(self):

        page = requests.get(self.base_url)
        soup = BeautifulSoup(page.content, 'html.parser')

        containers = soup.find_all('div', class_ = 'lister-item mode-advanced')
        print('[i] Website Reached...')
        data_set = pd.DataFrame()
        print('[i] Empty Dataframe Set...')
        for container in containers:
            print('[+] Inspect Page Container...')
            if container.find('div', class_ = 'ratings-metascore') is not None:

                num_reviews = 550
                url_mid = container.find('a')['href']
                review_page = requests.get(f'{self.url_header}{url_mid}{self.url_end}')
                review_soup = BeautifulSoup(review_page.text, 'html.parser')
                review_containers = review_soup.find_all('div', class_ = 'imdb-user-review')
                print('[+] Reviews in Container found...')
                if len(review_containers) < num_reviews:

                    num_reviews = len(review_containers)

                print(f'[i] Number of Reviews {num_reviews}')
                
                review_titles = []
                review_bodies = []

                for review_idx in range(num_reviews):
                    print('[+] Parsing Review...')
                    review_container = review_containers[review_idx]

                    review_title = review_container.find('a', class_ = 'title').text.strip()
                    review_body = review_container.find('div', class_ = 'text').text.strip()

                    review_titles.append(review_title)
                    review_bodies.append(review_body)

                name = container.h3.a.text
                names = [name for i in range(num_reviews)]

                year = container.h3.find('span', class_ = 'lister-item-year').text
                years = [year for i in range(num_reviews)]

                imdb_rating = float(container.strong.text)
                imdb_ratings = [imdb_rating for i in range(num_reviews)]

                metascore = container.find('span', class_ = 'metascore').text
                metascores = [metascore for i in range(num_reviews)]
                print('[i] Building Sentiment Labels...')
                sentiments = []
                for text in review_bodies:
                    senti = TextBlob(text).sentiment.polarity
                    print(f'[+] Review: {text}')
                    print(f'[+] Sentiment Polarity: {senti}')
                    if senti > 0:
                        sentiments.append('positive')
                    if senti == 0:
                        sentiments.append('neutral')
                    if senti < 0:
                        sentiments.append('negative')

                if data_set.empty:
                    data_set = pd.DataFrame({

                        'film': names,
                        'year': years,
                        'imdb': imdb_ratings,
                        'metascores': metascores,
                        'review_title': review_titles,
                        'review_text': review_bodies,
                        'sentiment': sentiments
                    })

                elif num_reviews > 0:

                    data_set = data_set.append(pd.DataFrame({

                        'film': names,
                        'year': years,
                        'imdb': imdb_ratings,
                        'metascores': metascores,
                        'review_title': review_titles,
                        'review_text': review_bodies,
                        'sentiment': sentiments
                    }))

        data_set.to_csv('data/labeled_dataset.csv')
        print('[+] Data saved in Data Folder...')
        return data_set


if __name__ == '__main__':

    Fetch().generate_dataset()





