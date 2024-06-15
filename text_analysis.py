import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from wordcloud import WordCloud
from textblob import TextBlob
import re
import logging

import duckdb


nltk.download('stopwords')
nltk.download('punkt')

crypto_config = {
    'project_name': 'cryptonews',
    'file_path': 'cryptonews_source.csv',
    'sep': ',',
    'encoding': 'utf-8',
    'analise_col': 'text',
    'time_col': 'date',
    'language': 'english'
}

log_filename = f'{crypto_config['project_name']}.log'
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')



# Read and clean Class A data
class CleanData:
    def __init__(self, read_config):
        self.read_config = read_config
        self.file_path = read_config['file_path']
        self.project_name = read_config['project_name']
        self.encoding = read_config['encoding']
        self.analise_col = read_config['analise_col']
        self.language = read_config['language']
        self.sep = read_config['sep']
        self.data = None
        self.normalized_data = None
        self.cleaned_text = None
        self.tokens = None
        self.cleaned_tokens = None
        logging.info('Initializing the ReadAndCleanData class')

    def read_csv(self):
        df = pd.read_csv(self.file_path)
        self.data = df.reset_index(drop=True)
        self.data = self.data.dropna()
        return self.data

    # clean text column in the dataframe
    def clean_text(self, text):
        cleaned_text = re.sub(r'\s+', ' ', text) # Remove extra spaces
        cleaned_text = re.sub(r'\n', ' ', cleaned_text) # Remove newline characters
        cleaned_text = re.sub(r'\d{1,2} [A-Za-z]+ \d{4}', '', cleaned_text) # Remove dates
        cleaned_text = re.sub(r'[^a-z\s]', '', cleaned_text) # Remove special characters
        self.cleaned_text = cleaned_text
        return self.cleaned_text

    def tokenization(self, text):
        """Tokenize the cleaned text column in the dataframe
        :return: tokens
        """
        self.tokens = word_tokenize(text)
        return self.tokens

    def stemming(self, tokens):
        """Stem the tokens in the dataframe
        :return: cleaned_tokens

        """
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words(self.language))
        self.cleaned_tokens = [stemmer.stem(word.lower()) for word in tokens if word.lower() not in stop_words]
        return self.cleaned_tokens

    def normilize_data(self):
        """Add cleaned_text column to the dataframe
        :param data: dataframe
        """
        self.data['cleaned_text'] = self.data[self.analise_col].apply(self.clean_text)
        logging.info('Added cleaned_text column to the dataframe')
        self.data['tokens'] = self.data['cleaned_text'].apply(self.tokenization)
        logging.info('Added tokens column to the dataframe')
        self.data['cleaned_tokens'] = self.data['tokens'].apply(self.stemming)
        logging.info('Added cleaned_tokens column to the dataframe')
        logging.info('Added cleaned_text column to the dataframe')
        return self.data

    def save_to_db(self):
        """Save the dataframe to a database
        :param data: dataframe
        :param table_name: name of the table
        """
        with duckdb.connect(f'{self.project_name}.db') as conn:
            conn.register(f'{self.project_name}', self.data)
            conn.execute(f'CREATE OR REPLACE TABLE {self.project_name} AS SELECT * FROM {self.project_name}')
        logging.info('Saved the dataframe to the database')


    def run(self):
        self.read_csv()
        cleaned_data = self.normilize_data()
        logging.info('Data cleaning completed')
        self.save_to_db()
        logging.info('Data saved to the database')



class TextAnalysis:
    def __init__(self, read_config):
        self.read_config = read_config
        self.project_name = read_config['project_name']
        self.file_path = read_config['file_path']
        self.analise_col = read_config['analise_col']
        self.time_col = read_config['time_col']
        self.language = read_config['language']
        self.data = None
        self.tokens = None
        self.wordcloud = None
        self.sentiment = None
        self.vectorizer = None
        self.model = None
        self.accuracy = None
        self.text = None
        self.cleaned_text = None
        self.cleaned_tokens = None
        self.data = None
        self.all_words = None
        self.pdf_pages = None
        logging.info('Initializing the TextAnalysis class')


    def read_from_db(self):
        with duckdb.connect(f'{self.project_name}.db') as conn:
            self.data = pd.read_sql_query(f"SELECT * FROM {self.project_name}", conn)
        return self.data


    def combine_all_words(self):
        self.all_words = []
        for tokens in self.data['cleaned_tokens']:
            self.all_words.extend(tokens)
        return self.all_words

    def create_wordcloud(self):
        wordcloud = WordCloud(width=800, height=400).generate(' '.join(self.all_words))
        plt.figure(figsize=(10, 5))
        plt.title('Wordcloud of Cryptonews')
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        self.pdf_pages.savefig()
        plt.show()
        plt.close()

    def sentiment_analysis(self, text):
        analysis = TextBlob(text)
        sentiment = analysis.sentiment.polarity
        if sentiment > 0:
            return "Positive"
        elif sentiment < 0:
            return "Negative"
        else:
            return "Neutral"

    def sentiment_visualization(self):
        sentiment = self.data['cleaned_text'].apply(self.sentiment_analysis)
        sentiment.value_counts().plot(kind='bar')
        plt.title('Sentiment Analysis of Cryptonews')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        self.pdf_pages.savefig()
        plt.show()
        plt.close()

    def sentiment_time_series(self):
        self.data['date_format'] = pd.to_datetime(self.data[self.time_col], format='mixed')
        self.data['year_month'] = self.data['date_format'].dt.strftime('%Y-%m')
        self.data = self.data.set_index('date_format')
        self.data['sentiment'] = self.data['cleaned_text'].apply(self.sentiment_analysis)
        self.data.groupby(['year_month', 'sentiment']).size().unstack().plot()
        plt.title('Sentiment Analysis of Cryptonews Over Time')
        plt.xlabel('Date')
        plt.ylabel('Count')
        self.pdf_pages.savefig()
        plt.show()
        plt.close()


    def run(self):
        self.data = self.read_from_db()
        self.combine_all_words()
        with PdfPages(f'{self.project_name}.pdf') as self.pdf_pages:
            self.create_wordcloud()
            self.sentiment_visualization()
            self.sentiment_time_series()
        logging.info('Text analysis completed')


CleanData(crypto_config).run()
TextAnalysis(crypto_config).run()
