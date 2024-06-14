import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import logging
from textblob import TextBlob


nltk.download('stopwords')
nltk.download('punkt')



log_filename = 'li_data.log'
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

crypto_config = {
    'file_path': '/Users/bartoszbiskupski/PycharmProjects/analiza_danych_txt_zaliczenie/cryptonews_source.csv',
    'sep': ',',
    'encoding': 'utf-8',
    'analise_col': 'text',
    'time_col': 'date',
    'language': 'english'
}

# Read and clean Class A data
class TextAnalysis:
    def __init__(self, read_config):
        self.read_config = read_config
        self.file_path = read_config['file_path']
        self.encoding = 'utf-8'
        self.analise_col = 'text'
        self.language = 'english'
        self.sep = ','
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


    def word_cloud(self, text):
        """Create a word cloud
        :param text: str
        """
        wordcloud = WordCloud(width=800, height=400).generate(' '.join(text))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()


    def run(self):
        self.read_csv()
        cleaned_data = self.normilize_data()
        return cleaned_data

