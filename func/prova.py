from bs4 import BeautifulSoup
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer


stop_words = stopwords.words('english') + list(punctuation) 

stemmer = SnowballStemmer("english")

def text_analyzer(text, stemmer, stop_words):
    text = re.sub("http\S+", " link ",text)
    text = word_tokenize(text)
    text = [token for token in text if token not in stop_words]
    text = [stemmer.stem(token) for token in text]

    return text
    
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(
    analyzer=lambda t: text_analyzer(t, tokenizer, stemmer, stop_words),
    min_df=5
)