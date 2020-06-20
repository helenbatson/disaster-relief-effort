'''
The pickle package needs the custom class(es) to be defined
in another module and then imported. So, create another
python package file (e.g. transformation.py) and then
import it like this from transformation import SelectBestPercFeats.
That will resolve the pickling error.
'''
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

# Custom transformer
class WordDeathExtractor(BaseEstimator, TransformerMixin):

    def find_death(self, text):
        terms = ['dead', 'dying', 'die', 'death', 'alive', 'life', 'living'] #search for terms related to death
        sentence_list = nltk.sent_tokenize(text)

        for sentence in sentence_list:

            words = sentence.split()              #split the sentence into individual words

            if terms in words:                    #see if one of the words in the sentence is related to death
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.find_death) # From Series to Numpy array
        return pd.DataFrame(X_tagged) # To Dataframe


# Custom transformer
class WordMinorExtractor(BaseEstimator, TransformerMixin):

    def find_minor(self, text):
        terms = ['kid', 'child', 'baby', 'toddler', 'teen', 'teenager', 'minor'] #search for terms related to minors
        sentence_list = nltk.sent_tokenize(text)

        for sentence in sentence_list:

            words = sentence.split()              #split the sentence into individual words

            if terms in words:                    #see if one of the words in the sentence is related to death
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.find_minor) # From Series to Numpy array
        return pd.DataFrame(X_tagged) # To Dataframe
