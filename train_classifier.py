# import libraries
import sys
import re
import numpy as np
import pandas as pd
import pickle

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

nltk.download(['punkt', 'wordnet'])


# load data from database
def load_data():
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('disaster_table', con=engine)
    X = df.text.values
    y = df.category.values
    return X, y


# tokenization function to process your text data
def tokenize(text):

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


#Build a machine learning pipeline
# This machine pipeline takes in the message column as input and outputs classification results
# on the other 36 categories in the dataset (multiple target variables).
def build_model():

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


    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('find_death', WordDeathExtractor()),

            ('find_minor', WordDeathExtractor())
        ])),

            ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


#Train pipeline
def evaluate_model(model, X_test, y_test, category_names):

    parameters = {
        "clf__estimator__n_estimators": [1,2,3,5],
        "clf__estimator__max_depth":[1,2,3,5],
        "clf__estimator__criterion": ["gini", "entropy"]
    }

    # Use grid search
    cv_gridsearch = GridSearchCV(pipeline, param_grid=parameters)


    # predict on the test data
    y_pred = pipeline.predict(X_test)

    # Report the f1 score, precision and recall for each output category of the dataset.
    # You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

    # Get one column from y_test and the corresponding column from y_pred and pass to classification_report

    test_matrix = []
    for col_y in y_test.columns:
        y_test_col=y_test[col_y]
        test_matrix.append(y_test_col.values)

    pred_matrix = []
    for col_p in range(y_pred.shape[1]):
        y_pred_col=y_pred[:,col_p]
        pred_matrix.append(y_pred_col)

    all_classification_reports = []

    for col in range(y_pred.shape[1]):
        all_classification_reports.append(classification_report(test_matrix[col], pred_matrix[col])) #output_dict=True is in sklearn 0.20


def save_model(model, model_filepath):
    #Export your model as a pickle file
    #After training we save the model in a pickle file to be used for future predictions
    pickle.dump(pipeline,open("models/classifier.pkl","wb"))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        # perform train test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
