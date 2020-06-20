# import libraries
import sys
import re
import numpy as np
import pandas as pd
import pickle
import transformation

from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
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

nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(database_filepath):
    '''
    Load data in from the database
    Split into X and y
    Category names are the columns names from y
    '''

    root = 'sqlite:///'
    engine = create_engine(root+database_filepath)
    df = pd.read_sql_table('disaster_table', con=engine)
    X = df.message.values
    y = df[['related',
       'request', 'offer', 'aid_related', 'medical_help', 'medical_products',
       'search_and_rescue', 'security', 'military', 'child_alone', 'water',
       'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees',
       'death', 'other_aid', 'infrastructure_related', 'transport',
       'buildings', 'electricity', 'tools', 'hospitals', 'shops',
       'aid_centers', 'other_infrastructure', 'weather_related', 'floods',
       'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']].values
    category_names = ['related',
       'request', 'offer', 'aid_related', 'medical_help', 'medical_products',
       'search_and_rescue', 'security', 'military', 'child_alone', 'water',
       'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees',
       'death', 'other_aid', 'infrastructure_related', 'transport',
       'buildings', 'electricity', 'tools', 'hospitals', 'shops',
       'aid_centers', 'other_infrastructure', 'weather_related', 'floods',
       'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']
    return X, y, category_names


def tokenize(text):
    '''
    Tokenize and lowercase text data
    Remove stopwords
    Leading and trailing whitespaces are removed
    '''
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # Remove stop words
    tokens = [t for t in tokens if t not in stopwords.words("english")]

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Build a machine learning pipeline
    This machine pipeline takes in the message column as input and outputs classification results
    on the other 36 categories in the dataset (multiple target variables).
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('find_death', transformation.WordDeathExtractor()),

            ('find_minor', transformation.WordDeathExtractor())
        ])),

            ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
    'features__text_pipeline__vect__max_df': [0.5, 1],
    'clf__estimator__bootstrap': [True, False],
    'clf__estimator__max_depth': [10, None],
    'clf__estimator__max_features': ['auto', 'sqrt'],
    'clf__estimator__min_samples_leaf': [1, 8],
    'clf__estimator__min_samples_split': [2, 10],
    'clf__estimator__n_estimators': [50, 400, 1000, 1400, 2000]
    }

    cv_gridsearch = GridSearchCV(pipeline, param_grid=parameters)

    return cv_gridsearch


def evaluate_model(model, X_test, y_test, category_names):
    '''
     Train pipeline
     Return classification report
         Report the f1 score, precision and recall for each output category of the dataset.
    '''

    y_test = pd.DataFrame(data=y_test, columns=category_names)

    # predict on the test data
    y_pred = model.predict(X_test)
    # convert to dataframe
    y_pred = pd.DataFrame(data=y_pred, columns=category_names)

    return classification_report(y_test, y_pred)


def save_model(model, model_filepath):
    '''
    Export the model as a pickle file
    After training we save the model in a pickle file to be used for future predictions
    '''

    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model,model_file)



def main():
'''
Run all functions to train the model, clean the data and save it to a database.
'''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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
