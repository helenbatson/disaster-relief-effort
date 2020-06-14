# import libraries
import sys
import re
import numpy as np
import pandas as pd
import pickle
import transformation
#from transformation import WordDeathExtractor

from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#import sklearn
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

def load_data(database_filepath):
# load data from database
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

    model = Pipeline([
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
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0)
    }

    cv_gridsearch = GridSearchCV(model, param_grid=parameters)

    return cv_gridsearch


#Train pipeline
def evaluate_model(model, X_test, y_test, category_names):

    y_test = pd.DataFrame(data=y_test, columns=category_names)

    # predict on the test data
    y_pred = model.predict(X_test)
    # convert to dataframe
    y_pred = pd.DataFrame(data=y_pred, columns=category_names)

    # Report the f1 score, precision and recall for each output category of the dataset.
    # You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

    # Get one column from y_test and the corresponding column from y_pred and pass to classification_report

    test_matrix = []
    for col_y in category_names:
        y_test_col=y_test[col_y]
        test_matrix.append(y_test_col.values)


    pred_matrix = []
    for col_p in category_names:
        y_pred_col=y_pred[col_p]
        pred_matrix.append(y_pred_col.values)

    all_classification_reports = []

    for col in range(y_pred.shape[1]):
        all_classification_reports.append(classification_report(test_matrix[col], pred_matrix[col]))


def save_model(model, model_filepath):
    #Export your model as a pickle file
    #After training we save the model in a pickle file to be used for future predictions

    #pickle.dump(model,open(model_filepath,"wb"))
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model,model_file)
        #pickle.dump(transformation.WordDeathExtractor,f)

    #---pickle.dump(model,open("models/classifier.pkl","wb"))
    #---joblib.dump(model, 'models/classifier.pkl')


def main():
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
