import json
import plotly
import joblib
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import plotly.express as px
from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet', 'stopwords'])


app = Flask(__name__)

def tokenize(text):
    '''
    Tokenize and lowercase text data
    Remove stopwords
    Leading and trailing whitespaces are removed
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # Remove stop words in multiple languages
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    tokens = [t for t in tokens if t not in stopwords.words("french")]
    tokens = [t for t in tokens if t not in stopwords.words("italian")]
    tokens = [t for t in tokens if t not in stopwords.words("spanish")]

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
#sqlite:///../data/DisasterResponse.db

df = pd.read_sql_table('disaster_table', engine)

# load model
model = joblib.load('models/classifier.pkl')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for bar chart visuals
    water_counts = df[df.water == 1].groupby(['genre']).count()
    food_counts = df[df.food == 1].groupby(['genre']).count()
    medical_help_counts = df[df.medical_help == 1].groupby(['genre']).count()
    category_names = list(water_counts.index)

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)


    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=water_counts,
                    name='Water',
                    marker=dict(color='#3498db')
                ),
                Bar(
                    x=category_names,
                    y=food_counts,
                    name='Food',
                    marker=dict(color='#f1c40f')
                ),
                Bar(
                    x=category_names,
                    y=medical_help_counts,
                    name='Medical',
                    marker=dict(color='#e74c3c')
                )
            ],
            'layout': {
                'title': 'Distribution of Vital Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre: Water, Food, Medical"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
