import json
import plotly
import numpy as np
import pandas as pd
import re
import os

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from collections import Counter

app = Flask(__name__)


def tokenize(text):
    """tokenize
    Perform text normalization.

    :param text: pure text

    :returns: tokens
    """
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # replace each url in text string with urlplaceholder
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, " ")

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    STOPWORDS = list(set(stopwords.words('english')))

    # remove short words
    tokens = [token for token in tokens if len(token) > 2]

    # remove stopwords
    tokens = [token for token in tokens if token not in STOPWORDS]

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster', engine)

# load model
model = joblib.load("../models/classifier.pkl")


def compute_word_counts(messages, load=True, filepath='../data/counts.npz'):
    '''
    input: (
        messages: list or numpy array
        load: Boolean value if load or run model
        filepath: filepath to save or load data
            )
    Function computes the top 20 words in the dataset with counts of each term
    output: (
        top_words: list
        top_counts: list
            )
    '''
    if load:
        # load arrays
        data = np.load(filepath)
        return list(data['top_words']), list(data['top_counts'])
    else:
        # get top words
        counter = Counter()
        for message in messages:
            tokens = tokenize(message)
            for token in tokens:
                counter[token] += 1
        # top 20 words
        top = counter.most_common(20)
        top_words = [word[0] for word in top]
        top_counts = [count[1] for count in top]
        # save arrays
        np.savez(filepath, top_words=top_words, top_counts=top_counts)
        return list(top_words), list(top_counts)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_counts = df[df.columns[4:]].sum()
    category_counts = category_counts.sort_values(ascending=False)
    category_names = list(category_counts.index)

    messages = df['message'].tolist()

    if not os.path.isfile('../data/counts.npz'):
        top_words, top_counts = compute_word_counts(messages, load=False)
    else:
        top_words, top_counts = compute_word_counts(None, load=True)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [{'data': [Bar(x=genre_names,
                            y=genre_counts,
                            )],
               'layout': {'title': 'Distribution of Message Genres',
                          'yaxis': {'title': "Count"},
                          'xaxis': {'title': "Genre"}}},
              {'data': [{"values": genre_counts,
                         "labels": genre_names,
                         "name": "Genre",
                         "hoverinfo": "label+percent+name",
                         "type": "pie"}],
               'layout': {}},
              {'data': [Bar(x=category_names,
                            y=category_counts)],
               'layout': {'title': 'Distribution of Message Categories',
                          'yaxis': {'title': "Count",
                                    'automargin': True},
                          'xaxis': {'title': "Category",
                                    'tickangle': -30,
                                    'automargin': True
                                    }}},
              {'data': [{"values": category_counts,
                         "labels": category_names,
                         "name": "Genre",
                         "hoverinfo": "label+percent+name",
                         "type": "pie"}],
               'layout': {}},
              {'data': [Bar(x=top_words,
                            y=top_counts)],
               'layout': {'title': 'Distribution of Top 20 Words',
                          'yaxis': {'title': "Count",
                                    'automargin': True},
                          'xaxis': {'title': "Words",
                                    'tickangle': -30,
                                    'automargin': True
                                    }}},
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
