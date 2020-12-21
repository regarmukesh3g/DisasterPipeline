#!/usr/bin/env python
import json
import joblib
import pandas as pd
import plotly
from flask import render_template, request
from plotly.graph_objs import Pie
from sqlalchemy import create_engine

from webapp.run import app

# load data
def get_df():
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql_table('disaster_messages', engine)
    return df

def get_model():
    model = joblib.load("../models/classifier.pkl")
    return model


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Homepage of the webapp.
    """
    # extract data needed for visuals
    df = get_df()
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names, values=genre_counts
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
    graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graph_json)


# web page that handles user query and displays model results
@app.route('/go')
def predict():
    """
    Page to show results.
    Returns:
        webpage to results.
    """
    # save user input in query
    query = request.args.get('query', '')

    model = get_model()
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )
