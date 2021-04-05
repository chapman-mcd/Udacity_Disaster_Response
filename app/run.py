import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # generate bar chart of top requests
    melted = df.melt(id_vars=['id','message','original', 'genre'], value_name='value')
    melted = melted[~(melted.variable.isin(['related', 'request', 'offer', 'direct_report' , 
                                        'aid_related', 'weather_related', 'infrastructure_related',
                                           'other_aid']))]
    top_requests = melted.groupby('variable').sum()['value'].sort_values(ascending=False).head(10)
    top_requests_names = top_requests.index.tolist()
    
    # bar chart of needle in a haystack - % of messages that request aid
    request_counts = df.groupby('request').count()['id']
    request_names = ["Sender doesn't need help", 'Sender needs help']
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
                # bar chart of requests for aid
                        {
            'data': [
                Bar(
                    x=top_requests_names,
                    y=top_requests
                )
            ],

            'layout': {
                'title': 'Top 10 Requests for Assistance',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Request Category"
                }
            }
        },
        # bar chart of top requests
                {
            'data': [
                Bar(
                    x=request_names,
                    y=request_counts
                )
            ],

            'layout': {
                'title': "Separating the Signal from the Noise - Which messages ask for help and which don't?",
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': ""
                }
            }
        },
# bar chart of genres of messages
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