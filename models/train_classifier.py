import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, fbeta_score, make_scorer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.exceptions import UndefinedMetricWarning
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import warnings



def load_data(database_filepath):
    """
    This function loads message and category data from a sqlite database
    stored at database_filepath.  Data should be stored in the 'messages'
    table as per the load_data routine from process_data.py.
    
    database_filepath: str
        the path to the database to be accessed
        
    returns: (X, y, category_names)
        X: numpy array of the messages
        y: numpy array of the categories
        category_names: a list of the category names
    """
    # initialize sqlalchemy engine
    engine = create_engine('sqlite:///' + database_filepath)
    # load data from table
    df = pd.read_sql_table('messages',engine)
    # extract category names from columns
    category_names = df.columns[4:].tolist()
    
    # return specified values in proper formats
    return (df['message'].values, df[category_names].values, category_names)

def tokenize(text):
    """
    This function tokenizes the text as part of preparation for classification.
    Steps are:
        - Convert to lowercase
        - Remove punctuation
        - Split into words
        - Remove extraneous white space
        - Remove english stopwords
        - Lemmatize for nouns, then verbs
        - Stem using porterstemmer
        
    text: str
        the text to be tokenized
    
    returns: words
        a list of the word tokens, in sequence
    """
    # remove punctuation using regex, convert to lowercase
    text = re.sub("[^a-zA-Z0-9]", " ", text.lower())
    # tokenize into words
    words = word_tokenize(text)
    # lemmatize nouns and verbs while removing stopwords and extraneous spaces
    lemmy = WordNetLemmatizer()
    words = [lemmy.lemmatize(lemmy.lemmatize(word.strip(), pos='v'))
             for word in words if word not in stopwords.words('english')]
    # stem the lemmatized text
    porter = PorterStemmer()
    words = [porter.stem(word) for word in words]
    # return result
    return words


def build_model():
    """
    This function constructs a machine learning pipeline to process
    the disaster response message data.
    
    returns: model
        the machine learning pipeline.  the pipeline takes in messages
        from load_data.py and outputs the class categorization.
    """
    # build model
    pipeline = Pipeline([
    # count vectorizer to count the occurrences of word tokens
    ('vect', CountVectorizer(tokenizer=tokenize)),
    # tfidf transformation
    ('tfidf', TfidfTransformer()),
    # multi-output classifier using random forests
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # define parameters for grid search testing
    # test minimum occurrence of 1 or 2 and
    # 2 different classifier algorithms with base parameters
    params = {'vect__min_df': [1, 2],
             'clf__estimator': [RandomForestClassifier(), MultinomialNB()]}
    
    # use f2 score to evaluate model - recall is more important than precision
    # (we want to make sure all messages with requests get through)
    # but precision is still important - can achieve recall of 1 by passing all messages through
    score = make_scorer(lambda x, y: fbeta_score(x, y, beta=2, average='macro'))
    
    cv = GridSearchCV(estimator=pipeline, param_grid=params, scoring=score)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluates the model and prints results to the console.
    
    model: sklearn.GridSearchCV
        the fitted GridSearch object
    X_test: np.array
        a numpy array of messages to be classified
    Y_test: np.array
        the true class categories of the test messages
    category_names:
        the names of the class categories, for printing
        
    returns: None
    """
    
    # predict class categories using the model
    y_pred = model.best_estimator_.predict(X_test)
    # print the result of sklearn's categorization_report
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    This function saves the best estimator from the provided grid search object
    to the provided filepath, using sklearn's joblib.dump and pickle format.
    
    model: sklearn.GridSearchCV object
        the fitted GridSearch object
    model_filepath: str
        the path where the pickle file is to be written
    """
    
    joblib.dump(model.best_estimator_, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        # Filter expected warnings to clean up console output
        # filter undefined metric warnings from printing
        # many appear because of categories with no true samples
        # or no predicted values
        warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
        # also filter RuntimeWarning, which appears due to divide-by-zero in 
        # Naive Bayes because one of the categories has no samples in training
        warnings.filterwarnings(action='ignore', category=RuntimeWarning)
        
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