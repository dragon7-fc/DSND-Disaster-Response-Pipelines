import sys
import re
import sqlite3
import pandas as pd
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support
import pickle


def load_data(database_filepath):
    """load_data
    Load the database_filepath file and extract X, Y and category_names.

    :param database_filepath: The SQLite database_filepath file

    :returns: X, Y, category_names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster', con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = df.columns[4:]

    return X, Y, category_names


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
        text = text.replace(url, "urlplaceholder")

    # tokenize text
    tokens = word_tokenize(text)

    STOPWORDS = list(set(stopwords.words('english')))
    # remove short words
    tokens = [token for token in tokens if len(tokens) > 2]
    # remove stopwords
    tokens = [token for token in tokens if token not in STOPWORDS]

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """build_model"""
    pipeline = Pipeline(
        [('text_pipeline',
          Pipeline(
              [('vect', CountVectorizer(tokenizer=tokenize)),
               ('tfidf', TfidfTransformer())])),
         ('clf',
          MultiOutputClassifier(
              OneVsRestClassifier(LinearSVC(random_state=0))))])

    parameters = {
            'clf__estimator__estimator__tol': [1e-2, 1e-4],
            'clf__estimator__estimator__C': [.5, 1, 2],

            }

    cv = GridSearchCV(
            estimator=pipeline,
            param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """evaluate_model
    Print the f1_score, precision and recall of the model.

    :param model: The model
    :param X_test: The X values of test sample
    :param Y_test: The Y values of test sample
    :param category_names: The Y_test category name
    """
    Y_pred = model.predict(X_test)

    results = pd.DataFrame(
        columns=[
            'Category',
            'f_score',
            'precision',
            'recall'])
    num = 0
    for cat in Y_test.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(
            Y_test[cat], Y_pred[:, num], average='weighted')
        results.set_value(num+1, 'Category', cat)
        results.set_value(num+1, 'f_score', f_score)
        results.set_value(num+1, 'precision', precision)
        results.set_value(num+1, 'recall', recall)
        num += 1
    print('Aggregated f_score:', results['f_score'].mean())
    print('Aggregated precision:', results['precision'].mean())
    print('Aggregated recall:', results['recall'].mean())
    print(results)


def save_model(model, model_filepath):
    """save_model
    Save model to model_filepath file.

    :param model: The model
    :param model_filepath: A model_filepath pickle file.
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
