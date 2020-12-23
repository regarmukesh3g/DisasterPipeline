import string
import sys

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from sklearn.ensemble import AdaBoostClassifier
nltk.download(['wordnet', 'stopwords', 'punkt'])


def load_data(database_filepath):
    """
    Load Data from database.
    Args:
        database_filepath: database path.

    Returns:
        Features, target feature, categories list
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_messages', engine)

    print(df['related'].value_counts())
    X = df['message']

    Y = df.drop(columns=['message', 'id', 'original', 'genre'])
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Lemmatize, normalize and tokenize text.
    Args:
        text: Text/string/corpus.
    Returns:
        Tokenized text.
    """
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')

    tokens = word_tokenize(text)
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in tokens]
    stop_words = stopwords.words("english")
    words = [w.lower().strip() for w in lemmed if w not in stop_words]
    return words


def build_model():
    """
    Build a model to be trained.
    Returns:
        Model for training.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    print(pipeline.get_params())
    parameters = {
        'clf__estimator__n_estimators': [10, 15, 20],
        'vect__min_df': [1, 2]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the metrics for model.
    Args:
        model: Classifier model object.
        X_test: Input features.
        Y_test: Target Feature.
        category_names: Category names list.
    """
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=Y_test.columns)
    for col in category_names:
        print("========================================================================")
        print("Evaluating Category: {}".format(col))
        print(classification_report(Y_test[col], y_pred_df[col], labels=[0, 1]))
        print("=========================================================================\n\n")


def save_model(model, model_filepath):
    """
    Save model into a pickle file.
    Args:
        model: Model to be saved.
        model_filepath: Filepath.
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
