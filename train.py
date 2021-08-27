import sklearn.naive_bayes
import functions
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
import settings
import os

if __name__ == '__main__':

    df = pd.read_csv(os.path.join(settings.DATA_DIR, 'train.csv'))

    df['author_id'] = df['author'].factorize()[0]

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['author_id'],
                                                        test_size=0.3, random_state=42)

    X_train, y_train, vec = functions.make_xy(X_train, y_train)
    X_test, y_test, _ = functions.make_xy(X_test, y_test, vectorizer=vec, train=False)
    model = sklearn.naive_bayes.MultinomialNB()
    model.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Train Accuracy: {functions.truncate(train_accuracy, 3)}")
    print(f"Test Accuracy: {functions.truncate(test_accuracy, 3)}")
    dump(vec, os.path.join(settings.MODEL_DIR, 'vectorizer.joblib'))
    dump(model, os.path.join(settings.MODEL_DIR, 'model.joblib'))