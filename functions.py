from sklearn.feature_extraction.text import CountVectorizer
import argparse
import math

def make_xy(var_col, tar_col, vectorizer=None, train=True):
    if vectorizer is None:
        vectorizer = CountVectorizer()
    if train == True:
        X = vectorizer.fit_transform(var_col)
    else:
        X = vectorizer.transform(var_col)
    X = X.tocsc()
    y = tar_col
    return X, y, vectorizer

def create_parser(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument(arguments[0])
    return parser.parse_args()

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper