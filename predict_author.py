import pandas as pd
import sklearn.naive_bayes
from sklearn.model_selection import train_test_split
import os
import settings
import functions
from joblib import load



if __name__ == '__main__':

    args = functions.create_parser(["sentence"])

    vec = load(os.path.join(settings.MODEL_DIR, 'vectorizer.joblib'))
    model = load(os.path.join(settings.MODEL_DIR, 'model.joblib'))

    X_example = vec.transform([args.sentence])
    probs = model.predict_proba(X_example)[0]
    print(f"Percentage similarity to Edgar Allen Poe: {functions.truncate(probs[0], 3)}")
    print(f"Percentage similarity to HP Lovecraft: {functions.truncate(probs[1], 3)}")
    print(f"Percentage similarity to Mary Shelley: {functions.truncate(probs[2], 3)}")
