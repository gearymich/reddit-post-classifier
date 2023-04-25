from collections import defaultdict
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from yellowbrick.text import freqdist
from utils import load_data

from yellowbrick.text.tsne import tsne
from sklearn.feature_extraction.text import TfidfVectorizer

def word_freq():
    # Load the text data
    X_tr, X_te, y_tr, y_te = load_data()

    X = pd.concat([X_tr, X_te])
    y = pd.concat([y_tr, y_te])

    # Create a dict to map target labels to documents of that category
    hobbies = defaultdict(list)
    for text, label in zip(X, y):
        hobbies[label].append(text)

    vectorizer = CountVectorizer(stop_words='english' )
    docs = vectorizer.fit_transform(X)
    features   = vectorizer.get_feature_names_out()

    freqdist(features, docs, orient='v')

def tsne_visualize():

    X_tr, X_te, y_tr, y_te = load_data(super_categories=True)
    X = pd.concat([X_tr, X_te])
    y = pd.concat([y_tr, y_te])

    tfidf = TfidfVectorizer()

    X = tfidf.fit_transform(X)

    tsne(X, y)

if __name__ == '__main__':
    tsne_visualize()