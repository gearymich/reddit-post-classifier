import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from utils import load_data

# Load the data
X_tr, X_te, y_tr, y_te = load_data()

# Vectorize the features using a bag-of-words model
vectorizer = CountVectorizer()
X_tr = vectorizer.fit_transform(X_tr)
X_te = vectorizer.transform(X_te)

# Train the classifier
clf = MultinomialNB()
clf.fit(X_tr, y_tr)

# Evaluate the classifier
accuracy = clf.score(X_te, y_te)
print('Accuracy:', accuracy)