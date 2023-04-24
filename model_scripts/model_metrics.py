import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from utils import load_data

models = [
    SVC(gamma='auto'), LinearSVC(),
    KNeighborsClassifier(),
    MultinomialNB(),
    BaggingClassifier(), ExtraTreesClassifier(n_estimators=300),
    RandomForestClassifier(n_estimators=300)
]

# Load the data
X_tr, X_te, y_tr, y_te = load_data()

# Vectorize the features using a bag-of-words model
vectorizer = CountVectorizer()
X_tr = vectorizer.fit_transform(X_tr)
X_te = vectorizer.transform(X_te)

for clf in models:
    # Train the classifier
    clf.fit(X_tr, y_tr)
    predicted = clf.predict(X_te)
    actual = y_te.values

    # Evaluate the classifier
    print(f"F1 score of {clf.__class__.__name__} : {f1_score(predicted, actual, average='micro')}")