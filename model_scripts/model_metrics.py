import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from utils import load_data

# models = [
#     SVC(gamma='auto'), LinearSVC(),
#     KNeighborsClassifier(n_neighbors=25),
#     MultinomialNB(),
#     BaggingClassifier(), ExtraTreesClassifier(n_estimators=300),
#     RandomForestClassifier(n_estimators=300)
# ]

models = [LinearSVC()]

# Load the data

for label in [True, False]:
    X_tr, X_te, y_tr, y_te = load_data(super_categories=label)

    # Vectorize the features using a bag-of-words model
    vectorizer = CountVectorizer()
    X_tr = vectorizer.fit_transform(X_tr)
    X_te = vectorizer.transform(X_te)

    for clf in models:
        avg_f1 = 0
        for _ in range(50):
            # Train the classifier
            clf.fit(X_tr, y_tr)
            predicted = clf.predict(X_te)
            actual = y_te.values
            score = f1_score(predicted, actual, average='micro')
            avg_f1 += score

        avg_f1 /= 50
        # Evaluate the classifier
        print(f"F1 score of {clf.__class__.__name__} (Super-catagories = {label}) : {avg_f1}")

        # save trained model
        import pickle
        label_text = 'super-catagory' if label else 'catagory'
        with open(f"models/{clf.__class__.__name__}_{label_text}.pkl", 'wb') as f:
            pickle.dump(clf, f)