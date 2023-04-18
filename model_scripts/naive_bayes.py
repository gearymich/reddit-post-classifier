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

print(f"Possible subreddits that model can classify text as: {set(y_tr)}")

# Evaluate the classifier
accuracy = clf.score(X_te, y_te)
print(f"Accuracy: {accuracy}")

a_subreddit_title_0 = "The 2020 US Presidential Election"
a_subreddit_title_1 = "Ghanaian footballer Asamoah Gyan has been named the best player in the 2019 Africa Cup of Nations"
a_subreddit_title_2 = "How to make a good cup of coffee"
a_subreddit_title_3 = "The Great Gatsby"

ans = clf.predict(vectorizer.transform([a_subreddit_title_0, a_subreddit_title_1, a_subreddit_title_2, a_subreddit_title_3]))
print(f"Input: {a_subreddit_title_0}, Model Prediction: {ans[0]}")
print(f"Input: {a_subreddit_title_1}, Model Prediction: {ans[1]}")
print(f"Input: {a_subreddit_title_2}, Model Prediction: {ans[2]}")
print(f"Input: {a_subreddit_title_3}, Model Prediction: {ans[3]}")