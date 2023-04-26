import pickle
from sklearn.feature_extraction.text import CountVectorizer
import typer
from model_scripts.utils import normalise_text, load_data

# main function of the script
def main(text: str = typer.Argument(..., help='The text to predict the catagory of'),
         super_catagory: bool = typer.Option(False, '--super-catagory', '-s', help='Use super-catagories instead of catagories')):
    
    path = 'models/LinearSVC_super-catagory.pkl' if super_catagory else 'models/LinearSVC_catagory.pkl'
    # Load pickled model from models/LinearSVC_super-catagory.pkl
    with open(path, 'rb') as f:
        clf = pickle.load(f)

    vectorizer = CountVectorizer()
    # Load the data
    X_tr, X_te, y_tr, y_te = load_data()
    # Vectorize the features using a bag-of-words model
    vectorizer = CountVectorizer()
    X_tr = vectorizer.fit_transform(X_tr)
    X_te = vectorizer.transform(X_te)

    # Predict the catagory of the text
    
    ans = clf.predict(vectorizer.transform([text]))
    print(ans)

if __name__ == '__main__':
    typer.run(main)