import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

root = Path(__file__).parent

REDDIT = root / '../assets/reddit.csv'
SUPER_CATEGORIES = root / '../assets/reddit_with_super_categories.csv'

def normalise_text(text):
    '''
    Normalise the text by removing punctuation, converting to lowercase, etc.
    '''
    text = text.str.lower() # lowercase
    text = text.str.replace(r"\#","") # replaces hashtags
    text = text.str.replace(r"http\S+","URL")  # remove URL addresses
    text = text.str.replace(r"@","")
    text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    text = text.str.replace("\s{2,}", " ")
    return text

def load_data(path=SUPER_CATEGORIES):
    '''
    Load the data from the csv file and return the train and test splits
    '''
    df = pd.read_csv(path)
    df = df.dropna()

    X = normalise_text(df['title']) 
    y = df['super_category']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_tr, X_te, y_tr, y_te

if __name__ == '__main__':
    tr, te = load_data()
    print(te.head())