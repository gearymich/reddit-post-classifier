import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

super_categories = ['Sports', 'Politics', 'Entertainment', 'News', 'Science']

def load_data(path='reddit_with_super_categories.csv'):
    '''
    Load the data from the csv file and return the train and test splits
    '''
    # csv columns
    # 0: id, 1: subreddit, 2: title, 3: body
    df = pd.read_csv(path)
    df = df.dropna()
    df = df[['super_category', 'subreddit', 'title', 'body']]

    # convert to numpycl
    X = df['title'].to_numpy()
    y = df['super_category'].to_numpy()

    # split into train and test
    X_tr, X_te, y_tr, y_te = train_test_split(X,  y, test_size = 0.2, random_state = 0)
    return X_tr, X_te, y_tr, y_te


if __name__ == '__main__':
    X_tr, X_te, y_tr, y_te = load_data(path='reddit_with_super_categories.csv')
    print(f"X: {X_tr[0][:20]}, y: {y_tr[0]}")

    
    