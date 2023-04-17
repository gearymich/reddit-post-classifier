import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

super_categories = ['Sports', 'Politics', 'Entertainment', 'News', 'Science']

# to clean data
def normalise_text (text):
    text = text.str.lower() # lowercase
    text = text.str.replace(r"\#","") # replaces hashtags
    text = text.str.replace(r"http\S+","URL")  # remove URL addresses
    text = text.str.replace(r"@","")
    text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    text = text.str.replace("\s{2,}", " ")
    return text

def load_data(path='reddit_with_super_categories.csv'):
    '''
    Load the data from the csv file and return the train and test splits
    '''
    df = pd.read_csv(path)
    df = df.dropna()

    # df = df[['super_category', 'subreddit', 'title', 'body']]

    df = df[['super_category', 'title']] # TODO: FIND GOOD WAY TO INCLUDE BODY TEXT
    df['title'] = normalise_text(df['title'])
    df['super_category'] = normalise_text(df['super_category'])

    tr, te = train_test_split(df, test_size = 0.2, random_state = 0)
    return tr, te

if __name__ == '__main__':
    tr, te = load_data(path='reddit_with_super_categories.csv')
    print(te.head())

    
    