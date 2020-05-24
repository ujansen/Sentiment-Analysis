# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('IMDB Dataset.csv')

# Distribution of data
dataset['sentiment'].value_counts()

# Cleaning the text

# Tokenize the text
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()

# Import BeautifulSoup which helps us act with html text
from bs4 import BeautifulSoup
def html_remove(review):
    return BeautifulSoup(review, 'html.parser').get_text()
dataset['review'] = dataset['review'].apply(html_remove)

# Import required library to remove non-alphanumeric characters
import re
def non_alpha_numeric_remove(review):
    return re.sub(pattern = '[^a-zA-Z0-9]', repl = ' ', string = review)
dataset['review'] = dataset['review'].apply(non_alpha_numeric_remove)

# Import the required library to stem each word to its root form
from nltk.stem import PorterStemmer
ps = PorterStemmer()
def stemming(review):
    rev = ' '.join([ps.stem(word) for word in review.split()])
    return rev
dataset['review'] = dataset['review'].apply(stemming)

# Convert everything to lowercase
def to_lower(review):
    return review.lower()
dataset['review'] = dataset['review'].apply(to_lower)

# Importing stopwords from nltk to remove them from reviews
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
print(stopwords.words('english'))
def remove_stopwords(review):
    tokens = tokenizer.tokenize(review)
    tokens = [token.strip() for token in tokens]
    new_tokens = [token for token in tokens if token not in stopwords.words('english')]
    new_review = ' '.join(new_tokens)
    return new_review
dataset['review'] = dataset['review'].apply(remove_stopwords)

# Save cleaned dataset to use later
dataset.to_csv('Cleaned dataset.csv')
