import pandas as pd
import numpy as np
import seaborn as sns
import string
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

test_data = pd.read_csv('Test.csv')
train_data = pd.read_csv('Train.csv')
valid_data = pd.read_csv('Valid.csv')

# merging the three data files
data = pd.concat([test_data, train_data, valid_data])

# Defining
data_class = data[(data['label'] == 0) | (data['label'] == 1)]
x = data_class['text']
y = data_class['label']

# Data example for vectorization
data_example = data['text'].iloc[50]

# Start of vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([data_example])
vectorizer.get_feature_names_out()
print(X.toarray())

# Clean the data using NLTK
nltk.download('stopwords')
nltk.download('words')

# Function to clean the text
def text_cleaning(text):
    # creates a list to gather all the characters that aren't already in string.punctuation
    text_without_punctuation = [char for char in text if char not in string.punctuation]
    # converts it back into a string
    text_without_punctuation = ''.join(text_without_punctuation)
    # creates another new list with words that aren't in the stopwords.words('english)'
    return [word for word in text_without_punctuation.split() if word.lower() not in stopwords.words('english')]

# Transforms the texts into numerical format
text_vectorizer = CountVectorizer(analyzer= text_cleaning).fit(x)
x = text_vectorizer.transform(x)

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=110)

# Using naive bayes to calculate the classification report and score

mnb = MultinomialNB()
mnb.fit(x_train, y_train)
predictmnb = mnb.predict(x_test)

# Scores it on overall accuracy
score = (accuracy_score(y_test, predictmnb))
print("Score: ")
print(score)


# Predict Positive
positive_review = data['text'][30]
positive_prediction = text_vectorizer.transform([positive_review])
pos_prediction = mnb.predict(positive_prediction)[0]
pos_actual = data['label'][30]

print("Review: ")
print(positive_review)

print("Prediction: ")
print(pos_prediction)

print("Actual:")
print(pos_actual)

# Predict Negative
negative_review = data['text'][100]
negative_prediction = text_vectorizer.transform([negative_review])
neg_prediction = mnb.predict(negative_prediction)[0]
neg_actual = data['label'][100]

print("Review: ")
print(negative_review)

print("Prediction: ")
print(neg_prediction)

print("Actual:")
print(neg_actual)
