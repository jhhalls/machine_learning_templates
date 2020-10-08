"""
@author: jhhalls

NATURAL LANGUAGE PROCESSING

"""
# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# load the data
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter= '\t', quoting=3)

#cleaning the texts
import re
import nltk

#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

# iterate through a range to split, and generalize the words to its root word
for i in range(0, 1000):
	#remove everything except the alphabets
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
	#convert to lowercase
    review = review.lower()
	# tokenize the sentence
    review = review.split()
	#reduce every word to its root word
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
	# join the words back
    review = ' '.join(review)
	#append the processed sentence to the corpus
    corpus.append(review)
    

#creating the bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X= cv.fit_transform(corpus).toarray()
y= dataset.iloc[:,1].values
              
#use naive bayes as model 
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#fitting NAIVE BAYES CLASSIFIER to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#predict test set result
y_pred = classifier.predict(X_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred)            
