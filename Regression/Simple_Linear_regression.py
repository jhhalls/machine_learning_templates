"""
@author: jhhalls

SIMPLE LINEAR REGRESSION

1. Import the Libraries
2. Import the dataset
3. Split the data between train and test
4. Standardize the data (feature scaling)
5. Build the model and fit the data
6. Make predictions
7. Visualize the predictions of train and test sets.

"""

# Simple Linear Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#feature scalining is done already by the library
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


#fitting linear model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()    
regressor.fit(X_train, y_train)  

#predecting the test result
#vector of predictions of y 
y_pred = regressor.predict(X_test)       

#visualizing the training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (trainings set)')
plt.xlabel ('Year of experience')
plt.ylabel ('Salary')
plt.show()   

#visualizing the test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (test set)')
plt.xlabel ('Year of experience')
plt.ylabel ('Salary')
plt.show()        
                        
                            
