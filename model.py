import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression

df = pd.read_csv('hiring.csv')

df['experience'].fillna(0, inplace=True)
df['test_score'].fillna(df['test_score'].mean(), inplace=True)

X = df.iloc[:,:3]

# converting words to integers

def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = df.iloc[:, -1]

# Since the dataset is very small we'll train with all data
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Save the model
pickle.dump(lin_reg, open('model.pkl', 'wb'))

# Load the model back and predict
model = pickle.load(open('model.pkl', 'rb'))
model.predict([[2,9,6]])
