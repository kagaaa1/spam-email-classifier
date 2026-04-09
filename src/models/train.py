import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import os

df = pd.read_csv('data/cleaned/spam_clean.csv')

X = df['text'] 
y = df['label']

vectorizer = CountVectorizer()
X_numeric = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(model, 'models/spam_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("The robot and its translator are saved in the models folder. ")
