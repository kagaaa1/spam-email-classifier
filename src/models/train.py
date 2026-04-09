import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

df = pd.read_csv('data/cleaned/spam_clean.csv')

X = df['text'] 
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()

model.fit(X_train, y_train)

joblib.dump(model, 'models/spam_model.pkl')

print("The robot has been trained and the brain is saved in the models folder. ")
