import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

model = joblib.load('models/spam_model.pkl')

df = pd.read_csv('data/cleaned/spam_clean.csv')

X = df['text']
y = df['label']

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

predictions = model.predict(X_test)

print(f"Accuracy Score: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nDetailed Performance Report:")
print(classification_report(y_test, predictions))
