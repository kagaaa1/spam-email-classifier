import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
model = joblib.load('models/spam_model.pkl')

predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)

print(f"The robot got a gradeof: {score * 100}%")
print("\nHere is the full report :")
print(classification_report(y_test, predictions))
