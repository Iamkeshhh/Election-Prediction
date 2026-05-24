import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Load dataset

df = pd.read_csv('eci_results_tamilnadu_2026.csv')

# Create Winner Column

max_votes = df.groupby('Constituency')['Total Votes'].transform('max')
df['Winner'] = np.where(df['Total Votes'] == max_votes, 1, 0)

# Copy winner column as future prediction target
# 1 = likely to win next election
# 0 = likely to lose

df['Next_Election_Prediction'] = df['Winner']

# Label Encoding

label_encoders = {}

categorical_columns = ['Constituency', 'Candidate', 'Party']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features

X = df[[
    'Constituency',
    'Candidate',
    'Party',
    'EVM Votes',
    'Postal Votes',
    'Total Votes',
    '% Votes'
]]

# Target

y = df['Next_Election_Prediction']

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Models

models = {
    'Logistic Regression': LogisticRegression(max_iter=500),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
}

best_model = None
best_accuracy = 0
best_model_name = ''

print('\nMODEL ACCURACY RESULTS\n')

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f'{name} Accuracy : {accuracy:.4f}')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print('\nBest Model :', best_model_name)
print('Best Accuracy :', best_accuracy)

# Save model

joblib.dump(best_model, 'election_prediction_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print('\nModel Saved Successfully')
