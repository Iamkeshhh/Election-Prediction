import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv('indian-national-level-election.csv')
print(df.head())
print(df.shape)
print(df.columns)
#Create Winner Column
# Find maximum votes in each constituency
max_votes = df.groupby(['year', 'pc_name'])['totvotpoll'].transform('max')

# Create winner column
df['winner'] = np.where(df['totvotpoll'] == max_votes, 1, 0)

print(df[['cand_name', 'totvotpoll', 'winner']].head())
#Handle Missing Values
print(df.isnull().sum())

# Remove missing values if any
df.dropna(inplace=True)
#Encode Categorical Data
label_encoder = LabelEncoder()

categorical_columns = ['st_name', 'pc_name', 'pc_type',
                       'cand_name', 'cand_sex',
                       'partyname', 'partyabbre']

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])
    #Feature Selection
X = df[['st_name', 'year', 'pc_no', 'pc_name',
        'pc_type', 'cand_sex', 'partyname',
        'partyabbre', 'totvotpoll', 'electors']]

y = df['winner']
#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
#Logistic Regression Model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(
    max_iter=5000,
    solver='lbfgs'
)
lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)

from sklearn.metrics import accuracy_score

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, lr_pred))
#Decision Tree Model
dt_model = DecisionTreeClassifier()

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print('Decision Tree Accuracy:',
      accuracy_score(y_test, dt_pred))
#Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print('Random Forest Accuracy:',
      accuracy_score(y_test, rf_pred))
#Model Comparison
models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracies = [
    accuracy_score(y_test, lr_pred),
    accuracy_score(y_test, dt_pred),
    accuracy_score(y_test, rf_pred)
]

comparison = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracies
})

print(comparison)
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

# Train Model
xgb_model.fit(X_train, y_train)

# Prediction
xgb_pred = xgb_model.predict(X_test)

# Accuracy
print("XGBoost Accuracy:",
      accuracy_score(y_test, xgb_pred))
#Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
#Classification Report
print(classification_report(y_test, rf_pred))
#Feature Importance
importance = rf_model.feature_importances_

feature_names = X.columns

feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

feature_df = feature_df.sort_values(by='Importance', ascending=False)

print(feature_df)
#Classification Report
print(classification_report(y_test, rf_pred))
#Votes Distribution
plt.figure(figsize=(10,5))
sns.histplot(df['totvotpoll'], bins=30)
plt.title('Votes Distribution')
plt.show()
#Gender Distribution
sns.countplot(x='cand_sex', data=df)
plt.title('Candidate Gender Distribution')
plt.show()
#Winners vs Losers
sns.countplot(x='winner', data=df)
plt.title('Winner vs Loser')
plt.show()
sample_input = pd.DataFrame([[
    1,      # st_name
    2014,   # year
    10,     # pc_no
    50,     # pc_name
    1,      # pc_type
    1,      # cand_sex
    20,     # partyname
    20,     # partyabbre
    500000, # totvotpoll
    1000000 # electors
]], columns=[
    'st_name',
    'year',
    'pc_no',
    'pc_name',
    'pc_type',
    'cand_sex',
    'partyname',
    'partyabbre',
    'totvotpoll',
    'electors'
])

prediction = rf_model.predict(sample_input)

if prediction[0] == 1:
    print("Predicted Result: Winner")
else:
    print("Predicted Result: Loser")