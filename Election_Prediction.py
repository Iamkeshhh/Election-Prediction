import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# -------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Election Prediction System",
    page_icon="🗳️",
    layout="wide"
)

# -------------------------------
# TITLE
# -------------------------------
st.title("🗳️ Indian Election Prediction System")
st.markdown("### Machine Learning Based Election Winner Prediction")

st.write(
    "This project predicts election winners using Machine Learning algorithms like "
    "Logistic Regression, Decision Tree, Random Forest, and XGBoost."
)

# -------------------------------
# FILE UPLOAD
# -------------------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Load Dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Columns")
    st.write(df.columns.tolist())

    # -------------------------------
    # CREATE WINNER COLUMN
    # -------------------------------
    max_votes = df.groupby(['year', 'pc_name'])['totvotpoll'].transform('max')
    df['winner'] = np.where(df['totvotpoll'] == max_votes, 1, 0)

    # -------------------------------
    # HANDLE MISSING VALUES
    # -------------------------------
    df.dropna(inplace=True)

    # -------------------------------
    # ENCODE CATEGORICAL COLUMNS
    # -------------------------------
    label_encoder = LabelEncoder()

    categorical_columns = [
        'st_name',
        'pc_name',
        'pc_type',
        'cand_name',
        'cand_sex',
        'partyname',
        'partyabbre'
    ]

    for col in categorical_columns:
        if col in df.columns:
            df[col] = label_encoder.fit_transform(df[col])

    # -------------------------------
    # FEATURE SELECTION
    # -------------------------------
    X = df[[
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
    ]]

    y = df['winner']

    # -------------------------------
    # TRAIN TEST SPLIT
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # -------------------------------
    # LOGISTIC REGRESSION
    # -------------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(
        max_iter=5000,
        solver='lbfgs'
    )

    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)

    lr_accuracy = accuracy_score(y_test, lr_pred)

    # -------------------------------
    # DECISION TREE
    # -------------------------------
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)

    dt_accuracy = accuracy_score(y_test, dt_pred)

    # -------------------------------
    # RANDOM FOREST
    # -------------------------------
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    rf_accuracy = accuracy_score(y_test, rf_pred)

    # -------------------------------
    # XGBOOST
    # -------------------------------
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )

    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    xgb_accuracy = accuracy_score(y_test, xgb_pred)

    # -------------------------------
    # MODEL COMPARISON
    # -------------------------------
    st.subheader("📈 Model Accuracy Comparison")

    comparison = pd.DataFrame({
        'Model': [
            'Logistic Regression',
            'Decision Tree',
            'Random Forest',
            'XGBoost'
        ],
        'Accuracy': [
            lr_accuracy,
            dt_accuracy,
            rf_accuracy,
            xgb_accuracy
        ]
    })

    st.dataframe(comparison)

    # -------------------------------
    # BAR CHART
    # -------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(comparison['Model'], comparison['Accuracy'])

    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')

    st.pyplot(fig)

    # -------------------------------
    # CONFUSION MATRIX
    # -------------------------------
    st.subheader("🧩 Random Forest Confusion Matrix")

    cm = confusion_matrix(y_test, rf_pred)

    fig2, ax2 = plt.subplots()

    cax = ax2.matshow(cm)
    plt.colorbar(cax)

    for (i, j), val in np.ndenumerate(cm):
        ax2.text(j, i, val, ha='center', va='center')

    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')

    st.pyplot(fig2)

    # -------------------------------
    # BEST MODEL
    # -------------------------------
    best_model = comparison.loc[
        comparison['Accuracy'].idxmax(),
        'Model'
    ]

    st.success(f'✅ Best Performing Model: {best_model}')

else:
    st.info("⬅️ Upload your election dataset CSV file to begin.")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Developed using Streamlit and Machine Learning")
