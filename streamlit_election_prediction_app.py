import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Election Prediction System",
    page_icon="🗳️",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

h1, h2, h3 {
    color: white;
}

.stApp {
    background: linear-gradient(to right, #0f172a, #1e293b);
}

.big-font {
    font-size:55px !important;
    font-weight: bold;
    color: #ffffff;
    text-align: center;
}

.sub-font {
    font-size:20px !important;
    color: #cbd5e1;
    text-align: center;
}

.card {
    background-color: #1e293b;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0px 0px 15px rgba(255,255,255,0.1);
}

.prediction-box {
    background-color: #111827;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 30px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown(
    '<p class="big-font">🗳️ Election Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-font">AI Powered Election Winner Prediction Web App</p>',
    unsafe_allow_html=True
)

st.write("")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ---------------------------------------------------
# IF FILE UPLOADED
# ---------------------------------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # ---------------------------------------------------
    # CREATE WINNER COLUMN
    # ---------------------------------------------------
    max_votes = df.groupby(
        ['year', 'pc_name']
    )['totvotpoll'].transform('max')

    df['winner'] = np.where(
        df['totvotpoll'] == max_votes,
        1,
        0
    )

    # ---------------------------------------------------
    # REMOVE NULL VALUES
    # ---------------------------------------------------
    df.dropna(inplace=True)

    # ---------------------------------------------------
    # ENCODING
    # ---------------------------------------------------
    encoder = LabelEncoder()

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
            df[col] = encoder.fit_transform(df[col])

    # ---------------------------------------------------
    # FEATURES
    # ---------------------------------------------------
    X = df[
        [
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
        ]
    ]

    y = df['winner']

    # ---------------------------------------------------
    # TRAIN TEST SPLIT
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ---------------------------------------------------
    # MODEL SELECTION
    # ---------------------------------------------------
    model_choice = st.sidebar.selectbox(
        "🤖 Select Prediction Model",
        [
            "Random Forest",
            "Decision Tree",
            "Logistic Regression",
            "XGBoost"
        ]
    )

    # ---------------------------------------------------
    # TRAIN MODEL
    # ---------------------------------------------------
    if model_choice == "Random Forest":

        model = RandomForestClassifier()

        model.fit(X_train, y_train)

    elif model_choice == "Decision Tree":

        model = DecisionTreeClassifier()

        model.fit(X_train, y_train)

    elif model_choice == "Logistic Regression":

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)

        model = LogisticRegression(max_iter=5000)

        model.fit(X_train, y_train)

    else:

        model = XGBClassifier()

        model.fit(X_train, y_train)

    # ---------------------------------------------------
    # USER INPUT SECTION
    # ---------------------------------------------------
    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🧠 Enter Candidate Details")

    col1, col2 = st.columns(2)

    with col1:

        st_name = st.number_input(
            "🏛️ State Code",
            min_value=0
        )

        year = st.number_input(
            "📅 Election Year",
            min_value=2000,
            max_value=2100
        )

        pc_no = st.number_input(
            "🗺️ Constituency Number",
            min_value=0
        )

        pc_name = st.number_input(
            "📍 Constituency Code",
            min_value=0
        )

        pc_type = st.number_input(
            "🏷️ PC Type",
            min_value=0
        )

    with col2:

        cand_sex = st.selectbox(
            "👤 Candidate Gender",
            [0, 1]
        )

        partyname = st.number_input(
            "🎉 Party Code",
            min_value=0
        )

        partyabbre = st.number_input(
            "🔠 Party Abbreviation Code",
            min_value=0
        )

        totvotpoll = st.number_input(
            "🗳️ Total Votes Polled",
            min_value=0
        )

        electors = st.number_input(
            "👥 Total Electors",
            min_value=0
        )

    st.write("")

    # ---------------------------------------------------
    # PREDICTION BUTTON
    # ---------------------------------------------------
    if st.button("🚀 Predict Election Result"):

        sample_input = pd.DataFrame(
            [[
                st_name,
                year,
                pc_no,
                pc_name,
                pc_type,
                cand_sex,
                partyname,
                partyabbre,
                totvotpoll,
                electors
            ]],
            columns=[
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
            ]
        )

        # Logistic Regression Scaling
        if model_choice == "Logistic Regression":

            sample_input = scaler.transform(
                sample_input
            )

        prediction = model.predict(
            sample_input
        )

        st.write("")

        # ---------------------------------------------------
        # OUTPUT
        # ---------------------------------------------------
        if prediction[0] == 1:

            st.markdown(
                '''
                <div class="prediction-box">
                🏆 Predicted Result: WINNER
                </div>
                ''',
                unsafe_allow_html=True
            )

            st.balloons()

        else:

            st.markdown(
                '''
                <div class="prediction-box">
                ❌ Predicted Result: LOSER
                </div>
                ''',
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# NO FILE MESSAGE
# ---------------------------------------------------
else:

    st.markdown("""
    <div class="card">
    <h2 style='text-align:center; color:white;'>
    📂 Upload your Election Dataset CSV File
    </h2>
    <p style='text-align:center; color:#cbd5e1;'>
    Start predicting election winners using AI & Machine Learning.
    </p>
    </div>
    """, unsafe_allow_html=True)
