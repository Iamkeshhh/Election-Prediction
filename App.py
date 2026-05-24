import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load dataset
df = pd.read_csv("eci_results_tamilnadu_2026.csv")

# Load model and encoders
model = joblib.load("election_prediction_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Tamil Nadu Election Prediction",
    page_icon="🗳️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #800000;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: gray;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}

.stButton>button {
    width: 100%;
    height: 55px;
    border-radius: 10px;
    background-color: #800000;
    color: white;
    font-size: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.markdown("<div class='title'>Tamil Nadu Election Prediction System</div>", unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Machine Learning Based Election Prediction Dashboard</div>", unsafe_allow_html=True)

st.write("---")

# ---------------- CONSTITUENCY DROPDOWN ----------------

constituencies = sorted(df['Constituency'].unique())

selected_constituency = st.selectbox(
    "Select Constituency",
    constituencies
)

# ---------------- FIND WINNER ----------------

constituency_data = df[df['Constituency'] == selected_constituency]

winner_row = constituency_data.loc[
    constituency_data['Total Votes'].idxmax()
]

winner_candidate = winner_row['Candidate']
winner_party = winner_row['Party']
winner_votes = winner_row['Total Votes']
winner_percentage = winner_row['% Votes']

# ---------------- SHOW WINNER DETAILS ----------------

st.subheader("Current Election Winner")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Winning Candidate", winner_candidate)

col2.metric("Party", winner_party)

col3.metric("Total Votes", int(winner_votes))

col4.metric("Vote Percentage", f"{winner_percentage}%")

st.write("---")

# ---------------- NEXT ELECTION PREDICTION ----------------

st.subheader("Next Election Prediction")

# Encode input values

constituency_encoded = label_encoders['Constituency'].transform(
    [winner_row['Constituency']]
)[0]

candidate_encoded = label_encoders['Candidate'].transform(
    [winner_candidate]
)[0]

party_encoded = label_encoders['Party'].transform(
    [winner_party]
)[0]

# Create Input Data

input_data = np.array([[

    constituency_encoded,
    candidate_encoded,
    party_encoded,
    winner_row['EVM Votes'],
    winner_row['Postal Votes'],
    winner_row['Total Votes'],
    winner_row['% Votes']

]])

# Predict Button

if st.button("Predict Next Election Result"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1] * 100

    st.write("---")

    if prediction == 1:

        st.success(
            f"✅ {winner_candidate} from {winner_party} is likely to WIN the next election."
        )

    else:

        st.error(
            f"❌ {winner_candidate} from {winner_party} is likely to LOSE the next election."
        )

    st.metric(
        "Winning Probability",
        f"{probability:.2f}%"
    )

# ---------------- DATASET PREVIEW ----------------

st.write("---")

st.subheader("Election Dataset Preview")

st.dataframe(df.head(20))
