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

st.markdown("""
<div class='title'>
🗳️ Tamil Nadu Election Prediction System
</div>

<div class='subtitle'>
Machine Learning Based Election Prediction Dashboard
</div>
""", unsafe_allow_html=True)

st.write("---")

import base64

# Background Video
video_file = open("CM.mp4", "rb")
video_bytes = video_file.read()
video_base64 = base64.b64encode(video_bytes).decode()

st.markdown(
    f"""
    <style>

    /* Full-screen background video */
    #bg-video {{
        position: fixed;
        right: 0;
        bottom: 0;
        min-width: 100%;
        min-height: 100%;
        object-fit: cover;
        z-index: -1;
        opacity: 0.35;   /* adjust visibility */
    }}

    /* Transparent overlay for readability */
    .overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.25);
        z-index: -1;
    }}

    /* Main content styling */
    .main {{
        background: transparent;
    }}

    .title {{
        text-align:center;
        font-size:55px;
        font-weight:bold;
        color:white;
        text-shadow:3px 3px 10px black;
        margin-top:20px;
    }}

    .subtitle {{
        text-align:center;
        font-size:24px;
        color:#f8f9fa;
        text-shadow:2px 2px 8px black;
    }}

    div[data-testid="stMetric"] {{
        background: rgba(255,255,255,0.85);
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
    }}

    div[data-testid="stDataFrame"] {{
        background: rgba(255,255,255,0.92);
        border-radius: 15px;
    }}

    </style>

    <video autoplay muted loop id="bg-video">
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    </video>

    <div class="overlay"></div>

    """,
    unsafe_allow_html=True
)

# ---------------- CONSTITUENCY DROPDOWN ----------------

constituencies = sorted(df['Constituency'].unique())

selected_constituency = st.selectbox(
    "Select Constituency",
    constituencies
)

st.markdown("""
<style>

.party-logo {
    width:120px;
    height:120px;
    animation: float 3s ease-in-out infinite;
    filter: drop-shadow(0px 0px 15px gold);
}

@keyframes float {
    0% {
        transform: translateY(0px) rotate(0deg);
    }
    50% {
        transform: translateY(-15px) rotate(5deg);
    }
    100% {
        transform: translateY(0px) rotate(0deg);
    }
}

.winner-container{
    text-align:center;
    margin-top:20px;
    margin-bottom:20px;
}

.winner-text{
    font-size:32px;
    font-weight:bold;
    color:white;
    text-shadow:2px 2px 10px black;
}

</style>
""", unsafe_allow_html=True)

logo_path = ""

if winner_party == "Tamilaga Vettri Kazhagam":
    logo_path = "tvk_logo.png"

elif winner_party == "Dravida Munnetra Kazhagam":
    logo_path = "dmk_logo.png"

elif winner_party == "All India Anna Dravida Munnetra Kazhagam":
    logo_path = "aiadmk_logo.png"

import base64

with open(logo_path, "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode()

st.markdown(
    f"""
    <div class="winner-container">

        <img class="party-logo"
             src="data:image/png;base64,{encoded}">

        <div class="winner-text">
            🏆 WINNER PARTY
            <br>
            {winner_party}
        </div>

    </div>
    """,
    unsafe_allow_html=True
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


# ---------------- CONSTITUENCY RESULTS ----------------

st.subheader(f"Election Results - {selected_constituency}")

# Sort by total votes descending
constituency_data = constituency_data.sort_values(
    by='Total Votes',
    ascending=False
)

# Winner Row
winner_row = constituency_data.iloc[0]

winner_candidate = winner_row['Candidate']
winner_party = winner_row['Party']
winner_votes = winner_row['Total Votes']
winner_percentage = winner_row['% Votes']

# ---------------- WINNER CARD ----------------

st.success(
    f"🏆 Winner: {winner_candidate} ({winner_party})"
)

# Metrics

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Winning Candidate",
    winner_candidate
)

col2.metric(
    "Party",
    winner_party
)

col3.metric(
    "Total Votes",
    int(winner_votes)
)

col4.metric(
    "Vote %",
    f"{winner_percentage}%"
)

st.write("---")


# ---------------- WINNER VIDEO ----------------



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

st.dataframe(df.head(4258))
