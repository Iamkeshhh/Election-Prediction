import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load files

model = joblib.load('election_prediction_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Load dataset

df = pd.read_csv('eci_results_tamilnadu_2026.csv')

# Page Config

st.set_page_config(
    page_title='Tamil Nadu Election Prediction',
    page_icon='🗳️',
    layout='wide'
)

# Custom CSS

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f7fa;
    }

    .title {
        text-align: center;
        color: #800000;
        font-size: 45px;
        font-weight: bold;
    }

    .sub {
        text-align: center;
        color: #444;
        font-size: 20px;
    }

    .stButton>button {
        background-color: #800000;
        color: white;
        font-size: 20px;
        border-radius: 10px;
        width: 100%;
        height: 55px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Title

st.markdown('<div class="title">Tamil Nadu Election Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Machine Learning Based Election Winner Prediction</div>', unsafe_allow_html=True)

st.write('---')

# Sidebar

st.sidebar.header('Election Analysis Dashboard')

# Dropdown Values

constituency_list = sorted(df['Constituency'].unique())
party_list = sorted(df['Party'].unique())
candidate_list = sorted(df['Candidate'].unique())

# Layout

col1, col2 = st.columns(2)

with col1:

    constituency = st.selectbox(
        'Select Constituency',
        constituency_list
    )

    candidate = st.selectbox(
        'Select Candidate',
        candidate_list
    )

    party = st.selectbox(
        'Select Party',
        party_list
    )

with col2:

    evm_votes = st.number_input(
        'EVM Votes',
        min_value=0,
        value=50000
    )

    postal_votes = st.number_input(
        'Postal Votes',
        min_value=0,
        value=500
    )

    total_votes = st.number_input(
        'Total Votes',
        min_value=0,
        value=50500
    )

    percentage_votes = st.slider(
        'Percentage Votes',
        0.0,
        100.0,
        45.0
    )

# Encode Inputs

constituency_encoded = label_encoders['Constituency'].transform([constituency])[0]
candidate_encoded = label_encoders['Candidate'].transform([candidate])[0]
party_encoded = label_encoders['Party'].transform([party])[0]

# Prediction Button

if st.button('Predict Election Result'):

    input_data = np.array([[
        constituency_encoded,
        candidate_encoded,
        party_encoded,
        evm_votes,
        postal_votes,
        total_votes,
        percentage_votes
    ]])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1] * 100

    st.write('---')

    if prediction == 1:

        st.success(f'''✅ {candidate} from {party} is likely to WIN the next election.''')

        st.metric(
            label='Winning Probability',
            value=f'{probability:.2f}%'
        )

    else:

        st.error(f'''❌ {candidate} from {party} is likely to LOSE the next election.''')

        st.metric(
            label='Winning Probability',
            value=f'{probability:.2f}%'
        )

# Dataset Preview

st.write('---')

st.subheader('Election Dataset Preview')

st.dataframe(df.head(20))
