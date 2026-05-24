import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

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

.stApp {
    background: linear-gradient(to right, #0f172a, #1e293b);
}

.big-font {
    font-size:55px !important;
    font-weight: bold;
    color: white;
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
    box-shadow: 0px 0px 20px rgba(255,255,255,0.1);
}

.prediction-box {
    background-color: #111827;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: white;
    font-size: 32px;
    font-weight: bold;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.markdown(
    '<p class="big-font">🗳️ Election Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-font">AI Powered Election Winner Prediction</p>',
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
    # SAVE ORIGINAL VALUES
    # ---------------------------------------------------
    original_df = df.copy()

    # ---------------------------------------------------
    # LABEL ENCODING
    # ---------------------------------------------------
    encoder_dict = {}

    categorical_columns = [
        'st_name',
        'pc_name',
        'cand_name',
        'cand_sex',
        'partyname',
        'partyabbre'
    ]

    for col in categorical_columns:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(df[col])

        encoder_dict[col] = encoder

    # ---------------------------------------------------
    # FEATURES
    # ---------------------------------------------------
    X = df[
        [
            'st_name',
            'year',
            'pc_no',
            'pc_name',
            'cand_name',
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
    # MODEL TRAINING
    # ---------------------------------------------------
    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    # ---------------------------------------------------
    # USER INPUT SECTION
    # ---------------------------------------------------
    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🧠 Candidate Information Form")

    col1, col2 = st.columns(2)

    # ---------------------------------------------------
    # COLUMN 1
    # ---------------------------------------------------
    with col1:

        st_name_option = st.selectbox(
            "🏛️ State Name",
            original_df['st_name'].unique()
        )

        year_option = st.selectbox(
            "📅 Election Year",
            sorted(original_df['year'].unique())
        )

        pc_no_option = st.selectbox(
            "🗺️ Constituency Number",
            sorted(original_df['pc_no'].unique())
        )

        pc_name_option = st.selectbox(
            "📍 Constituency Name",
            original_df['pc_name'].unique()
        )

        cand_name_option = st.selectbox(
            "👤 Candidate Name",
            original_df['cand_name'].unique()
        )

    # ---------------------------------------------------
    # COLUMN 2
    # ---------------------------------------------------
    with col2:

        cand_sex_option = st.selectbox(
            "⚧ Candidate Gender",
            original_df['cand_sex'].unique()
        )

        partyname_option = st.selectbox(
            "🎉 Party Name",
            original_df['partyname'].unique()
        )

        partyabbre_option = st.selectbox(
            "🔠 Party Abbreviation",
            original_df['partyabbre'].unique()
        )

        totvotpoll_option = st.number_input(
            "🗳️ Total Votes Polled",
            min_value=0,
            step=1000
        )

        electors_option = st.number_input(
            "👥 Total Electors",
            min_value=0,
            step=1000
        )

    st.write("")

    # ---------------------------------------------------
    # PREDICT BUTTON
    # ---------------------------------------------------
    if st.button("🚀 Predict Election Result"):

        # Encode Inputs
        st_name_encoded = encoder_dict['st_name'].transform(
            [st_name_option]
        )[0]

        pc_name_encoded = encoder_dict['pc_name'].transform(
            [pc_name_option]
        )[0]

        cand_name_encoded = encoder_dict['cand_name'].transform(
            [cand_name_option]
        )[0]

        cand_sex_encoded = encoder_dict['cand_sex'].transform(
            [cand_sex_option]
        )[0]

        partyname_encoded = encoder_dict['partyname'].transform(
            [partyname_option]
        )[0]

        partyabbre_encoded = encoder_dict['partyabbre'].transform(
            [partyabbre_option]
        )[0]

        # Create Input DataFrame
        sample_input = pd.DataFrame(
            [[
                st_name_encoded,
                year_option,
                pc_no_option,
                pc_name_encoded,
                cand_name_encoded,
                cand_sex_encoded,
                partyname_encoded,
                partyabbre_encoded,
                totvotpoll_option,
                electors_option
            ]],
            columns=[
                'st_name',
                'year',
                'pc_no',
                'pc_name',
                'cand_name',
                'cand_sex',
                'partyname',
                'partyabbre',
                'totvotpoll',
                'electors'
            ]
        )

        # Prediction
        prediction = model.predict(sample_input)

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
# NO FILE
# ---------------------------------------------------
else:

    st.markdown("""
    <div class="card">
    <h2 style='text-align:center; color:white;'>
    📂 Upload Election Dataset CSV File
    </h2>

    <p style='text-align:center; color:#cbd5e1;'>
    Predict election winners using Artificial Intelligence
    and Machine Learning.
    </p>

    </div>
    """, unsafe_allow_html=True)
