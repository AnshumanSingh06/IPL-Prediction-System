import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="IPL Winner Predictor",
    page_icon="🏏",
    layout="wide"
)

# =====================================
# LOAD MODEL & ENCODERS
# =====================================

import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'models', 'ipl_model.pkl'))
le_team = joblib.load(os.path.join(BASE_DIR, 'models', 'le_team.pkl'))
le_venue = joblib.load(os.path.join(BASE_DIR, 'models', 'le_venue.pkl'))
le_decision = joblib.load(os.path.join(BASE_DIR, 'models', 'le_decision.pkl'))



# =====================================
# TITLE
# =====================================

st.markdown("""
<h1 style='text-align:center; color:#ff4b4b;'>
🏏 IPL Match Winner Predictor
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center; font-size:18px;'>
Predict IPL match winners using Machine Learning
</p>
""", unsafe_allow_html=True)

st.divider()

# =====================================
# INPUTS
# =====================================

teams = sorted(le_team.classes_)
venues = sorted(le_venue.classes_)

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Select Team 1", teams)

with col2:
    team2 = st.selectbox("Select Team 2", teams)

col3, col4 = st.columns(2)

with col3:
    toss_winner = st.selectbox(
        "Toss Winner",
        [team1, team2]
    )

with col4:
    toss_decision = st.selectbox(
        "Toss Decision",
        le_decision.classes_
    )

venue = st.selectbox(
    "Select Venue",
    venues
)

st.divider()

# =====================================
# PREDICTION
# =====================================

if st.button("Predict Winner"):

    if team1 == team2:
        st.error("Please select different teams.")
    else:

        # Encode inputs
        input_data = pd.DataFrame({
            'team1': [le_team.transform([team1])[0]],
            'team2': [le_team.transform([team2])[0]],
            'toss_winner': [le_team.transform([toss_winner])[0]],
            'toss_decision': [le_decision.transform([toss_decision])[0]],
            'venue': [le_venue.transform([venue])[0]]
        })

        # Predict
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        predicted_winner = le_team.inverse_transform([prediction])[0]

        # Team probabilities
        team1_index = le_team.transform([team1])[0]
        team2_index = le_team.transform([team2])[0]

        team1_prob = round(probabilities[team1_index] * 100, 2)
        team2_prob = round(probabilities[team2_index] * 100, 2)

        # Result
        st.success(f'🏆 Predicted Winner: {predicted_winner}')

        # Probability chart
        fig = go.Figure(data=[
            go.Bar(
                x=[team1, team2],
                y=[team1_prob, team2_prob],
                text=[
                    f'{team1_prob}%',
                    f'{team2_prob}%'
                ],
                textposition='auto'
            )
        ])

        fig.update_layout(
            title='Winning Probability',
            yaxis_title='Probability %',
            xaxis_title='Teams'
        )

        st.plotly_chart(fig, use_container_width=True)

# =====================================
# FOOTER
# =====================================

st.divider()

st.markdown("""
<p style='text-align:center;'>
Built using Python, Scikit-learn, Streamlit & Plotly
</p>
""", unsafe_allow_html=True)