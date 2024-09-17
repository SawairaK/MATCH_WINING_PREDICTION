import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import streamlit as st

# Load the model and encoders
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        model_data = pickle.load(file)
    return model_data

data = load_model()

# Unpack the model and encoders
rf_classifier = data["model"]
le_team1 = data["le_team1"]
le_team2 = data["le_team2"]
le_venue = data["le_venue"]
le_toss_winner = data["le_toss_winner"]
le_winner = data["le_winner"]

# Load the dataset to extract all teams and venues
def load_data():
    df = pd.read_csv('ODI_Match_info.csv')
    return df

df = load_data()

# Extract unique teams and venues from the dataset
teams = np.unique(df[['team1', 'team2']].values)
venues = df['venue'].unique()

# Define the predict page
def show_predict_page():
    st.title("Cricket Match Outcome Prediction")

    st.write("""### We need some information to predict the match outcome""")

    # User input for team1, team2, and venue
    team1 = st.selectbox("Team 1", teams)
    team2 = st.selectbox("Team 2", teams)
    venue = st.selectbox("Venue", venues)

    ok = st.button("Predict Outcome")
    if ok:
        try:
            # Prepare the input data for prediction
            X = np.array([[team1, team2, venue, team1]])  # Ensure the feature count matches
            X[:, 0] = le_team1.transform(X[:, 0])
            X[:, 1] = le_team2.transform(X[:, 1])
            X[:, 2] = le_venue.transform(X[:, 2])
            X[:, 3] = le_toss_winner.transform(X[:, 3])
            X = X.astype(float)

            # Predict the toss winner and match winner
            predictions = rf_classifier.predict(X)

            # Debugging: Print prediction output
            st.write("Predictions:", predictions)
            st.write("Predictions shape:", predictions.shape)

            if predictions.shape[0] == 1:
                # Single prediction scenario
                match_winner_encoded = predictions[0]
                match_winner = le_team2.inverse_transform([int(match_winner_encoded)])[0]
                st.subheader(f"The predicted match winner is: {match_winner}")

            elif predictions.shape[1] == 2:
                # Two predictions scenario
                toss_winner_encoded = predictions[0][0]
                match_winner_encoded = predictions[0][1]
                toss_winner = le_toss_winner.inverse_transform([int(toss_winner_encoded)])[0]
                match_winner = le_team2.inverse_transform([int(match_winner_encoded)])[0]
                st.subheader(f"The predicted toss winner is: {toss_winner}")
                st.subheader(f"The predicted match winner is: {match_winner}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Display the prediction page
if __name__ == "__main__":
    show_predict_page()
