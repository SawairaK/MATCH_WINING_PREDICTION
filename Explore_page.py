import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    # Load the cricket match dataset
    df = pd.read_csv("ODI_Match_info.csv")
    # Filter only relevant columns for exploration
    df = df[['team1', 'team2', 'venue', 'toss_winner', 'winner']].dropna()
    return df

df = load_data()

def show_explore_page():
    st.title("Explore Cricket Match Data")

    st.write(
        """
        ### ODI Match Info Dataset
        """
    )

    # Plot the distribution of matches by venue
    st.write("#### Number of Matches Played at Each Venue")
    venue_data = df['venue'].value_counts()

    fig1, ax1 = plt.subplots(figsize=(28, 22)) 
    ax1.pie(venue_data, labels=venue_data.index, autopct="%1.2f%%", startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures the pie is drawn as a circle.
    st.pyplot(fig1)

    # Bar chart of toss wins per team
    st.write("#### Toss Wins by Team")
    toss_wins = df['toss_winner'].value_counts()
    st.bar_chart(toss_wins)

    # Bar chart of match wins by team
    st.write("#### Match Wins by Team")
    match_wins = df['winner'].value_counts()
    st.bar_chart(match_wins)

    # Display mean toss win percentage per team
    st.write("#### Toss Win Percentage by Team")
    toss_win_percentage = (df['toss_winner'].value_counts() / (df['team1'].value_counts() + df['team2'].value_counts())) * 100
    st.bar_chart(toss_win_percentage)

    # Display win/loss analysis per team
    st.write("#### Team Win/Loss Ratio")
    win_loss_ratio = df['winner'].value_counts() / (df['team1'].value_counts() + df['team2'].value_counts())
    st.bar_chart(win_loss_ratio)

# Call this function in the main script to show the explore page
if __name__ == "__main__":
    show_explore_page()
