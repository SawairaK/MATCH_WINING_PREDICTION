import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
import os

# Loading dataset
df = pd.read_csv("ODI_Match_info.csv")

# Checking the dataset
print(df.head())
print(df.info())

# features for prediction are 'team1', 'team2', 'venue', and 'toss_winner'
# Target label is 'winner'
# Preprocess the dataset: drop rows with missing values
df = df[['team1', 'team2', 'venue', 'toss_winner', 'winner']].dropna()

# Convert categorical data to numerical using LabelEncoder
le_team1 = LabelEncoder().fit(df['team1'])
le_team2 = LabelEncoder().fit(df['team2'])
le_venue = LabelEncoder().fit(df['venue'])
le_toss_winner = LabelEncoder().fit(df['toss_winner'])
le_winner = LabelEncoder().fit(df['winner'])
# LabelEncoder for each column
df['team1'] = le_team1.transform(df['team1'])
df['team2'] = le_team2.transform(df['team2'])
df['venue'] = le_venue.transform(df['venue'])
df['toss_winner'] = le_toss_winner.transform(df['toss_winner'])
df['winner'] = le_winner.transform(df['winner'])

# Define features(input) and target(output) variable
X = df[['team1', 'team2', 'venue', 'toss_winner']]
y = df['winner']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Classifier Accuracy: {accuracy * 100:.2f}%")

# Feature Importance Analysis Graph Visual Representation
importances = rf_classifier.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title('Feature Importance')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Save the model and encoders
model_data = {
    "model": rf_classifier,  # your trained RandomForest model
    "le_team1": le_team1,  # encoder for team1
    "le_team2": le_team2,  # encoder for team2
    "le_venue": le_venue,  # encoder for venues
    "le_toss_winner": le_toss_winner,  # encoder for toss winners
    "le_winner": le_winner  # encoder for the 'winner' column
}

with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(model_data, file)

# Load the model and encoders
with open('saved_steps.pkl', 'rb') as file:
    model_data = pickle.load(file)

# Unpack the model and encoders
rf_classifier_loaded = model_data["model"]
le_team1 = model_data["le_team1"]
le_team2 = model_data["le_team2"]
le_venue = model_data["le_venue"]
le_toss_winner = model_data["le_toss_winner"]
le_winner = model_data["le_winner"]

# Make predictions on the test set
y_pred_loaded = rf_classifier_loaded.predict(X_test)

# Calculate accuracy for the loaded model
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
print(f"Loaded Random Forest Classifier Accuracy: {accuracy_loaded * 100:.2f}%")

# Output the predicted values
print("Predicted values:", y_pred_loaded)
