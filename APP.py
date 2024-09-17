import streamlit as st
from show_predict_page import show_predict_page
from Explore_page import show_explore_page

page = st.sidebar.selectbox("Explore or Predict", ("Predict", "Explore"))

if page == "Predict":
    show_predict_page()
else:
    show_explore_page()
