import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# streamlit run .\project.py

mXmh_df = pd.read_csv('mxmh_survey_results.csv')

mXmh_df.head(10)

st.header('Music & Mental Healt')
st.subheader('Survey results on music taste and self-reported mental health')