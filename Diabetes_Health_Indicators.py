import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

diabetes_df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

# cd Documenti/Project of Programming and Database
# streamlit run .\Diabetes_Health_Indicators.py