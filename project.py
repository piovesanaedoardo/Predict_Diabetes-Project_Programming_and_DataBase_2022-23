import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# streamlit run .\project.py

# -------------- DataFrame & cleaning --------------
mXmh_df = pd.read_csv('mxmh_survey_results.csv')

# Column 'Age': replacing the null values (only one) with the mean from the Age column
mXmh_df['Age'] = mXmh_df['Age'].fillna(mXmh_df['Age'].mean())
# Column 'Primary streaming service': replacing the null values (only one) with the string 'No one'
mXmh_df['Primary streaming service'] = mXmh_df['Primary streaming service'].fillna('No one')
# Column 'While working': replacing the null values with the string 'Not specified'
mXmh_df['While working'] = mXmh_df['While working'].fillna('Not specified')
# Column 'Instrumentalist': replacing the null values with the string 'Not specified'
mXmh_df['Instrumentalist'] = mXmh_df['Instrumentalist'].fillna('Not specified')
# Column 'Composer': replacing the null values with the string 'Not specified'
mXmh_df['Composer'] = mXmh_df['Composer'].fillna('Not specified')
# Column 'Foreign languages': replacing the null values with the string 'Not specified'
mXmh_df['Foreign languages'] = mXmh_df['Foreign languages'].fillna('Not specified')
# Column 'BPM': replacing the null values with the mean from the BPM column
mXmh_df['BPM'] = mXmh_df['BPM'].fillna(mXmh_df['BPM'].mean())
# Column 'Music effects': replacing the null values with the string 'Not specified'
mXmh_df['Music effects'] = mXmh_df['Music effects'].fillna('Not specified')

# delete columns 'Timestamp', 'Permissions'
mXmh_df = mXmh_df.drop(['Timestamp', 'Permissions'], axis=1, inplace=True)

mXmh_df.info()

st.header('Music & Mental Healt')
st.subheader('Survey results on music taste and self-reported mental health')

st.sidebar.subheader('Settings')

if st.sidebar.checkbox('Display DataFrame'):
    st.write('The DataFrame')
    st.write(mXmh_df)