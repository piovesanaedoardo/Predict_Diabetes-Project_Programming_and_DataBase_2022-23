import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# streamlit run .\project.py

# import csv
mXmh_df = pd.read_csv('mxmh_survey_results.csv')

################## 1. Explore the dataset  ##################
print(mXmh_df.corr())
#############################################################

################## 2. Clean up the dataset ##################
# ---------- Cleaning NaN ----------

# Column 'Age': replacing the null values (only one) with the mean from the Age column
mXmh_df['Age'] = mXmh_df['Age'].fillna(mXmh_df['Age'].mean())
# Column 'Primary streaming service': replacing the null values (only one) with the string 'I do not use a streaming service.'
mXmh_df['Primary streaming service'] = mXmh_df['Primary streaming service'].fillna('I do not use a streaming service.')
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

# ---------- Delete columns 'Timestamp', 'Permissions' ----------
mXmh_df.drop(['Timestamp', 'Permissions'], axis=1, inplace=True)

# ---------- Rename columns 'Primary streaming service' to 'Streaming_Platform' ----------
mXmh_df.rename(columns={'Primary streaming service':'Streaming_Platform'}, inplace=True)

#############################################################

# print(mXmh_df.columns)

# mXmh_df.info()

st.header('Music & Mental Healt')
st.subheader('Survey results on music taste and self-reported mental health')

st.sidebar.subheader('Settings')

if st.sidebar.checkbox('Display DataFrame'):
    st.write('The DataFrame')
    st.write(mXmh_df)

st.subheader('Plots')
# print('Value counts:', mXmh_df['Streaming_Platform'].value_counts())

# st.write(plt.pie(mXmh_df['Streaming_Platform'].value_counts(), labels=mXmh_df['Streaming_Platform']))

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    mXmh_df.Age.hist(ax=ax, bins=30)
    plt.xlabel('Age')
    plt.ylabel('Number of People')
    st.write(fig)
    st.caption('Age distribution')

with col2:
    platform_label = ['Spotify', 'YouTube Music', 'I do not use a streaming service.', 'Apple Music', 'Other streaming service', 'Pandora']
    platform_count = list(mXmh_df['Streaming_Platform'].value_counts())
    platform_colors = ['#2ca02c', 'red', 'grey', 'purple', '#1f77b4', '#ff7f0e']
    explode = (0.1, 0, 0, 0, 0, 0)  # only "explode" the 1st slice ('Spotify')
    fig1, ax1 = plt.subplots()
    ax1.pie(platform_count, explode=explode, autopct='%.1f%%',
            shadow=False, startangle=90, colors=platform_colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.legend(platform_label, loc='best')
    st.pyplot(fig1)
    st.caption('Streaming Platform distribution')



