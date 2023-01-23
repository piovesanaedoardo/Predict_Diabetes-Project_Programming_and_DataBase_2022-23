import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from PIL import Image

# cd Documenti/Project of Programming and Database
# streamlit run titanic.py

@st.cache(allow_output_mutation=True)
def get_data(url):
    titanic_df = pd.read_csv(url)
    return titanic_df

@st.cache
def get_downloadable_data(df):
    return df.to_csv().encode('utf-8')


url = 'https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv'
titanic_df_static = get_data(url)
titanic_df = titanic_df_static.copy()

st.header('Titanic Classification')
st.write("__Let's guess who dies__")
st.write('Dataset source: [click link](' + url + ')')
st.download_button('DOWNLOAD RAW DATA', get_downloadable_data(titanic_df_static), file_name='titanic_raw.csv')

st.sidebar.subheader('Controls')
show_raw_data = st.sidebar.checkbox('Show raw data')

if show_raw_data:
    st.subheader('Raw data')
    st.write(titanic_df)

st.sidebar.code('''
@st.cache(allow_output_mutation=True)
def get_data(url):
    titanic_df = pd.read_csv(url)
    return titanic_df
'''
)

st.sidebar.download_button('DOWNLOAD', get_downloadable_data(titanic_df), file_name='titanic.csv')


st.write(titanic_df.info()) #doesn't work because info() returns none

titanic_df.Sex.replace( { 'male':0, 'female':1}, inplace=True )
titanic_corr = titanic_df.corr()


st.subheader('Correlation Matrix')
fig, ax = plt.subplots(figsize=(10,6)) # show what happens when you change sizes
sns.heatmap(titanic_corr, annot=True, ax=ax)
st.write(fig)

st.subheader('What are the distributions of ages and fares?')

col_1, col_2 = st.columns(2)
with col_1:
    fig, ax = plt.subplots(figsize=(10,6))
    titanic_df.Age.hist(ax=ax)
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')

    st.pyplot(fig)
    st.caption('Ages distribution')

with col_2:
    fig, ax = plt.subplots(figsize=(10,6))
    titanic_df.Fare.hist(ax=ax)
    ax.set_xlabel('Fare')
    ax.set_ylabel('Count')

    st.pyplot(fig)
    st.caption('Fares distribution')

#age_hist = Image.open('age_hist.png')
#st.image(age_hist, caption='Ages Histogram')


st.subheader('Is class correlated?')
st.write(titanic_df[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean())

with st.expander('Show model'):

    st.subheader('A model to predict who dies')

    y = titanic_df.Survived

    select_model = st.selectbox('Select model:', ['RandomForest','GaussianNB'])

    model = RandomForestClassifier()

    if select_model == 'GaussianNB':
        model = GaussianNB()

    choices = st.multiselect('Select features', ['Sex','Pclass','Fare'])

    test_size = st.slider('Test size: ', min_value=0.1, max_value=0.9, step =0.1)

    if len(choices) > 0 and st.button('RUN MODEL'):
        with st.spinner('Training...'):
            x = titanic_df[choices]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=2)

            x_train = x_train.to_numpy().reshape(-1, len(choices))
            model.fit(x_train, y_train)

            x_test = x_test.to_numpy().reshape(-1, len(choices))
            y_pred = model.predict(x_test)

            accuracy = accuracy_score(y_test, y_pred)

            st.write(f'Accuracy = {accuracy:.2f}')