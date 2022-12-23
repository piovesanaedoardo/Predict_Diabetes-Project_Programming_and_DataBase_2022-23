import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# import csv
diabetes_df = pd.read_csv('diabetes.csv')

# print(diabetes_df.describe())


# split df in 2: diabets and not diabets
diabetic_people_df = diabetes_df.query('Outcome == 1')
not_diabetic_people_df = diabetes_df.query('Outcome == 0')

print(diabetic_people_df.describe())
print(not_diabetic_people_df.describe())

# Glucose
glucose_d = diabetic_people_df['Glucose']
glucose_nd = not_diabetic_people_df['Glucose']

# Insulin
insulin_d = diabetic_people_df['Insulin']
insulin_nd = not_diabetic_people_df['Insulin']

# BMI
BMI_d = diabetic_people_df['BMI']
BMI_nd = not_diabetic_people_df['BMI']

st.header('Diabetes')
st.subheader('Bla bla bla')

st.sidebar.subheader('Settings')

if st.sidebar.checkbox('Display DataFrame'):
    st.write('The DataFrame')
    st.write(diabetes_df)

st.subheader('Plots')
# print('Value counts:', diabetes_df['Streaming_Platform'].value_counts())

# st.write(plt.pie(diabetes_df['Streaming_Platform'].value_counts(), labels=diabetes_df['Streaming_Platform']))
f, ax = plt.subplots(figsize=(10, 8))
corr = diabetes_df.corr()
sns.heatmap(corr, 
cmap=sns.diverging_palette(220, 10, as_cmap=True),
vmin=-1.0, vmax=1.0,
square=True, ax=ax)
st.write(f)
st.caption('Matrix of Correlation')




col1_1, col1_2 = st.columns(2)
with col1_1:
    diabetic_label = ['Non-Diabetic', 'Diabetic']
    diabetic_count = list(diabetes_df['Outcome'].value_counts())
    colors_df = ['#becee6', '#4287f5'] # not / diab
    explode = (0.1, 0)
    fig, ax = plt.subplots()
    ax.pie(diabetic_count, explode=explode, autopct='%.1f%%',
            shadow=False, startangle=90, colors=colors_df)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.legend(diabetic_label, loc='best')
    st.pyplot(fig)
    st.caption('Diabetic distribution')
with col1_2:
    fig, ax = plt.subplots()
    diabetes_df.Age.hist(ax=ax, bins=30)
    plt.xlabel('Age')
    plt.ylabel('Number of Women')
    st.write(fig)
    st.caption('Age distribution')


col2_1, col2_2, col2_3 = st.columns(3)
with col2_1:
    fig, ax = plt.subplots()
    ax.hist([glucose_d, glucose_nd], label=['diabetic', 'not_diabetic'], color=['#4287f5', '#becee6'], bins=50)
    ax.set_ylabel("Glucose level")
    plt.legend(loc='upper right')
    st.write(fig)
    st.caption('Glucose of People with Diabet and without Diabet. Two hours after drinking the glucose solution, a normal blood glucose level is lower than 155 mg/dL (8.6 mmol/L).')
with col2_2:
    fig, ax = plt.subplots()
    ax.hist([insulin_d, insulin_nd], label=['diabetic', 'not_diabetic'], color=['#4287f5', '#becee6'], bins=100)
    ax.set_ylabel("Insulin level")
    plt.legend(loc='upper right')
    st.write(fig)
    st.caption('Insuline of People with Diabet and without Diabet')
with col2_3:
    fig, ax = plt.subplots()
    ax.hist([BMI_d, BMI_nd], label=['diabetic', 'not_diabetic'], color=['#4287f5', '#becee6'], bins=100)
    ax.set_ylabel("BMI level")
    plt.legend(loc='upper right')
    st.write(fig)
    st.caption('BMI of People with Diabet and without Diabet')

# col3_1, col3_2 = st.columns(2)

# with col3_1:
#     fig, ax = plt.subplots(figsize=(10, 8))
#     corr = diabetic_people_df.corr()
#     sns.heatmap(corr, 
#     cmap=sns.diverging_palette(220, 10, as_cmap=True),
#     vmin=-1.0, vmax=1.0,
#     square=True, ax=ax)
#     st.write(fig)
#     st.caption('Matrix of Correlation of Diabetic')

# with col3_2:
#     fig, ax = plt.subplots(figsize=(10, 8))
#     corr = not_diabetic_people_df.corr()
#     sns.heatmap(corr, 
#     cmap=sns.diverging_palette(220, 10, as_cmap=True),
#     vmin=-1.0, vmax=1.0,
#     square=True, ax=ax)
#     st.write(fig)
#     st.caption('Matrix of Correlation of NOT Diabetic')


# fig, ax = plt.subplots()
# ax.hist([insulin_d, insulin_nd], label=['diabetic', 'not_diabetic'], color=['#4287f5', '#becee6'], bins=100)
# ax.set_ylabel("Insulin level")
# plt.legend(loc='upper right')
# st.write(fig)
# st.caption('Insuline of People with Diabet and without Diabet')