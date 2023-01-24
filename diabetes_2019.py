import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

diabetes_df = pd.read_csv('diabetes_dataset__2019.csv')

# cd Documenti/Project of Programming and Database
# streamlit run .\diabetes_2019.py

# ---------------------------- 1.1 Explore the dataset ----------------------------
print(diabetes_df.info())
# Data columns (total 18 columns):
#  #   Column            Non-Null Count  Dtype
# ---  ------            --------------  -----
#  0   Age               952 non-null    object
#  1   Gender            952 non-null    object
#  2   Family_Diabetes   952 non-null    object
#  3   highBP            952 non-null    object
#  4   PhysicallyActive  952 non-null    object
#  5   BMI               948 non-null    float64
#  6   Smoking           952 non-null    object
#  7   Alcohol           952 non-null    object
#  8   Sleep             952 non-null    int64
#  9   SoundSleep        952 non-null    int64
#  10  RegularMedicine   952 non-null    object
#  11  JunkFood          952 non-null    object
#  12  Stress            952 non-null    object
#  13  BPLevel           952 non-null    object
#  14  Pregancies        910 non-null    float64
#  15  Pdiabetes         951 non-null    object
#  16  UriationFreq      952 non-null    object
#  17  Diabetic          951 non-null    object

print(diabetes_df.describe())

# print(diabetes_df.corr())
#                  BMI     Sleep  SoundSleep  Pregancies
# BMI         1.000000 -0.067896   -0.298074   -0.045124
# Sleep      -0.067896  1.000000    0.534686    0.041959
# SoundSleep -0.298074  0.534686    1.000000    0.150531
# Pregancies -0.045124  0.041959    0.150531    1.000000
# 
# There are too much columns with no numerical data, so I have to fix this problem
# (i.e. the column 'Family_Diabetes' has value 'yes'/'no'. I change this values by replacing 1/0)


# ---------------------------- 1.2 Clean up the dataset ----------------------------
# see the value of each column
# print('see the value of each column')
# for col in diabetes_df.columns:
#     print(col)
#     print(diabetes_df[col].value_counts())
#     print("\n")

# Replacing in col 'Age' less than 40 with 0-40
diabetes_df['Age'].replace('less than 40','0-40', inplace=True)
# Replacing in col 'Age' 40-49 with 40,49
# diabetes_df['Age'].replace('40-49','40-49', inplace=True)
# Replacing in col 'Age' 50-59 with 50,59
# diabetes_df['Age'].replace('50-59','50-59', inplace=True)
# Replacing in col 'Gender' Female with 1
diabetes_df['Age'].replace('60 or older','60-99', inplace=True)
# print(diabetes_df['Age'].unique())

# Replacing in col 'Gender' Male with 0
diabetes_df['Gender'].replace('Male','0', inplace=True)
# Replacing in col 'Gender' Female with 1
diabetes_df['Gender'].replace('Female','1', inplace=True)

# Replacing in col 'Family_Diabetes' no with 0
diabetes_df['Family_Diabetes'].replace('no','0', inplace=True)
# Replacing in col 'Family_Diabetes' yes with 1
diabetes_df['Family_Diabetes'].replace('yes','1', inplace=True)

# Replacing in col 'highBP' no with 0
diabetes_df['highBP'].replace('no','0', inplace=True)
# Replacing in col 'highBP' yes with 1
diabetes_df['highBP'].replace('yes','1', inplace=True)

# Replacing in col 'PhysicallyActive' none with 0
diabetes_df['PhysicallyActive'].replace('none',0, inplace=True)
# Replacing in col 'PhysicallyActive' less than half an hr with 1
diabetes_df['PhysicallyActive'].replace('less than half an hr',1, inplace=True)
# Replacing in col 'PhysicallyActive' more than half an hr with 2
diabetes_df['PhysicallyActive'].replace('more than half an hr',2, inplace=True)
# Replacing in col 'PhysicallyActive' one hr or more with 3
diabetes_df['PhysicallyActive'].replace('one hr or more',3, inplace=True)

# Replacing in col 'Smoking' no with 0
diabetes_df['Smoking'].replace('no','0', inplace=True)
# Replacing in col 'Smoking' yes with 1
diabetes_df['Smoking'].replace('yes','1', inplace=True)

# Replacing in col 'Alcohol' no with 0
diabetes_df['Alcohol'].replace('no','0', inplace=True)
# Replacing in col 'Alcohol' yes with 1
diabetes_df['Alcohol'].replace('yes','1', inplace=True)

# Replacing in col 'RegularMedicine' no/o with 0
diabetes_df['RegularMedicine'].replace('no','0', inplace=True)
diabetes_df['RegularMedicine'].replace('o','0', inplace=True)
# Replacing in col 'RegularMedicine' yes with 1
diabetes_df['RegularMedicine'].replace('yes','1', inplace=True)

# Replacing in col 'JunkFood' occasionally with 0
diabetes_df['JunkFood'].replace('occasionally','0', inplace=True)
# Replacing in col 'JunkFood' often with 1
diabetes_df['JunkFood'].replace('often','1', inplace=True)
# Replacing in col 'JunkFood' very often with 2
diabetes_df['JunkFood'].replace('very often','2', inplace=True)
# Replacing in col 'JunkFood' always with 3
diabetes_df['JunkFood'].replace('always','3', inplace=True)

# Replacing in col 'Stress' not at all with 0
diabetes_df['Stress'].replace('not at all','0', inplace=True)
# Replacing in col 'Stress' sometimes with 1
diabetes_df['Stress'].replace('sometimes','1', inplace=True)
# Replacing in col 'Stress' very often with 2
diabetes_df['Stress'].replace('very often','2', inplace=True)
# Replacing in col 'Stress' always with 3
diabetes_df['Stress'].replace('always','3', inplace=True)

# remove capital letter and spacing
diabetes_df['BPLevel'] = diabetes_df['BPLevel'].str.lower().str.strip()
# Replacing in col 'BPLevel' low with 0
diabetes_df['BPLevel'].replace('low',0, inplace=True)
# Replacing in col 'BPLevel' normal with 1
diabetes_df['BPLevel'].replace('normal',1, inplace=True)
# Replacing in col 'BPLevel' high with 2
diabetes_df['BPLevel'].replace('high',2, inplace=True)

# Column 'Pregancies': replacing the null values with 0
diabetes_df['Pdiabetes'] = diabetes_df['Pdiabetes'].fillna(0)
# Replacing in col 'Pdiabetes' no with 0
diabetes_df['Pdiabetes'].replace('no','0', inplace=True)
# Replacing in col 'Pdiabetes' yes with 1
diabetes_df['Pdiabetes'].replace('yes','1', inplace=True)

# Replacing in col 'UriationFreq' not much with 0
diabetes_df['UriationFreq'].replace('not much','0', inplace=True)
# Replacing in col 'UriationFreq' quite often with 1
diabetes_df['UriationFreq'].replace('quite often','1', inplace=True)

# remove capital letter in col 'Diabetic'
diabetes_df['Diabetic'] = diabetes_df['Diabetic'].str.strip()
# Replace null value with 0
diabetes_df['Diabetic'] = diabetes_df['Diabetic'].fillna(0)
# Replacing in col 'Diabetic' no with 0
diabetes_df['Diabetic'].replace('no', '0', inplace=True)
# Replacing in col 'Diabetic' yes with 1
diabetes_df['Diabetic'].replace('yes', '1', inplace=True)

# Column 'Pregancies': replacing the null values with 0
diabetes_df['Pregancies'] = diabetes_df['Pregancies'].fillna(0)

# Remove rows with nullvalues in BMI col because i think that BMI is an important variable, so I don't want to replace nan with the mean of this col like this:
# diabetes_df['BMI'] = diabetes_df['BMI'].fillna(diabetes_df['BMI'].mean())
# So i take only the rows where BMI is not null
diabetes_df = diabetes_df[diabetes_df['BMI'].notna()]

# print(diabetes_df.info())

# split df in 2: diabets and not diabets
diabetic_mask = diabetes_df['Diabetic'] == '1'
diabetic_df = diabetes_df[diabetic_mask]

not_diabetic_mask = diabetes_df['Diabetic'] == '0'
not_diabetic_df = diabetes_df[not_diabetic_mask]

diabetic_df.to_csv('diab_clean.csv', encoding='utf-8', index=False)
diabetic_clean_df = pd.read_csv('diab_clean.csv') #diab_clean_try
diabetic_clean_df['Diabetic'] = diabetic_clean_df['Diabetic'].fillna(1)
# print(diabetic_clean_df.info())
# print(diabetic_clean_df.corr())
not_diabetic_df.to_csv('not_diab_clean.csv', encoding='utf-8', index=False)
not_diabetic_clean_df = pd.read_csv('not_diab_clean.csv')
not_diabetic_clean_df['Diabetic'] = not_diabetic_clean_df['Diabetic'].fillna(0)
# print(not_diabetic_clean_df.info())
# print(not_diabetic_clean_df.corr())

# diabetic_clean_df.Age = pd.Categorical(diabetic_clean_df.Age, categories=['0-40','40-49','50-59','60-99'], ordered=True)
# not_diabetic_clean_df.Age = pd.Categorical(not_diabetic_clean_df.Age, categories=['0-40','40-49','50-59','60-99'], ordered=True)

frames = [diabetic_clean_df, not_diabetic_clean_df]
diabetes_clean_df = pd.concat(frames)
diabetes_clean_df.to_csv('diabetes_clean_df.csv', encoding='utf-8', index=False)
# print(diabetes_clean_df.info())
# print(diabetes_clean_df.corr())

diab_df = pd.read_csv('diab_clean.csv')
not_diab_df = pd.read_csv('not_diab_clean.csv')

# ---------------------------- 1.3 Show some interesting plots ----------------------------
st.header('Diabetes Dataset 2019')
st.subheader('Write a caption.')
st.write('The aim of this ...')

st.sidebar.subheader('Settings')
if st.sidebar.checkbox('Display DataFrame'):
    st.write('The DataFrame')
    st.write(diabetes_clean_df)

st.header('1.3 - Some interesting plots')
with st.expander("Show some interesting plots"):
    st.subheader('Correlation Matrix')
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = diabetes_clean_df.corr()
    sns.heatmap(corr, 
        cmap=sns.color_palette("light:#124683", n_colors=20), # 220, 10 # diverging_palette(150, 10, as_cmap=True)
        vmin=-1.0, vmax=1.0,
        square=True, ax=ax)
    st.write(fig)
    st.caption('Write something about the Correlation Matrix...')

    st.subheader('Diabetes and Age')
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        label = ['Non-Diabetic', 'Diabetic']
        diabetic_count = list(diabetes_clean_df['Diabetic'].value_counts())
        colors_df = ['#becee6', '#4287f5'] # not, diab
        explode = (0.1, 0)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(diabetic_count, explode=explode, autopct='%.1f%%',
                shadow=False, startangle=90, colors=colors_df)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.legend(label, loc='best')
        st.pyplot(fig)
        st.caption('Diabetic distribution')
    with col1_2:
        # sort the dataframe by the age column
        diab_df = diab_df.sort_values(by='Age')
        not_diab_df = not_diab_df.sort_values(by='Age')
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist([diab_df.Age, not_diab_df.Age], 
            label=['Diabetic', 'Non-Diabetic'], 
            color=['#4287f5', '#becee6'], 
            bins=len(diab_df['Age'].unique())
        )
        ax.set_xlabel("Age")
        ax.set_ylabel("Number of people")
        # set the x-axis labels to the unique values in the age column
        ax.set_xticks(diab_df['Age'].unique())
        ax.set_xticklabels(diab_df['Age'].unique())
        plt.legend(loc='upper right')
        st.write(fig)
        st.caption('Diabetes and Age')

    st.subheader('Diabetic and BMI')
    fig, ax = plt.subplots()
    ax.hist([diabetic_clean_df.BMI, not_diabetic_clean_df.BMI], label=['Diabetic', 'Non-Diabetic'], color=['#4287f5', '#becee6'], bins=10)
    ax.set_xlabel("BMI level")
    ax.set_ylabel("Number of people")
    plt.legend(loc='upper right')
    st.write(fig)
    st.caption('BMI of People with Diabet and without Diabet. If BMI is less than 18.5: underweight range. If BMI is 18.5 to <25: healthy weight range. If BMI is 25.0 to <30: overweight range. If BMI is 30.0 or higher: obesity range.')

    col2_1, col2_2 = st.columns(2)
    with col2_1:
        diabetic_label = ['Diabetic with Regular Medicine', 'Diabetic without Regular Medicine']
        diabetic_count = list(diabetic_clean_df['RegularMedicine'].value_counts())
        colors_df = ['#4287f5', '#becee6']
        explode = (0.1, 0)
        fig, ax = plt.subplots()
        ax.pie(diabetic_count, explode=explode, autopct='%.1f%%',
                shadow=False, startangle=90, colors=colors_df)
        ax.axis('equal')
        plt.legend(diabetic_label, loc='best')
        st.pyplot(fig)
        st.caption('Diabetic Regular Medicine distribution')
    with col2_2:
        diabetic_label = ['Diabetic with highBP', 'Diabetic without highBP']
        diabetic_count = list(diabetic_clean_df['highBP'].value_counts())
        colors_df = ['#4287f5', '#becee6']
        explode = (0.1, 0)
        fig, ax = plt.subplots()
        ax.pie(diabetic_count, explode=explode, autopct='%.1f%%',
                shadow=False, startangle=90, colors=colors_df)
        ax.axis('equal')
        plt.legend(diabetic_label, loc='best')
        st.pyplot(fig)
        st.caption('Diabetic highBP distribution')

    col3_1, col3_2 = st.columns(2)
    with col3_1:
        diabetic_label = ['Diabetic Male', 'Diabetic Female']
        diabetic_count = list(diabetic_clean_df['Gender'].value_counts())
        colors_df = ['#4287f5', '#becee6']
        explode = (0.1, 0)
        fig, ax = plt.subplots()
        ax.pie(diabetic_count, explode=explode, autopct='%.1f%%',
                shadow=False, startangle=90, colors=colors_df)
        ax.axis('equal')
        plt.legend(diabetic_label, loc='best')
        st.pyplot(fig)
        st.caption('Diabetic gender distribution')
    with col3_2:
        fig, ax = plt.subplots()
        ax.hist([diabetic_clean_df.PhysicallyActive, not_diabetic_clean_df.PhysicallyActive], 
                label=['Diabetic', 'Non-Diabetic'], 
                color=['#4287f5', '#becee6'], 
                bins=10,
                ) #.loc[['none', 'less than half an hr', 'more than half an hr', 'one hr or more']]
        ax.set_ylabel("PhysicallyActive level")
        plt.legend(loc='upper right')
        st.write(fig)
        st.caption('PhysicallyActive People with Diabet and without Diabet')

    col4_1, col4_2 = st.columns(2)
    with col4_1:
        fig, ax = plt.subplots()
        ax.hist([diabetic_clean_df.Stress, not_diabetic_clean_df.Stress], 
                label=['Diabetic', 'Non-Diabetic'], 
                color=['#4287f5', '#becee6'], 
                bins=10,
                ) # not at all, sometimes, very often, always
        ax.set_ylabel("Stress level")
        plt.legend(loc='upper right')
        st.write(fig)
        st.caption('Stress People with Diabet and without Diabet')
    with col4_2:
        fig, ax = plt.subplots()
        ax.hist([diabetic_clean_df.BPLevel, not_diabetic_clean_df.BPLevel], 
                label=['Diabetic', 'Non-Diabetic'], 
                color=['#4287f5', '#becee6'], 
                bins=10,
                ) #.loc[['none', 'less than half an hr', 'more than half an hr', 'one hr or more']]
        ax.set_ylabel("BP level")
        plt.legend(loc='upper right')
        st.write(fig)
        st.caption('BPLevel People with Diabet and without Diabet')

    col5_1, col5_2 = st.columns(2)
    with col5_1:
        diabetic_label = ['Diabetic Pdiabetes NO', 'Diabetic Pdiabetes YES']
        diabetic_count = list(diabetic_clean_df['Pdiabetes'].value_counts())
        # print(diabetic_clean_df['Pdiabetes'].value_counts())
        colors_df = ['#4287f5', '#becee6']
        explode = (0.1, 0)
        fig, ax = plt.subplots()
        ax.pie(diabetic_count, explode=explode, autopct='%.1f%%',
                shadow=False, startangle=90, colors=colors_df)
        ax.axis('equal')
        plt.legend(diabetic_label, loc='best')
        st.pyplot(fig)
        st.caption('Diabetic Pdiabetes distribution')
    with col5_2:
        fig, ax = plt.subplots()
        ax.hist([diabetic_clean_df.UriationFreq, not_diabetic_clean_df.UriationFreq], 
                label=['Diabetic', 'Non-Diabetic'], 
                color=['#4287f5', '#becee6'], 
                bins=10,
                ) #.loc[['none', 'less than half an hr', 'more than half an hr', 'one hr or more']]
        ax.set_ylabel("UriationFreq")
        plt.legend(loc='upper right')
        st.write(fig)
        st.caption('UriationFreq People with Diabet and without Diabet')

# ---------------------------- 2 Find a model that explains tha data ----------------------------
st.header('2 - Models')
with st.expander('Show model'):

    st.subheader('A model to predict who has the Diabetes')

    y = diabetes_clean_df.Diabetic

    select_model = st.selectbox('Select model:', ['RandomForest','LogisticRegression'])

    model = RandomForestClassifier()

    if select_model == 'LogisticRegression':
        model = LogisticRegression()

    choices = st.multiselect('Select features', ["Gender","Family_Diabetes","highBP","PhysicallyActive","BMI","Pdiabetes","UriationFreq", "Smoking","Alcohol","Sleep","SoundSleep","RegularMedicine","JunkFood","Stress","BPLevel","Pregancies"])

    test_size = st.slider('Test size: ', min_value=0.1, max_value=0.9, step =0.1)

    if len(choices) > 0 and st.button('RUN MODEL'):
        with st.spinner('Training...'):
            x = diabetes_clean_df[choices]
            x_train, x_test, y_train, y_test = train_test_split(x, 
                                                                y,
                                                                test_size=test_size,
                                                                random_state=2)

            model.fit(x_train[choices], y_train)

            x_test = x_test.to_numpy().reshape(-1, len(choices))
            y_pred = model.predict(x_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)

            st.write("Accuracy: ", acc)
            st.write("Precision: ", prec)
            st.write("Recall: ", rec)