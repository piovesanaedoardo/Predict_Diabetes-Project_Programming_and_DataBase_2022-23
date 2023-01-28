import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import heapq

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
print('Value of each column')
for col in diabetes_df.columns:
    print(col)
    print(diabetes_df[col].value_counts())
    print("\n")

# Replacing in col 'Age' less than 40 with 0-40
diabetes_df['Age'].replace('less than 40','0-40', inplace=True)
# Replacing in col '60 or older' Female with 60-99
diabetes_df['Age'].replace('60 or older','60-99', inplace=True)

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

# remove capital letter and space
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

print('--------- Cleaned dataset ---------')
print(diabetes_df.info())

# split df in 2: diabets and not diabets
diabetic_mask = diabetes_df['Diabetic'] == '1'
diabetic_df = diabetes_df[diabetic_mask]

not_diabetic_mask = diabetes_df['Diabetic'] == '0'
not_diabetic_df = diabetes_df[not_diabetic_mask]

diabetic_df.to_csv('diab_clean.csv', encoding='utf-8', index=False)
diabetic_clean_df = pd.read_csv('diab_clean.csv')
diabetic_clean_df['Diabetic'] = diabetic_clean_df['Diabetic'].fillna(1)

not_diabetic_df.to_csv('not_diab_clean.csv', encoding='utf-8', index=False)
not_diabetic_clean_df = pd.read_csv('not_diab_clean.csv')
not_diabetic_clean_df['Diabetic'] = not_diabetic_clean_df['Diabetic'].fillna(0)

frames = [diabetic_clean_df, not_diabetic_clean_df]
diabetes_clean_df = pd.concat(frames)
diabetes_clean_df.to_csv('diabetes_clean_df.csv', encoding='utf-8', index=False)

diab_df = pd.read_csv('diab_clean.csv')
not_diab_df = pd.read_csv('not_diab_clean.csv')

print('--------- Cleaned dataset merged ---------')
print(diabetes_clean_df.info())
print(diabetes_clean_df.corr())

# ---------------------------- 1.3 Show some interesting plots ----------------------------
st.title("Diabetes Dataset's 2019")
st.subheader('Does the patient have Diabetes?')
st.write("""The aim of this project is analyse the Diabetes dataset's in order to predict if a person has the Diabetes Type 2. 
            In this dataset there are 17 independent variables and one binary dependent, Diabetes.""")

st.sidebar.subheader('Settings')
if st.sidebar.checkbox('Display DataFrame'):
    st.write('The DataFrame')
    st.write(diabetes_clean_df)

st.header('Interesting plots')

st.subheader('Diabetes, Age & Gender')
with st.expander("Show Diabetes, Age & Gender plot's"):
    st.subheader('Diabetic distribution')
    fig, ax = plt.subplots(figsize=(8, 6))
    label = ['Non-Diabetic', 'Diabetic']
    diabetic_count = list(diabetes_clean_df['Diabetic'].value_counts())
    colors_df = ['#becee6', '#4287f5'] # not, diab
    explode = (0.1, 0)
    ax.pie(diabetic_count, explode=explode, autopct='%.1f%%',
            shadow=False, startangle=90, colors=colors_df)
    ax.axis('equal')  # equal aspect ratio ensures that pie is drawn as a circle
    plt.legend(label, loc='best')
    st.pyplot(fig)
    
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        st.subheader('Diabetes & Gender')
        diabetic_label = ['Diabetic Male', 'Diabetic Female']
        diabetic_count = list(diab_df['Gender'].value_counts())
        colors_df = ['#4287f5', '#D991C3']
        explode = (0.1, 0)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(diabetic_count, explode=explode, autopct='%.1f%%',
                shadow=False, startangle=90, colors=colors_df)
        ax.axis('equal')
        plt.legend(diabetic_label, loc='best')
        st.pyplot(fig)
    with col1_2:
        st.subheader('Age')
        # sort the dataframe by the age column
        diab_df = diab_df.sort_values(by='Age')
        not_diab_df = not_diab_df.sort_values(by='Age')
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist([diab_df.Age, not_diab_df.Age], 
            label=['Diabetic', 'Non-Diabetic'], 
            color=['#4287f5', '#becee6'], 
            bins=len(diab_df['Age'].unique())
        )
        ax.set_xlabel("Age range")
        ax.set_ylabel("Number of people")
        # set the x-axis labels to the unique values in the age column
        ax.set_xticks(diab_df['Age'].unique())
        ax.set_xticklabels(diab_df['Age'].unique())
        plt.legend(loc='upper right')
        st.write(fig)

st.subheader("Healt factors's")
with st.expander("Show Healt factors plot's"):
    col2_1, col2_2 = st.columns(2)
    col2_3, col2_4 = st.columns(2)
    with col2_1:
        st.subheader('Body Mass Index')
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist([diab_df.BMI, not_diab_df.BMI], 
                label=['Diabetic', 'Non-Diabetic'], 
                color=['#4287f5', '#becee6'], 
                bins=15)
        ax.set_xlabel("BMI level")
        ax.set_ylabel("Number of people")
        plt.legend(loc='upper right')
        st.write(fig)
        st.caption("""If BMI is less than 18.5: underweight range. 
                    If BMI is 18.5 to <25: healthy weight range. If BMI is 25.0 to <30: overweight range. 
                    If BMI is 30.0 or higher: obesity range.""")

    with col2_2:
        fig, ax = plt.subplots(figsize=(8, 6))
        st.subheader('Blood Pressure')
        ax.hist([diab_df.BPLevel, not_diab_df.BPLevel], 
                label=['Diabetic', 'Non-Diabetic'], 
                color=['#4287f5', '#becee6'], 
                bins=10,
                )
        ax.set_xlabel("BP level")
        ax.set_ylabel("Number of people")
        plt.xticks(range(0, 3))
        plt.legend(loc='upper right')
        st.write(fig)
        st.caption("""0 corresponds to low BP, 
                      1 to normal BP and 
                      2 to high BP.""")

    with col2_3:
        fig, ax = plt.subplots(figsize=(8, 6))
        st.subheader('Physical Activity')
        ax.hist([diab_df.PhysicallyActive, not_diab_df.PhysicallyActive], 
                label=['Diabetic', 'Non-Diabetic'], 
                color=['#4287f5', '#becee6'], 
                bins=10,
                )
        ax.set_xlabel("PhysicallyActive level")
        ax.set_ylabel("Number of people")
        plt.xticks(range(0, 4))
        plt.legend(loc='upper right')
        st.write(fig)
        st.caption("""0: none, 
                      1: less than half an hr, 
                      2: more than half an hr, 
                      3: one hr or more.""")

    with col2_4:
        fig, ax = plt.subplots(figsize=(8, 6))
        st.subheader('Stress')
        ax.hist([diab_df.Stress, not_diab_df.Stress], 
                label=['Diabetic', 'Non-Diabetic'], 
                color=['#4287f5', '#becee6'], 
                bins=10,
                )
        ax.set_xlabel("Stress level")
        ax.set_ylabel("Number of people")
        plt.xticks(range(0, 4))
        plt.legend(loc='upper right')
        st.write(fig)
        st.caption("""0: not at all, 
                      1: sometimes, 
                      2: very often, 
                      3: always.""")

st.subheader("Focus on Diabetic people")
with st.expander("Show the focus on Diabetic people"):
    col3_1, col3_2 = st.columns(2)
    col3_3, col3_4 = st.columns(2)
    col3_5, col3_6 = st.columns(2)

    with col3_1:
        st.subheader('Family Diabetes')
        diabetic_label = ['Diabetic with family Diabetes', 'Diabetic without family Diabetes']
        diabetic_count = list(diab_df['Family_Diabetes'].value_counts())
        colors_df = ['#4287f5', '#becee6']
        explode = (0.1, 0)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(diabetic_count, explode=explode, autopct='%.1f%%',
                shadow=False, startangle=90, colors=colors_df)
        ax.axis('equal')
        plt.legend(diabetic_label, loc='best')
        st.pyplot(fig)
        st.caption('Diabetic patients and family history of Diabetes.')
        
    with col3_2:
        st.subheader('Regular Medicine')
        diabetic_label = ['Diabetic with Regular Medicine', 'Diabetic without Regular Medicine']
        diabetic_count = list(diab_df['RegularMedicine'].value_counts())
        colors_df = ['#4287f5', '#becee6']
        explode = (0.1, 0)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(diabetic_count, explode=explode, autopct='%.1f%%',
                shadow=False, startangle=90, colors=colors_df)
        ax.axis('equal')
        plt.legend(diabetic_label, loc='best')
        st.pyplot(fig)
        st.caption('Diabetic patients and the use of regular Medicine.')

    with col3_3:
        st.subheader('Smoking')
        diabetic_label = ['Diabetic smoker', 'Diabetic not smoker']
        diabetic_count = list(diab_df['Smoking'].value_counts())
        colors_df = ['#4287f5', '#becee6']
        explode = (0.1, 0)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(diabetic_count, explode=explode, autopct='%.1f%%',
                shadow=False, startangle=90, colors=colors_df)
        ax.axis('equal')
        plt.legend(diabetic_label, loc='best')
        st.pyplot(fig)
        st.caption('Smoking and nonsmoking Diabetic patients.')
    with col3_4:
        st.subheader('Alcohol')
        diabetic_label = ['Diabetic Alcohol', 'Diabetic not Alcohol']
        diabetic_count = list(diab_df['Alcohol'].value_counts())
        colors_df = ['#4287f5', '#becee6']
        explode = (0.1, 0)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(diabetic_count, explode=explode, autopct='%.1f%%',
                shadow=False, startangle=90, colors=colors_df)
        ax.axis('equal')
        plt.legend(diabetic_label, loc='best')
        st.pyplot(fig)
        st.caption('Diabetic patients who are alchol users and non-alchol users.')

    with col3_5:
        st.subheader('JunkFood')
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(diab_df.JunkFood.unique(), diab_df.JunkFood.value_counts(), 
                color=['#8dade0', '#b6ccf0', '#4287f5', '#659af0']) # 4 1 3 2
        plt.xlabel('Number of diab peopl')
        plt.ylabel('JunkFood level')
        plt.yticks(range(0, 4))
        plt.legend(loc='upper right')
        st.write(fig)
        st.caption("""0: occasionally, 
                      1: often, 
                      2: very often, 
                      3: always""")

    with col3_6:
        st.subheader('Sleep')
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(diab_df.Sleep.unique(), diab_df.Sleep.value_counts(),
                color=['#788ae3', '#9eaae6', '#052beb', '#3d58e0', '#02115e', '#6179ed', '#122aa3'])
        plt.xlabel('Number of diab peopl')
        plt.ylabel('Sleep level')
        plt.yticks(range(min(diab_df.Sleep.unique()), max(diab_df.Sleep.unique())+1))
        plt.legend(loc='upper right')
        st.write(fig)
        st.caption("Sleep hours of Diabetic patients.")

st.subheader('Correlation Matrix')
with st.expander('Show Correlation Matrix plot'):
    st.title('Correlation Matrix')
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = diabetes_clean_df.corr()
    sns.heatmap(corr, 
        cmap=sns.color_palette("light:#124683", n_colors=20),
        vmin=-1.0, vmax=1.0,
        square=True, ax=ax,
        annot=True, fmt=".2f")
    st.write(fig)
    # show the 5 highest_values (excluding 1 and duplicates)
    highest_values = [float(item.get_text()) for item in ax.texts]
    filtered_highest_values = [val for val in highest_values if val != 1]
    unique_highest_values = set(filtered_highest_values)
    top_5_highest_values = heapq.nlargest(5, unique_highest_values)

    print(top_5_highest_values)
    st.caption('The most correlations between the variable "Diabetic" are the variables "RegularMedicine" and "BPLevel".')

# ---------------------------- 2 Find a model that explains tha data ----------------------------
st.header('Model that explains the data')
with st.expander('Show model'):

    st.subheader('A model to predict who has the Diabetes')

    y = diabetes_clean_df.Diabetic

    model = LogisticRegression()

    choices = st.multiselect('Select the features to construct the model:', ["Gender","Family_Diabetes","highBP","PhysicallyActive","BMI",
                                                 "Pdiabetes","UriationFreq", "Smoking","Alcohol","Sleep","SoundSleep",
                                                 "RegularMedicine","JunkFood","Stress","BPLevel","Pregancies"])

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