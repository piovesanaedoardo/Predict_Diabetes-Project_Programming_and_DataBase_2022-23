import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

diabetes_df = pd.read_csv('diabetes_dataset__2019.csv')

# cd Documenti/Project of Programming and Database
# streamlit run .\diabetes_2019.py

# ---------------------------- 1. Explore the dataset ----------------------------
# print(diabetes_df.info())
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

# print(diabetes_df.describe())

# print(diabetes_df.corr())
#                  BMI     Sleep  SoundSleep  Pregancies
# BMI         1.000000 -0.067896   -0.298074   -0.045124
# Sleep      -0.067896  1.000000    0.534686    0.041959
# SoundSleep -0.298074  0.534686    1.000000    0.150531
# Pregancies -0.045124  0.041959    0.150531    1.000000
# 
# There are too much columns with no numerical data, so I have to fix this problem
# (i.e. the column 'Family_Diabetes' has value 'yes'/'no'. I change this values by replacing 1/0)


# ---------------------------- 2. Clean up the dataset ----------------------------
# see the value of each column
# print('see the value of each column')
# for col in diabetes_df.columns:
#     print(col)
#     print(diabetes_df[col].value_counts())
#     print("\n")

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

# Column 'Pregancies': replacing the null values with 0
diabetes_df['Pdiabetes'] = diabetes_df['Pdiabetes'].fillna(0)
# Replacing in col 'Pdiabetes' no with 0
diabetes_df['Pdiabetes'].replace('no','0', inplace=True)
# Replacing in col 'Pdiabetes' yes with 1
diabetes_df['Pdiabetes'].replace('yes','1', inplace=True)

# Replacing in col 'UriationFreq' no with 0
diabetes_df['UriationFreq'].replace('no','0', inplace=True)
# Replacing in col 'Pdiabetes' yes with 1
diabetes_df['UriationFreq'].replace('yes','1', inplace=True)

# remove capital letter in col 'Diabetic'
diabetes_df['Diabetic'] = diabetes_df['Diabetic'].str.strip()
# Replacing in col 'Diabetic' no with 0
diabetes_df['Diabetic'].replace('no','0', inplace=True)
# Replacing in col 'Diabetic' ' no' with 0
diabetes_df['Diabetic'].replace(' no','0', inplace=True)
# Replacing in col 'Diabetic' yes with 1
diabetes_df['Diabetic'].replace('yes','1', inplace=True)
# Replace null value with 0
diabetes_df['Diabetic'] = diabetes_df['Diabetic'].fillna(0)

# Column 'Pregancies': replacing the null values with 0
diabetes_df['Pregancies'] = diabetes_df['Pregancies'].fillna(0)

# Remove rows with nullvalues in BMI col because i think that BMI is an important variable, so I don't want to replace nan with the mean of this col like this:
# diabetes_df['BMI'] = diabetes_df['BMI'].fillna(diabetes_df['BMI'].mean())
# So i take only the rows where BMI is not null
diabetes_df = diabetes_df[diabetes_df['BMI'].notna()]

# print(diabetes_df.info())

