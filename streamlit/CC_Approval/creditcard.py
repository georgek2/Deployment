import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn import metrics

st.write('''
    # Credit Risk Analysis App
    ## Predict the probability of a customer paying back a loan based on their profile data
    
    Using client information to measure the level of risk of loan default
''')

st.write('''
    ### Using data from Kaggle
    Pre-requsite knowledge of pandas is required
''')

data = pd.read_csv('train.csv')

st.write('The pandas dataframe', data.head(10))

y = data.IsUnderRisk
X = data.drop('IsUnderRisk', axis = 1)

features = X.columns

st.write('+ Features to use', features)

st.write('Some Descriptive Statistics', X.describe())

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = .2)

st.write(f'''Split the training and validation sets 

    X_train {X_train.shape}
    X_valid {X_valid.shape}

''')


st.write('''
    ## Predictive modelling using 
    + ### Logistic Regression
''')

# Logistic Regression
model_lr = LogisticRegression()
f_model = model_lr.fit(X_train, y_train)
preds_lr = model_lr.predict(X_valid)

acc_lr = metrics.accuracy_score(y_valid, preds_lr)
conf_matrix = metrics.confusion_matrix(y_valid, preds_lr)
st.write(f_model, f'''
    \nAccuracy: {acc_lr}
    \nConfusion Matrix
''')
st.write(conf_matrix)

# DecisionTree
st.write('+ ### DecisionTree')
model_dt = DecisionTreeClassifier(random_state=11)
f_dt = model_dt.fit(X_train, y_train)
preds_dt = model_dt.predict(X_valid)

acc_dt = metrics.accuracy_score(y_valid, preds_dt)
dt_cmatrix = metrics.confusion_matrix(y_valid, preds_dt)

st.write(f_dt, f'''
    \n Accuracy: {acc_dt}
    \n Confusion Matrix
''')

st.write(dt_cmatrix)


# KNeighbors
st.write('+ ### KNeighbors')
model_knn = KNeighborsClassifier()
f_knn = model_knn.fit(X_train, y_train)
preds_knn = f_knn.predict(X_valid)

acc_knn = metrics.accuracy_score(y_valid, preds_knn)
knn_cm = metrics.confusion_matrix(y_valid, preds_knn)

st.write(f_knn, f'''
    \n Accuracy: {acc_knn}
    \n Confusion Matrix
''')

st.write(knn_cm)

# LightGBM

# model_lgb = lgb()
