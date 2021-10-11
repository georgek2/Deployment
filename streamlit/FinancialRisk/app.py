import streamlit as st
import pandas as pd

# import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt 

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.write('''
    # Financial Risk Analysis Web App

    Predict whether a loan request is under risk of default and make 
    informed decisions whether to approve or deny client loan requests.
''')

st.write('''
    ## Technologies Used

    + Streamlit: This web app is made using streamlit
    + Sklearn: Machine learning Algorithms
    + Pandas: Library for data analysis and manipulation
''')

st.write('''
    ## Dataset Used

    The data used in this application was acquired from kaggle which is 
    an amazing free online learning resource for data scientists.

    It contains information about different clients who applied for loans.
    Also, the data highlights whether a request is under high or low 
    risk of loan default leading to financial loss.
''')

# Reading the data
data = pd.read_csv('Train.csv')

st.write(data.head())

st.write(f'''
    ### Exploratory Data Analysis
    Dimensions of the data : {data.shape} 
    
    DataFrame contains {data.shape[0]} rows and {data.shape[1]} columns
    
    ''')

st.write('+ Descriptive statistics', data.describe())

st.write('+ Missing values per column', data.isnull().sum())

st.bar_chart(data['IsUnderRisk'].value_counts())

model = st.selectbox('Select Model', ('DecisionTreeClassifier', 'LogisticRegression', 'SupportVectorClassifier', 'KNeighborsClassifier'))

st.write('Selected Model: ', model)

# Data preparation

train = data.sample(frac = 0.8)
valid = data.drop(train.index)

X_train, y_train = train.drop('IsUnderRisk', axis=1), train.IsUnderRisk

X_valid, y_valid = valid.drop('IsUnderRisk', axis=1), valid.IsUnderRisk

# Model Selection

if model == 'DecisionTreeClassifier':
    st.write('### Training a DecisionTreeCassifier')
    model = DecisionTreeClassifier(random_state = 11)

elif model == 'LogisticRegression':
    st.write('### Training a LogisticRegression model')
    model = LogisticRegression()

elif model == 'SupportVectorClassifier':
    st.write('### Training a SupportVectorClassifier')
    model = SVC()

elif model == 'KNeighborsClassifier':
    st.write('### Training a KNeighborsClassifier')
    model = KNeighborsClassifier()

else:
    st.write('Please select a model above')


st.write(f'''
    Train set: {train.shape} ------- Validation set: {valid.shape}
''')

tr_model = model.fit(X_train, y_train)

st.write(tr_model)

# Model Evaluation
preds = tr_model.predict(X_valid)
st.write('Predictions', preds[:5])

# Accuracy
acc = accuracy_score(y_valid, preds)
st.write('Model Accuracy: ', acc)

# Confusion Matrix
cm = confusion_matrix(y_valid, preds)
st.write('Confusion Matrix', cm)




