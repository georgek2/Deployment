
import sklearn
import streamlit as st
import numpy as np
import pandas as pd 

from sklearn import datasets

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

st.title("Supervised Machine Learning")

st.write("""
## Classification Models...
+ Find the best performing model
"""
)

dataset = st.sidebar.selectbox("Select Dataset", ('Iris', 'Breast Cancer', 'Wine Dataset'))

st.write('Selected Dataset >> ', dataset)

classifier = st.sidebar.selectbox("Select Classification Model", ('Decision Tree', 'Random Forest', 'KNN', 'SVM'))

if classifier == 'Decision Tree':
    clf = DecisionTreeClassifier(random_state=11)

else:
    clf = RandomForestClassifier(random_state=12)

st.write('Classification Model >> ', classifier)

if dataset == 'Iris':
    data = datasets.load_iris()

elif dataset == 'Breast Cancer':
    data = datasets.load_breast_cancer()

else: 
    data = datasets.load_wine()

X = data.data
y = data.target 


st.write(X[:7])
st.write("Shape of the data >> ", X.shape)
st.write("Number of classes >> ", len(np.unique(y)))

st.write('## Model Building...')

def train_clf(df, clf):
    train = df.sample(frac = 0.8)
    valid = df.drop(train.index)

    return train, valid 

st.write(f"Training Data: \n")

# train, valid = train_clf(data, clf)
