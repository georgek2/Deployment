import pandas as pd 
import numpy as np
import streamlit as st

from sklearn import datasets

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
st.title("Supervised Machine Learning")

st.write("""
## Classification Models...
> Find the best performing model
"""
)

dataset_name = st.sidebar.selectbox("Select Dataset", ('Iris', 'Breast Cancer', 'Wine Dataset'))

st.write('Selected Dataset >> ', dataset_name)

classifier = st.sidebar.selectbox("Select Classification Model", ('Decision Tree', 'Random Forest', 'KNN', 'SVM'))

st.write('Classification Model >> ', classifier)


def get_data(dataset_name):
    if dataset_name == 'Iris'.lower:
        data = datasets.load_iris()

    elif dataset_name == 'Breast Cancer'.lower:
        data = datasets.load_breast_cancer()

    else: 
        dataset_name == 'Wine Dataset'.lower
        data = datasets.load_wine()

    X = data.data
    y = data.target

    return X, y

X, y = get_data(dataset_name)






