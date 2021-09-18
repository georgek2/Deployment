
import streamlit as st
import numpy as np
import pandas as pd 

from sklearn import datasets

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn import metrics

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

elif classifier == 'Random Forest':
    clf = RandomForestClassifier(random_state=12)

elif classifier == 'KNN':
    clf = KNeighborsClassifier()

else: 
    clf = SVC()

st.write('Classification Model >> ', classifier)

if dataset == 'Iris':
    data = datasets.load_iris()

elif dataset == 'Breast Cancer':
    data = datasets.load_breast_cancer()

else: 
    data = datasets.load_wine()

st.write(f"+ Overview of {dataset} dataset", data)

st.write("Training Featuers: ", )
# Training Features
X = pd.DataFrame(data.data, columns = data.feature_names)

y = np.array(data.target) 

features = X.columns

st.write(X[:7])
st.write("Shape of the data >> ", X.shape)
st.write("Number of classes >> ", len(np.unique(y)))
st.write('Target Values: ', data.target_names)

st.write('## Model Building & Validation')

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2)

st.write(f"""
    + Training Data : {X_train.shape}
    + Validation Data : {X_valid.shape}
""")

model = clf.fit(X_train, y_train)

st.write("Fitted ", model)

model_preds = model.predict(X_valid)
confusion_matrix = metrics.confusion_matrix(y_valid, model_preds)
model_acc = metrics.accuracy_score(y_valid, model_preds)

st.write("Confusion Matrix", confusion_matrix)
st.write('Accuracy: ', model_acc)

