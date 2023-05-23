from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler


from sklearn import metrics

data = pd.read_csv('heart.csv')

st.write('''
    # Heart Disease Analysis App..
    Predicting the risk of heart disease using some vital health signals

    Dataset used is from kaggle
''')

st.write(data.head(8), '''Data Labels

    Age : Age of the patient
    Sex : Sex of the patient
    exang: exercise induced angina (1 = yes; 0 = no)
    ca: number of major vessels (0-3)
    cp : Chest Pain type chest pain type
        Value 0: typical angina
        Value 1: atypical angina
        Value 2: non-anginal pain
        Value 3: asymptomatic
    trtbps : resting blood pressure (in mm Hg)
    chol : cholestoral in mg/dl fetched via BMI sensor
    fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
    rest_ecg : resting electrocardiographic results
        Value 0: normal
        Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
        Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
    thalach : maximum heart rate achieved
    target : risk or chance of heart attack
        0 = less chance of heart attack
        1 = more chance of heart attack

''')


st.write('''
    ## Supervised Machine Learning Classifiers used
    + DecisionTreeClassifier
    + RandomForestClassifier
    + KNNClassifier
    + SVClassifier
''')

st.write('## Exploratory Data Science')

st.write(f'+ Correlation Matrix')
fig, ax = plt.subplots()
sns.heatmap(data.corr(), ax=ax)
st.write(fig)

st.write(f'''+ Output Distribution
''')
fig, ax = plt.subplots()
sns.histplot(data.output, ax=ax)
st.write(fig)

st.write('### DecisionTreeClassifier')

X = data.drop('output', axis=1)
y = data.output

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

model_dt = DecisionTreeClassifier()

st.write(model_dt.fit(X_train, y_train))

preds_dt = model_dt.predict(X_valid)

acc_dt = metrics.accuracy_score(y_valid, preds_dt)
st.write('DecisionTree Accuracy: ', acc_dt)

st.write('### RandomForestClassifier')

model_rf = RandomForestClassifier()
st.write(model_rf.fit(X_train, y_train))

preds_rf = model_rf.predict(X_valid)

acc_rf = metrics.accuracy_score(y_valid, preds_rf)
st.write('RandomForest Accuracy: ', acc_rf)


st.write('## Using Distance-Based Models')
st.write('### KNN')

model_knn = KNeighborsClassifier()

st.write(model_knn.fit(X_train, y_train))

preds_knn = model_knn.predict(X_valid)
acc_knn = metrics.accuracy_score(y_valid, preds_knn)

st.write('#### Accuracy: ',acc_knn)

st.write('### SVC')

model_svc = SVC()

st.write(model_svc.fit(X_train, y_train))

preds_svc = model_svc.predict(X_valid)
acc_svc = metrics.accuracy_score(y_valid, preds_svc)

st.write('#### Accuracy: ',acc_svc)

st.write('## Using Scaled features')

scaler = StandardScaler()

# Scaling input features
sx_train = scaler.fit_transform(X_train)
sx_valid = scaler.transform(X_valid)

#KNN
knn_2 = KNeighborsClassifier()
knn_2.fit(sx_train, y_train)

preds_knn2 = knn_2.predict(sx_valid)
k_acc = metrics.accuracy_score(y_valid, preds_knn2)

st.write('### KNN - Scaled')
st.write('Accuracy: ', k_acc)

# SVC
svc_2 = SVC()
svc_2.fit(sx_train, y_train)

preds_svc2 = svc_2.predict(sx_valid)
svc_acc = metrics.accuracy_score(y_valid, preds_svc2)

st.write('### SVC - Scaled')
st.write('Accuracy: ', svc_acc)


st.write('The accuracy for the distance-based machine learning models improves significantly when input features are scaled')
