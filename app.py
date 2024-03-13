
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st
#loading the diabetes dataset to pandas data frame.
diabetes_dataset=pd.read_csv('diabetes.csv')
diabetes_dataset.head()
diabetes_dataset.shape
diabetes_dataset.describe()
diabetes_dataset['Outcome'].value_counts()
diabetes_dataset.groupby('Outcome').mean()
x=diabetes_dataset.drop(columns='Outcome', axis=1)
y=diabetes_dataset['Outcome']
scaler=StandardScaler()
scaler.fit(x)
standerdized_data=scaler.transform(x)
x=standerdized_data
y=diabetes_dataset['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
classifier=svm.SVC(kernel='linear')
#Training the suppot vector classifier



classifier.fit(x_train,y_train)

x_train_prediction=classifier.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)

x_test_prediction=classifier.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)

filename='trained_model.sav'
pickle.dump(classifier,open(filename,'wb'))
loading_model=pickle.load(open('trained_model.sav','rb'))


def diabetes_prediction(input_data):
    
    #scaler=StandardScaler()
    #scaler.fit(input_data)
    
    # changing the input data to numpy array
    input_data_as_numpy_array=np.asarray(input_data)

    #Reshape array for predicting for instances.
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    print(input_data_reshaped)

    
    std_data=scaler.transform(input_data_reshaped)
    print(std_data)

    prediction=loading_model.predict(std_data)
    print(prediction)
    if prediction[0]==0:
        return "The person is not diabetic."
    else:
        return "The person is diabetic."

def main():
    
    
    #title for web page
    
    st.title("Diabetes prediction web app")
    
    #getting the input from the user

    input_data=[]
    

    input_data.append(st.text_input("Number of pregnencies:"))
    input_data.append(st.text_input("Glucose Level:"))
    input_data.append(st.text_input("Blood pressure Value:"))
    input_data.append(st.text_input("Skin Thickness value:"))
    input_data.append(st.text_input("Insulin level:"))
    input_data.append(st.text_input("BMI Value:"))
    input_data.append(st.text_input("Diabetes pedegree function:"))
    input_data.append(st.text_input("Age of the person:"))
    
    Diagnosis=''
    
    #Creatinng button
    if st.button("Diabetes test Result:"):
        Diagnosis=diabetes_prediction(input_data)
        
    st.success(Diagnosis)
    
    
if __name__=='__main__':
    #input_data=(5,166,72,19,175,25.8,0.587,51)
    main()
