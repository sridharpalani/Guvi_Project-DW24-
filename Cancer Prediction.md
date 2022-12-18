# Cancer Prediction

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv('cancer.csv')
df=df.drop(['Unnamed: 32'],axis=1)
le=LabelEncoder()
df['diagnosis']=le.fit_transform(df['diagnosis'])

#0--->Benign
#1--->Malignant

#df['diagnosis'].value_counts()
#df.groupby('diagnosis').mean()

X=df.drop(columns='diagnosis',axis=1)
Y=df['diagnosis']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

model=LogisticRegression()
model.fit(X_train,Y_train)

#To findtraining data Accuracy

X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(Y_train,X_train_prediction)
print('Accuracy on training data = ',training_data_accuracy)


X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(Y_test,X_test_prediction)
print('Accuracy on testing data = ',testing_data_accuracy)

User_input=(
843786,12.45,15.7,82.57,477.1,0.1278,0.17,0.1578,0.08089,0.2087,0.07613,0.3345,0.8902,2.217,27.19,0.00751,0.03345,0.03672,0.01137,0.02165,0.005082,15.47,23.75,103.4,741.6,0.1791,0.5249,0.5355,0.1741,0.3985,0.1244)
input_data=np.asarray(User_input)
input_data_1=input_data.reshape(1,-1)
prediction=model.predict(input_data_1)
print(prediction)

if (prediction[0]==0):
    print('The breast cancer is Benign')
else:
    print('The breast cancer is Malignant')

