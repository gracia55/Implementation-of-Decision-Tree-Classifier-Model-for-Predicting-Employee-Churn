# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
# AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

# Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
# Algorithm
1.Import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics. 10.Find the accuracy of our model and predict the require values.

# Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:Gracia Ravi
RegisterNumber: 212222040047

import pandas as pd
data=pd.read_csv("/content/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

*/
```
# Output:
# data.head()
![image](https://github.com/gracia55/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/129026838/746db0ed-1128-4a6d-81f6-f14f380a12f6)


# data.info()
![image](https://github.com/gracia55/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/129026838/0d94a369-dac3-40ce-b1ce-900aa71ebe79)


# isnull() and sum ()
![image](https://github.com/gracia55/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/129026838/eeb1dd94-f3af-4d9d-b0b1-056605fe4529)


# data value counts()
![image](https://github.com/gracia55/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/129026838/54c907a8-6908-49fd-a579-2930faf68ae4)


# data.head() for salary
![image](https://github.com/gracia55/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/129026838/83b0caef-a15b-4caa-aaf7-06e9856eec7d)


# x.head()
![image](https://github.com/gracia55/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/129026838/3287cab7-3e29-4ea0-96f5-8de59f53bbc3)


# accuracy value
![image](https://github.com/gracia55/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/129026838/6f1e540e-41a5-4854-a98e-5fd6502f9166)


# data prediction
![image](https://github.com/gracia55/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/129026838/d1b50a36-55fd-4bd7-8319-d8b8bc7b2f15)


# Result:
Thus the program to implement the Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
