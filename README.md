# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import required libraries(For eg:Pandas,Numpy,etc).

2.Read the dataset using Pandas Library.

3.Preview the data (first 5 rows) and check for missing values.

4.Convert categorical 'salary' column into numerical form using label encoding.

5.Select relevant features for training (x) and target variable (y).

6.Split the dataset into training and testing sets (80% train, 20% test).

7.Train a Decision Tree Classifier using the training data.

8.Make predictions on the test data and a new sample input.

9.Evaluate the model accuracy.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: ROHAN J
RegisterNumber: 212223040171
*/
import pandas as pd
df=pd.read_csv("Employee.csv")
df

df.head()

df.info()

df.isnull().sum()

df["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])
df.head()

x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=df["left"]


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
```

## Output:
### Data head:
![image](https://github.com/user-attachments/assets/421dbe07-8c08-41c0-85bf-b644608a1a1f)

### Data info:
![image](https://github.com/user-attachments/assets/db783fa0-2d5f-4597-8fc3-504eedd00e26)


### Null values:
![image](https://github.com/user-attachments/assets/cb052405-a2fc-4163-bc56-076f3b361b9b)



### Dataset transformed head:
![image](https://github.com/user-attachments/assets/54b7bea5-09ef-4584-92d9-4ded3ee077a1)


### x.head():
![image](https://github.com/user-attachments/assets/04e0c19c-e8fe-42f7-b25c-cb69b4c177e2)


### Accuracy:
![image](https://github.com/user-attachments/assets/bb0b07a6-4113-42da-89bb-fd3a6204faf5)


### Data Prediction:
![image](https://github.com/user-attachments/assets/1b4011bb-65b7-4a48-a5b1-c107d70db1ac)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
