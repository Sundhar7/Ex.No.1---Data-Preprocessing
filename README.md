# Ex.No.1---Data-Preprocessing
##AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


##ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

##PROGRAM:
import pandas as pd

import numpy as np

df = pd.read_csv("Churn_Modelling.csv") df

df.isnull().sum()

#Check for Duplicate Values

df.duplicated()

df.describe()

#Detect the Outliers

Outliers are any abnormal values going beyond
df['Exited'].describe()

""" Normalize the data - There are range of values in different columns of x are different.

To get a correct ne plot the data of x between 0 and 1

LabelEncoder can be used to normalize labels. It can also be used to transform non-numerical labels to numerical labels. """ from sklearn.preprocessing import LabelEncoder le = LabelEncoder()

df1 = df.copy()

df1["Geography"] = le.fit_transform(df1["Geography"]) df1["Gender"] = le.fit_transform(df1["Gender"])

''' MinMaxScaler - Transform features by scaling each feature to a given range. When we normalize the dataset it brings the value of all the features between 0 and 1 so that all the columns are in the same range, and thus there is no dominant feature.'''

from sklearn.preprocessing import StandardScaler from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]] = pd.DataFrame(scaler.fit_transform(df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]]))

df1

df1.describe()

Since values like Row Number, Customer Id and surname doesn't affect the output y(Exited).
#So those are not considered in the x values X = df1[["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]].values print(X)

y = df1.iloc[:,-1].values print(y)

from sklearn.model_selection import train_test_split X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train) print("Size of X_train: ",len(X_train))

print(X_test) print("Size of X_test: ",len(X_test))

X_train.shape

##OUTPUT:


![192950859-0b6f64ca-6e76-4c10-9830-287232ba545a](https://user-images.githubusercontent.com/83111884/194223889-ad38251e-476e-4149-ab3d-7d9cc4ae1ce2.png)


![192951831-c32485fa-ba7b-4625-ab0c-4361657d6d0b](https://user-images.githubusercontent.com/83111884/194223900-4df45f3f-b32f-4b0f-bb2f-40df2d4048a8.png)


![192952680-9c2cf2d3-858b-4bdf-98cf-f356d84a571a](https://user-images.githubusercontent.com/83111884/194223921-145f9b18-34a7-4c26-8095-9416915ec217.png)

![192953487-609c4eef-5d89-46d5-bb2b-0b15a4553082](https://user-images.githubusercontent.com/83111884/194223935-7fe0bae9-e59c-4967-9007-71f6e0a634d1.png)



![192953810-78b49e03-da5d-44d3-9613-09ddb65b7ef7](https://user-images.githubusercontent.com/83111884/194223970-167d913b-539e-42e0-9feb-35cc251ef8c3.png)



![192954241-b0d4947c-b534-48ff-b279-375acc32ff23](https://user-images.githubusercontent.com/83111884/194223983-5dada296-bc18-4eac-a788-471d47b69566.png)


![192954334-58f7419a-c49d-4b1a-acf8-36ef015a6f27](https://user-images.githubusercontent.com/83111884/194223989-6a6db178-9c2c-4191-92d6-96a61e60aa8a.png)


![193093641-4afb1906-fd06-4e81-940e-6938cd451ca0](https://user-images.githubusercontent.com/83111884/194224004-1c50a274-7ca7-45c0-875a-bf1aa7d2d674.png)

![193093728-d4f44ff6-5d77-4499-8854-0341552e95c5](https://user-images.githubusercontent.com/83111884/194224010-43d61043-dd2f-4df2-ab0b-2402cdff4cb5.png)


![193093857-0a8f82c7-89ef-4b0d-96e9-226cc31e9877](https://user-images.githubusercontent.com/83111884/194224020-f0bbba0e-2e7d-4819-a4ab-2f34ffdcd98d.png)


![193093944-603601da-9bd7-4f7c-819d-76dcb72f9bcc](https://user-images.githubusercontent.com/83111884/194224035-b7e69d19-e1b1-4f84-a1b3-f28f4fbb2eee.png)



##RESULT
Data preprocessing is performed in a data set downloaded from Kaggle
