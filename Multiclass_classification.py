import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score
import ssl
import warnings
warnings.filterwarnings("ignore")

ssl._create_default_https_context = ssl._create_unverified_context

#load the dataset

file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)
print(data.sample(2))
# print("this is data head",data.head())


#distribution of target variables 
# sns.countplot(y="NObeyesdad", data=data)
# plt.title("Distribution of obesity levell")
# plt.show()
# print(data.isnull().sum())
# print(data.info())
# print("describing",data.describe())

#give me all column in data that are of type of decimal and then turn into list
continuous_columns = data.select_dtypes(include="float64").columns.tolist()
print("this is continous",continuous_columns)

scaler = StandardScaler()
#fit() Learns the mean and standard deviation of each column
#transform() Applies the standardization to every value
#Now scaled_features is a NumPy array of standardized numbers.
#data[['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']] is equivalent data[continuous_columns]
#which 2d array what fit_transform expects
scaled_features =scaler.fit_transform(data[continuous_columns])
print("scaled features:",scaled_features)


#You wrap the scaled array into a new DataFrame with proper column names.
#pd.DataFrame(data_array, columns=column_names)
scaled_df = pd.DataFrame(scaled_features, columns=continuous_columns)
print("dataFrame",scaled_df)

#Standardization of data is important to better define the decision boundaries between classes by making sure that the feature variations are in similar scales.
#The data is now ready to be used for training and testing.
#combining with the original dataset
scaled_data = pd.concat([data.drop(columns=continuous_columns),scaled_df],axis=1)


#Now convert categorical into numerical format using one hot encoding 

#identfying categorical columns 
categorical_columns = scaled_data.select_dtypes(include=["object"]).columns.tolist()
print(categorical_columns)
categorical_columns.remove("NObeyesdad") #Exclude target columns 
print(categorical_columns)

# #Applying one-hot encoding 
encoder= OneHotEncoder(sparse_output=False,drop="first")
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])


#converting to a DataFrame
encoded_df = pd.DataFrame(encoded_features,columns=encoder.get_feature_names_out(categorical_columns))

#combining with the original dataset
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)
print("final data",prepped_data)


#Encoding the target variable
prepped_data["NObeyesdad"] = prepped_data["NObeyesdad"].astype('category').cat.codes
print("encoding",prepped_data.head())

#seperate input and target data
#Preparing finall dataset
X = prepped_data.drop("NObeyesdad",axis=1)
y = prepped_data["NObeyesdad"]


#MODEL TRAINING AND EVALUATION

#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42,stratify=y)


#training logistic regression model using One vs all
#logistic regression model for multi-class classification using the One-vs-Rest (OvR) strategy.
model_ova = LogisticRegression(multi_class='ovr',max_iter=1000)
model_ova.fit(X_train,y_train)

#Predictions
y_pred_ova = model_ova.predict(X_test)

#Evaluations for One vs Rest
print("One-vs-All (OvA) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")

#training logistic regression model using One vs one 
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ova.fit(X_train,y_train)

#Evaluate the accuracy of the trained model as a measurre of its performance on unseen testing data
y_pred_ovo = model_ova.predict(X_test)

#Evaluation metrics for ovo 
print("One-vs-One (OvO) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")