import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')
import ssl


ssl._create_default_https_context = ssl._create_unverified_context


url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)
print(churn_df.sample(9))



churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip','churn']]
print(churn_df.sample(9))

#What Is .astype()? It’s a Pandas method to change the data type of a column.
#df['age'].astype('float')  changes to decimall numbers
#df['is_active'].astype('bool')  changes to true false
churn_df.churn = churn_df['churn'].astype('int')

#np.asarray takes an input (like a list, tuple, or other array-like object)
#transforms it into a NumPy ndarray

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]

Y = np.asarray(churn_df['churn'])
Y[0:5]

#StandardScaler function from scikit learn library 
# I believe we standardize to numbers on a same scale 
# find the average and spread fairly that all features become equal 
X_norm = StandardScaler().fit(X).transform(X)
X_norm[0:5]


#SO what we basically do here is training and testing our model
#Suppose i have 100 customers,I splitted 80 for training and 20 for testing 
#So X_norm goes for X train  and Y goes for Y train
X_train, X_test, y_train, y_test = train_test_split(X_norm,Y,test_size=0.2,random_state=4)

#fitting or in a simple terms training 
LR = LogisticRegression().fit(X_train,y_train)

#the model calculates probability and then converts it to 0 or 1
#calcuation pribably happens if greater than 0.5 is 1 otherway is 0
yhat = LR.predict(X_test)
yhat[:10]
print(yhat)


yhat_prob = LR.predict_proba(X_test)
yhat_prob[:10]
print(yhat_prob)