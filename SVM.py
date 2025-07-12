# Import the libraries we need to use in this lab
from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC
import ssl
import warnings
warnings.filterwarnings('ignore')
ssl._create_default_https_context = ssl._create_unverified_context

url = url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"
bank_data = pd.read_csv(url)

print(bank_data.sample(3))
# print("info: ",bank_data.info())

#get the set of distinct classes
labels = bank_data.Class.unique()
print("labels: ",labels)

#get the count of each class
sizes = bank_data.Class.value_counts().values
print('sizes: ',sizes)

fig,ax = plt.subplots()
ax.pie(sizes,labels=labels,autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')
# plt.show()

correlation_values = bank_data.corr()['Class'].drop('Class')
correlation_values.plot(kind='barh',figsize=(10,6))
# plt.show()

#standardize features by removing the mean and scaling unit variance
bank_data.iloc[:, 1:30] = StandardScaler().fit_transform(bank_data.iloc[:, 1:30])
data_matrix = bank_data.values

# X: feature matrix (for this analysis  we exclude the Time variable from the dataset)
X = data_matrix[:, 1:30]

#y: labels vector
y = data_matrix[:, 30]

#data normalization 
X = normalize(X, norm='l1')

#split dataset into test-train
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#compute the sample weights for taking account class imbalance in dataset 
w_train = compute_sample_weight('balanced',y_train)


dt = DecisionTreeClassifier(max_depth=4,random_state=35)
dt.fit(X_train,y_train,sample_weight=w_train)


#Build Support Vector Machine model with Scikit-Learn
svm = LinearSVC(class_weight='balanced',random_state=31,loss='hinge',fit_intercept=False)
svm.fit(X_train,y_train)

#Evaluate the Decision Tree classifier model 
y_pred_dt = dt.predict_proba(X_test)[:,1]
roc_auc_dt = roc_auc_score(y_test,y_pred_dt)
print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dt))

#Evaluate Support Vector machine models 
y_pred_svm = svm.decision_function(X_test)

#Evaluate accuracy of SVM
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm))

#top six feature of dataset to train model

correlation_values2 = abs(bank_data.corr()['Class']).drop('Class')
correlation_values2 = correlation_values2.sort_values(ascending=False)[:6]
print(correlation_values2)