import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv'
tele_data = pd.read_csv(url)
print("data sample: ",tele_data.sample(4))
print("data head: ",tele_data.head())
print("data info: ", tele_data.info())

#class-wise distribution of dataset 
print(tele_data['custcat'].value_counts())

#visualize corelation
correlation_matrix = tele_data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt='.2f',linewidths=0.5)
# plt.show()

#list of features sorted in the descending orders with respect ot the target field
correlation_values = abs(tele_data.corr()['custcat'].drop('custcat')).sort_values(ascending=False)
print(correlation_values)

X = tele_data.drop('custcat',axis=1)
y = tele_data['custcat']
print(y.sample(6))

#normalize our data 
X_norm = StandardScaler().fit_transform(X)

#train test-split 
X_train,X_test,y_train,y_test = train_test_split(X_norm,y,test_size=0.2,random_state=4)

#training 
k = 99
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_model = knn_classifier.fit(X_train,y_train)

#predicting
knn_predict = knn_model.predict(X_test)

#Accuracy Evaluation
print("Test set accuracy: ",accuracy_score(y_test,knn_predict))


#check the performance of model for 10 values of k
Ks = 10
acc = np.zeros((Ks))
std_acc = np.zeros((Ks))
for n in range(1,Ks+1):
    #train model and predict
    knn_model_n = KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    knn_predict = knn_model.predict(X_test)
    acc[n-1] = accuracy_score(y_test,knn_predict)
    std_acc[n-1] = np.std(knn_predict==y_test)/np.sqrt(knn_predict.shape[0])


#plot the model accuracy for a different number of neighbors
plt.plot(range(1,Ks+1),acc,'g')
plt.fill_between(range(1,Ks+1),acc - 1 * std_acc,acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Acuracy')
plt.xlabel('Number of neighbors (K)')
plt.tight_layout()
print( "The best accuracy was with", acc.max(), "with k =", acc.argmax()+1) 
plt.show()

