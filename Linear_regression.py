import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import ssl


url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

#skipped verified version of python with this line 
ssl._create_default_https_context = ssl._create_unverified_context

#pandas read method for reading database 
df = pd.read_csv(url)

#statistical summary of data
# print(df.describe())

#select few features that might be indicative for CO2 emissions
#double square brackets for pandas to pick multiple columns as data frame 
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.sample(9))

#visualize data to better interpret
# viz = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

#A histogram is a graphical representation of the distribution of numerical data
# viz.hist()
# plt.show()



#display some scatter plots(graph) of these features against the CO2 emissions, to see how linear their relationships are.
# plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS, color='blue')
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("Emission")
#Three car groups each have a strong linear relationship between their combined fuel consumption and their CO2 emissions. 
#Their intercepts are similar, while they noticeably differ in their slopes.
# plt.show()

# plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS, color = 'purple')
# plt.xlabel('ENGINESIZE')
# plt.ylabel('CO2EMISSIONS')
# plt.xlim(0,50)
# plt.show()

# plt.scatter(cdf.CYLINDERS,cdf.CO2EMISSIONS, color='red')
# plt.xlabel('CYLINDERS')
# plt.ylabel('CO2EMISSIONS')
# plt.show()

X = cdf.ENGINESIZE.to_numpy()
Y = cdf.CO2EMISSIONS.to_numpy()

# print("x : ",X, "y :",Y)


from sklearn.model_selection import train_test_split

#X,Y is my whole dataset(inpyts and labels)
#test size is 20 percent so 80 percent remained for traning
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#It's a numpy.ndarray (a type of array in Python)
#It has 853 rows (samples), and it’s 1D.
print(type(X_train)), print(np.shape(X_train))

from sklearn import linear_model
print(X_train)
#create a model objects
regressor = linear_model.LinearRegression()
print("regressionss : ",regressor)

# X_train is 1D array but sklearn models expect 2D array as input for the trainig data
# with shape (n_observations, n_features)
# So we need to reshape it. We can let it infer the number of observations using '-1'.
regressor.fit(X_train.reshape(-1,1),y_train)


# Print the coefficients
# Here, Coefficient and Intercept are the regression parameters determined by the model.
# They define the slope and intercept of the 'best-fit' line to the training data.
print ('Coefficients: ', regressor.coef_[0]) # with simple linear regression there is only one coefficient, here we extract it from the 1 by 1 array.
print ('Intercept: ',regressor.intercept_)



# plt.scatter(X_train,y_train, color='blue')
# # probably formula 
# plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
# plt.xlabel("ENGINESIZE")
# plt.ylabel("CO2EMISSIONS")
# plt.show()




from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Use the predict method to make test prediction
y_test_ = regressor.predict(X_test.reshape(-1,1))

#Evaluation
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_test_))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_test_)))
print("R2-score: %.2f" % r2_score(y_test, y_test_))


X = cdf.FUELCONSUMPTION_COMB.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state=42)


regressor_2 = linear_model.LinearRegression()

regressor_2.fit(X_train.reshape(-1,1),y_train)
#we use the model to make test prediction in here 
y_test_ = regressor_2.predict(X_test.reshape(-1,1))

print("Mean absolute error %.2f" % mean_absolute_error(X_test,y_test_))
