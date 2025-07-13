import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import ssl 

ssl._create_default_https_context = ssl._create_unverified_context


#Load the California housing dataset
data = fetch_california_housing()
X,y = data.data , data.target
# print('our data: ',data)

#split data into training and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#observations and feature of dataset 
N_observations,N_features = X.shape
print('Number of Observations: ' + str(N_observations))
print('Number of Features : ' + str(N_features))

#initialize model
n_estimators = 100
rf = RandomForestRegressor(n_estimators=n_estimators,random_state=42)
xgb = XGBRegressor(n_estimators=n_estimators,random_state=42)

#Fit models and measure training time for RandomForestRegressor
start_time_rf = time.time()
rf.fit(X_train,y_train)
end_time_rf = time.time()
rf_train_time = end_time_rf - start_time_rf
# print('training time for RandomForestRegressor: ',rf_train_time)

#Fit models and measure training itme fot XGBRegressor
start_time_xgb = time.time()
xgb.fit(X_train,y_train)
end_time_xgb = time.time()
xgb_train_time = end_time_xgb - start_time_xgb
# print('XGBRegressor training time: ',xgb_train_time)

# Measure prediction time for Random Forest
start_time_rf2 = time.time()
y_pred_rf = rf.predict(X_test)
end_time_rf2 = time.time()
rf_pred_time = start_time_rf - end_time_rf

# Measure prediciton time for XGBoost
start_time_xgb2 = time.time()
y_pred_xgb = xgb.predict(X_test)
end_time_xgb = time.time()
xgb_pred_time = end_time_xgb - start_time_xgb

#Calculate MSE
mse_rf = mean_squared_error(y_test,y_pred_rf)
mse_xgb = mean_squared_error(y_test,y_pred_xgb)

#Calculate R^2 
r2_rf = r2_score(y_test,y_pred_xgb)
r2_xgb = r2_score(y_test,y_pred_xgb)

all_errors = f"""
mean squared error rf : {mse_rf},
mean squared error xgb : {mse_xgb}
r2 error rf : {r2_rf}
r2 error xgb : {r2_xgb}
"""
print(all_errors)

#better syntax for errrors
print(f'Random Forest:  MSE = {mse_rf:.4f}, R^2 = {r2_rf:.4f}')
print(f'      XGBoost:  MSE = {mse_xgb:.4f}, R^2 = {r2_xgb:.4f}')

#better syntax for training time
print(f'Random Forest:  Training Time = {rf_train_time:.3f} seconds, Testing time = {rf_pred_time:.3f} seconds')
print(f'      XGBoost:  Training Time = {xgb_train_time:.3f} seconds, Testing time = {xgb_pred_time:.3f} seconds')

"""as you can see there is a big difference training time between xgb and rf
XGB has better computation time"""

#calculate standard deviation of the test 
std_y = np.std(y_test)

#Visualize the results
plt.figure(figsize=(14,6))

#Random forest plot
plt.subplot(1,2,1)
plt.scatter(y_test,y_pred_rf, alpha=0.5, color='blue',ec='k')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--',lw=2,label='perfect model')
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1, )
plt.ylim(0,6)
plt.title("Random Forest Predictions vs Actual")
plt.xlabel("Actual labels")
plt.ylabel("Predicted values")
plt.legend()
# plt.show()

#XGBost plot
plt.subplot(1,2,2)
plt.scatter(y_test,y_pred_rf,alpha=0.5,color='purple',ec='k')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--',lw=2,label='perfect model')
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1, )
plt.ylim(0,6)
plt.title("XGBoost Predictions vs Actual")
plt.xlabel("Actual Labels")
# plt.ylabel('Predicited Values')
plt.legend()
plt.tight_layout()
plt.show()
