import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import ssl

print(plt.get_backend())

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

#skipped verified version of python with this line 
ssl._create_default_https_context = ssl._create_unverified_context

#pandas read method for reading database 
df = pd.read_csv(url)

#statistical summary of data
print(df.describe())

#select few features that might be indicative for CO2 emissions
#double square brackets for pandas to pick multiple columns as data frame 
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.sample(9))

#visualize data to better interpret
viz = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

#A histogram is a graphical representation of the distribution of numerical data
viz.hist()
plt.show()



#display some scatter plots(graph) of these features against the CO2 emissions, to see how linear their relationships are.
plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
#Three car groups each have a strong linear relationship between their combined fuel consumption and their CO2 emissions. 
#Their intercepts are similar, while they noticeably differ in their slopes.
plt.show()

plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS, color = 'purple')
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2EMISSIONS')
plt.xlim(0,50)
plt.show()

plt.scatter(cdf.CYLINDERS,cdf.CO2EMISSIONS, color='red')
plt.xlabel('CYLINDERS')
plt.ylabel('CO2EMISSIONS')
plt.show()