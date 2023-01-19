# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

# Load and clean the emission data
emission_pc = pd.read_csv('API_EN.ATM.CO2E.PC_DS2_en_csv_v2_4770565.csv', skiprows =4)

# Check for missing values
emission_pc.isnull().sum().T

# Drop missing values
emission_pc = emission_pc.dropna()

# Define a function to load and clean data
def load_and_clean(data_file):
    data = pd.read_csv(data_file, skiprows = 4)
    # Drop unnecessary column
    data = data.drop('Unnamed: 66', axis =1)
    # Reshape the dataframe
    long = data.melt(id_vars = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                    var_name = 'year', value_name = 'value')
    return long

# Use the function to load and clean emission data
emission_pc = load_and_clean('API_EN.ATM.CO2E.PC_DS2_en_csv_v2_4770565.csv')

# Load and clean GDP per capita data
gdp_pc = load_and_clean('API_NY.GDP.PCAP.CD_DS2_en_csv_v2_4770417.csv')

# Concatenate the two dataframes
data = pd.concat([emission_pc, gdp_pc])

# Keep only necessary columns
data = data[['Country Name', 'Indicator Name', 'year', 'value']].copy()

# Pivot the dataframe
data = data.pivot(index=['Country Name', 'year'], columns='Indicator Name', values='value').reset_index()

# Convert year column to integer
data['year'] = data['year'].astype(int)



categories = ['Africa Eastern and Southern','Arab World','Caribbean small states','Central African Republic', 'Central Europe and the Baltics',
'Early-demographic dividend', 'East Asia & Pacific',
       'East Asia & Pacific (IDA & IBRD countries)',
       'East Asia & Pacific (excluding high income)','Europe & Central Asia',
       'Europe & Central Asia (IDA & IBRD countries)',
       'Europe & Central Asia (excluding high income)', 'European Union',
 'Fragile and conflict affected situations','French Polynesia','Heavily indebted poor countries (HIPC)',
 'High income', 'IBRD only',
       'IDA & IBRD total', 'IDA blend', 'IDA only', 'IDA total','Late-demographic dividend',
 'Latin America & Caribbean',
       'Latin America & Caribbean (excluding high income)',
       'Latin America & the Caribbean (IDA & IBRD countries)',
       'Least developed countries: UN classification', 'Low & middle income', 'Low income', 'Lower middle income',
 'Middle East & North Africa',
       'Middle East & North Africa (IDA & IBRD countries)',
       'Middle East & North Africa (excluding high income)',
       'Middle income', 'Not classified',
       'OECD members', 'Other small states',
       'Pacific island small states','Post-demographic dividend',
       'Pre-demographic dividend','Small states','South Asia (IDA & IBRD)','Sub-Saharan Africa', 
 'Sub-Saharan Africa (IDA & IBRD countries)',
       'Sub-Saharan Africa (excluding high income)','Upper middle income', 'West Bank and Gaza',
                 'World','Africa Western and Central'
]

# Remove rows that have categories in the 'Country Name' column
data = data[~data['Country Name'].isin(categories)].copy()

# Remove rows that have missing values
data = data.dropna()

# Filter data for year 2018
data_2019 = data[data['year']==2018].copy()

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_2019.drop(['Country Name','year'], axis =1))

# Calculate silhouette scores for different number of clusters
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(scaled_data)
    y_pred = kmeans.predict(scaled_data)
    score = silhouette_score(scaled_data, y_pred)
    silhouette_scores.append(score)

# Plot the silhouette scores
plt.plot(range(2, 11), silhouette_scores)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Silhouette Score for Optimum Cluster')
plt.show()

# This code is performing k-means clustering on a dataset with two features (scaled_data) and determining the optimal number of clusters using the elbow method.

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot the WCSS values to determine the optimal number of clusters
import matplotlib.pyplot as plt
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Initialize the model with the optimal number of clusters
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)

# Fit the model to the data
kmeans.fit(scaled_data)

# Use the silhouette score to evaluate the quality of the clusters
from sklearn.metrics import silhouette_score
print(f'Silhouette Score: {silhouette_score(scaled_data, kmeans.labels_)}')

# Plot the data points with the assigned cluster labels
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=kmeans.labels_)
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# predicting
y_pred = kmeans.fit_predict(data_2019.drop(['Country Name','year'], axis =1))
data_2019['cluster'] = y_pred +1

#Plotting the clusters
plt.figure(figsize=(12,6))
sns.scatterplot(y= data_2019['GDP per capita (current US$)'],
                x= data_2019['CO2 emissions (metric tons per capita)'], 
                hue= data_2019['cluster'], 
                palette='bright')
plt.title('Country Clusters Based on GDP per Capita and CO2 Emissions Per Capita', fontsize = 18)

# Extracting countries that are in cluster 1
data_2019[data_2019['cluster']==1]

# Defining logistic function
def logistic(t, n0, g, t0):
  """Calculates the logistic function with scale factor n0 and growth rate g"""
  
  f = n0 / (1 + np.exp(-g*(t - t0)))
  
  return f

# This code defines two functions for fitting a data set: an exponential function and an error range function. 
# The exponential function takes in a time variable, and initial value, and a growth rate, and returns a value based on the exponential equation. 
# The error range function takes in a variable, function, parameters, and sigmas and calculates the upper and lower limits for the function given the parameters and sigmas. 
# It then uses these functions to plot and fit the CO2 emission trend for China.

def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    
    t = t - 1960.0
    f = n0 * np.exp(g*t)
    return f


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters, and sigmas for single value or array x. 
    Functions values are calculated for all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    """

    import itertools as iter
    lower = func(x, *param)
    upper = lower
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
    return lower, upper   


n = [1,2,3,4,5]
for i in n:
    print(f"Cluster {i}")
    print(data_2019[data_2019['cluster']==i]['Country Name'].unique())
    print("**")


china_data = data[data['Country Name']=='China']
china_data.plot("year","CO2 emissions (metric tons per capita)")
plt.title('China CO2 Emission Trend (Cluster 1)')
plt.show()

param, covar = opt.curve_fit(logistic, china_data["year"], 
                             china_data["CO2 emissions (metric tons per capita)"], 
                             p0=(2000000000, 0.03,2005))

sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)

china_data["fit"] = logistic(china_data["year"], *param)

china_data.plot("year", ["CO2 emissions (metric tons per capita)", "fit"])
plt.title('Switzerland GDP Trend')
plt.show()

# create array of years from 1990 to 2030
year = np.arange(1990, 2030)

# fit data to logistic function
param, covar = opt.curve_fit(logistic, china_data["year"], china_data["CO2 emissions (metric tons per capita)"], p0=(5.102798, 0.03, 2005.00))

# calculate standard deviation
sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)

# add fit column to data
china_data["fit"] = logistic(china_data["year"], *param)

# plot original data and fitted logistic function
china_data.plot("year", ["CO2 emissions (metric tons per capita)", "fit"])
plt.title('China CO2 Emission Trend')
plt.show()

# forecast using logistic function
forecast = logistic(year, *param)

# plot forecast and original data
plt.figure()
china_data.plot("year","CO2 emissions (metric tons per capita)", label = 'CO2 emissions (metric tons per capita)')
plt.plot(year, forecast, label="forecast")

plt.xlabel("CO2 emissions (metric tons per capita)")
plt.ylabel("year")
plt.legend()
plt.title('China CO2 Emision Forecast')
plt.show()

# calculate error ranges
low, up = err_ranges(year, logistic, param, sigma)

# plot forecast with error ranges
plt.figure()
china_data.plot("year","CO2 emissions (metric tons per capita)", label = 'CO2 emissions (metric tons per capita)')
plt.plot(year, forecast, label="forecast")

plt.xlabel("CO2 emissions (metric tons per capita)")
plt.ylabel("year")
plt.legend()
plt.title('China CO2 Emision Forecast')

plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.legend()
plt.title('China GDP Forecast and Confidence Interval')
plt.show()

# print error ranges for a specific value
print(err_ranges(10143.838196, logistic, param, sigma))

# select data for USA
usa_data = data[data['Country Name']=='United States']

# plot original USA data
usa_data.plot("year","CO2 emissions (metric tons per capita)")
plt.title('USA CO2 Emission Trend (Cluster 2)')
plt.show()

# fit data to exponential function
param, covar = opt.curve_fit(exponential,usa_data["year"],
usa_data["CO2 emissions (metric tons per capita)"], p0=(5.102798, -0.02))

# calculate standard deviation
sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)

# add fit column to data
usa_data["fit"] = exponential(usa_data["year"], *param)

# plot original data and fitted exponential function
usa_data.plot("year", ["CO2 emissions (metric tons per capita)", "fit"])
plt.title('USA CO2 Emission Trend ')
plt.show()

# create array of years from 1990 to 2030
year = np.arange(1990, 2030)

# forecast using exponential function
forecast = exponential(year, *param)

# plot forecast and original data
plt.figure()
usa_data.plot("year","CO2 emissions (metric tons per capita)", label = 'CO2 emissions (metric tons per capita)')
plt.plot(year, forecast, label="forecast")

plt.xlabel("CO2 emissions (metric tons per capita)")
plt.ylabel("year")
plt.legend()
plt.title('USA CO2 Emission Forecast')
plt.show()

# calculate error ranges
low, up = err_ranges(year, exponential, param, sigma)

# plot forecast with error ranges
plt.figure()
usa_data.plot("year","CO2 emissions (metric tons per capita)", label = 'CO2 emissions (metric tons per capita)')
plt.plot(year, forecast, label="forecast")

plt.xlabel("CO2 emissions (metric tons per capita)")
plt.ylabel("year")
plt.legend()
plt.title('USA CO2 Emission Forecast')

plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.legend()
plt.title('USA CO2 Emission Forecast and Confidence Interval')
plt.show()

# print error ranges for a specific year
print(err_ranges(2010, exponential, param, sigma))
