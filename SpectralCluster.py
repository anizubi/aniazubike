

import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import SpectralClustering 
from sklearn.preprocessing import StandardScaler, normalize 
from sklearn.decomposition import PCA 
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Load the data
df = pd.read_csv('dow_jones_index.data')

# Print the first few rows of the data
df.head()

# Check the data types
df.dtypes

# Remove the dollar sign from the data and convert to float
df = df.applymap(lambda x: x.replace('$', '') if isinstance(x, str) else x)
df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float,
                'next_weeks_open': float, 'next_weeks_close': float})

# Check the data types again
df.dtypes

# Check for missing values
df.isnull().sum()

# Remove the missing values
df = df.dropna()

# Preprocessing the data to make it visualizable 

# Scaling the Data 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(df.drop(['stock', 'date'], axis =1)) 
  
# Normalizing the Data 
X_normalized = normalize(X_scaled) 
  
# Converting the numpy array into a pandas DataFrame 
X_normalized = pd.DataFrame(X_normalized) 
  
# Reducing the dimensions of the data 
pca = PCA(n_components = 2) 
X_principal = pca.fit_transform(X_normalized) 
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['feature1', 'feature2'] 

# Print the first few rows of the principal data
X_principal.head(6) 

# Initialize an empty list to store the Calinski Harabasz index
CH_index = []

# Loop through a range of possible number of clusters
for k in range(2, 11):
    # Create a spectral clustering model with the current number of clusters
    model = SpectralClustering(n_clusters=k, assign_labels="discretize")
    # Fit the model to the data
    model.fit(X_principal)
    # Get the cluster labels
    labels = model.labels_
    # Append the CH index of the model to the list
    CH_index.append(calinski_harabasz_score(X_principal, labels))

# Plot the CH index against the number of clusters
plt.plot(range(2, 11), CH_index)
plt.xlabel('Number of clusters')
plt.ylabel('Calinski Harabasz Index')
plt.show()

# Building the clustering model 
spectral_model_rbf = SpectralClustering(n_clusters = 7, affinity ='rbf') 
  
# Training the model and Storing the predicted cluster labels 
labels_rbf = spectral_model_rbf.fit_predict(X_principal)
# Visualizing the clustering 

# Visualizing the clustering 
plt.scatter(X_principal['feature1'], X_principal['feature2'],  
           c = SpectralClustering(n_clusters = 7, affinity ='rbf') .fit_predict(X_principal), cmap ='rainbow') 
plt.title('rbf Spectral Clusters')

plt.show() 

# Building the clustering model 
spectral_model_nn = SpectralClustering(n_clusters = 7, affinity ='nearest_neighbors') 
  
# Training the model and Storing the predicted cluster labels 
labels_nn = spectral_model_nn.fit_predict(X_principal)

# Visualizing the clustering 
plt.scatter(X_principal['feature1'], X_principal['feature2'],  
           c = SpectralClustering(n_clusters = 7, affinity ='nearest_neighbors') .fit_predict(X_principal),
            cmap ='rainbow') 
plt.title('Nearest Neighbour Spectral CLusters')
plt.show() 



