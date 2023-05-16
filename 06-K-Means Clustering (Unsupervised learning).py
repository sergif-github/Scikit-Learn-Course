print("Types of learning")
print("Supervised learning:      Making predictions using dada. Input labeled data to predict labels of new data")
print("Unsupervised learning:    Extracting structure from data. Input no labeled data to learn patrons or structures.")
print("                          (Clustering, dimensionality reduction, data association)")
print("Reinforcement learning:   Input states and actions to learn how to improve on best states and actions")

# We don't know what we are looking for in our features, we will make clustering
print("\nK-Means")
print("1 - K-Means start with K different random points or centroids")
print("2 - Compute distance of every feature to centroids. Cluster then accordingly")
print("3 - Adjust centroids to become center of gravity for given cluster")
print("4 - Repeat 2 - 3 until centroids position doesn't change")

print("\nTo find out how many K centroids we chose we use the sum of squared errors (SSE) and the elbow technique")
print("SSE = Compute for each centroid the distances squared of each clustered point")
print("Get the total SSE of all clusters")
print("Graphically, if we chose a rather K centroids the SSE is going to decrease")
print("Using the elbow technique, we chose the elbow corresponding point from the SSE - K-centroids graphic")

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# Problem to solve: Extract structure from data. Clustering data.

# Read and plot features
df = pd.read_csv("income.csv")
print("\nFirst 5 dataset rows:\n", df.head())
plt.scatter(df['Age'], df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.show()

# Preprocessing
scaler = MinMaxScaler()
scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])
scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])

# Find best K (Elbow technique)
k_rng = range(1, 10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age', 'Income($)']])
    sse.append(km.inertia_)  # SSE
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(k_rng, sse)
plt.show()

# Clustering (Using best K found using elbow technique)
km = KMeans(n_clusters=3)
df['cluster'] = km.fit_predict(df[['Age', 'Income($)']])
print("\nFirst 5 dataset rows:\n", df.head())
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
plt.scatter(df1.Age, df1['Income($)'], color='green')
plt.scatter(df2.Age, df2['Income($)'], color='red')
plt.scatter(df3.Age, df3['Income($)'], color='black')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='centroid')
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.show()
