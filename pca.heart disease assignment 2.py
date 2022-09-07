import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.linalg import svd
from sklearn.cluster import	KMeans

disease = pd.read_csv('C:\\Users\\heart disease.csv')
disease.describe()
disease.info()

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(disease.iloc[:, :])

#hierarchical clustering
# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 3 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters =3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)

disease['hclust'] = cluster_labels # creating a new column and assigning it to new column 
disease = disease.iloc[:, [14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]

#kmeans
###### scree plot or elbow curve ############
TWSS = []
k = list(range(1,15))
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)

TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")
# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
disease['kclust'] = mb # creating a  new column and assigning it to new column 
disease.head()
disease = disease.iloc[:, [15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]


#pca
from sklearn.preprocessing import scale 

disease_normal = scale(disease)
disease_normal
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
pca_values = pca.fit_transform(disease_normal)

var = pca.explained_variance_ratio_
var
# PCA weights

pca.components_
pca.components_[0]
# Cumulative variance 
var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1

plt.plot(var1, color = "red")
# PCA scores
pca_values
pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0", "comp1", "comp2"
final = pd.concat([disease, pca_data.iloc[:, 0:3]], axis = 1) 

#After PCA
#hierarchical and kmeans clustering on new data set 

#hierarchical clustering
# for creating dendrogram 
z = linkage(final, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 3 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters =3, linkage = 'complete', affinity = "euclidean").fit(final) 
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)

final['newhclust'] = cluster_labels # creating a new column and assigning it to new column 
final = final.iloc[:, [19,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]

#kmeans
###### scree plot or elbow curve ############
TWSS = []
k = list(range(1,21))
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(final)
    TWSS.append(kmeans.inertia_)

TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")
# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(final)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
final['newkclust'] = mb # creating a  new column and assigning it to new column fu
final.head()
final = final.iloc[:, [20,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]

#results of clusters before and after PCA or not similar the no. of cluster are different and changed
