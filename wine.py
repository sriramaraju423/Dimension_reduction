#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


wine = pd.read_csv(r"C:\Users\srira\Desktop\Ram\Data science\Course - Assignments\Module 14 - PCA\Dataset\wine.csv")
wine.head(10)


# ### EDA

# In[5]:


wine.Type.value_counts()


# In[6]:


wine.isnull().sum()


# In[7]:


wine.describe()


# In[8]:


describe = wine.describe()
outliers = []
i = 0
while i<len(describe.columns):
    Q1 = describe[describe.columns[i]]['25%']
    Q3 = describe[describe.columns[i]]['75%']
    Upper_fence = describe[describe.columns[i]]['max']
    Lower_fence = describe[describe.columns[i]]['min']
    IQR = Q3-Q1
    min_fence,max_fence = (Q1-1.5*IQR),(Q3+1.5*IQR)
    outlier_cat = []
    if Lower_fence > min_fence:
        outlier_cat.append('Lower Outliers')
    if Upper_fence > max_fence:
        outlier_cat.append('Upper Outliers')
    if(Upper_fence < max_fence and Lower_fence < min_fence):
        outliers.append('No outliers')
    else:
        outliers.append(outlier_cat)
    i += 1    

df_outliers = pd.DataFrame(np.column_stack([describe.columns.tolist(),outliers]),columns=['columns','Outliers'])
df_outliers


# In[9]:


#Let's handle outliers in a moment after completing whole PCA and clustering


# In[10]:


from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import matplotlib.pylab as plt


# In[11]:


wine_scale = scale(wine)
wine_scale


# In[12]:


pca = PCA()
pca_values = pca.fit_transform(wine_scale)
pca_values


# In[13]:


pca.components_


# In[14]:


var = pca.explained_variance_ratio_
var


# In[15]:


var_sum = np.cumsum(np.round(var,decimals=4)*100)
var_sum


# In[16]:


#Plotting variance
plt.plot(var_sum)


# In[17]:


#Plotting PC1 vs PC2
PC1 = pca_values[:,0]
PC2 = pca_values[:,1]
plt.plot(PC1,PC2,"bo")


# In[18]:


#hierarchical clustering: Let's take 7 out of 14 columns which has 90% of the data without correlation


# In[19]:


wine_pca = pd.DataFrame(pca_values[:,0:7])
wine_pca


# In[20]:


import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering


# In[21]:


z = linkage(wine_pca,method='complete',metric='euclidean')
z


# In[22]:


import matplotlib.pyplot as plt


# In[23]:


plt.figure(figsize=(15,5))
sch.dendrogram(z,leaf_rotation=0.,leaf_font_size=12.)
plt.show()


# In[24]:


h_clustering = AgglomerativeClustering(n_clusters=4,linkage='complete',affinity='euclidean').fit(wine_pca)
h_clustering.labels_


# In[25]:


#Let's try KMeans. Let's try with normal data first and then with PCA


# In[26]:


def norm(i):
    x = (i - i.min())/(i.max() - i.min())
    return x


# In[27]:


wine_norm = norm(wine.iloc[:,:])
wine_norm.head(10)


# In[28]:


#Finding the optimum k value using scree plot


# In[29]:


k = range(2,10)
TWSS = []


# In[30]:


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# In[31]:


for i in k:
    kmeans = KMeans(n_clusters=i).fit(wine_norm)
    WSS = []
    for j in range(i):
        WSS.append(sum(cdist(wine_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,wine.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# In[32]:


len(TWSS)


# In[33]:


import matplotlib.pyplot as plt
plt.plot(k,TWSS,'ro-')
plt.show()


# In[34]:


#Looking at the screen plot we can either take 3 or 4 clusters


# In[35]:


kmeans = KMeans(n_clusters=3).fit(wine_norm)
kmeans.labels_


# In[36]:


#Now let's do Kmeans using PCA data


# In[37]:


k = range(2,10)
TWSS = []


# In[38]:


for i in k:
    kmeans = KMeans(n_clusters=i).fit(wine_pca)
    WSS = []
    for j in range(i):
        WSS.append(sum(cdist(wine_pca.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,wine_pca.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# In[39]:


len(TWSS)


# In[40]:


plt.plot(k,TWSS,'ro-')
plt.show()


# In[41]:


#This clearely says the no of clusters we can have are 3


# In[42]:


#Let's try and now solve by taking just 3 PCA's


# In[43]:


wine_pca_3 = pd.DataFrame(pca_values[:,0:3])
wine_pca_3.head(10)


# In[47]:


k = range(2,10)
TWSS_3 = []


# In[48]:


for i in k:
    kmeans = KMeans(n_clusters=i).fit(wine_pca)
    WSS = []
    for j in range(i):
        WSS.append(sum(cdist(wine_pca.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,wine_pca.shape[1]),"euclidean")))
    TWSS_3.append(sum(WSS))


# In[49]:


plt.plot(k,TWSS_3,'ro-')
plt.show()


# In[ ]:


#From my perspective still 3 looks good k value for clustering


# In[ ]:


#I don't see any improvement in doing clustering post PCA. Practically i guess apart from accurate clustering and stuff, i guess the power of PCA lies with computational ability.


# In[44]:


wine_pca.shape


# In[45]:


wine.shape


# In[46]:


wine.columns

