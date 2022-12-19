#E-Commerce Customer Segmetation


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df1=pd.read_csv('cust_data1.csv')

df1=df1.drop(columns=['Gender','Cust_ID'],axis=1)
df1['sums']=df1.iloc[:,1:37].sum(axis=1)

df=df1.iloc[:,[0,36]].values



wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(df)
    
    
    wcss.append(kmeans.inertia_)
    
sns.set()
plt.plot(range(1,11),wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


#No of clusters=4

kmeans=KMeans(n_clusters=4,init='k-means++',random_state=0)
y=kmeans.fit_predict(df)
print(y)


plt.figure(figsize=(10,10))
plt.scatter(df[y==0,0], df[y==0,1],s=50,c='green',label='cluster 1')
plt.scatter(df[y==1,0], df[y==1,1],s=50,c='red',label='cluster 2')
plt.scatter(df[y==2,0], df[y==2,1],s=50,c='blue',label='cluster 3')
plt.scatter(df[y==2,0], df[y==2,1],s=50,c='yellow',label='cluster 4')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='cyan',label='centroids')
plt.title('Customer groups')
plt.xlabel('Number of orders')
plt.ylabel('total no of times searched')
plt.show()




    
