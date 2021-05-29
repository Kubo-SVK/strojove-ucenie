import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift,estimate_bandwidth
from itertools import cycle


important = ['appid','developer','publisher','genres','positive_ratings','negative_ratings','average_playtime','median_playtime','achievements','price']
usefull_cols = ['developer','publisher','genres','positive_ratings','negative_ratings','average_playtime']

def eda(data):
    print(data.describe())
    print(data['developer'].value_counts())
    print(data['publisher'].value_counts())
    print(data['genres'].value_counts())

def load():
    dataset = pd.read_csv('steam.csv')
    dataset2 = pd.read_csv('steamspy_tag_data.csv')

    dataset.dropna(inplace=True)
    
    #developer
    le_dev = preprocessing.LabelEncoder()
    dataset['developer'] = le_dev.fit_transform(dataset['developer'])

    #publisher
    le_pub = preprocessing.LabelEncoder()
    dataset['publisher'] = le_pub.fit_transform(dataset['publisher'])

    #genres
    le_gen = preprocessing.LabelEncoder()
    dataset['genres'] = le_gen.fit_transform(dataset['genres'])

    dataset = dataset[dataset.negative_ratings < 35]
    dataset = dataset[dataset.positive_ratings < 20000]
    dataset = dataset[dataset.positive_ratings > 0]
    dataset = dataset[dataset.negative_ratings > 0]
    dataset = dataset[dataset.price < 90]
     
    data = pd.merge(dataset,dataset2,on="appid")
    data.drop(['name', 'release_date','platforms','categories','owners'], axis=1, inplace=True)
    data.dropna(inplace=True)

    return data


data = load()
important.remove("appid")

def heatMap(data):
    correlation = data[important].corr()
    sns.heatmap(correlation, annot=True, cmap='cubehelix')
    plt.show()

def meanShift(data):
    X = data[usefull_cols].to_numpy()
    
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    analyzer = MeanShift(bandwidth=bandwidth, bin_seeding=True) 
    analyzer.fit(X)
    labels = analyzer.labels_

    cluster_centers = analyzer.cluster_centers_
    
    # Plot result
    plt.figure(1)
    plt.clf()
    
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

def meanShiftPCA(dataset):
    df=dataset
    pca = PCA(n_components=2)
    pca.fit(df)
    scores_pca = pca.transform(df)
    new_dim = list()
    for comp1, comp2 in scores_pca:
        new_dim.append([comp1, comp2])

    new_df = pd.DataFrame(new_dim)

    k_means = KMeans(n_clusters=8, init="k-means++", random_state=42)
    k_means = k_means.fit_predict(new_dim)
    new_series = pd.Series(k_means, dtype=np.int64, name='Segment K-means')
    new_series = new_series.map(
        {0: 'first', 1: 'second', 2: 'third', 3: 'fourth', 4: 'fifth', 5: 'sixth', 6: 'seventh', 7: 'eight'})
    plt.figure(figsize=(12, 9))
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    x_axis = new_df[0]
    y_axis = new_df[1]
    sns.scatterplot(x_axis, y_axis, c=k_means, hue=new_series)
    plt.title("MeanShift PCA")
    plt.show()

def elbow(data):
    mms = MinMaxScaler()
    mms.fit(data)
    data_transformed = mms.transform(data)

    Sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data_transformed)
        Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

def visualise(data):
    plt.figure(figsize=(12,9))
    plt.scatter(data['positive_ratings'],data['achievements'])
    plt.xlabel('Positive ratings')
    plt.ylabel('Achievements')
    plt.title("Visualisation")
    plt.show()


def kMeansPca(data):
    scaler = MinMaxScaler()
    scaled_dataset=scaler.fit_transform(data)
    pca=PCA(n_components=200)
    pca.fit(scaled_dataset)
    
    scores_pca=pca.transform(scaled_dataset)
    print(len(scores_pca))
    
    k_means_pca=KMeans(n_clusters=6,init="k-means++",random_state=42)
    k_means_pca.fit(scores_pca)
    segm_pca_kmeans=pd.concat([data.reset_index(drop=True),pd.DataFrame(scores_pca)],axis=1)
    segm_pca_kmeans.columns.values[-3: ]=['genres','average_playtime','positive_ratings']
    segm_pca_kmeans['Segment K-means PCA']=k_means_pca.labels_
    segm_pca_kmeans['Segment']=segm_pca_kmeans['Segment K-means PCA'].map({0:'first',1:'second',2:'third',3:'fourth',4:'fifth',5:'sixth'})
    print(segm_pca_kmeans["genres"])
    x_axis=segm_pca_kmeans[6]
    y_axis=segm_pca_kmeans[10]
    
    plt.figure(figsize=(10,8))
    sns.scatterplot(x_axis,y_axis,hue=segm_pca_kmeans['Segment'])
    plt.title('Clusters by PCA')
    plt.show()

"""All functions are called here"""
heatMap(data)
meanShift(data)
meanShiftPCA(data[usefull_cols])
elbow(data)
visualise(data)
kMeansPca(data)