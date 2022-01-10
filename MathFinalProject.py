import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.spatial import distance 
warnings.filterwarnings('ignore')

#~~~~Reading in Data~~~~

def goodDataIn(filePath):
  data = pd.read_csv(filePath)
  data.head()  
  

  data = data.loc[:, ['exports', 'gdpp']]
  data.head(2)  

  X = data.values
  
  return X


def badDataIn(filePath):
  data = pd.read_csv(filePath)
  data.head()  
  
  data = data.loc[:, ['Annual Income (k$)', 'Spending Score (1-100)', 'Age']]
  data.head(2)  
  X = data.values
  
  return X
  



#~~~~K-Means Algorithm~~~~

#actually implements the K-means clustering algorithm
#returns array of centroids and the cluster where each index is the 
#same index in the data, but with the value of which centroid is the closest
def kmeans(X, k):
  
  diff = 1
  
  #Array of zeros based on the shape of the data
  cluster = np.zeros(X.shape[0])

  # select k random centroids
  random_indices = np.random.choice(len(X), size=k, replace=False)
  
  #picks random points in X and makes that into an array
  centroids = X[random_indices, :]

  while diff:

    # for each point
    for i, row in enumerate(X):

      mn_dist = float('inf')
     
      # dist of the point from all centroids
      for idx, centroid in enumerate(centroids):
        d = distance.euclidean(centroid, row)

 
        # store closest centroid 
        if mn_dist > d:
          mn_dist = d
          cluster[i] = idx

    new_centroids = pd.DataFrame(X).groupby(by=cluster).mean().values

    # if centroids are same then leave
    if np.count_nonzero(centroids-new_centroids) == 0:
      diff = 0
    else:
      centroids = new_centroids
  return centroids, cluster  

    
    
#~~~~Analysis~~~~


#   ~~Elbow Method~~

#used to figure out cost of using different k's
#returns the sum of each cost
# calculates the average distance of each point in a cluster to its centroid 
def calculate_cost(X, centroids, cluster):
  sum = 0
  for i, val in enumerate(X):
    sum += np.sqrt((centroids[int(cluster[i]), 0]-val[0])**2 +(centroids[int(cluster[i]), 1]-val[1])**2)
  return sum

#finding optimal K using elbow method and the cost (WCSS)
def findOptimalK(X):
  cost_list = []
  for k in range(1, 10):
    centroids, cluster = kmeans(X, k)
  
    # WCSS (Within cluster sum of square)
    cost = calculate_cost(X, centroids, cluster)
    cost_list.append(cost)

  # displaying elbow method 
  sns.lineplot(x=range(1,10), y=cost_list, marker='o')
  plt.xlabel('k')
  plt.ylabel('WCSS')
  plt.show()

#   ~~Silhouette Analysis~~

#Calculates the silhouette coefficient for each point, based on the average distance between a point and the other points in a cluster
# minus the average distance of the point and other points in a neighboring cluster, and divided by the max of the two
def silhouetteAnalysis(X, centroids, cluster):
  #need to return dictionary of lists, where each cluster has its own list of coefficients
  silhouetteCoefficients = {}
  #for every cluster
  for idx, c in enumerate(np.unique(cluster)):
    #for every point in cluster
    currentSilhouetteCoefficient = []
    for i, point in enumerate(X):
      
      if cluster[i] == c :
  
        arrayAverageDistanceSameCluster = []
        arrayAverageDistanceNeighborCluster = []
        
        # go through and find distance for that point to every point associated
        # with the same cluster    
        for i2, row2 in enumerate(X):
          if cluster[i2] == cluster[i]:
            distanceToPointSameCluster = distance.euclidean(point, row2)
            arrayAverageDistanceSameCluster.append(distanceToPointSameCluster)
        
      
        #first find the second closest centroid to the point
        secondClosestClusterDict = {}
        for idx, centroid in enumerate(centroids):
          d = distance.euclidean(centroid, point)
          secondClosestClusterDict[idx] = d
   
        chosenNearestClusterIndex = min(secondClosestClusterDict, key=secondClosestClusterDict.get)
        secondClosestClusterDict.pop(chosenNearestClusterIndex, None)
        chosenNearestClusterIndex = min(secondClosestClusterDict, key=secondClosestClusterDict.get)
        
        #then go through and find average distance for that point to every point associated
        #with a closest neigbhoring cluster to that specific point (could have different neighboring clusters for different points in the 
        #same cluster)     
        for j, point2 in enumerate(X):
          if cluster[j] == chosenNearestClusterIndex:
            distanceToPointNeighborCluster = distance.euclidean(point, point2)
            arrayAverageDistanceNeighborCluster.append(distanceToPointNeighborCluster)      
          
        
        #calculate averages
        averageDistanceSameCluster = np.mean(arrayAverageDistanceSameCluster)
        averageDistanceNeighborCluster = np.mean(arrayAverageDistanceNeighborCluster)
        
        #plug into equation for silhouette coefficient
        currentSilhouetteCoefficient.append((averageDistanceNeighborCluster - averageDistanceSameCluster) / max(averageDistanceSameCluster, averageDistanceNeighborCluster))
        
        
    #add to overall array, for display in another function
    silhouetteCoefficients[c] = currentSilhouetteCoefficient
      
  return silhouetteCoefficients
  
  
#~~~~Displaying Results~~~~

#Run and Display results based on inputted data and number of clusters
#will also print out the average silhouette coefficients for each cluster for a particular k
def displayKMeansResults(X, k):
  kRangeArray = []
  for i in range(k):
    if (i >= 1):
      kRangeArray.append(i+1)
  silhouetteValAverage = {}    
  for i, k in enumerate(kRangeArray):
      fig, (ax1, ax2) = plt.subplots(1, 2)
      fig.set_size_inches(5, 3)
      
      # Run the Kmeans algorithm
      centroids, cluster = kmeans(X, k)
  
      # Get silhouette samples
      silhouette_vals = silhouetteAnalysis(X, centroids, cluster)
  
      # Silhouette plot
      y_ticks = []
      y_lower, y_upper = 0, 0
      for i in range(len(centroids)):
          cluster_silhouette_vals = silhouette_vals[i]
          cluster_silhouette_vals.sort()
          y_upper += len(cluster_silhouette_vals)
          ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
          ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
          y_lower += len(cluster_silhouette_vals)
  
      # Get the average silhouette score and plot it
      sumScore = 0;
      lenScore = 0;
      silhouetteValAverageArray = []
      for item in silhouette_vals:
        sumScore += sum(silhouette_vals[item])
        lenScore += len(silhouette_vals[item])
        silhouetteValAverageArray.append(sum(silhouette_vals[item]) / len(silhouette_vals[item]))
      avg_score = sumScore / lenScore
      ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
      ax1.set_yticks([])
      ax1.set_xlim([-0.1, 1])
      ax1.set_xlabel('Silhouette coefficient values')
      ax1.set_ylabel('Cluster labels')
      ax1.set_title('Silhouette plot for the various clusters', y=1.02);
      plt.tight_layout()
      plt.suptitle(f'Silhouette analysis using k = {k}',
                   fontsize=16, fontweight='semibold', y=1.05);        
      
      # Scatter plot of data colored with labels
      sns.scatterplot(X[:,0], X[:, 1], hue=cluster, ax = ax2)
      sns.scatterplot(centroids[:,0], centroids[:, 1], s=100, color='y', ax = ax2)      
      silhouetteValAverage[k] = silhouetteValAverageArray
  print(silhouetteValAverage)


#~~~~Running Program~~~~

goodData = goodDataIn('file:/Users/leshe/Downloads/Country-data.csv') 
badData = badDataIn('file:/Users/leshe/Downloads/Mall_Customers.csv')
findOptimalK(goodData)
displayKMeansResults(badDataIn, 4)



