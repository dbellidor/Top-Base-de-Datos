import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy import genfromtxt
from sklearn.metrics import silhouette_score
import math 

data = genfromtxt('data.csv', delimiter=',')
df=pd.DataFrame({'x':data[:,0],'y':data[:,1]})

xval=np.array(df['x'])
yval=np.array(df['y'])

#kmeans = KMeans(n_clusters=100,init='random',n_init=1, max_iter=10,tol=0.0001,precompute_distances='False'
kmeans = KMeans(n_clusters=100,init='random',n_init=1,max_iter=5,tol=0.5)
kmeans.fit(df)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_

colmap = {1: 'r', 2: 'g', 3: 'b'}
colors = map(lambda x: colmap[x+1], labels)
lbl=np.array(labels)

print("Label cluster para : ",len(labels),"datos")
for i in range(10):
	print("i: ",i,labels[i])

print("Numero de centroides: ",len(centroids))
#for i in (len(centroids)):

print ("0: ",centroids[0][0]," ",centroids[0][1])
print ("1: ",centroids[1][0]," ",centroids[1][1])
print ("2: ",centroids[2][0]," ",centroids[2][1])
print ("3: ",centroids[3][0]," ",centroids[3][1])
print ("4: ",centroids[4][0]," ",centroids[4][1])
print ("5: ",centroids[5][0]," ",centroids[5][1])


###Grafica


print(xval.shape[0])
for i in range(xval.shape[0]):
    plt.scatter(xval[i],yval.max(axis=0)-yval[i],c=plt.cm.RdYlBu((lbl[i]/20)+0.04))
    print(xval[i],' ',yval[i],' ',lbl[i]+1)

plt.show()



##########################################################################################
#Metrica 1
dist = dict()
distancia=0.0
for i in range (len(xval)):
	#(abs(xval[i]-centroids[i][0])+ abs(yval[i]-centroids[i][1]))
	get_centroid=labels[i]
	#print("Get centroide: ",get_centroid)
	val_cen=centroids[get_centroid]
	#print("x: ",xval[i],"y: ",yval[i]," label: ",labels[i]," ","centroid: ",val_cen[0]," ",val_cen[1])
	#Manhattan
	distancia=(abs(val_cen[0]-xval[i])+abs(val_cen[1]-yval[i]))
	#print("x: %.2f y: %.2f Dist: %.5f" % (xval[i],yval[i],distancia))
	#Euclideana
	#distancia=math.sqrt(pow(val_cen[0]-xval[i],2) + pow(val_cen[1]-yval[i],2))
	if get_centroid in dist.keys():
		dist[get_centroid].append(distancia)
	else: 
        	dist[get_centroid]=[distancia]

sumatoria=0.0
sum_temp=dict()
for key in dist.keys():
	#print("Tam: ",len(dist[key]))
	for value in dist[key]:
		sumatoria +=value
		#print (key," -- ",value)
	sum_temp[key]=sumatoria/len(dist[key])
	#print("Suma de cluster: ",key," : ",sum_temp[key])
	sumatoria=0.0
	
print("--------------------------------------")
sum_final=0.0
for key in sum_temp.keys():
	sum_final +=sum_temp[key]
	#print (key," -- ",sum_temp[key])
print("Metric 1: ",sum_final)

			


##########################################################################################
#Metrica 2
n_cluster=101
sum_metric2=0.0
for i in range (2,n_cluster):
	clusterer = KMeans(n_clusters=i)
	cluster_labels = clusterer.fit_predict(df)
	centers = clusterer.cluster_centers_
	score = silhouette_score (df, cluster_labels)
	sum_metric2 +=score
print("Metrica 2 : Para n_clusteres: ",i," el valor de silhouette es: ",sum_metric2/(i-1))

##########################################################################################

sumatoria3=0.0
sum_temp3=dict()
for i in range(len(dist)):
	temp3=dist[i]
	for j in range(len(temp3)):
		sumatoria3 +=temp3[j]
	sum_temp3[i]=sumatoria3
	sumatoria3=0.0

print("--------------------------------------")
sum_final3=0.0
for key in sum_temp3.keys():
	sum_final3 +=sum_temp3[key]
print("Metrica 3: ",sum_final3/len(xval))


#Metrica 3
##########################################################################################
sumatoria3=0.0
sum_temp3=dict()
for key in dist.keys():
	#print("Tam: ",len(dist[key]))
	for value in dist[key]:
		sumatoria3 +=value
		#print (key," -- ",value)
	sum_temp3[key]=sumatoria3
	#print("Suma de cluster: ",key," : ",sum_temp[key])
	sumatoria3=0.0
	
print("--------------------------------------")
sum_final3=0.0
for key in sum_temp3.keys():
	sum_final3 +=sum_temp3[key]
	#print (key," -- ",sum_temp[key])

print("Metric 3: ",sum_final3/len(xval))
##########################################################################################



#Metrica 4
temp=100;
##########################################################################################
n_clusters=100;
for i in range (n_clusters):
	if ((i in lbl) == False):
		temp-=1
	
print("Metrica 4: ",temp/n_clusters)	

##########################################################################################


