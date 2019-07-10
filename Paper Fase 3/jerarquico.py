import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as jerarqui
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.metrics import silhouette_score

a = genfromtxt('data.csv', delimiter=',')
df=pd.DataFrame({'x':a[:,0],'y':a[:,1]})
xval=np.array(df['x'])
yval=np.array(df['y'])

fig, axes23 = plt.subplots(1, 1)

for axes in [axes23]:
    #z = jerarqui.linkage(a, method='single')
    ##Otros metodos
    z = jerarqui.linkage(a, method='complete')
    #z = jerarqui.linkage(a, method='average')
    #z = jerarqui.linkage(a, method='weighted')
    #z = jerarqui.linkage(a, method='centroid')
    #z = jerarqui.linkage(a, method='median')
    #z = jerarqui.linkage(a, method='ward')

    num_clust1 = 100

    label = jerarqui.fcluster(z, num_clust1, 'maxclust')
    label = np.array(label)
    print("Label para los: ",len(label))
    print("CLUSTER PARA 5 -----------------------------")
    for k in range(10):
         print(label[k])
    print("Longitud de los puntos: ",len(a))
    print("PUNTO Y CLUSTER PARA 5 -----------------------------")
    for j in range(10):
         print(a[j,0],' ',a[j,1],' ',label[j])
    clr = ['#2200CC' ,'#D9007E' ,'#FF6600' ,'#FFCC00' ,'#ACE600' ,'#0099CC' ,
    '#8900CC' ,'#FF0000' ,'#FF9900' ,'#FFFF00' ,'#00CC01' ,'#0055CC']
    print("PUNTO Y CLUSTER PARA TODOS -----------------------------")
    print(label.shape[0])
    #for i in range(label.shape[0]):
        #plt.scatter(a[i,0], a[i, 1],c=plt.cm.RdYlBu((label[i]/20)+0.04))
        #print(a[i,0],' ',a[i,1],' ',label[i])
    #plt.setp(axes, title='{} Clusters'.format(num_clust1))

#plt.show()

puntos=np.zeros(shape=(len(xval),2))
for i in range (len(xval)):
	puntos[i][0]=xval[i]
	puntos[i][1]=yval[i]

puntos = puntos.tolist()


##########################################################################################

#Metrica 1
clusters = dict()
centroids=dict()
for i in range (len(xval)):
	temp=label[i]-1
	if temp in clusters.keys():
		clusters[temp].append(puntos[i])
	else: 
        	clusters[temp]=[puntos[i]]


print (label)

sum_x=0.0
sum_y=0.0
temp_ptcentroid=np.zeros(shape=(len(label),2))
temp_ptcentroid = temp_ptcentroid.tolist()
for i in range (len(clusters)):
	temp=(clusters[i])
	#print(temp)
	for j in range (len(temp)):
		sum_x +=temp[j][0]
		sum_y +=temp[j][1]
	#print("sumx: ",sum_x)
	#print("sumy: ",sum_y)
	print("\n")
	print("Cluster: ",i)
	temp_ptcentroid[i][0]=(sum_x)/len(temp)
	temp_ptcentroid[i][1]=(sum_y)/len(temp)
	sum_x=0.0
	sum_y=0.0
	centroids[i]=temp_ptcentroid[i]
	print("i: ",i,centroids[i])
		
print("############################################")



dist = dict()
distancia=0.0
for i in range (len(centroids)):
	val_cx=centroids[i][0]
	val_cy=centroids[i][1]
	#print("Cluster: ",i)
	#print("Longitud: ",len(clusters[i]))
	for j in range (len(clusters[i])):
		temp=clusters[i]
		distancia=(abs(val_cx-temp[j][0])+abs(val_cy-temp[j][1]))
		if i in dist.keys():
			dist[i].append(distancia)
		else: 
	        	dist[i]=[distancia]


sumatoria=0.0
sum_temp=dict()
for i in range(len(dist)):
	temp=dist[i]
	for j in range(len(temp)):
		sumatoria +=temp[j]
	sum_temp[i]=sumatoria/len(temp)
	sumatoria=0.0

print("--------------------------------------")
sum_final=0.0
for key in sum_temp.keys():
	sum_final +=sum_temp[key]
print("Metrica 1: ",sum_final)

##########################################################################################

	

##########################################################################################
#Metrica 2
n_cluster=101
sum_metric2=0.0
for i in range (2,n_cluster):
	z = jerarqui.linkage(df, method='complete')
	lbl_cluster = jerarqui.fcluster(z, i , 'maxclust')
	#clusterer = KMeans(n_clusters=i)
	#cluster_labels = clusterer.fit_predict(df)
	#centers = clusterer.cluster_centers_
	score = silhouette_score (df, lbl_cluster)
	#print("El valor de silhouette es: ",score)
	sum_metric2 +=score
print("Metrica 2 : Para n_clusteres: ",i," el valor de silhouette es: ",sum_metric2/(i-1))

##########################################################################################

#Metrica 3
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
##########################################################################################



#################################################################################
#Metrica 4
temp=100;
n_clusters=100;
for i in range (n_clusters):
	if ((i in label) == False):
		temp-=1
	
print("Metrica 4: ",temp/n_clusters)	
####################################################################################


