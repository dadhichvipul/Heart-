from sklearn.cluster import KMeans
import csv

from random import seed
from random import randrange
from csv import reader
import csv
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix


#id	Age	Sex	ChestPain	RestBP	Chol	Fbs	RestECG	MaxHR	ExAng	Oldpeak	Slope	Ca	Thal	AHD

data = pd.read_csv('data.csv')
print(data.head())

names=list(data.columns)



names=list(data.columns)



correlations = data.corr()
# plot correlation matrix
fig = plt.figure()
fig.canvas.set_window_title('Correlation Matrix')
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
#fig.savefig('Correlation Matrix.png')
    
 
#scatterplot
scatter_matrix(data)
    
plt.show()

ncols=3
plt.clf()
f = plt.figure(1)
f.suptitle(" Data Histograms", fontsize=12)
vlist = list(data.columns)
nrows = len(vlist) // ncols
if len(vlist) % ncols > 0:
	nrows += 1
for i, var in enumerate(vlist):
	plt.subplot(nrows, ncols, i+1)
	plt.hist(data[var].values, bins=15)
	plt.title(var, fontsize=10)
	plt.tick_params(labelbottom='off', labelleft='off')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()



k=3
kmeans = KMeans(n_clusters=k).fit(data)
labels = kmeans.labels_
print(labels)
centroids = kmeans.cluster_centers_
print("Centroids Value")
print(centroids)    	





ay1=[]
ay2=[]
ay3=[]
N=1
for i in range(0,len(centroids[0])):
    ay1 = np.append(ay1,0.6 + 0.6 * np.random.rand(N))
    ay2 = np.append(ay2,0.4+0.3 * np.random.rand(N))
    ay3 = np.append(ay3,0.3*np.random.rand(N))


datax = (centroids[0],centroids[1],centroids[2])
datay = (ay1,ay2,ay3)
colors = ("red", "green", "blue")
groups = ("Cluster1", "Cluster2", "Cluster3") 
 
# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
 
for datx, daty, color, group in zip(datax,datay, colors, groups):
    x, y = datx,daty
    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
 
plt.title('Cluster Centroids')
plt.legend(loc=2)
plt.show()


clusters = {}
n = 0

for item in labels:
	if item in clusters:
		clusters[item].append(data.iloc[n])
	else:
		clusters[item] = [data.iloc[n]]
	n +=1
	
with open('Cluster0.csv','w',newline='') as file:
    writer = csv.writer(file, delimiter=',')
    for line in clusters[0]:
    	writer.writerow(line)
    	
with open('Cluster1.csv','w',newline='') as file:
    writer = csv.writer(file, delimiter=',')
    for line in clusters[1]:
    	writer.writerow(line)	


with open('Cluster2.csv','w',newline='') as file:
    writer = csv.writer(file, delimiter=',')
    for line in clusters[2]:
    	writer.writerow(line)	
    	
print("Finished")    	   	