 
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split




import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from pandas.tools.plotting import scatter_matrix
import scipy.stats as stats
import pylab as pl
from pandas import Series
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.feature_selection import RFE
from tkinter import filedialog
from tkinter import*
import random
import time
import datetime

from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import csv

from random import seed
from random import randrange
from csv import reader
import csv
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas.tools.plotting import scatter_matrix



mse=[]
mae=[]
rsq=[]
rmse=[]
acy=[]


data = pd.read_csv('data.csv')
print(data.head())

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


clf = linear_model.LogisticRegression(C=1e5)
sclf = svm.LinearSVC()
yyy=""
yyysvm="";


dataset=pd.read_csv('data.csv').values

x1 = dataset[:,0:13]
y1 = dataset[:,14] # define the target variable (dependent variable) as y
print(x1)
print(y1)
X, XX, y, YY = train_test_split(x1, y1)



print(X) 
print(y) 

print(XX) 
print(YY)

clf.fit(X,y)
c
yyylog = clf.predict(X)
print(yyylog)


sclf.fit(X,y)
yysvm = sclf.predict(XX)
yyysvm = sclf.predict(X)
print(yyysvm)
   

    
result1=open("resultLogisticRegression.csv","w")
result1.write("ID,Predicted Value" + "\n")
for j in range(len(yylog)):
    result1.write(str(j+1) + "," + str(yylog[j]) + "\n")
result1.close()


result2=open("resultSVM.csv","w")
result2.write("ID,Predicted Value" + "\n")
for j in range(len(yysvm)):
    result2.write(str(j+1) + "," + str(yysvm[j]) + "\n")
result2.close()



print("---------------------------------------------------------")
print("MSE VALUE FOR LogisticRegression IS %f "  % mean_squared_error(YY, yylog))
print("MAE VALUE FOR LogisticRegression IS %f "  % mean_absolute_error(YY, yylog))
print("R-SQUARED VALUE FOR LogisticRegression IS %f "  % r2_score(YY, yylog))
rms = np.sqrt(mean_squared_error(YY, yylog))
print("RMSE VALUE FOR LogisticRegression IS %f "  % rms)
ac=accuracy_score(YY,yylog)
print ("ACCURACY VALUE LogisticRegression IS %f" % ac)
print("---------------------------------------------------------")
mse.append(mean_squared_error(YY, yylog))
mae.append(mean_absolute_error(YY, yylog))
rsq.append(r2_score(YY, yylog))
rmse.append(rms)
acy.append(ac)	 
    


print("---------------------------------------------------------")
print("MSE VALUE FOR SVM IS %f "  % mean_squared_error(YY, yysvm))
print("MAE VALUE FOR SVM IS %f "  % mean_absolute_error(YY, yysvm))
print("R-SQUARED VALUE FOR SVM IS %f "  % r2_score(YY, yysvm))
rms = np.sqrt(mean_squared_error(YY, yysvm))
print("RMSE VALUE FOR SVM IS %f "  % rms)
ac=accuracy_score(YY,yysvm)
print ("ACCURACY VALUE SVM IS %f" % ac)
print("---------------------------------------------------------")
mse.append(mean_squared_error(YY, yysvm))
mae.append(mean_absolute_error(YY, yysvm))
rsq.append(r2_score(YY, yysvm))
rmse.append(rms)
acy.append(ac)



from pandas import read_csv
data = read_csv('resultLogisticRegression.csv', header=0)
data = data.dropna()
names=list(data.columns)
 

#Barplot for the dependent variable
fig = plt.figure(0)
fig.canvas.set_window_title('Predicted Value For Regression')
sns.barplot(y='Predicted Value',x='ID', data=data, palette='hls')
plt.xlabel('ID')
plt.ylabel('AHD')
plt.title("Predicted Value For Regression");
fig.savefig('Predicted Value For Regression.png')
plt.show()
   
data1 = read_csv('resultsvm.csv', header=0)
data1 = data1.dropna()
names=list(data1.columns)
   
   
   
#Barplot for the dependent variable
fig = plt.figure(0)
fig.canvas.set_window_title('Predicted Value For SVM')
sns.barplot(y='Predicted Value',x='ID', data=data1, palette='hls')
plt.xlabel('ID')
plt.ylabel('AHD')
plt.title("Predicted Value For SVM");
fig.savefig('Predicted Value For SVM.png')
plt.show()
    
al = ['Logistic Regression', 'SVM']
    
    
result2=open('MSE.csv', 'w')
result2.write("Algorithm,MSE" + "\n")
for i in range(0,len(mse)):
    result2.write(al[i] + "," +str(mse[i]) + "\n")
result2.close()
    
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
explode = (0.1, 0, 0, 0, 0)  
       
    
#Barplot for the dependent variable
fig = plt.figure(0)
df =  pd.read_csv('MSE.csv')
acc = df["MSE"]
alc = df["Algorithm"]
plt.bar(alc,acc,align='center', alpha=0.5,color=colors)
plt.xlabel('Algorithm')
plt.ylabel('MSE')
plt.title("MSE Value");
fig.savefig('MSE.png')
plt.show()
    
    
    
result2=open('MAE.csv', 'w')
result2.write("Algorithm,MAE" + "\n")
for i in range(0,len(mae)):
    result2.write(al[i] + "," +str(mae[i]) + "\n")
result2.close()
                
fig = plt.figure(0)            
df =  pd.read_csv('MAE.csv')
acc = df["MAE"]
alc = df["Algorithm"]
plt.bar(alc,acc,align='center', alpha=0.5,color=colors)
plt.xlabel('Algorithm')
plt.ylabel('MAE')
plt.title('MAE Value')
fig.savefig('MAE.png')
plt.show()
    
result2=open('R-SQUARED.csv', 'w')
result2.write("Algorithm,R-SQUARED" + "\n")
for i in range(0,len(rsq)):
    result2.write(al[i] + "," +str(rsq[i]) + "\n")
result2.close()
            
fig = plt.figure(0)        
df =  pd.read_csv('R-SQUARED.csv')
acc = df["R-SQUARED"]
alc = df["Algorithm"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
explode = (0.1, 0, 0, 0, 0)  
plt.bar(alc,acc,align='center', alpha=0.5,color=colors)
plt.xlabel('Algorithm')
plt.ylabel('R-SQUARED')
plt.title('R-SQUARED Value')
fig.savefig('R-SQUARED.png')
plt.show()
    
result2=open('RMSE.csv', 'w')
result2.write("Algorithm,RMSE" + "\n")
for i in range(0,len(rmse)):
    result2.write(al[i] + "," +str(rmse[i]) + "\n")
result2.close()
      
fig = plt.figure(0)    
df =  pd.read_csv('RMSE.csv')
acc = df["RMSE"]
alc = df["Algorithm"]
plt.bar(alc, acc, align='center', alpha=0.5,color=colors)
plt.xlabel('Algorithm')
plt.ylabel('RMSE')
plt.title('RMSE Value')
fig.savefig('RMSE.png')
plt.show()
    
result2=open('Accuracy1.csv', 'w')
result2.write("Algorithm,Accuracy" + "\n")
for i in range(0,len(acy)):
    result2.write(al[i] + "," +str(acy[i]) + "\n")
result2.close()
    
fig = plt.figure(0)
df =  pd.read_csv('Accuracy1.csv')
acc = df["Accuracy"]
alc = df["Algorithm"]
plt.bar(alc, acc, align='center', alpha=0.5,color=colors)
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('Accuracy Value')
fig.savefig('Accuracy.png')
plt.show()
    
