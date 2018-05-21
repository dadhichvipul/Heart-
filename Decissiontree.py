from random import seed
from random import randrange
from csv import reader
import csv
import numpy as np
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




ac=[]
 
# Load a CSV file
def load_csv(filename):
	file = open('c0.csv', "rb")
	lines = reader(file)
	print(lines)
	dataset = list(lines)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, clust, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		#train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		
		
		if clust == 0:
			with open('Cluster0-Train-Set.csv','w',newline='') as file:
				writer = csv.writer(file, delimiter=',')
				for line in train_set:
					writer.writerow(line)
			with open('Cluster0-Test-Set.csv','w',newline='') as file:
				writer = csv.writer(file, delimiter=',')
				for line in test_set:
					writer.writerow(line)
			with open('Cluster0-predicted.csv','w',newline='') as file:
				for line in predicted:
					file.write(str(line))
					file.write('\n')
		if clust == 1:
			with open('Cluster1-Train-Set.csv','w',newline='') as file:
			++	writer = csv.writer(file, delimiter=',')
				for line in train_set:
					writer.writerow(line)
			with open('Cluster1-Test-Set.csv','w',newline='') as file:
				writer = csv.writer(file, delimiter=',')
				for line in test_set:
					writer.writerow(line)
			with open('Cluster1-predicted.csv','w',newline='') as file:
				for line in predicted:
					file.write(str(line))
					file.write('\n')
		if clust == 2:
			with open('Cluster2-Train-Set.csv','w',newline='') as file:
				writer = csv.writer(file, delimiter=',')
				for line in train_set:
					writer.writerow(line)
			with open('Cluster2-Test-Set.csv','w',newline='') as file:
				writer = csv.writer(file, delimiter=',')
				for line in test_set:
					writer.writerow(line)
			with open('Cluster2-predicted.csv','w',newline='') as file:
				for line in predicted:
					file.write(str(line))
					file.write('\n')
		
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores
 
# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right
 
# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini
 
# Select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)
 
# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root
 
# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']
 
# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
		
	return(predictions)
 
# Test CART on  dataset
seed(1)

reader=csv.reader(open("Cluster0.csv","r"),delimiter=",")
X=list(reader)
X=np.array(X)
dataset=X.astype(np.float)



# evaluate algorithm
n_folds = 5
max_depth = 5
min_size = 10
clust = 0
scores = evaluate_algorithm(dataset, decision_tree, n_folds, clust, max_depth, min_size)
#print('Scores: %s' % scores)
a1=sum(scores)/float(len(scores))
ac.append(a1)
print('Mean Accuracy for Cluster0: %.3f%%' % (sum(scores)/float(len(scores))))




reader=csv.reader(open("Cluster1.csv","r"),delimiter=",")
X=list(reader)
X=np.array(X)
dataset=X.astype(np.float)


# evaluate algorithm
n_folds = 5
max_depth = 5
min_size = 10
clust = 1
scores = evaluate_algorithm(dataset, decision_tree, n_folds, clust, max_depth, min_size)
#print('Scores: %s' % scores)
a2=sum(scores)/float(len(scores))
ac.append(a2)
print('Mean Accuracy for Cluster1: %.3f%%' % (sum(scores)/float(len(scores))))


reader=csv.reader(open("Cluster2.csv","r"),delimiter=",")
X=list(reader)
X=np.array(X)
dataset=X.astype(np.float)


# evaluate algorithm
n_folds = 5
max_depth = 5
min_size = 10
clust = 2
scores = evaluate_algorithm(dataset, decision_tree, n_folds, clust, max_depth, min_size)
#print('Scores: %s' % scores)
a3=sum(scores)/float(len(scores))
ac.append(a3)
print('Mean Accuracy for Cluster2: %.3f%%' % (sum(scores)/float(len(scores))))


al = ['Cluster0', 'Cluster1', 'Cluster2']


result2=open('Accuracy.csv', 'w')
result2.write("Cluster,Accuracy" + "\n")
for i in range(0,len(ac)):
    print(ac[i])
    print(al[i])
    result2.write(al[i] + "," +str(ac[i]) + "\n")
result2.close()


df =  pd.read_csv('Accuracy.csv')
acc = df["Accuracy"]
alc = df["Cluster"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
explode = (0.1, 0, 0, 0, 0)  


plt.bar(alc, acc, align='center', alpha=0.5,color=colors)
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('Accuracy Value')
 
plt.show()
