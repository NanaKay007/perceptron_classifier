# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 10:42:24 2018

@author: jnran
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron
from plot_decision_regions import plot_decision_regions

data_file = pd.read_csv('iris.data',header=None)
data_file.tail()

"""
Training data set
"""
#[names]
y = data_file.iloc[:100, 4].values
y = np.where(y=='Iris-setosa',-1,1)

#[sepal length,petal length]
X = data_file.iloc[:100,[0,2]].values
plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1], color='blue', marker='x',label='veriscolor')

plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()


ppn = Perceptron()
#ppn.fit(X,y)
#plt.plot(range(1,len(ppn.errors_) + 1),ppn.errors_,marker='o')
#plt.xlabel('Epochs')
#plt.ylabel('Number of misclassifications')
#plt.show()

plot_decision_regions(X,y,classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()