#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sophie j
"""

import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


c528 = pd.read_csv('COSC528data_set.csv')

X=c528.iloc[:,0:2];
y=c528.iloc[:,2];
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)
#random state added for reproducibility 
y_train=y

#plot the decision tree
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X_train, y_train)
plt.figure()
plot_tree(clf, filled=True)

sns.relplot( data=c528, x="#_hr_self-study", y="#_attended_lectures", hue="K_score", size="K_score", palette="bright" ).set(title='Knowledge Scores')
#add the regions of the decision tree
plt.plot([3.5,3.5 ] , [1,5] , linewidth=2, color='red') 
plt.plot([3.5, 5 ] , [4.5,4.5] , linewidth=2, color='red') 
plt.plot([1, 3.5 ] , [3.5,3.5] , linewidth=2, color='red') 


#%%
#re-define y for classification
y=c528.iloc[:,3];
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
#random state added for reproducibility 

X_train=X
y_train=y

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

plt.figure()
plot_tree(clf, filled=True)

sns.relplot( data=c528, x="#_hr_self-study", y="#_attended_lectures", hue="Pass1_Fail0", size="Pass1_Fail0", palette="bright" ).set(title='Pass/Fail')
plt.plot([3.5,3.5 ] , [1,5] , linewidth=2, color='red') 
plt.plot([1, 3.5 ] , [3.5,3.5] , linewidth=2, color='red') 



