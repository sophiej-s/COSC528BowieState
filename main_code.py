#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sophie j
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
import itertools
from sklearn.metrics import confusion_matrix


#%%
# Reference: 
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
    #    print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



#%%Load data
spam_data = pd.read_csv('spambase_wnames.data')
X=spam_data.iloc[:,0:57];
y=spam_data.iloc[:,57];

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Decision Tree 
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)
plt.figure()
plot_tree(clf, filled=True)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred) 
print(accuracy_score(y_test, y_pred)  )

cnf_matrix1 = confusion_matrix(y_test,y_pred,labels=[1,0])

#Random Forest  
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred) 
cnf_matrix2 = confusion_matrix(y_test,y_pred,labels=[1,0])


#------begin scikit-learn.org borrowed code------
#https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
import numpy as np
importances = clf.feature_importances_
std = np.std([ tree.feature_importances_ for tree in clf.estimators_], axis=0)
feature_names = [f'feature {i}' for i in range(X.shape[1])]
feature_names_spam=spam_data.columns.values.tolist()
feature_names_spam.pop()

forest_importances = pd.Series(importances, index=feature_names_spam)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
#------end scikit-learn.org borrowed code------



#AdaBoost 
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred) 
cnf_matrix3 = confusion_matrix(y_test,y_pred,labels=[1,0])

#Gradient Boosting 
clf = GradientBoostingClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

cnf_matrix4 = confusion_matrix(y_test,y_pred,labels=[1,0])


plt.subplot(2,2, 1)
plot_confusion_matrix(cnf_matrix1, classes=['Email','Spam'],title='Decision Tree')

plt.subplot(2,2, 2)
plot_confusion_matrix(cnf_matrix2, classes=['Email','Spam'],title='Random Forest')

plt.subplot(2,2, 3)

plot_confusion_matrix(cnf_matrix3,  classes=['Email','Spam'],title='AdaBoost')

plt.subplot(2,2, 4)
plot_confusion_matrix(cnf_matrix4, classes=['Email','Spam'],title='Gradient Boosting')









