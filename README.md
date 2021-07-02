# COSC528BowieState


# Decision Trees and Ensemble Methods 
## Abstract 
In this paper, we examine decision trees and their enhancements, known as ensemble methods, which are supervised learning algorithms that closely mimic human decision making. Due to their interpretability and computationally affordable costs, decision trees supply simple and useful interpretations of data and are suitable for both regression and classification problems. Decision trees, however, generally suffer from overfitting and have lower accuracy than other supervised learning methods. Tree ensemble methods (bootstrap aggregation or bagging and boosting) are enhancements of the decision tree algorithm. They can often improve accuracy and decrease overfitting by combining information from trees grown from segments of the data set. Bootstrapping can be thought of as combining tree models in parallel where the result is found by averaging or a majority vote. Boosting can be thought of as a sequential tree method in which a next tree is grown using information from the previously grown trees. Decision trees and ensemble methods are incorporated into modern open-source software and widely used by data scientists and machine learning professionals. We will use python and scikit-learn library to show how these tree methods can be used to solve a spam classification problem. 


## Decision Trees: Intuition
Decision trees are method of recursively segmenting  a set of data into multiple distinct and non-overlapping sub-sets (or regions) based on setting feature thresholds determining the segmentations. Decision trees allow making predictions for new observations based on estimating the mean value (in regression problems) or most common class (in classification problems) in a region where the future data point would be placed.
To build our intuition for trees, we begin with an example of a relationship where two features (the number of hours of self-study and the number of attended lectures) predict a student knowledge score, or passing/failing a course. The knowledge score can take values between 0 and 100 and the passing or failing is represented as 0 for failing and 1 for passing. Figures 1-2 show decision trees for the data and plots of the data with superimposed tree regions to illustrate the concept. The data are self-generated and was intentionally made simple. The data as well as python code used to generate Figures 1-2 are shown in Appendix A.

![image](https://user-images.githubusercontent.com/20401990/124323867-fe489100-db4f-11eb-8b8c-c953423c93d5.png)
![image](https://user-images.githubusercontent.com/20401990/124323881-043e7200-db50-11eb-9a7b-49288c35f02a.png)
*Figure 1. Top plot: knowledge score as a function of the number of attended lectures and hours of self-study. The red lines are add to illustrate the segmentation of the data into three regions. Bottom plot: decision tree.*


Like other supervised learning algorithms, decision trees have a number of benefits and drawbacks. The benefits of trees are ([2]):
* Trees are explainable, easily interpretable
* Trees can be displayed graphically
* Trees mirror human decision-making (this can be an impression or intuition possibly because trees are interpretable and easy to understand) 

Trees also have drawbacks; they are ([2]):
* Trees are generally prone to overfitting (higher variance)
* Trees have lower accuracy than other supervised learning algorithms

To address the issue of overfitting and sup-optimal accuracy, tree ensemble algorithms (bagging and boosting) can be utilized.


## Regression Decision Trees: Formal Introduction

To derive a regression decision tree, we divide the data set  into distinct regions and estimate a future observation from a given region as the mean of the data points of that region. Predictions for new observations in regression problems are based on estimating the mean value in the region, in which the data point belongs. When should these splits be performed? Ideally, the goal should be to minimize the square error between the calculated mean (y-hat) in each region and the training data points (y) in that region:
Thus we want to minimize the following:
![image](https://user-images.githubusercontent.com/20401990/124324034-449df000-db50-11eb-99b7-349a3c047195.png) ([1], [6])

This is computationally costly because we would be forced to consider every possible way to segment the training data set.  Therefore, we will use recursive binary splitting instead where, at each split, we only minimize errors in two resulting regions due to that particular split, essentially needing to minimize the following (in each step):

![image](https://user-images.githubusercontent.com/20401990/124324068-597a8380-db50-11eb-92c4-97db5bd36c34.png). [1]

We continue the binary splitting (using the above criterium) until a stopping criterium (i.e., tree depth or the number of samples in each leaf) is met. 
One of the drawbacks of trees is that they can overfit (grow excessively deep and complex in an attempt to model the training data) thus suffering from high variance and high model complexity. To address that, a stopping criterium can be tightened (for example, one can decrease the tree depth) ([2]). Another solution is to grow a complex tree initially and then perform tree using the cost complexity method. In the cost complexity method, we do not consider every possible sub-tree. Instead, we create a sequence of trees associated with a nonnegative tuning parameter α to which we apply cross-validation  to select a sub tree with an optimal prediction error ([1], [2], [6]).

## Classification Decision Trees: Formal Introduction

To derive a classification outcome, we follow a similar procedure as before: we recursively split our data into two regions until a stopping criterium is met. Unlike in regression trees, predictions for new classification values is based on the most commonly occurring class of training observations in the region to which the new value belongs ([6]). To guide our decisions in tree growing and data splitting, we use recursive binary splitting that minimizes either misclassification error rate, Ginni index or Entropy scores.

Misclassification error rate E represents the fraction of the training observations in that region that do not belong to the most common class:
![image](https://user-images.githubusercontent.com/20401990/124324112-6eefad80-db50-11eb-8795-9cd71f7325c1.png). ([1], [5], [6])

In the formula for E, p-hatmk represents the proportion of training data in the “m”-th region that are from the “k”-th class, i.e., the proportion of the correctly classified data points ([1]),  and subtracting the value from one is the error rate. 
The formulas for the Ginnie and Entropy scores are as follows:

![image](https://user-images.githubusercontent.com/20401990/124324152-7fa02380-db50-11eb-8832-91dc8925d309.png)  ([1], [5], [6])

![image](https://user-images.githubusercontent.com/20401990/124324170-87f85e80-db50-11eb-90a4-197f0ef1bea8.png). ([1], [5], [6])


The Ginnie scope G and entropy score D represent a split purity score because G and D are small when the proportion of correctly classified data points, p-hatmk,  is 1 or 0 ([1]). Ginnie score is used in tree building in scikit-learn library which can be seen when a tree is graphically plotted for  visual inspection ([2]). In tree pruning, minimization of the misclassification error rate is typically used since the goal is to have nodes with low misclassification rate ([1]).

## Tree Ensemble Method: Bagging

Bootstrap aggregation, or bagging, is a method for reducing the variance in decision trees that relies on drawing a sub-set of samples from the master training set and training decision trees on these sub-sets. The resulting information is combined the following way: we average the prediction (for regression problems) or take a majority vote for the predicted class (for classification problem) ([6]). 
Bagging relies on the fact that in a set of n independent observations Z1,...,Zn, each with variance σ2 ; the variance of the mean of the Zs is σ2/n ([1]). When we draw data points with replacement, the data points among the sub-sets not independent since some data points will occur in a number of sub-sets. We expect some correlation “p” among the sub-sets; therefore, in the tree bagging method, the variance of the mean of the Zs sub-sets is pσ2 + σ2(1-p)/n ([3]).

Graphically interpreting bagged decision trees becomes not an easy task because a number of trees are combed and visually tracing splits is not possible. To gain an intuitive understanding, one can analyze the relative importance of each feature by analyzing the decreases in the squared error (for regression trees) or Ginnie scores due to tree splits over the features (for all used trees), where a large value indicates a large decrease implying the corresponding feature was important. 

A further improvement of bagging is the random forest method. It allows bagged decision tree models to draw features that are more de-correlated by further restricting bagging by allowing only sampling of a sub-set of features from the total number of features each time a split in a tree is considered ([6]). This approach allows to improve models of relationships that include a strong feature (predictor). The strong feature would dominate bagged trees since it would likely be relied on as one of the top predictors (i..e, be near the top of bagged trees) thus not decreasing the variance in the resulting model ([1]). In random forest, we are restricting our selection of features (“m” features out of “p” total features), thus “on average (p − m)/p of the splits will not even consider the strong predictor” ([1]).

## Tree Ensemble Method: Boosting 

Unlike bagging (which can be thought of as a method of growing trees in parallel), boosting is a sequential method where subsequent trees are built using information from previous trees. Moreover, in boosting, the whole training data set is used (in place of sub-sets that are used in bagging). Subsequent trees are trained on modified training data though; the data points that were not correctly identified by the previous model are assigned larger weights so that the present model better captures the relationship. The output is a combination of the initial tree and re-weighted subsequent trees (built upon modified training data):

<img width="163" alt="image" src="https://user-images.githubusercontent.com/20401990/124324269-abbba480-db50-11eb-86a5-72ff510e9163.png"> ([6])


In the equation, f-hat b(xi) consists of “b” trees trained on re-weighted versions of the training set, and λ  is the weight of each additive (subsequent) tree in the final model ([1]). Noteworthy, in boosting trees of depth 1 are recommended to be one (i.e., trees become stumps) ([1] and [3]). An intuitive explanation for why stumps are effective is that each subsequent tree captures a relationship involving one feature at a time allowing the overall model to learn slowly adding these relationship one at a time.

## Spam Classification Example Using scikit-learn Library


To examine an application of the decision trees and their ensemble methods, we will examine the binary classification problem for classifying an incoming email as either spam or legitimate email using the data set “SPAM E-mail Database” from the UCI Irvine ([7]). Applications of decision trees to this data set were demonstrated in [4]. In this paper, we will apply four algorithms (single decision tree, random forest and two boosting algorithms) to the data set and compare them.

The data contains 4601 tuples representing emails, 1813 (39.4%) of which are classified as spam; the data includes 46 features such as frequencies of certain words (i.e., “remove”, “business”, “free”) or characters (i.e., “$”, “!”, “:”) as well as other quantitative scores such as the length of the longest uninterrupted (CAPMAX), the sum of the length of uninterrupted sequences of capital (CAPTOT), and the average length of uninterrupted sequences of capital letters (CAPAVE).

We randomly split the data set into a training set (3680 data tuples, or 80% of the data) and a test set (921 data tuples, or 20% of the data).  Building a model using the “DecisionTreeClassifier” of the scikit-learn library and testing it on the test set shows the accuracy ranging from ~0.81 (with the tree depth of one, i.e., a stump tree) to ~0.92 (with the tree depth of seven to eight). Without any depth restriction, the tree grows excessively large (left, Figure 3) while producing a sub-optimal accuracy (the accuracy is ~0.9 and is lower than of a seven- to eight-level tree). Limiting the tree to three levels, allows us to see the features determined to be most important in the algorithm (right, Figure 3); these are the frequency of character “$” (feature 52), frequency of word “remove” (feature 6), frequency of “hp”(feature 24),  frequency of character “!” (feature 51), frequency of word “george” (feature 26), frequency of word “edu” (feature 45).


![image](https://user-images.githubusercontent.com/20401990/124324318-c2fa9200-db50-11eb-9275-cd787c7e3689.png)

![image](https://user-images.githubusercontent.com/20401990/124324332-c8f07300-db50-11eb-8328-731737798ec4.png)
*Figure 2. Top: DecisionTreeClassifier produces an excessively deep tree using the training set. Bottom: a three-level tree allows us to examine the six critical features in the data set.*


Using the random forest method “RandomForestClassifier” allows us to improve the accuracy to ~0.95. We plot the  feature importance (Figure 4) and notice that a partial overlap with most important features noted in Figure 1; these are frequency of words “remove”, “free”, the frequency of characters “$” and “!”.

![image](https://user-images.githubusercontent.com/20401990/124324363-d60d6200-db50-11eb-9b86-92bfb18a4562.png)
*Figure 4.  Plotting feature importance allows us to note the features deemed important by the random forest algorithm.*


Using the “AdaBoostClassifier” produces the accuracy  of ~0.93, while the gradient boosting algorithm “GradientBoostingClassifier” produces the accuracy of ~0.95.  Based on the accuracies, we note the gradient boosting and the random forest algorithms performed better than a single tree or ADA-boost. To further examine the four algorithms’ performance on the data, we also examine their confusion matrices and confirm the gradient boosting and random forest have the best chance of correctly identifying both true negatives and true positives (Figure 5). 


![image](https://user-images.githubusercontent.com/20401990/124324412-e6bdd800-db50-11eb-9dff-de76fd4eff38.png)

*Figure 5. Confusion matrices show the count of correctly classified and misclassified samples.*

Python code used to analyze the “SPAM E-mail Database” data, generate Figures 3-5 and calculate accuracies for the four algorithms is shown in Appendix B.

## Conclusion

The gradient boosting and random forest algorithms have the best chance of correctly identifying both true negatives and true positives in the data set. 

## References
[1] Gareth James,  Daniela Witten Trevor Hastie, Robert Tibshirani, “An Introduction to Statistical Learning with Applications in R”, 7th ed.

[2] scikit-learn.org, “Decision Trees — scikit-learn 0.24.2 documentation”, https://scikit-learn.org/stable/modules/tree.html 

[3] Stanford CS229 Lecture 10 “Decision Trees and Ensemble Methods: Machine Learning (Autumn 2018)” at https://www.youtube.com/watch?v=wr9gUr-eWdA&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=10/ 

[4]  Trevor Hastie, “Boosting”, https://www.cc.gatech.edu/~hic/CS7616/pdf/lecture5.pdf

[5] Trevor Hastie, Robert Tibshirani, Jerome Friedman, “The Elements of Statistical Learning: Data Mining, Inference, and Prediction”, 12th ed., 2017. Springer.

[6] Trevor Hastie, “Tree-based methods”, https://web.stanford.edu/~hastie/MOOC-Slides/trees.pdf

[7] UCI Machine Learning Repository, “Spambase Data Set”,  https://archive.ics.uci.edu/ml/datasets/spambase

[8]   scikit-learn.org, “Confusion matrix”, https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

[9] scikit-learn.org, “Feature importances with a forest of trees”, https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html 


## Appendix A
* Self-generated data used to produce Figures 1-2 is COSC528data_set.csv and is be located in this repo. 
* Python code used for generating Figures 1-2 is COSC528parser.py and is be located in this repo.


## Appendix B
* Python code used to process the “SPAM E-mail Database” and generate Figures 3-5 is main_code.py and is be located in this repo.
*The plot_confusion_matrix() function is taken from the scikit-learn.org website, which is also noted within the code. A portion of the syntax for plotting feature importnace in random forest algorithm is also taken from scikit-learn.org, which is also noted within the code.*
*  The data is spambase_wnames.data and is be located in this repo. The data source is “Spambase Data Set”,  https://archive.ics.uci.edu/ml/datasets/spambase.










