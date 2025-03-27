---
layout: post
title: gbm Implementation
date: 2023-01-26
categories: coding
tags: AI GBM
---
# Implementation of Friedman's GBM with Custom Objective

In this notebook, I figure out the hacks needed to implement Friedman's original GBM algorithm using sklearn DecisionTreeRegressor as the weak learner and scipy minimize as the argmin method.
Basically we just need to be able to modify the tree predictions to predict the best prediction value according to the argmin of the loss function.
This page on the [decision tree structure](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html)  in the sklearn documentation is super helpful.

## sklearn decision trees


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
%matplotlib inline

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
```

## Prepare the data ##


```python
rng = np.random.default_rng()
x = np.linspace(0, 10, 200)
y = np.where(x < 5, x, 5) + rng.normal(0, 0.4, size=x.shape)
x = x.reshape(-1,1)
```


```python
plt.plot(x, y, 'o')
```




    [<matplotlib.lines.Line2D at 0x7f02cfa50a90>]




    
![png](/assets/2023-09-10-gbm-implementation_files/2023-09-10-gbm-implementation_6_1.png)
    


## Friedman's Generic Gradient Boosting Algorithm ##  
![image.png](/assets/2023-09-10-gbm-implementation_files/c06163db-b825-476f-9d08-a24c9eb58564.png)

## 简单的GMB ##


```python
class GradientBoostingFromScratch():
    
    def __init__(self, n_trees, learning_rate, max_depth=1):
        self.n_trees=n_trees; self.learning_rate=learning_rate; self.max_depth=max_depth;
        
    def fit(self, x, y):
        self.trees = []
        # set the F0(x)
        self.F0 = y.mean()
        Fm = self.F0 
        for _ in range(self.n_trees):
            #set up regression tree 
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            #residual = y - Fm for training
            tree.fit(x, y - Fm)  
            # the tree.predict(x) = w = mean of y - y_hat 
            Fm += self.learning_rate * tree.predict(x) 
            self.trees.append(tree)
            
    def predict(self, x):
        return self.F0 + self.learning_rate * np.sum([tree.predict(x) for tree in self.trees], axis=0)
```


```python
from sklearn.tree import DecisionTreeRegressor

# model hyperparameters
learning_rate = 0.3
n_trees = 10
max_depth = 2

# Training
F0 = y.mean() 
Fm = F0
trees = []
for _ in range(n_trees):
    tree = DecisionTreeRegressor(max_depth=max_depth)
    tree.fit(x, y - Fm)
    Fm += learning_rate * tree.predict(x)
    trees.append(tree)

# Prediction
y_hat = F0 + learning_rate * np.sum([t.predict(x) for t in trees], axis=0)
```


```python
#hide_input
plt.plot(x,y,'o', label='y')
plt.plot(x,y_hat,'-k', label='GBM')
plt.legend()
plt.xlabel('x');
```


    
![png](/assets/2023-09-10-gbm-implementation_files/2023-09-10-gbm-implementation_11_0.png)
    



```python
reg = DecisionTreeRegressor(max_depth=2)
reg.fit(x, y)
y_hat = reg.predict(x)
```


```python
# parallel arrays that give info on the nodes
pd.DataFrame({
    'children_left': reg.tree_.children_left
    , 'children_right': reg.tree_.children_right
    , 'feature': reg.tree_.feature 
    , 'threshold': reg.tree_.threshold
    , 'n_node_samples': reg.tree_.n_node_samples 
    , 'impurity': reg.tree_.impurity
})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>children_left</th>
      <th>children_right</th>
      <th>feature</th>
      <th>threshold</th>
      <th>n_node_samples</th>
      <th>impurity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>3.492462</td>
      <td>200</td>
      <td>2.842706</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>1.884422</td>
      <td>70</td>
      <td>1.097194</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>-2.000000</td>
      <td>38</td>
      <td>0.359338</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>-2.000000</td>
      <td>32</td>
      <td>0.348038</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>6</td>
      <td>0</td>
      <td>4.296482</td>
      <td>130</td>
      <td>0.300465</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>-2.000000</td>
      <td>16</td>
      <td>0.133329</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>-2.000000</td>
      <td>114</td>
      <td>0.180591</td>
    </tr>
  </tbody>
</table>
</div>



The index corresponds to the nodes in the tree.
`children_left` and `children_right` give the index of the left and right children of the given node. 
They are set to -1 on the terminal nodes.
Looks like the tree is indexed in a depth-first order.


```python
plot_tree(reg);
```


    
![png](/assets/2023-09-10-gbm-implementation_files/2023-09-10-gbm-implementation_15_0.png)
    



```python
reg.tree_.node_count
```




    7




```python
# find the terminal nodes that each observation lands in.
reg.apply(x)
```




    array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
           3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
           3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6,
           6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
           6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
           6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
           6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
           6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
           6, 6])




```python
# find the terminal nodes that each observation lands in.
# it works on the tree_ object too
reg.tree_.apply(x.astype(np.float32))
```




    array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
           3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
           3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6,
           6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
           6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
           6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
           6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
           6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
           6, 6])




```python
# terminal node id's
np.nonzero(reg.tree_.children_left == reg.tree_.children_right)[0]
```




    array([2, 3, 5, 6])




```python
# the prediction values for each node (including non terminal ones)
reg.tree_.value
```




    array([[[3.75928113]],
    
           [[1.7090544 ]],
    
           [[0.91804084]],
    
           [[2.648383  ]],
    
           [[4.86324936]],
    
           [[3.91691426]],
    
           [[4.99606833]]])



Not sure why `value` has two other dimensions.


```python
# the prediction values for each node (including non terminal ones)
reg.tree_.value[:, 0, 0]
```




    array([3.75928113, 1.7090544 , 0.91804084, 2.648383  , 4.86324936,
           3.91691426, 4.99606833])




```python
# manually get predicted values for given feature vector observations
reg.tree_.value[:, 0, 0][reg.apply(x)]
```




    array([0.91804084, 0.91804084, 0.91804084, 0.91804084, 0.91804084,
           0.91804084, 0.91804084, 0.91804084, 0.91804084, 0.91804084,
           0.91804084, 0.91804084, 0.91804084, 0.91804084, 0.91804084,
           0.91804084, 0.91804084, 0.91804084, 0.91804084, 0.91804084,
           0.91804084, 0.91804084, 0.91804084, 0.91804084, 0.91804084,
           0.91804084, 0.91804084, 0.91804084, 0.91804084, 0.91804084,
           0.91804084, 0.91804084, 0.91804084, 0.91804084, 0.91804084,
           0.91804084, 0.91804084, 0.91804084, 2.648383  , 2.648383  ,
           2.648383  , 2.648383  , 2.648383  , 2.648383  , 2.648383  ,
           2.648383  , 2.648383  , 2.648383  , 2.648383  , 2.648383  ,
           2.648383  , 2.648383  , 2.648383  , 2.648383  , 2.648383  ,
           2.648383  , 2.648383  , 2.648383  , 2.648383  , 2.648383  ,
           2.648383  , 2.648383  , 2.648383  , 2.648383  , 2.648383  ,
           2.648383  , 2.648383  , 2.648383  , 2.648383  , 2.648383  ,
           3.91691426, 3.91691426, 3.91691426, 3.91691426, 3.91691426,
           3.91691426, 3.91691426, 3.91691426, 3.91691426, 3.91691426,
           3.91691426, 3.91691426, 3.91691426, 3.91691426, 3.91691426,
           3.91691426, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833])




```python
# modifying one of the terminal node prediction values
reg.tree_.value[3, 0, 0] = 0.0
```


```python
#built in predict method
reg.predict(x)
```




    array([0.91804084, 0.91804084, 0.91804084, 0.91804084, 0.91804084,
           0.91804084, 0.91804084, 0.91804084, 0.91804084, 0.91804084,
           0.91804084, 0.91804084, 0.91804084, 0.91804084, 0.91804084,
           0.91804084, 0.91804084, 0.91804084, 0.91804084, 0.91804084,
           0.91804084, 0.91804084, 0.91804084, 0.91804084, 0.91804084,
           0.91804084, 0.91804084, 0.91804084, 0.91804084, 0.91804084,
           0.91804084, 0.91804084, 0.91804084, 0.91804084, 0.91804084,
           0.91804084, 0.91804084, 0.91804084, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           3.91691426, 3.91691426, 3.91691426, 3.91691426, 3.91691426,
           3.91691426, 3.91691426, 3.91691426, 3.91691426, 3.91691426,
           3.91691426, 3.91691426, 3.91691426, 3.91691426, 3.91691426,
           3.91691426, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833,
           4.99606833, 4.99606833, 4.99606833, 4.99606833, 4.99606833])



## scipy minimize


```python
from scipy.optimize import minimize
```


```python
my_fun = lambda t: (t - 4)**2 + 3
t0 = 0
res = minimize(my_fun, t0)
res.x[0]
```




    3.999999987147814




```python
res
```




          fun: 3.0
     hess_inv: array([[0.5]])
          jac: array([0.])
      message: 'Optimization terminated successfully.'
         nfev: 8
          nit: 3
         njev: 4
       status: 0
      success: True
            x: array([3.99999999])



That wasn't so bad.

## Loss Function Classes

I think we'll implement loss functions as a class that the user supplies.
The class should have two methods, `loss` and `negative_gradient`, which both take two arguments, `y` and `y_hat`.
The `loss` method should return a scalar, while the `negative_gradient` method should return an array the same size as `y` and `y_hat`.


```python
class SquaredErrorLoss():
    
    def loss(self, y, y_hat):
        return np.mean((y - y_hat)**2)
    
    def negative_gradient(self, y, y_hat):
        return y - y_hat
```


```python
# make an instance
obj = SquaredErrorLoss()
```


```python
# loss method should return a number
obj.loss(y, y_hat)
```




    0.23756349425442555




```python
# negative_gradient method should return an array, same shape as y and y_hat (get from previouse one dicision tree)
obj.negative_gradient(y, y_hat)
```




    array([-0.49255678, -0.32066061, -1.1784309 , -0.53375033, -0.98815151,
           -1.09888638, -0.67857424, -0.25830859, -0.49204353, -0.02359868,
           -0.54589252, -0.75128065, -0.58031253,  0.03874785, -0.19066955,
            0.88048938,  0.32790997, -0.26344076, -0.69209517,  0.14841553,
            0.07083979, -0.09176096,  0.79218655, -0.13984208, -0.17380496,
            0.03267205,  0.57492123,  0.38596099,  0.23825812,  0.19322872,
            1.17191446,  1.06500213,  0.74360901,  0.67556879,  0.66238355,
            0.61572065,  0.37655057,  0.49968143,  0.22022114, -0.95836931,
           -0.9939993 , -0.37475115, -1.32978096, -0.70896935, -0.21635722,
            0.4337836 , -0.42981613,  0.17449931, -1.15174471, -0.48872236,
           -0.08142043,  0.70285175, -0.30088331,  0.22241331,  0.06431568,
            0.01334207,  0.11388726, -0.45009487,  0.94884203,  0.0932048 ,
           -0.13219572,  0.88661359,  0.28153107,  0.91233939,  0.4425616 ,
            0.63421352,  0.13398699,  0.5170198 ,  0.56861933,  0.25285859,
           -0.33065244,  0.21256738, -0.52482522, -0.60505751,  0.13767718,
            0.04713186, -0.14208287, -0.18659448,  0.58410822, -0.05318493,
            0.21359337, -0.32987855,  0.36567509,  0.72201231,  0.16998206,
           -0.28047147, -0.48537644,  0.04290149, -0.45361261, -0.31398043,
           -0.66198662, -0.97769314,  0.32799293, -0.04483615,  0.24082134,
           -0.18200017, -0.12835403, -0.30143722,  0.28010512, -0.43800651,
           -0.10314913,  0.74029697,  0.65379024,  0.21386612,  0.70071287,
            0.11352387, -0.3890135 , -0.28621517,  0.58323056, -0.58283824,
            0.4959617 , -0.06211422, -0.24294399,  0.29456231,  0.64665966,
           -0.26489692,  0.4224831 ,  0.20412148,  0.07502915, -0.24005835,
           -1.02023288, -0.02901337, -0.01020261, -0.52236571,  0.08092122,
           -0.77523367,  0.95022498,  0.14674102, -0.22035964, -0.29973747,
           -0.40185016,  0.83946797,  0.09904628,  0.67040872,  0.37628663,
           -0.35042556, -0.54743247, -0.34088812,  0.09844891,  0.45265303,
           -0.09083517,  0.06610746, -0.11653813, -0.13520666, -0.56615654,
           -0.16687041, -0.03071953, -0.17698441,  0.08663586, -0.95440681,
           -0.5104037 , -0.69415237,  0.58766281,  0.6615037 ,  0.46522491,
            0.2720463 , -0.36316134,  0.43569145, -0.41965952, -0.67616288,
            0.35125389, -0.22111344,  0.73099068, -0.31794585,  0.65117165,
            0.17472301,  0.3810294 , -0.13664031,  0.12362699, -0.02757297,
            0.04766081,  0.110813  ,  0.0460294 , -0.60778078, -0.57101885,
           -0.02281148,  0.43300191,  0.38623513, -0.09828116,  0.51236255,
            0.63552504, -0.0574628 , -0.20213633, -0.21869942, -0.25043074,
            0.20255221,  0.30607896,  0.17446274, -0.09110075, -0.21181668,
           -0.41422072,  0.22664063,  0.81457072,  0.00713662,  0.7704969 ,
            0.11464773, -0.15686365, -0.26182471, -0.09909693,  0.01818937])



## GBM Implementation


```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor 
from scipy.optimize import minimize

class GradientBoostingMachine():
    '''Gradient Boosting Machine supporting any user-supplied loss function.
    
    Parameters
    ----------
    n_trees : int
        number of boosting rounds
        
    learning_rate : float
        learning rate hyperparameter
        
    max_depth : int
        maximum tree depth
    '''
    
    def __init__(self, n_trees, learning_rate=0.1, max_depth=1):
        self.n_trees=n_trees; 
        self.learning_rate=learning_rate
        self.max_depth=max_depth;
    
    def fit(self, X, y, objective):
        '''Fit the GBM using the specified loss function.
        
        Parameters
        ----------
        X : ndarray of size (number observations, number features)
            design matrix
            
        y : ndarray of size (number observations,)
            target values
            
        objective : loss function class instance
            Class specifying the loss function for training.
            Should implement two methods:
                loss(labels: ndarray, predictions: ndarray) -> float
                negative_gradient(labels: ndarray, predictions: ndarray) -> ndarray
        '''
        
        self.trees = []
        # initial the F0
        self.base_prediction = self._get_optimal_base_value(y, objective.loss)
        # convert to matrix
        current_predictions = self.base_prediction * np.ones(shape=y.shape)
        # start to run boosting
        for _ in range(self.n_trees):
            
            # using object function return the residual = negative gradient (partial deritive of the loss function) 
            pseudo_residuals = objective.negative_gradient(y, current_predictions)
            # here setup the tree still with MSE
            tree = DecisionTreeRegressor(max_depth=self.max_depth) 
            
            #using redidual to train the tree and get the predicted (minimized) delta
            tree.fit(X, pseudo_residuals)
            
            # replace orignal loss with the object function loss/residual or target
            self._update_terminal_nodes(tree, X, y, current_predictions, objective.loss)
            
            #adding the delta part into whole function y
            current_predictions += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
     
    #get the intial value (F0) to minimize the all data loss
    def _get_optimal_base_value(self, y, loss):
        '''Find the optimal initial prediction for the base model.'''
        fun = lambda c: loss(y, c)
        c0 = y.mean() # give the initial value as mean of the data
        return minimize(fun=fun, x0=c0).x[0]
        
    def _update_terminal_nodes(self, tree, X, y, current_predictions, loss):
        '''Update the tree's predictions according to the loss function.'''
        # terminal node id's
        leaf_nodes = np.nonzero(tree.tree_.children_left == -1)[0]
        # compute leaf for each sample in ``X``.
        leaf_node_for_each_sample = tree.apply(X)
        for leaf in leaf_nodes:
            samples_in_this_leaf = np.where(leaf_node_for_each_sample == leaf)[0]
            y_in_leaf = y.take(samples_in_this_leaf, axis=0)
            preds_in_leaf = current_predictions.take(samples_in_this_leaf, axis=0)
            # get the predict value w
            val = self._get_optimal_leaf_value(y_in_leaf, 
                                               preds_in_leaf,
                                               loss)
            tree.tree_.value[leaf, 0, 0] = val
            
    #get the optimized loss target to the  w/c or residual: y-(current_preddiction+w) = (y-current_predictin) -w 
    # = residual - w ..
    def _get_optimal_leaf_value(self, y, current_predictions, loss):
        '''Find the optimal prediction value for a given leaf.'''
        fun = lambda c: loss(y, current_predictions + c)
        c0 = y.mean()       
        return minimize(fun=fun, x0=c0).x[0]
          
    def predict(self, X):
        '''Generate predictions for the given input data.'''
        return (self.base_prediction 
                + self.learning_rate 
                * np.sum([tree.predict(X) for tree in self.trees], axis=0))
```

### Mean Squared Error


```python
rng = np.random.default_rng()
x = np.linspace(0, 10, 500)
y = np.where(x < 5, x, 5) + rng.normal(0, 0.4, size=x.shape)
x = x.reshape(-1,1)
```


```python
class SquaredErrorLoss():
    
    def loss(self, y, y_hat):
        return np.mean((y - y_hat)**2)
    
    def negative_gradient(self, y, y_hat):
        return y - y_hat
```


```python
gbm = GradientBoostingMachine(n_trees=10,
                              learning_rate=0.5)
gbm.fit(x, y, SquaredErrorLoss())
```


```python
# fig, ax = plt.subplot()
plt.plot(x.ravel(), y, 'o', label='data')
plt.plot(x.ravel(), gbm.predict(x), '-k', label='model')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('model predicting mean or y | x')
```




    Text(0.5, 1.0, 'model predicting mean or y | x')




    
![png](/assets/2023-09-10-gbm-implementation_files/2023-09-10-gbm-implementation_43_1.png)
    


### Mean Absolute Error


```python
rng = np.random.default_rng()
x = np.linspace(0, 10, 500)
y = np.where(x < 5, x, 5) + rng.normal(0, 0.4, size=x.shape)
x = x.reshape(-1,1)
```


```python
class AbsoluteErrorLoss():
    
    def loss(self, y, y_hat):
        return np.mean(np.abs(y - y_hat))
    
    def negative_gradient(self, y, y_hat):
        return np.sign(y - y_hat)
```


```python
gbm = GradientBoostingMachine(n_trees=10,
                              learning_rate=0.5)
gbm.fit(x, y, AbsoluteErrorLoss())
```


```python
# fig, ax = plt.subplot()
plt.plot(x.ravel(), y, 'o', label='data')
plt.plot(x.ravel(), gbm.predict(x), '-k', label='model')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('model predicting median of y | x')
```




    Text(0.5, 1.0, 'model predicting median of y | x')




    
![png](/assets/2023-09-10-gbm-implementation_files/2023-09-10-gbm-implementation_48_1.png)
    



```python

```

### Quantile Loss
$$
L_\gamma\left(y, y^p\right)=\sum_{i: y_i<y_i^p}(1-\gamma)\left|y_i-y_i^p\right|+\sum_{i: y_i \geq y_i^p} \gamma\left|y_i-y_i^p\right|
$$
γ是所需的分位数，其值介于0和1之间。 Y轴：分位数损失。X轴：预测值。Y的真值为0。
许多商业问题的决策通常希望了解预测中的不确定性，更关注区间预测而不仅是点预测时，分位数损失函数就很有用。

![image.png](/assets/2023-09-10-gbm-implementation_files/39c4b069-0662-40d5-b135-c8ad94f6008a.png)


```python
rng = np.random.default_rng()
x = np.linspace(0, 10, 500)
# y = np.where(x < 5, x, 5) + rng.uniform(-2, 2, size=x.shape)
y = np.where(x < 5, x, 5) + rng.normal(0, 1, size=x.shape)
x = x.reshape(-1,1)
```


```python
class QuantileLoss():
    
    def __init__(self, alpha):
        if alpha < 0 or alpha >1:
            raise ValueError('alpha must be between 0 and 1')
        self.alpha = alpha
        
    def loss(self, y, y_hat):
        e = y - y_hat
        return np.mean(np.where(e > 0, self.alpha * e, (self.alpha - 1) * e))
    
    def negative_gradient(self, y, y_hat):
        e = y - y_hat 
        return np.where(e > 0, self.alpha, self.alpha - 1)
```


```python
gbm_up = GradientBoostingMachine(n_trees=10,
                              learning_rate=0.5)
gbm_up.fit(x, y, QuantileLoss(alpha=0.9))

gbm_low = GradientBoostingMachine(n_trees=10,
                              learning_rate=0.5)

gbm_low.fit(x, y, QuantileLoss(alpha=0.05))
```


```python
plt.plot(x, y, 'o', label='data')
plt.plot(x, gbm_up.predict(x), 'r-', label='0.95')
plt.plot(x, gbm_low.predict(x), 'g-', label='0.05')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('model predicting 0.9 quantile of y | x')
```




    Text(0.5, 1.0, 'model predicting 0.9 quantile of y | x')




    
![png](/assets/2023-09-10-gbm-implementation_files/2023-09-10-gbm-implementation_54_1.png)
    



```python

```

### Binary Cross Entropy


```python
rng = np.random.default_rng()
x = np.linspace(-3, 3, 500)
expit = lambda t: np.exp(t) / (1 + np.exp(t))
p = expit(x)
y = rng.binomial(1, p, size=p.shape)
x = x.reshape(-1,1)
```


```python
class BinaryCrossEntropyLoss():
    # in these methods, y_hat gives the log odds ratio
    
    def __init__(self):
        self.expit = lambda t: np.exp(t) / (1 + np.exp(t))
    
    def loss(self, y, y_hat):
        p = self.expit(y_hat)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    def negative_gradient(self, y, y_hat):
        p = self.expit(y_hat)
        return y / p - (1 - y) / (1 - p)
```


```python
gbm = GradientBoostingMachine(n_trees=10,
                              learning_rate=0.5)
gbm.fit(x, y, BinaryCrossEntropyLoss())
```


```python
plt.plot(x, y, 'o', label='data')
plt.plot(x, p, '-r', label='P(y=1|x)')
plt.plot(x, expit(gbm.predict(x)), '-k', label='model')
plt.title('model predicting P(y = 1 | x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f02cee90b90>




    
![png](/assets/2023-09-10-gbm-implementation_files/2023-09-10-gbm-implementation_60_1.png)
    



```python

```
