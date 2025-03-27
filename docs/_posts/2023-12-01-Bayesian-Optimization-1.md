---
layout: post
title: Bayesian Optimization with GPyOpt
date: 2021-01-26
categories: coding
tags: AI BayesianOptimization
---

**Bayesian Optimization with GPyOpt**  
**This example only show how the GPyOpt API work**

Try to find optimal hyperparameters to XGBoost model using Bayesian optimization with GP, with the diabetes dataset (from sklearn) as input. Let’s first load the dataset with the following python code snippet:


```python
from sklearn import datasets
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
dataset = datasets.load_diabetes()
X = dataset['data']
y = dataset['target']
```
We will use cross-validation score to estimate accuracy and our goal will be to tune: max_depth, learning_rate, n_estimators parameters. First, we have to define optimization function and domains, as shown in the code below.

```python
# Optimizer will try to find minimum, so let's add a "-" sign.
def f(parameters):
    parameters = parameters[0]
    score = -cross_val_score(
        XGBRegressor(learning_rate=parameters[0],
                     max_depth=int(parameters[2]),
                     n_estimators=int(parameters[3]),
                     gamma=int(parameters[1]),
                     min_child_weight = parameters[4]), 
        X, y, scoring='neg_root_mean_squared_error'
    ).mean()
    score = np.array(score)
    return score
     
# Bounds (define continuous variables first, then discrete!)
bounds = [
    {'name': 'learning_rate',
     'type': 'continuous',
     'domain': (0, 1)},
    {'name': 'gamma',
     'type': 'continuous',
     'domain': (0, 5)},
    {'name': 'max_depth',
     'type': 'discrete',
     'domain': (1, 50)},
    {'name': 'n_estimators',
     'type': 'discrete',
     'domain': (1, 300)},
    {'name': 'min_child_weight',
     'type': 'discrete',
     'domain': (1, 10)}
]
```

Let’s find the baseline RMSE with default XGBoost parameters is . Let’s see if we can do better.


```python
baseline = -cross_val_score(
    XGBRegressor(), X, y, scoring='neg_root_mean_squared_error'
).mean()
baseline
# 64.90693011829266
```




    64.90693033120199



Now, run the Bayesian optimization with GPyOpt and plot convergence, as in the next code snippet:


```python
import numpy as np
import GPyOpt
optimizer = GPyOpt.methods.BayesianOptimization(
    f=f, domain=bounds,
    acquisition_type ='MPI', ## method to optimize the acq. function
    acquisition_par = 0.1,
    exact_eval=True
)
max_iter = 50
max_time = 60
optimizer.run_optimization(max_iter, max_time)
optimizer.plot_convergence()
```


    
![png](/assets/2023-12-01-Bayesian-Optimization-1_files/2023-12-01-Bayesian-Optimization-1_7_0.png)
    


Extract the best values of the parameters and compute the RMSE / gain obtained with Bayesian Optimization, using the following code.


```python
optimizer.X[np.argmin(optimizer.Y)]
# array([2.01515532e-01, 1.35401092e+00, 1.00000000e+00, 
# 3.00000000e+02, 1.00000000e+00])
```




    array([5.00047461e-02, 3.10049677e+00, 1.00000000e+00, 3.00000000e+02,
           1.00000000e+01])




```python
print('RMSE:', np.min(optimizer.Y),
      'Gain:', baseline/np.min(optimizer.Y)*100)
# RMSE: 57.6844355488563 Gain: 112.52069904249859
```

    RMSE: 56.1566484514227 Gain: 125.44111632088877


Paramerter Tuning for SVR

-Now, let’s tune a Support Vector Regressor model with Bayesian Optimization and find the optimal values for three parameters: C, epsilon and gamma.

-Let’s use range (1e-5, 1000) for C, (1e-5, 10) for epsilon and gamma.

-Let’s use MPI as an acquisition function with weight 0.1.


```python
from sklearn.svm import SVR# Bounds (define continuous variables first, then discrete!)
bounds = [
    {'name': 'C',
     'type': 'continuous',
     'domain': (1e-5, 1000)},    {'name': 'epsilon',
     'type': 'continuous',
     'domain': (1e-5, 10)},    {'name': 'gamma',
     'type': 'continuous',
     'domain': (1e-5, 10)}
]
 
# Score. Optimizer will try to find minimum, so we will add a "-" sign.
def f(parameters):
    parameters = parameters[0]
    score = -cross_val_score(
        SVR(C = parameters[0],
            epsilon = parameters[1],
            gamma = parameters[2]), 
        X, y, scoring='neg_root_mean_squared_error'
    ).mean()
    score = np.array(score)
    scoreoptimizer = GPyOpt.methods.BayesianOptimization(
            f=f, domain=bounds,
            acquisition_type ='MPI',
            acquisition_par = 0.1,
            exact_eval=True
            )
    return scoreoptimizer

max_iter = 50*4
max_time = 60*4

optimizer.run_optimization(max_iter, max_time)
baseline = -cross_val_score( SVR(), X, y, scoring='neg_root_mean_squared_error').mean()
print(baseline)

# 70.44352670586173print(optimizer.X[np.argmin(optimizer.Y)])
# [126.64337652   8.49323372   8.59189135]
print('RMSE:', np.min(optimizer.Y),
      'Gain:', baseline/np.min(optimizer.Y)*100)
# RMSE: 54.02576574389976 Gain: 130.38876124364006     best_epsilon = optimizer.X[np.argmin(optimizer.Y)][1] 
optimizer.plot_convergence()
```

    70.44352670586173
    RMSE: 56.1566484514227 Gain: 125.44111632088877



    
![png](/assets/2023-12-01-Bayesian-Optimization-1_files/2023-12-01-Bayesian-Optimization-1_12_1.png)
    


Surrogate Function
The surrogate function is a technique used to best approximate the mapping of input examples to an output score.

## How to Implement Bayesian Optimization from Scratch in Python ##

https://machinelearningmastery.com/what-is-bayesian-optimization/


```python
# example of a gaussian process surrogate function
from math import sin
from math import pi
from numpy import arange
from numpy import asarray
from numpy.random import normal
from numpy.random import random
from matplotlib import pyplot
from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process import GaussianProcessRegressor

# objective function
def objective(x, noise=0.1):
	noise = normal(loc=0, scale=noise)
	return (x**2 * sin(5 * pi * x)**6.0) + noise

# surrogate or approximation for the objective function
def surrogate(model, X):
	# catch any warning generated when making a prediction
	with catch_warnings():
		# ignore generated warnings
		simplefilter("ignore")
		return model.predict(X, return_std=True)

# plot real observations vs surrogate function
def plot(X, y, model):
	# scatter plot of inputs and real objective function
	pyplot.scatter(X, y)
	# line plot of surrogate function across domain
	Xsamples = asarray(arange(0, 1, 0.001))
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	ysamples, _ = surrogate(model, Xsamples)
	pyplot.plot(Xsamples, ysamples)
	# show the plot
	pyplot.show()

# sample the domain sparsely with noise
X = random(100)
y = asarray([objective(x) for x in X])
# reshape into rows and cols
X = X.reshape(len(X), 1)
y = y.reshape(len(y), 1)
# define the model
model = GaussianProcessRegressor()
# fit the model
model.fit(X, y)
# plot the surrogate function
plot(X, y, model)
```


    
![png](/assets/2023-12-01-Bayesian-Optimization-1_files/2023-12-01-Bayesian-Optimization-1_14_0.png)
    



```python
# example of bayesian optimization for a 1d function from scratch
# this is pure gaussian bayesian optimization to find the next x to sampling...
from math import sin
from math import pi
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from numpy.random import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot

# the true objective loss function??
def objective(x, noise=0.1):
	noise = normal(loc=0, scale=noise)
	return (x**2 * sin(5 * pi * x)**6.0) + noise

# surrogate or approximation for the objective function
def surrogate(model, X):
	# catch any warning generated when making a prediction
	with catch_warnings():
		# ignore generated warnings
		simplefilter("ignore")
		return model.predict(X, return_std=True)

# probability of improvement acquisition function
def acquisition(X, Xsamples, model):
	# calculate the best surrogate score found so far
	yhat, _ = surrogate(model, X)
	best = max(yhat)
	# calculate mean and stdev via surrogate function
	mu, std = surrogate(model, Xsamples)
	mu = mu[:, 0]
	# calculate the probability of improvement
	probs = norm.cdf((mu - best) / (std+1E-9))
	return probs

# optimize the acquisition function
def opt_acquisition(X, y, model):
	# random search, generate random samples
	Xsamples = random(100)
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	# calculate the acquisition function for each sample
	scores = acquisition(X, Xsamples, model)
	# locate the index of the largest scores
	ix = argmax(scores)
	return Xsamples[ix, 0]

# plot real observations vs surrogate function
def plot(X, y, model):
	# scatter plot of inputs and real objective function
	pyplot.scatter(X, y)
	# line plot of surrogate function across domain
	Xsamples = asarray(arange(0, 1, 0.001))
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	ysamples, _ = surrogate(model, Xsamples)
	pyplot.plot(Xsamples, ysamples)
	# show the plot
	pyplot.show()

# sample the domain sparsely with noise
X = random(100)
y = asarray([objective(x) for x in X])
# reshape into rows and cols
X = X.reshape(len(X), 1)
y = y.reshape(len(y), 1)
# define the model
model = GaussianProcessRegressor()
# fit the model
model.fit(X, y)
# plot before hand
plot(X, y, model)

# perform the optimization process
for i in range(100):
	# select the next point to sample
	x = opt_acquisition(X, y, model)
	# if hyperparamters , here need to apply the paramters into the object function
    # sample the point
	actual = objective(x)
	# summarize the finding
	est, _ = surrogate(model, [[x]])
	print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
	# add the data to the dataset
	X = vstack((X, [[x]]))
	y = vstack((y, [[actual]]))
	# update the model
	model.fit(X, y)

# plot all samples and the final surrogate function
plot(X, y, model)
# best result
ix = argmax(y)
print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))
```


    
![png](/assets/2023-12-01-Bayesian-Optimization-1_files/2023-12-01-Bayesian-Optimization-1_15_0.png)
    


    >x=0.753, f()=0.130899, actual=0.202
    >x=0.914, f()=0.441535, actual=0.830
    >x=0.914, f()=0.468772, actual=0.614
    >x=0.448, f()=0.072738, actual=0.213
    >x=0.255, f()=-0.005875, actual=-0.006
    >x=0.546, f()=0.125458, actual=0.071
    >x=0.644, f()=0.099045, actual=-0.099
    >x=0.546, f()=0.115388, actual=0.125
    >x=0.913, f()=0.480679, actual=0.718
    >x=0.724, f()=0.094843, actual=0.413
    >x=0.108, f()=0.029385, actual=-0.000
    >x=0.366, f()=0.019852, actual=0.169
    >x=0.935, f()=0.466089, actual=0.308
    >x=0.912, f()=0.484408, actual=0.802
    >x=0.913, f()=0.501859, actual=0.812
    >x=0.846, f()=0.379995, actual=0.132
    >x=0.661, f()=0.088295, actual=0.177
    >x=0.383, f()=0.037196, actual=-0.013
    >x=0.113, f()=0.026929, actual=-0.118
    >x=0.981, f()=0.147262, actual=-0.012
    >x=0.868, f()=0.436012, actual=0.285
    >x=0.911, f()=0.503125, actual=0.739
    >x=0.646, f()=0.095198, actual=-0.099
    >x=0.355, f()=0.019695, actual=0.077
    >x=0.535, f()=0.120008, actual=0.179
    >x=0.729, f()=0.099606, actual=0.180
    >x=0.913, f()=0.515075, actual=0.530
    >x=0.309, f()=0.003572, actual=0.045
    >x=0.950, f()=0.416246, actual=0.119
    >x=0.912, f()=0.506245, actual=0.845
    >x=0.911, f()=0.520569, actual=0.806
    >x=0.911, f()=0.532109, actual=0.848
    >x=0.135, f()=0.018783, actual=-0.117
    >x=0.911, f()=0.544241, actual=0.610
    >x=0.148, f()=0.011521, actual=-0.119
    >x=0.002, f()=0.093856, actual=0.073
    >x=0.221, f()=-0.003341, actual=-0.009
    >x=0.072, f()=0.008024, actual=-0.175
    >x=0.293, f()=0.001839, actual=0.134
    >x=0.804, f()=0.259678, actual=-0.110
    >x=0.111, f()=-0.004541, actual=0.012
    >x=0.679, f()=0.071661, actual=0.215
    >x=0.426, f()=0.079402, actual=-0.100
    >x=0.967, f()=0.280364, actual=-0.133
    >x=0.814, f()=0.283361, actual=-0.118
    >x=0.529, f()=0.126789, actual=0.145
    >x=0.664, f()=0.077135, actual=0.231
    >x=0.914, f()=0.528958, actual=0.622
    >x=0.124, f()=0.001293, actual=0.042
    >x=0.911, f()=0.533073, actual=0.884
    >x=0.357, f()=0.020477, actual=-0.036
    >x=0.393, f()=0.043710, actual=-0.044
    >x=0.638, f()=0.094322, actual=0.031
    >x=0.469, f()=0.101346, actual=0.175
    >x=0.584, f()=0.122175, actual=0.081
    >x=0.911, f()=0.545548, actual=0.825
    >x=0.701, f()=0.072042, actual=0.487
    >x=0.911, f()=0.552639, actual=0.811
    >x=0.641, f()=0.098095, actual=-0.014
    >x=0.305, f()=-0.004428, actual=0.003
    >x=0.295, f()=-0.005267, actual=0.150
    >x=0.767, f()=0.153709, actual=-0.174
    >x=0.658, f()=0.081160, actual=0.100
    >x=0.171, f()=0.014084, actual=-0.156
    >x=0.657, f()=0.081738, actual=0.422
    >x=0.640, f()=0.100931, actual=-0.157
    >x=0.911, f()=0.559911, actual=0.786
    >x=0.194, f()=0.007003, actual=0.065
    >x=0.884, f()=0.528691, actual=0.647
    >x=0.927, f()=0.550419, actual=0.328
    >x=0.682, f()=0.075016, actual=0.308
    >x=0.531, f()=0.135375, actual=0.071
    >x=0.079, f()=-0.014623, actual=0.022
    >x=0.918, f()=0.559093, actual=0.659
    >x=0.826, f()=0.329826, actual=0.135
    >x=0.380, f()=0.031751, actual=-0.054
    >x=0.255, f()=-0.002043, actual=-0.072
    >x=0.755, f()=0.124829, actual=0.248
    >x=0.431, f()=0.070760, actual=-0.057
    >x=0.437, f()=0.071749, actual=-0.102
    >x=0.115, f()=-0.000257, actual=0.042
    >x=0.549, f()=0.129516, actual=0.278
    >x=0.053, f()=-0.009525, actual=-0.098
    >x=0.348, f()=-0.000176, actual=0.010
    >x=0.133, f()=0.005985, actual=0.021
    >x=0.912, f()=0.562865, actual=0.720
    >x=0.911, f()=0.567181, actual=0.835
    >x=0.716, f()=0.087171, actual=0.275
    >x=0.052, f()=-0.015777, actual=-0.162
    >x=0.743, f()=0.114964, actual=0.195
    >x=0.335, f()=-0.005154, actual=0.003
    >x=0.910, f()=0.573601, actual=0.816
    >x=0.278, f()=-0.007704, actual=0.097
    >x=0.692, f()=0.087360, actual=0.370
    >x=0.560, f()=0.138122, actual=0.218
    >x=0.583, f()=0.137465, actual=-0.197
    >x=0.530, f()=0.126498, actual=0.055
    >x=0.963, f()=0.324050, actual=0.177
    >x=0.241, f()=0.007597, actual=-0.173
    >x=0.910, f()=0.577252, actual=0.785



    
![png](/assets/2023-12-01-Bayesian-Optimization-1_files/2023-12-01-Bayesian-Optimization-1_15_2.png)
    


    Best Result: x=0.911, y=0.884


In this case, the model via mean 5-fold cross-validation


```python
# example of bayesian optimization with scikit-optimize
from numpy import mean
from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

# generate 2d classification dataset
X, y = make_blobs(n_samples=500, centers=3, n_features=2)
# define the model
model = KNeighborsClassifier()
# define the space of hyperparameters to search
search_space = [Integer(1, 5, name='n_neighbors'), Integer(1, 2, name='p')]

# define the function used to evaluate a given configuration
@use_named_args(search_space)
def evaluate_model(**params):
	# something
	model.set_params(**params)
	# calculate 5-fold cross validation
	result = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='accuracy')
	# calculate the mean of the scores
	estimate = mean(result)
	return 1.0 - estimate

# perform optimization
result = gp_minimize(evaluate_model, search_space)
# summarizing finding:
print('Best Accuracy: %.3f' % (1.0 - result.fun))
print('Best Parameters: n_neighbors=%d, p=%d' % (result.x[0], result.x[1]))
```

    Best Accuracy: 1.000
    Best Parameters: n_neighbors=3, p=2



```python
result.fun
```




    0.0




```python

```
