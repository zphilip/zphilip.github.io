---
layout: post
title: Machine Learning Notes
date: 2021-01-26
categories: LEARNING
tags: AI
---

# Machine Learning (some parts are still need updated)

## Linear regression/prediction

<img src="/assets/Machine_LearningNote.assets/image-20230505160733435.png" alt="image-20230505160733435" style="zoom:80%;" />

formula the Lost function $J(\theta)$ and fine the parameters $\theta$ by the data (x, y)

Matrix format is :  $\widehat{y} = X\ \theta\ $, each x have d dimension features , total have n data pair ( y_i, x_i)

<img src="/assets/Machine_LearningNote.assets/image-20230505160748702.png" alt="image-20230505160748702" style="zoom:80%;" />

Assume the each $y_i$ is Gaussian distributed with the mean $x_i^T\theta$  and variance $\sigma^2$ 

$$y_i = N(x_i^T\theta, \sigma^2) = x_i^T\theta + N(0, \sigma^2)$$

<img src="/assets/Machine_LearningNote.assets/image-20230505193953662.png" alt="image-20230505193953662" style="zoom: 50%;" />

the likelihood for the linear regression will be 
$$
\begin{aligned}
p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\theta}, \sigma) & =\prod_{i=1}^n p\left(y_i \mid \mathbf{x}_i, \boldsymbol{\theta}, \sigma\right) \\
& =\prod_{i=1}^n\left(2 \pi \sigma^2\right)^{-1 / 2} e^{-\frac{1}{2 \sigma^2}\left(y_i-\mathbf{x}_i^T \boldsymbol{\theta}\right)^2} \\
& =\left(2 \pi \sigma^2\right)^{-n / 2} e^{-\frac{1}{2 \sigma^2} \sum_{i=1}^n\left(y_i-\mathbf{x}_i^T \boldsymbol{\theta}\right)^2}
\end{aligned}
$$

or using the log likelihood, $$L(\theta) = \  - \frac{n}{2}\log{2\pi\sigma^{2} - \frac{1}{2\sigma^{2}}(y - X\theta)^{T}(y - X\theta)}$$

$$\frac{\partial L(\theta)}{\partial\theta} = \ 0 - \frac{1}{2\sigma^{2}}\left\lbrack 0 - 2X^{T}Y + \ 2X^{T}X\theta \right\rbrack = 0$$

so we get $\widehat{\theta} = {X^{T}X}^{- 1}X^{T}y$ =\> $\widehat{\theta} = {X^{T}X + \delta^{2}Ι_{d}}^{- 1}X^{T}y$

## Regularization $\delta^2 \boldsymbol{\theta}^T \boldsymbol{\theta}$

$$
\widehat{\boldsymbol{\theta}}=\left(\mathbf{X}^T \mathbf{X}+\delta^2 I_d\right)^{-1} \mathbf{X}^T \mathbf{y}
$$

$$
J(\boldsymbol{\theta})=(\mathbf{y}-\mathbf{X} \boldsymbol{\theta})^T(\mathbf{y}-\mathbf{X} \boldsymbol{\theta})+\delta^2 \boldsymbol{\theta}^T \boldsymbol{\theta}
$$

$$
\frac{\partial J(\theta)}{\partial \theta}=2 x^{\top} x \theta-2 x^{\top} y+2 \delta^2 I \theta
$$

<img src="/assets/Machine_LearningNote.assets/image-20230506124401185.png" alt="image-20230506124401185" style="zoom: 33%;" />

the optmization point will be in the blue curve between the $\hat{\theta_{ML}}$ and $\hat{\theta_R(while\ \sigma^2 -> \infin)}$

## Extend to nonlinear via basis function 

$$y(x) = \phi(x)\theta + \epsilon $$  using the $\phi(x)$ to deal with nonlinearity

$$\color{red}\hat{\theta}_{ML} = [\phi(x)^T \phi(x)]^{-1} \phi(x)^T y $$

<img src="/assets/Machine_LearningNote.assets/image-20230506130455991.png" alt="image-20230506130455991" style="zoom:50%;" />

<img src="/assets/Machine_LearningNote.assets/image-20230506130654112.png" alt="image-20230506130654112" style="zoom:50%;" />

The $\sigma$ will make some $\theta$ goto zero. 

## Kernel regression and RBF (with same variance)

$$
\begin{gathered}
\phi(\mathrm{x})=\left[\kappa\left(\mathrm{x}, \boldsymbol{\mu}_1, \lambda\right), \ldots, \kappa\left(\mathrm{x}, \boldsymbol{\mu}_d, \lambda\right)\right], \quad \text { e.g. } \kappa\left(\mathrm{x}, \boldsymbol{\mu}_i, \lambda\right)=e^{\left(-\frac{1}{\lambda}\left\|\mathrm{x}-\mu_i\right\|^2\right)} \\
\hat{y}\left(x_i\right)=\phi\left(x_i\right) \theta=1 \theta_0+k\left(x_i, \mu_1, \lambda\right) \theta_1+\ldots+k\left(x_i, \mu_d, \lambda\right) \theta_d
\end{gathered}
$$

<img src="/assets/Machine_LearningNote.assets/image-20230506132230224.png" alt="image-20230506132230224" style="zoom:50%;" />

**Example: **
$$
\phi\left(x_i\right)=\left[\begin{array}{llll}
1 & k\left(x_i, \mu_1, \lambda\right) \quad k\left(x_i, \mu_2, \lambda\right) \quad k\left(x_i, \mu_3, \lambda\right)
\end{array}\right]
$$
$\phi\left(x_i\right)$ is a vector with 4 entries. There are 3 bases.
The corresponding vector of parameters is $\underline{\theta}=\left[\begin{array}{llll}\theta_0 & \theta_1 & \theta_2 & \theta_3\end{array}\right]^{\top}$
$$
\hat{y}_i=\phi\left(x_i\right) \underline{\theta}
$$
If we have $i=1, \cdots, N$ data, let
$$
Y=\left[\begin{array}{c}
y_1 \\
y_2 \\
\vdots \\
y_N
\end{array}\right] \quad \Phi=\left[\begin{array}{c}
\phi\left(x_1\right) \\
\phi\left(x_2\right) \\
\vdots \\
\phi\left(x_N\right.
\end{array}\right]
$$
Then
$$
\begin{aligned}
& \hat{Y}=\Phi \underline\theta \\
& \text { and } \\
& \hat{\theta}_{l s}=\left(\Phi^{\top} \Phi\right)^{-1} \Phi^{\top} Y
\end{aligned}
$$
or $\hat{\theta}_{\text {ridge }}=\left(\Phi^{\top} \Phi+\delta^2 I\right)^{-1} \Phi^{\top} y$
Hence, this is still linear regression, with' $X$ replaced by $\Phi$.

**in practise , we choose the location $\mu$ of the base function to be the inputs: $\mu_i=x_i $ , but then we need choose the $\lambda$ **

## <font color=red>**Using Cross-validation to get the regularization parameters $\delta^2$**</font>

1) given training data (x,y) and some $\delta^2$guess , compute $\hat{\theta}$
2) $\hat{y}_{train}=x_{train}\hat{\theta}$
3) $\hat{y}_{test}=x_{test}\hat{\theta}$ or do this on validation data set

so the paramters $\delta^2$ could be picked accoridngly differently - average or min_max

![image-20230506201640724](/assets/Machine_LearningNote.assets/image-20230506201640724.png)

<img src="C:\Users\zphil\AppData\Roaming\Typora\typora-user-images\image-20230506201845243.png" alt="image-20230506201845243" style="zoom:50%;" />

<img src="/assets/Machine_LearningNote.assets/image-20230506202305896.png" alt="image-20230506202305896" style="zoom:50%;" />

## Bayesian learning 

<img src="/assets/Machine_LearningNote.assets/image-20230521171459119.png" alt="image-20230521171459119" style="zoom: 33%;" />

Given a dataset $\mathcal{D}=\left\{x_1, \ldots, x_n\right\}$ : here  the $x_i$ is one random variable . 
Bayes Rule:
$$
\begin{array}{rll}
P(\theta \mid \mathcal{D})=\frac{P(D \mid \theta) P(\theta)}{P(\mathcal{D})} &p(D|\theta) & \text {likelihood function of } \theta \\
& P(\theta) & \text { Prior probability of } \theta \\
& P(\theta \mid \mathcal{D}) & \text { Posterior distribution over } \theta \\
&p(D) & \text {marginal likelihood function} \\
\end{array}
$$
Computing posterior distribution is known as the inference problem. But:
$$
P(\mathcal{D})=\int P(\mathcal{D}, \theta) d \theta =\sum_{h^{\prime} \in H} p\left(d \mid h^{\prime}\right) p\left(h^{\prime}\right)
$$
This integral can be very high-dimensional and difficult to compute.
$$
p\left( h \middle| d \right) = \ \frac{p\left( d \middle| h \right)p(h)}{\sum_{h^{'} \in H}^{}{p\left( d \middle| h^{'} \right)p\left( h^{'} \right)}}
$$


Model parameters: 

<img src="C:\Users\zphil\AppData\Roaming\Typora\typora-user-images\image-20230506205856792.png" alt="image-20230506205856792" style="zoom: 80%;" />

<font color=red>**Give the parameters $\theta$ uncertainty , only give the prior distribution (prior knowledge)**</font>

### Learning material 

[Lecture 7. Bayesian Learning — ML Engineering (ml-course.github.io)

[[Gaussian processes (3/3) - exploring kernels (peterroelants.github.io)](https://peterroelants.github.io/posts/gaussian-process-kernels/)

[[Optimizing Parameters (mlatcl.github.io)](https://mlatcl.github.io/gpss/lectures/04-optimizing-parameters.html)](https://ml-course.github.io/master/notebooks/07 - Bayesian Learning.html)

### **Bayesian linear regression** 

Understanding the training data <font color=red>$D = (X, y)$ </font>, ($X$，$y$可以理解为坐标系, the D is random variable. 在GPR中 $y$ is function of $x$, $x$ is not random variable, y is the random variable)

With likelihood is Gaussian $N(y|X\theta,\ \sigma^{2}I_{n})$, the conjugate prior is also a gaussian, which we will donate by $p(\theta) = N\left( \theta \middle| \theta_{0},\ V_{0} \right)$ --- probably just a guess or random choose, using Bayes rule for Gaussians the posterior is given by
$$
\begin{aligned}
p\left( \theta \middle| X,Y,\sigma^{2} \right) &\propto \ p\left( Y \middle| X, \theta,\ \sigma^{2} \right)p(\theta) = N\left( \theta \middle| \theta_{0},\ V_{0} \right)N\left( y \middle| X\theta,\ \sigma^{2}I_{n} \right) \\
&= N\left( \theta \middle| \theta_{n},\ V_{n} \right) , \sigma^{2}\ is\ given \\

\theta_{n} &= {V_{n}V}_{0}^{- 1}\ \ \theta_{0} + \frac{1}{\sigma^{2}}V_{n}X^{T}y \\

V_{n}^{- 1} &= V_{0}^{- 1} + \frac{1}{\sigma^{2}}X^{T}X, \text {**Note: it is $\theta$ ’s variants**} \\
\end{aligned}
$$

- $p\left( y \middle| x,\theta,\ \sigma^{2} \right)$ *--- maximum likehood* $\left( 2\pi\sigma^{2} \right)^{- \frac{n}{2}}e^{- \frac{1}{2\sigma^{2}}(y - X\theta)^{T}(y - X\theta)}$
- $p(y) = (2\pi\Sigma)^{- 1\text{/}2}e^{- \frac{1}{2}(y - \mu)^{T}\Sigma^{- 1}(y - \mu)}$

**here is how the cacluation: **
$$
\begin{aligned}
p\left( \theta \middle| x,y,\sigma^{2} \right) &\propto e^{- \frac{1}{2}(y - X\theta)^{T}\left( \sigma^{2}I \right)^{- 1}(y - X\theta)}\ e^{- \frac{1}{2}\left( \theta - \theta_{0} \right)^{T}{V_{0}}^{- 1}\left( \theta - \theta_{0} \right)} \\

&=e^{-\frac{1}{2}\left\{\left(y - X\theta\right)_0^{T}\left( \sigma^{2}I \right)(y- X \theta)+\left(\theta-\theta_0\right)^{T} V_0^{-1}\left(\theta-\theta_0\right)\right\}} \\

&=e^{-\frac{1}{2}\left\{\color{blue} y^{T}(\sigma^2I)^{-1} y - 
\color{red}2 y^{T} (\sigma^2I)^{-1} x \theta + 
\color{green}\theta^{T} x^{T} (\sigma^2I)^{-1} x \theta + \theta^{T} v_0^{-1} \theta - 
\color{red}2 \theta_0^{T} V_0^{-1} \theta + 
\color{blue}\theta_0^{T} V_0^{-1} \theta_0\right\}} \\

&= e^{-\frac{1}{2}\left\{ \color{blue} const + \color{green}\Theta^{\top} \overbrace{\left(x^{\top}\left(\sigma^2 I\right)^{-1} x+V_0^{-1}\right)\theta} ^{\text {call this $V_n^{-1}$ }}  - \color{red}2\left(Y^{\top}\left(\sigma^2 I\right)^{-1} x+\theta_0^{\top} V_0^{-1}\right) \theta \right\}} \\

& =e^{-\frac{1}{2}\left\{\text { const }+\Theta^{\top} V_n^{-1} \theta-2\left(\frac{y^{\top} x}{\sigma^2}+\theta_0^{\top} V_0^{-1}\right) \theta\right\}} \\
& =e^{-\frac{1}{2}\left\{\text { const }+\theta^{\top} V_n^{-1} \theta-2 \theta_n^{\top} V_n^{-1} \theta+2 \theta_n^{\top} V_n^{-1} \theta-2\left(\frac{y^{\top} x}{\sigma^2}+\theta_0^{\top} V_0^{-1}\right) \theta\right\}} \\
& =e^{-\frac{1}{2}}\left\{\operatorname{const}_2+\left(\theta-\theta_n\right)^{\top} V_n^{-1}\left(\theta-\theta_n\right)+2\left[\theta_n^{\top} v_n^{-1}-\frac{y^{\top} x}{\sigma^2}-\theta_0^{\top} V_0^{-1}\right]\theta\right\} \\

\end{aligned} \\
$$
in above steps we get $\color{red}V_{n}^{- 1} = V_{0}^{- 1} + \frac{1}{\sigma^{2}}X^{T}X$,  To make $\theta_n^{\top} v_n^{-1}-\frac{y^{\top} x}{\sigma^2}-\theta_0^{\top} V_0^{-1}=0$, so we get  $\color{red}\theta_{n} = V_{n}\left\lbrack {V_{0}}^{- 1}\theta_{0} + \frac{X^{T}Y}{\sigma^{2}} \right\rbrack$ to get ride of the last term in above equation,  and **we know gaussian’s integral , ****normalization constant** as below
$$
\int_{}^{}\mathbf{e}^{\mathbf{-}\frac{\mathbf{1}}{\mathbf{2}}\left( \mathbf{\theta -}\mathbf{\theta}_{\mathbf{n}} \right)^{\mathbf{T}}\mathbf{V}_{\mathbf{n}}^{\mathbf{- 1}}\left( \mathbf{\theta -}\mathbf{\theta}_{\mathbf{n}} \right)}\mathbf{d\theta}\mathbf{=}\left| \mathbf{2}\mathbf{\pi}\mathbf{V}_{\mathbf{n}} \right|^{\frac{\mathbf{1}}{\mathbf{2}}}
$$
The posterior $p(\theta │x,y,\sigma^{2}) = \left| 2\pi V_{n} \right|^{- \frac{1}{2}}e^{- {\frac{1}{2}\left( \theta - \theta_{0} \right)}^{T}V_{n}^{- 1}\left( \theta - \theta_{0} \right)\ }$, so the $\color{red}const_2 = ln(2\pi V_n)$ ?

when $\theta_{0} = 0$ and $V_{0} = \tau^{2}I_{d}\ $ which means the Gaussian prior is a spherical,  $\theta_{n} = V_{n}\left\lbrack \frac{X^{T}Y}{\sigma^{2}} \right\rbrack = \frac{1}{\sigma^{2}}V_{n}X^{T}Y = \left( \lambda I_{d} + X^{T}X \right)^{- 1}X^{T}y$ , $\lambda = \ \frac{\sigma^{2}}{\tau^{2}}$, it is ridge regression

if  $\tau \rightarrow \infty$ ($\tau$ is prior gaussian variance , gaussian is flat... uniform distribution..we don't know the prior) , also it means $\lambda \rightarrow 0$, it is maximum likelihood

### Theorem 4.4.1 (Bayes rule for linear Gaussian systems) (KPM)

Given condition，Likelihood $p(y|x)$and Prior distribution $p(x)$ (in Bayesian learning the x is the parameters, y is the data)
$$
\begin{split}
p(\mathbf{x}) & =\mathcal{N}\left(\mathbf{x} \mid \boldsymbol{\mu}_x, \mathbf{\Sigma}_x\right) \\
p(\mathbf{y} \mid \mathbf{x}) & =\mathcal{N}\left(\mathbf{y} \mid \mathbf{A} \mathbf{x}+\mathbf{b}, \mathbf{\Sigma}_y\right) 
\end{split}\tag{4.124}
$$
Theorem 4.4.1 (Bayes rule for linear Gaussian systems). Given a linear Gaussian system, as in Equation 4.124, the posterior $p(x|y)$ is given by the following:

posterior distribution
$$
\begin{split}
p(\mathbf{x} \mid \mathbf{y}) & =\mathcal{N}\left(\mathbf{x} \mid \boldsymbol{\mu}_{x \mid y}, \boldsymbol{\Sigma}_{x \mid y}\right) \\
\boldsymbol{\Sigma}_{x \mid y}^{-1} & =\boldsymbol{\Sigma}_x^{-1}+\mathbf{A}^T \boldsymbol{\Sigma}_y^{-1} \mathbf{A} \\
\boldsymbol{\mu}_{x \mid y} & =\boldsymbol{\Sigma}_{x \mid y}\left[\mathbf{A}^T \boldsymbol{\Sigma}_y^{-1}(\mathbf{y}-\mathbf{b})+\boldsymbol{\Sigma}_x^{-1} \boldsymbol{\mu}_x\right]
\end{split}\tag{4.125}
$$
after learning, the predict function (convolution of two gaussian ) $p(y)=\int p(y|x^*,\theta)p(\theta)d\theta)$

$$
p(\mathbf{y})=\mathcal{N}\left(\mathbf{y} \mid \mathbf{A} \boldsymbol{\mu}_x+\mathbf{b}, \mathbf{\Sigma}_y+\mathbf{A} \mathbf{\Sigma}_x \mathbf{A}^T\right) \tag{4.126}
$$
beside above proof, **please also refer to chapter 4.4.3 (KPM)**, utilize the 

where the precision matrix of the joint is defined as
$$
\boldsymbol{\Sigma}^{-1}=\left(\begin{array}{cc}
\boldsymbol{\Sigma}_x^{-1}+\mathbf{A}^T \boldsymbol{\Sigma}_y^{-1} \mathbf{A} & -\mathbf{A}^T \boldsymbol{\Sigma}_y^{-1} \\
-\boldsymbol{\Sigma}_y^{-1} \mathbf{A} & \boldsymbol{\Sigma}_y^{-1}
\end{array}\right) \triangleq \boldsymbol{\Lambda}=\left(\begin{array}{cc}
\boldsymbol{\Lambda}_{x x} & \boldsymbol{\Lambda}_{x y} \\
\boldsymbol{\Lambda}_{y x} & \boldsymbol{\Lambda}_{y y}
\end{array}\right)
$$
From Equation 4.69  and using the fact that $\boldsymbol{\mu}_y=\mathbf{A} \boldsymbol{\mu}_x+\mathbf{b}$, we have
$$
\begin{aligned}
p(\mathbf{x} \mid \mathbf{y}) & =\mathcal{N}\left(\boldsymbol{\mu}_{x \mid y}, \boldsymbol{\Sigma}_{x \mid y}\right) \\
\boldsymbol{\Sigma}_{x \mid y} & =\boldsymbol{\Lambda}_{x x}^{-1}=\left(\boldsymbol{\Sigma}_x^{-1}+\mathbf{A}^T \boldsymbol{\Sigma}_y^{-1} \mathbf{A}\right)^{-1} \\
\boldsymbol{\mu}_{x \mid y} & =\boldsymbol{\Sigma}_{x \mid y}\left(\boldsymbol{\Lambda}_{x x} \boldsymbol{\mu}_x-\boldsymbol{\Lambda}_{x y}\left(\mathbf{y}-\boldsymbol{\mu}_y\right)\right) \\
& =\boldsymbol{\Sigma}_{x \mid y}\left(\boldsymbol{\Sigma}_x^{-1} \boldsymbol{\mu}+\mathbf{A}^T \boldsymbol{\Sigma}_y^{-1}(\mathbf{y}-\mathbf{b})\right)
\end{aligned}
$$

### Naive  Bayesian learning

- Predict the probability that a point belongs to a certain class, using Bayes’ Theorem

$P(c|\textbf{x}) = \frac{P(\textbf{x}|c)P(c)}{P(\textbf{x})}$

- Problem: since x is a vector, computing $P(\textbf{x}|c)$can be very complex
- Naively assume that all features are conditionally independent from each other, in which case:
  $P(\mathbf{x}|c) = P(x_1|c) \times P(x_2|c) \times ... \times P(x_n|c)$
- Very fast: only needs to extract statistics from each feature.

![ml](/assets/Machine_LearningNote.assets/06_bayes_example.png)

### **Bayesian prediction Vs ML prediction**

Posterior mean: $\quad \theta_n=\left(\lambda \mathbf{I}_d+\mathbf{X}^T \mathbf{X}\right)^{-1} \mathbf{X}^T \mathbf{y}$
Posterior variance: $\quad V_n=\sigma^2\left(\lambda \mathbf{I}_d+\mathbf{X}^T \mathbf{X}\right)^{-1}$

**The given training data <font color=red>$D = (X, y)$</font>**.  To predict, Bayesians marginalize over the posterior $\theta_{ml}$ distribution.  $X_*$ is the new input:
$$
\begin{aligned}
P\left(y \mid x_*, D, \sigma^2\right) & =\int \mathcal{N}\left(y \mid x_*^T \theta, \sigma^2\right) \mathcal{N}\left(\theta \mid \theta_n, V_n\right) d \theta \\
& = \int N\left(y \mid x_{*}^T\theta, \sigma^2\right) \delta_{ml}(\theta) d \theta \\
& =\mathcal{N}\left(y \mid x_*^T \theta_n, \sigma^2+x_*^T V_n x_*\right)
\end{aligned}
$$
ML predictor have , which believe there only one true $\theta_{ML}$
$$
P\left(y \mid x_*, D, \sigma^2\right)=\mathcal{N}(y \mid \underbrace{x_*^T \theta_{M L}}, \sigma^2)
$$
$X\  = \ \left( x_{1},x_{2},\ \ldots x_{d} \right)$ $D\  = \ \left\{ \left( X_{1},\ \ Y_{1} \right),\ \left( X_{2},\ \ Y_{2} \right),\ \left( X_{3},\ \ Y_{3} \right),\ \ldots\left( X_{n},\ \ Y_{n} \right) \right\}$

$\mu\  = \ \left( \mu_{1},\mu_{2},\ \ldots\mu_{d} \right)$*,* $\Sigma\  = \begin{pmatrix}
\Sigma_{11} & \cdots & \Sigma_{1d} \\
 \vdots & \ddots & \vdots \\
\Sigma_{n1} & \cdots & \Sigma_{nd} \\
\end{pmatrix}$

### Bayesian Networks(TBU)

### Bayesian optimization 

[Bayesian Hyperparameter Optimization using Gaussian Processes | Brendan Hasz](https://brendanhasz.github.io/2019/03/28/hyperparameter-optimization.html)

[How to Implement Bayesian Optimization from Scratch in Python - MachineLearningMastery.com](https://machinelearningmastery.com/what-is-bayesian-optimization/)

[Tutorial #8: Bayesian optimization - Borealis AI](https://www.borealisai.com/research-blogs/tutorial-8-bayesian-optimization/#Incorporating_noisy_measurements)

- Bayesian Optimization is used when there is **no explicit objective function** and it’s expensive to evaluate the objective function.
- As shown in the next figure, **a GP is used along with an acquisition (utility) function to choose the next point to sample**, where it’s more likely to find the maximum value in an unknown objective function.
- A GP is constructed from the points already sampled and the next point is sampled from the region where **the GP posterior has higher mean (to exploit) and larger variance (to explore),** which is determined by the maximum value of the acquisition function (which is a function of GP posterior mean and variance).
- To choose the next point to be sampled, the above process is repeated.
- The next couple of figures show the basic concepts of Bayesian optimization using GP, the algorithm, how it works, along with a few popular acquisition functions.

<img src="/assets/Machine_LearningNote.assets/61e02-1ts1koxhkx84baw2oagenia.png" alt="Image for post" style="zoom:80%;" />

<img src="/assets/Machine_LearningNote.assets/8cfe5-1dwmhejlfkfqueob2xlumoa.png" alt="Image for post" style="zoom:80%;" />



**Bayesian Optimization Algorithm:**

**For t=1,2,… do**

1.  **Find $x_{t}$ by combining attributes of the posterior distribution in a utility function u and maximizing: $x_{i} = argmax_{x}u(x|D_{1:t - 1})$**

2.  **Sample the objective function $y_{t} = f\left( x_{t} \right) + \ \varepsilon_{t}$**

3.  **Augment the data $D_{1:t} = \{ D_{1:t - 1},\ (x_{t},\ y_{t}\}$ and update the GP**

**End for**
$$
\begin{aligned}
P\left(y_{t+1} \mid \mathcal{D}_{1: t}, \mathbf{x}_{t+1}\right) & =\mathcal{N}\left(\mu_t\left(\mathbf{x}_{t+1}\right), \sigma_t^2\left(\mathbf{x}_{t+1}\right)+\sigma_{\text {noise }}^2\right) \\
\mu_t\left(\mathbf{x}_{t+1}\right) & =\mathbf{k}^T\left[\mathbf{K}+\sigma_{\text {noise }}^2 \mathbf{I}\right]^{-1} \mathbf{y}_{1: t} \\
\sigma_t^2\left(\mathbf{x}_{t+1}\right) & =k\left(\mathbf{x}_{t+1}, \mathbf{x}_{t+1}\right)-\mathbf{k}^T\left[\mathbf{K}+\sigma_{\text {noise }}^2 \mathbf{I}\right]^{-1} \mathbf{k}
\end{aligned}
$$
basic we should choose the next point x where the 

- mean is high (exploitation)
- the variance is high (exploration)

**The acquisition function** 
$$
\mu(\mathbf{x})+\kappa \sigma(\mathbf{x})
$$
The acquisition function takes in the obtained GP fit (mean and variance at each point) and returns the hyper-parameter value for the next run of the machine learning algorithm. For more details on the derivation refer to this [article](https://distill.pub/2020/bayesian-optimization/).  More kernel visualizations can be found [here](https://www.cs.toronto.edu/~duvenaud/cookbook/).

#### **Probability of Improvements** 

$$
\begin{aligned}
\mathrm{PI}(\mathbf{x}) & =P\left(f(\mathbf{x}) \geq \mu^{+}+\xi\right) \\
& =\Phi\left(\frac{\mu(\mathbf{x})-\mu^{+}-\xi}{\sigma(\mathbf{x})}\right)
\end{aligned}
$$

​			The $\Phi $ is the probability cumulative function.

#### Expected improvement  (Expected utility criterion/Bayesian and decision theory)

$EU(a) = \sum_{x}^{}{u(x,a)p(x|data)}$, each action weighted by the probability 

a is action, $u(x,a)$ is cost/reward model.  **we want to expected improvement is max --- quantify the amount of the improvement.**

At iteration $n+1$, choose the point that minimizes the distance o the objective evaluated at the maximum $x$ *:
$$
\begin{aligned}
\mathbf{x}_{n+1} & =\arg \min _{\mathbf{x}} \mathbb{E}\left(\left\|f_{n+1}(\mathbf{x})-f\left(\mathbf{x}^{\star}\right)\right\| \mid \mathcal{D}_n\right) \\
& =\arg \min _{\mathbf{x}} \int\left\|f_{n+1}(\mathbf{x})-f\left(\mathbf{x}^{\star}\right)\right\| p\left(f_{n+1} \mid \mathcal{D}_n\right) d f_{n+1}
\end{aligned}
$$
we don't have the true objective ,  so we get expected improvement function as following:
$$
\mathbf{x}=\arg \max _{\mathbf{x}} \mathbb{E}\left(\max \left\{0, f_{n+1}(\mathbf{x})-f^{\max }\right\} \mid \mathcal{D}_n\right)
$$

$$
\begin{aligned}
\mathrm{EI}(\mathbf{x}) & = \begin{cases}\left(\mu(\mathbf{x})-\mu^{+}-\xi\right) \Phi(Z)+\sigma(\mathbf{x}) \phi(Z) & \text { if } \sigma(\mathbf{x})>0 \\
0 & \text { if } \sigma(\mathbf{x})=0\end{cases} \\
Z & =\frac{\mu(\mathbf{x})-\mu^{+}-\xi}{\sigma(\mathbf{x})}
\end{aligned}
$$
where $\phi(\cdot)$ and $\Phi(\cdot)$ denote the PDF and CDF of the standard Normal

<img src="/assets/Machine_LearningNote.assets/image-20230514211657419-1684472111352-17.png" alt="image-20230514211657419" style="zoom:50%;" />

```python
# Generate function prediction from the parameters and the loss (run bayesian regression)
loss_predicted,sigma,loss_evaluated = generate_prediction()

# Calculate expected improvement (finding the maximum of the information gain function)
expected_improvement = calculate_expected_improvement(loss_predicted, sigma, loss_evaluated)

def generate_prediction():
    list_of_parameters_stack, list_of_parameters = normalize_current_values()
    loss_evaluated = self.parameters_and_loss_dict['loss']
    if len(self.parameters_and_loss_dict['loss'])>1:
        loss_evaluated = (parameters_and_loss_dict['loss']-		
                          np.mean(self.parameters_and_loss_dict['loss']))/(np.std(self.parameters_and_loss_dict['loss'])+1e-6)

    loss_predicted, sigma  = fit_gp(list_of_parameters_stack, list_of_parameters, loss_evaluated)
    return loss_predicted,sigma,loss_evaluated
    
def calculate_expected_improvement(loss_predicted, sigma, loss_evaluated):

        # Calculate the expected improvement
        eps = 1e-6
        num =(loss_predicted-max(loss_evaluated)-eps)
        Z=num/sigma
        expected_improvement = num*scs.norm(0,1).cdf(Z)+sigma*scs.norm(0,1).pdf(Z)
        expected_improvement[sigma==0.0] = 0.0

        return expected_improvement
```



#### GP-UCB (Upper/lower confidence band)

Define the regret and cumulative regret as follows:
$$
\begin{aligned}
r(\mathbf{x}) & =f\left(\mathbf{x}^{\star}\right)-f(\mathbf{x}) \\
R_T & =r\left(\mathbf{x}_1\right)+\cdots+r\left(\mathbf{x}_T\right)
\end{aligned}
$$
The GP-UCB criterion is as follows:
$$
\operatorname{GP}-\mathrm{UCB}(\mathbf{x})=\mu(\mathbf{x})+\sqrt{\nu \beta_t} \sigma(\mathbf{x})
$$
Beta is set using a simple concentration bound:
With $\nu=1$ and $\beta_t=2 \log \left(t^{d / 2+2} \pi^2 / 3 \delta\right)$, it can be shown ${ }^2$ with high probability that this method is no regret, i.e. $\lim _{T \rightarrow \infty} R_T / T=0$. This in turn implies a lower-bound on the convergence rate for the optimization problem.

<img src="/assets/Machine_LearningNote.assets/image-20230514211800165.png" alt="image-20230514211800165" style="zoom:50%;" />

<img src="/assets/Machine_LearningNote.assets/image-20230514165712997.png" alt="image-20230514165712997" style="zoom:50%;" />

#### Thompson sampling (TBS)

$$
\alpha_{\text {THOMP. }}(\mathbf{x} ; \theta, \mathcal{D})=g(\mathbf{x}) \text {, where } g(\mathbf{x}) \text { is sampled form } \mathcal{G} \mathcal{P}\left(\mu(x), k\left(x, x^{\prime}\right)\right) \text {. }
$$

<img src="/assets/Machine_LearningNote.assets/image-20230514213022169.png" alt="image-20230514213022169" style="zoom:50%;" />

#### Entropy search and predictive entropy search (TBS)

$$
\alpha_{E S}(\mathbf{x} ; \theta, \mathcal{D})=H\left[p\left(x_{\text {min }} \mid \mathcal{D}\right)\right]-\mathbb{E}_{p(y \mid \mathcal{D}, \mathbf{x})}\left[H\left[p\left(x_{\text {min }} \mid \mathcal{D} \cup\{\mathbf{x}, y\}\right)\right]\right]
$$

### Bayesian Hyperparameter Optimization

- [A Conceptual Explanation of Bayesian Hyperparameter Optimization for Machine Learning | by Will Koehrsen | Towards Data Science](https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f)
- [Hyper-parameter tuning with Bayesian optimization or how I carved boats from wood | by Aliaksei Mikhailiuk | Towards Data Science](https://towardsdatascience.com/bayesian-optimization-or-how-i-carved-boats-from-wood-examples-and-code-78b9c79b31e5)

[Bayesian approaches](https://www.iro.umontreal.ca/~bengioy/cifar/NCAP2014-summerschool/slides/Ryan_adams_140814_bayesopt_ncap.pdf), in contrast to random or grid search, keep track of past evaluation results which they use to form a probabilistic model mapping hyperparameters to a probability of a score on the objective function:
$$
P(\text { score } \mid \text { hyperparameters })
$$
[In the literature](https://sigopt.com/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf), **this model is called a “surrogate” for the objective function and is represented as p(y | x)**. The surrogate is much easier to optimize than the objective function and Bayesian methods work by finding the next set of hyperparameters to evaluate on the actual objective function by selecting hyperparameters that perform best on the surrogate function. In other words:

  1. **Build a surrogate probability model of the objective function**
2. **Find the hyperparameters that perform best on the surrogate** -- have one best guess from the acquisition function. 
3. **Apply these hyperparameters to the true objective function** -- get the new loss data from loss function. (scores)
4. **Update the surrogate model incorporating the new results** --- put the x (hyperparameters) and y (loss data from true object function) to fit/update the model
5. **Repeat steps 2–4 until max iterations or time is reached**

At a high-level, Bayesian optimization methods are efficient because they choose the next hyperparameters in an *informed manner***.** The basic idea is: **spend a little more time selecting the next hyperparameters in order to <font color=red>make fewer calls to the objective function.</font>**

### Sequential Model-Based Optimization

[Sequential model-based optimization (SMBO) methods (SMBO)](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) are a formalization of Bayesian optimization. The sequential refers to running trials one after another, each time trying better hyperparameters by applying Bayesian reasoning and updating a probability model (surrogate).

There are five aspects of model-based hyperparameter optimization:

1. **A domain of hyperparameters over which to search**
2. **An objective function which takes in hyperparameters and outputs a score that we want to minimize (or maximize)**
3. **The surrogate model of the objective function**
4. **A criteria, called a selection function, for evaluating which hyperparameters to choose next from the surrogate model**
5. **A history consisting of (score, hyperparameter) pairs used by the algorithm to update the surrogate model**

There are several variants of [SMBO methods that differ](https://sigopt.com/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf) in steps 3–4, namely, how they build a surrogate of the objective function and the criteria used to select the next hyperparameters. Several common choices for the surrogate model are [Gaussian Processes](https://en.wikipedia.org/wiki/Gaussian_process), [Random Forest Regressions](http://aad.informatik.uni-freiburg.de/papers/13-GECCO-BBOB_SMAC.pdf), and Tree Parzen Estimators (TPE) while the most common choice for step 4 is Expected Improvement. In this post, we will focus on TPE and Expected Improvement.

#### Domain

In the case of random search and grid search, the domain of hyperparameters we search is a grid. 

#### Objective Function -- the true objective function

The objective function takes in hyperparameters and outputs a single real-valued score that we want to minimize (or maximize) --- loss function or maximum likelihood function. As an example, let’s consider the case of building a random forest for a regression problem. The hyperparameters we want to optimize are shown in the hyperparameter grid above and the score to **minimize is the Root Mean Squared Error**. 

While the objective function looks simple, it is very expensive to compute! If the objective function could be quickly calculated, then we could try every single possible hyperparameter combination (like in grid search). If we are using a simple model, a small hyperparameter grid, and a small dataset, then this might be the best way to go. However, in cases where the objective function may take hours or even days to evaluate, we want to limit calls to it.

**The entire concept of Bayesian model-based optimization is to reduce the number of times the objective function needs to be run by choosing only the most promising set of hyperparameters to evaluate based on previous calls to the evaluation function.** The next set of hyperparameters are selected based on a model of the objective function called a surrogate.

#### Surrogate Function (Probability Model)

The surrogate function, also called the response surface, is the probability representation of the objective function built using previous evaluations. This is called sometimes called a response surface because it is a **high-dimensional mapping of hyperparameters to the probability of a score on the objective function**. Below is a simple example with only two hyperparameters:

![img](/assets/Machine_LearningNote.assets/0aBsprZzniYMB0KWc.png)

There are several different forms of the surrogate function including Gaussian Processes and Random Forest regression. However, in this post we will focus on the Tree-structured Parzen Estimator as [put forward by Bergstra et al](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) in the paper “Algorithms for Hyper-Parameter Optimization”. 

#### Selection Function

please refer to the **The acquisition function** 

The selection function is the criteria by which the next set of hyperparameters are chosen from the surrogate function.

Moreover, because the surrogate is just an estimate of the objective function, the selected hyperparameters may not actually yield an improvement when evaluated and the surrogate model will have to be updated. This updating is done based on the current surrogate model and the history of objective function evaluations.

#### History

Each time the algorithm proposes a new set of candidate hyperparameters, **it evaluates them with the actual objective function and records the result in a pair (score, hyperparameters)**. These records form the **history**. The algorithm builds "Surrogate function" using the history to come up with a probability model of the objective function that improves with each iteration.

### Bayesian optimization with GPs in practice

consider how to deal with noisy observations, **how to choose a kernel, how to learn the parameters of that kernel, how to exploit parallel sampling of the function**. 

in noise GP model, We no longer observe the function values $\mbox{f}[\mathbf{x}]$ directly, but observe noisy corruptions $y[\mathbf{x}] = \mbox{f}[\mathbf{x}]+\epsilon$of them. The joint distribution of previously observed noisy function values $\mathbf{y}$ and a new unobserved point $f^{*}$ becomes: 
$$
\begin{equation}Pr\left(\begin{bmatrix}\mathbf{y}\\f^{*}\end{bmatrix}\right) = \mbox{Norm}\left[\mathbf{0}, \begin{bmatrix}\mathbf{K}[\mathbf{X},\mathbf{X}]+\sigma^{2}_{n}\mathbf{I} & \mathbf{K}[\mathbf{X},\mathbf{x}^{*}]\\ \mathbf{K}[\mathbf{x}^{*},\mathbf{X}]& \mathbf{K}[\mathbf{x}^{*},\mathbf{x}^{*}]\end{bmatrix}\right],  \tag{13}\end{equation}
$$
and the conditional probability of a new point becomes:
$$
\begin{eqnarray}\label{eq:noisy_gp_posterior}Pr(f^{*}|\mathbf{y}) &=& \mbox{Norm}[\mu[\mathbf{x}^{*}],\sigma^{2}[\mathbf{x}^{*}]],  \tag{14}\end{eqnarray}
$$
where
$$
\begin{eqnarray}\mu[\mathbf{x}^{*}]&=& \mathbf{K}[\mathbf{x}^{*},\mathbf{X}](\mathbf{K}[\mathbf{X},\mathbf{X}]+\sigma^{2}_{n}\mathbf{I})^{-1}\mathbf{f}\nonumber \\\sigma^{2}[\mathbf{x}^{*}] &=& \mathbf{K}[\mathbf{x}^{*},\mathbf{x}^{*}]\!-\!\mathbf{K}[\mathbf{x}^{*}, \mathbf{X}](\mathbf{K}[\mathbf{X},\mathbf{X}]+\sigma^{2}_{n}\mathbf{I})^{-1}\mathbf{K}[\mathbf{X},\mathbf{x}^{*}].  \tag{15}\end{eqnarray}
$$
Incorporating noise means that there is uncertainty about the function even where we have already sampled points , and so sampling twice at the same position or at very similar positions could be sensible.



![Tutorial #8: Bayesian optimization](/assets/Machine_LearningNote.assets/T9_6.png)

#### Kernel choice

When we build the model of the function and its uncertainty, we are assuming that the function is smooth. If this was not the case, then we could say nothing at all about the function between the sampled points. The details of this smoothness assumption are embodied in the choice of kernel covariance function. 

We can visualize the covariance function by drawing samples from the Gaussian process prior. In one dimension, we do this by defining an evenly spaced set of points $\mathbf{X}=\begin{bmatrix}\mathbf{x}_{1},& \mathbf{x}_{2},&\cdots,& \mathbf{x}_{I}\end{bmatrix}$, drawing a sample from $\mbox{Norm}[\mathbf{0}, \mathbf{K}[\mathbf{X},\mathbf{X}]]$ and then plotting the results. In this section, we’ll consider several different choices of covariance function, and use this method to visualize each.

**Squared Exponential Kernel:** include the amplitude $\alpha$ which controls the overall amount of variability and the length scale $\lambda$ which controls the amount of smoothness:
$$
\begin{equation}\label{eq:bo_squared_exp}\mbox{k}[\mathbf{x},\mathbf{x}’] = \alpha^{2}\cdot \mbox{exp}\left[-\frac{d^{2}}{2\lambda}\right],\nonumber \end{equation}
$$
where $d$ is the Euclidean distance between the points: $\begin{equation}d = \sqrt {\left(\mathbf{x}-\mathbf{x}’\right)^{T}\left(\mathbf{x}-\mathbf{x}’\right)}. \end{equation}$

**Matérn kernel**:
$$
\begin{equation}\mbox{k}[\mathbf{x},\mathbf{x}’] = \alpha^{2}\cdot \exp\left[-\frac{d}{\lambda^{2}}\right], \tag{17}\end{equation}
$$


**Periodic Kernel:**  $\tau$ is the period of the oscillation and the other parameters have the same meanings as before.

$\begin{equation}\mbox{k}[\mathbf{x},\mathbf{x}^\prime] = \alpha^{2} \cdot \exp \left[ \frac{-2(\sin[\pi d/\tau])^{2}}{\lambda^2} \right], \tag{20}\end{equation}$

#### Learning GP parameters

**1. Maximum likelihood:** similar to training ML models, **we can choose these parameters by maximizing the marginal likelihood** (i.e., the likelihood of the data after marginalizing over the possible values of the function): 
$$
\begin{eqnarray}\label{eq:bo_learning}        Pr(\mathbf{y}|\mathbf{x},\boldsymbol\theta)&=&\int Pr(\mathbf{y}|\mathbf{f},\mathbf{x},\boldsymbol\theta)d\mathbf{f}\nonumber\\
        &=& \mbox{Norm}_{y}[\mathbf{0}, \mathbf{K}[\mathbf{X},\mathbf{X}]+\sigma^{2}_{n}\mathbf{I}], \tag{21}
    \end{eqnarray}
$$
where $\boldsymbol\theta$ contains the unknown parameters in the kernel function and the measurement noise $\sigma^{2}_{n}$. 

<font color="red">**In Bayesian optimization, we are collecting the observations sequentially, and where we collect them will depend on the kernel parameters, and we would have to interleave the processes of acquiring new points and optimizing the kernel parameters.**</font>

**2. Full Bayesian approach:** here we would choose a prior distribution $Pr(\boldsymbol\theta)$on the kernel parameters of the Gaussian process and combine this with the likelihood in equation (21) to compute the posterior. We then weight the acquisition functions according to this posterior:
$$
\begin{equation}\label{eq:snoek_post}        \hat{a}[\mathbf{x}^{*}]\propto \int a[\mathbf{x}^{*}|\boldsymbol\theta]Pr(\mathbf{y}|\mathbf{x},\boldsymbol\theta)Pr(\boldsymbol\theta).  \tag{22}    \end{equation}
$$
In practice this would usually be done using an Monte Carlo approach in which the posterior is represented by a set of samples (see [Snoek *et* al., 2012](https://arxiv.org/abs/1206.2944)) and we sum together multiple acquisition functions derived from these kernel parameter samples (figure 9).

<img src="/assets/Machine_LearningNote.assets/T9_9.png" alt="Tutorial #8: Bayesian optimization" style="zoom: 50%;" />

Figure: Bayesian approach for handling length scale of kernel from Snoek et al., 2012. a-c) We fit the model with three different length scales and compute the acquisition function for each. d) We compute a weighted sum of these acquisition functions (black curve) where the weight is given by posterior probability of the data at each scale (see equation 22). We choose the next point by **finding the maximum of this weighted function (black arrow)**. In this way, we approximately marginalize out the length scale.



## Optimization approach

> cost function
>
> $J(\theta) = (Y - X\theta)^{T}(Y - X\theta) = \sum_{i = 1}^{n}\left( y_{i} - x_{i}^{T}\theta \right)^{2}(remove\ the\ data\ batch\ n\ here) = \ \sum_{i = 1}^{n}\left( y_{i} - {\widehat{y}}_{i} \right)^{2}$
>
> Gradient or solve this directly (=0, when equal to 0, it mean reach the minimal point):
>
> 非矩阵求解：
>
> <img src="C:\Users\zphil\AppData\Roaming\Typora\typora-user-images\image-20230505160844056.png" alt="image-20230505160844056" style="zoom:80%;" />
>
> 矩阵求解：
>
> 
>
> $\frac{\partial J(\theta)}{\partial\theta} = \frac{\partial}{\partial\theta}\lbrack Y^{T}Y + \theta^{T}X^{T}X\theta - 2Y^{T}X\theta\rbrack = 0 + 2X^{T}X - 2X^{T}Y = 0$
>
> $$\theta = \left( X^{T}X \right)^{- 1}X^{T}Y$$

### Gradient 

> <img src="/assets/Machine_LearningNote.assets/image-20230505160859363.png" alt="image-20230505160859363" style="zoom:80%;" />
>
> The gradient vector of $f(\theta)$ , $\nabla_{\theta}f(\theta) = \begin{bmatrix}
> \begin{matrix}
> \frac{\partial f(\theta)}{\partial\theta_{1}} \\
> \frac{\partial f(\theta)}{\partial\theta_{2}} \\
> \end{matrix} \\
>  \vdots \\
> \frac{\partial f(\theta)}{\partial\theta_{n}} \\
> \end{bmatrix}$, the

### Hessian 

> <img src="/assets/Machine_LearningNote.assets/image-20230505194109147.png" alt="image-20230505194109147" style="zoom:80%;" />
>
> <img src="/assets/Machine_LearningNote.assets/image-20230505194137307.png" alt="image-20230505194137307" style="zoom:80%;" />
>
> <img src="/assets/Machine_LearningNote.assets/image-20230505194200413.png" alt="image-20230505194200413" style="zoom:80%;" />
>
> ![image-20230505194215095](/assets/Machine_LearningNote.assets/image-20230505194215095.png)

### Online learning/Stochastic gradient descent (SGD)

Batch $\theta_{k + 1} = \theta_{k} + \eta\sum_{i = 1}^{n}{x_{i}^{T}\left( y_{i} - x_{i}\theta_{k} \right)}$

Online $\theta_{k + 1} = \theta_{k} + \eta x_{k}^{T}\left( y_{k} - x_{k}\theta_{k} \right)$

Mini-batch $\theta_{k + 1} = \theta_{k} + \eta\sum_{i = 1}^{20}{x_{i}^{T}\left( y_{i} - x_{i}\theta_{k} \right)}$

### 蒙特卡洛梯度估计方法（MCGE）

机器学习中最常见的优化算法是基于梯度的优化方法，当目标函数是一个类似如下结构的随机函数 F(θ) 时：

![img](/assets/Machine_LearningNote.assets/539de60bf87244eaada0a6e617246600.png)

优化该类目标函数，最核心的计算问题是对随机函数 F(θ) 的梯度进行估计，即：

![img](/assets/Machine_LearningNote.assets/02a93e1f8c3e49fd8ffaa61615f1093c.png)

随机函数梯度估计在机器学习以及诸多领域都是核心计算问题，比如：变分推断，一种常见的近似贝叶斯推断方法；强化学习中的策略梯度算法；实验设计中的贝叶斯优化和主动学习方法等。其中，对于函数期望类目标问题，最常见的是基于蒙特卡洛采样的方法。

**蒙特卡洛采样（MCS）**

MCS 是一种经典的求解积分方法，公式（1）中的问题通常可以用 MCS 近似求解如下：

![img](/assets/Machine_LearningNote.assets/1e450b2228dc413a8e37f7f921400f54.png)

其中，$\hat{x}^{(n)}$ 采样自分布 $p(x;θ)$，由于采样的不确定性和有限性，这里 $\bar{F}_N$ 是一个随机变量，公式（3）是公式（1）的蒙特卡洛估计器（MCE）。这类方法非常适用于求解形式如公式（1）的积分问题，尤其是当分布 $p(x;θ)$ 非常容易进行采样的时候。

## The least squares 

> estimates: $\widehat{\theta} = {X^{T}X}^{- 1}X^{T}y$

### 

## ERM, Empirical risk minimization

![image-20230505235902121](C:\Users\zphil\AppData\Roaming\Typora\typora-user-images\image-20230505235902121.png)

![image-20230505235911011](C:\Users\zphil\AppData\Roaming\Typora\typora-user-images\image-20230505235911011.png)

## Univariate/Multivariate Gaussian distribution 

### Reference material 

https://gaussianprocess.org/gpml/

https://zhuanlan.zhihu.com/p/31427491

https://www.cnblogs.com/tornadomeet/archive/2013/06/15/3137239.html

https://sandipanweb.wordpress.com/2020/12/08/gaussian-process-regression-with-python/

### **Univariate Gaussian distribution probability density function(pdf)**

 $p(x) = \left( 2\pi\sigma^{2} \right)^{- 1\text{/}2}e^{- \frac{1}{2\sigma^{2}}(x - \mu)^{2}}$ $x\  \sim \mathcal{N}\left( \mu,\sigma^{2} \right)$

 in python sampling normal distribution like :   x = sigma * np.random.randn(10) + mu

sampling the distribution using the uniform distribution generate the random number and mapping it to the cumulative Distribution Functions (CDFs) function to get the samples 

<img src="/assets/Machine_LearningNote.assets/image-20230505135313111.png" alt="image-20230505135313111" style="zoom: 67%;" />

### **Multivariate Gaussian distribution**

$X = (x_1, x_2....)^T$ is multiple random variables, The pdf of an n-dimensional Gaussian is : $$p(X) = (2\pi\Sigma)^{- 1\text{/}2}e^{- \frac{1}{2}(X - \mu)^{T}\Sigma^{- 1}(X - \mu)}$$

$\mu = \ \binom{\mu_{1}}{\begin{matrix}
\vdots \\
\mu_{n} \\
\end{matrix}}\  = \binom{E\left( x_{1} \right)}{\begin{matrix}
\vdots \\
E\left( x_{n} \right) \\
\end{matrix}}\ $ *and* $\Sigma\  = \begin{pmatrix}
\sigma_{11} & \cdots & \sigma_{1n} \\
\vdots & \ddots & \vdots \\
\sigma_{n1} & \cdots & \sigma_{nn} \\
\end{pmatrix}\mathbb{= E}\left\lbrack (X - \mu)(X - \mu)^{T} \right\rbrack$

in case two independent univariate Gaussian variables , $x_1 = N(\mu_1, \sigma^2), x_2 = N(\mu_2, \sigma^2)$

their joint distribution is 
$$
\begin{aligned}

 p(x_1, x_2) &= p(x_1)p(x_2) \\
 		  	&=(2\pi\sigma^2)^{-1}e^{-1/2 [(x_1-\mu_1),(x_2-\mu_2)]
 		  	\begin{bmatrix}
 		  	\sigma^2, 0\\
 		  	0, \sigma^2\\
 		  	\end{bmatrix}^{-1}
 		  	\begin{bmatrix}
 		  	x_1-\mu_1\\
 		  	x_2-\mu_2\\
 		  	\end{bmatrix}}
\end{aligned}
$$
$$ \Sigma = \begin{bmatrix}
 		  	\sigma^2, 0\\
 		  	0, \sigma^2\\
 		  	\end{bmatrix}^{-1} $$ and $$ \mu = \begin{bmatrix} \mu_1 \\\mu_2 \\\end{bmatrix} $$
$$
X  \sim N(\mu, \Sigma), \Sigma = BB^T (cholesky rule) \\
X  \sim \mu + BN(0, 1)
$$

### Theorem 4.3.1 (marginal and conditionals of an MVN)

**Theorem 4.3.1 (marginal and conditionals of an MVN), suppose** $\mathbf{X =}\left( \mathbf{X}_{\mathbf{1}}\mathbf{,}\mathbf{X}_{\mathbf{2}} \right)\mathbf{\ }$ **is a jointly Gaussian with parameters**

$$\mu = \begin{pmatrix}
\mu_{1} \\
\mu_{2} \\
\end{pmatrix},\ \Sigma = \begin{pmatrix}
\Sigma_{11} & \Sigma_{12} \\
\Sigma_{21} & \Sigma_{22} \\
\end{pmatrix},\ \Lambda = \Sigma^{- 1} = \begin{pmatrix}
\Lambda_{11} & \Lambda_{12} \\
\Lambda_{21} & \Lambda_{22} \\
\end{pmatrix}$$

then the marginals are given by

$$p\left( x_{1} \right) = N\left( x_{1} \middle| \mu_{1},\ \Sigma_{11} \right)$$

$$p\left( x_{2} \right) = N\left( x_{2} \middle| \mu_{2},\ \Sigma_{22} \right)$$

and the posterior conditional distribution  is given by (refer to KPM to know following equations detailly)
$$
\begin{split}
p\left(\mathbf{x}_1 \mid \mathbf{x}_2\right) & =\mathcal{N}\left(\mathbf{x}_1 \mid \boldsymbol{\mu}_{1 \mid 2}, \boldsymbol{\Sigma}_{1 \mid 2}\right) \\
\boldsymbol{\mu}_{1 \mid 2} & =\boldsymbol{\mu}_1+\boldsymbol{\Sigma}_{12} \boldsymbol{\Sigma}_{22}^{-1}\left(\mathbf{x}_2-\boldsymbol{\mu}_2\right) \\
& =\boldsymbol{\mu}_1-\boldsymbol{\Lambda}_{11}^{-1} \boldsymbol{\Lambda}_{12}\left(\mathbf{x}_2-\boldsymbol{\mu}_2\right) \\
& =\boldsymbol{\Sigma}_{1 \mid 2}\left(\boldsymbol{\Lambda}_{11} \boldsymbol{\mu}_1-\boldsymbol{\Lambda}_{12}\left(\mathbf{x}_2-\boldsymbol{\mu}_2\right)\right) \\
\boldsymbol{\Sigma}_{1 \mid 2} & =\boldsymbol{\Sigma}_{11}-\boldsymbol{\Sigma}_{12} \boldsymbol{\Sigma}_{22}^{-1} \boldsymbol{\Sigma}_{21}=\boldsymbol{\Lambda}_{11}^{-1} 
\end{split}\tag{4.69}
$$
推导过程看KPM chapter 4.3.4.3.. **this 4.69 also used for the proof of the Theorem 4.4.1(Bayes rule for linear Gaussian systems).**
$$
\begin{eqnarray*}
p\left(\mathbf{x}_1, \mathbf{x}_2\right) &=&p\left(\mathbf{x}_1 \mid \mathbf{x}_2\right) p\left(\mathbf{x}_2\right) \tag{4.118} \\
&=&\mathcal{N}\left(\mathrm{x}_1 \mid \boldsymbol{\mu}_{1 \mid 2}, \Sigma_{1 \mid 2}\right) \mathcal{N}\left(\mathrm{x}_2 \mid \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_{22}\right) \tag{4.119}\\
\boldsymbol{\mu}_{1 \mid 2} &=& \boldsymbol{\mu}_1+\Sigma_{12} \Sigma_{22}^{-1}\left(\mathrm{x}_2-\boldsymbol{\mu}_2\right) \tag{4.120}\\
\Sigma_{1 \mid 2} &=& \Sigma / \Sigma_{22}=\Sigma_{11}-\Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}\tag{4.121}
\end{eqnarray*}
$$
也即是根据Joint distribution $p\left(\mathbf{x}_1, \mathbf{x}_2\right)$ 的mean，covariance 能求出各个conditional 或者又叫marginal distribution 的$\mu_{1 \mid 2}$和$\Sigma_{1 \mid 2}$ 

### Gaussian process (random process)

高斯过程，从字面上分解，我们就可以看出他包含两部分

\- 高斯，指的是高斯分布

\- 过程，指的是随机过程

<img src="assets/v2-13bb48ae827530521164f00f9a89311e_720w-1683710287318-2.png" alt="v2-13bb48ae827530521164f00f9a89311e_720w" style="zoom: 67%;" />

首先当随机变量是1维的时候，我们称之为一维高斯分布，概率密度函数 $p(x)=N(μ,σ2)$当随机变量的维度上升到有限的 p 维的时候，就称之为高维高斯分布， $p(x)=N(μ,Σp×p)$. 而高斯过程则更进一步，他是一个定义在连续域上的无限多个高斯随机变量所组成的随机
过程，换句话说，高斯过程是一个无限维的高斯分布

对于一个连续域 T （假设他是一个时间轴），如果我们在连续域上任选 n 个时刻： t1,t2,t3,...,tn∈T ，使得获得的一个 n 维向量 ${ξ_1,ξ_2,ξ_3,...,ξ_n}$ 都满足其是一个 n 维高斯分布，那么这个 ${ξ_t}$ 就是一个高斯过程

对于一个 p 维的高斯分布而言，决定他的分布是两个参数，一个是 p 维的均值向量 μp ，他反映了 p 维高斯分布中每一维随机变量的期望，另一个就是 p×p 的协方差矩阵 Σp×p ，他反映了高维分布中，每一维自身的方差，以及不同维度之间的协方差

定义在连续域 T 上的高斯过程其实也是一样，他是无限维的高斯分布，他同样需要描述每一个时间点 t 上的均值，但是这个时候就不能用向量了，因为是在连续域上的，维数是无限的，因此就应该定义成一个关于时刻 t 的函数： $m(t)$ 。

协方差矩阵也是同理，无限维的情况下就定义为一个核函数 $k(s,t)$ ，其中 s 和 t 表示任意两个时刻，核函数也称协方差函数。核函数是一个高斯过程的核心，他决定了高斯过程的性质，在研究和实践中，核函数有很多种不同的类型，他们对高斯过程的衡量方法也不尽相同，这里我们介绍和使用最为常见的一个核函数：径向基函数 RBF - Squared Exponential Kernel ，其定义如下

$$k(x_a, x_b) = \sigma^2 \exp \left(-\frac{ \left\Vert x_a - x_b \right\Vert^2}{2\ell^2}\right)$$

这里面的 σ 和 l 是径向基函数的超参数，使我们提前可以设置好的，例如我们可以让 σ=1 ， l=1 ，从这个式子中，我们可以解读出他的思想

和 t 表示高斯过程连续域上的两个不同的时间点， $||s−t||^2$ 是一个二范式，简单点说就是 s 和 t 之间的距离，径向基函数输出的是一个标量，他代表的就是 s 和 t 两个时间点各自所代表的高斯分布之间的协方差值，很明显径向基函数是一个关于 s，t 距离负相关的函数，两个点距离越大，两个分布之间的协方差值越小，即相关性越小，反之越靠近的两个时间点对应的分布其协方差值就越大。

由此，高斯过程的两个核心要素：均值函数和核函数的定义我们就描述清楚了，按照高斯过程存在性定理，一旦这两个要素确定了，那么整个高斯过程就确定了：

$$ξ_t∼GP(m(t),k(t,s))$$



**The random process** is the $f(x)$, x is not random variable , for example it could time $t$. but the $f(x)$ is random process which give a certain $x_i$, the $f(x_i)$ is one random variable . 

$f(x) \sim GP(m(x), k(x,x'))$ 

 means the multiple random $ f(x) = ((f(x_1),f(x_2)...f(x_n))$ have joint gaussian distribution

$\mu = \ \binom{\mu_1={m(1)}}{\begin{matrix}
\vdots \\
\mu_n={m(X_n)} \end{matrix}}$ *and* $\Sigma\  = \begin{pmatrix}
\sigma_{11}=k(x_1,x_1) & \cdots & \sigma_{1n}=k(x_1,x_n) \\
\vdots & \ddots & \vdots \\
\sigma_{n1}=k(x_n,x_1) & \cdots & \sigma_{nn}=k(x_n,x_n) \\
\end{pmatrix}$ 

and $$k(x_a, x_b) = \sigma^2 \exp \left(-\frac{ \left\Vert x_a - x_b \right\Vert^2}{2\ell^2}\right)$$ , those $\sigma^2$ and $\ell$ need to learn by data.

<img src="assets/image-20230505145229063.png" alt="image-20230505145229063" style="zoom:80%;" />

 **Cholesky Decomposition **
定理: 若 $A \in R^{n \times n}$ 是对称正定矩阵 $Q$ ，则存在一个对角元全为正数的下三角矩阵 $L \in R^{n \times n}$ ，使得 $A=L^T$ 成立。
$\mathrm{L}$ 是一个下三角形，形式是这样的:
$$
L=\left(\begin{array}{cccc}
l_{11} & 0 & \cdots & 0 \\
l_{21} & l_{22} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
l_{n 1} & l_{n 2} & \cdots & l_{n n}
\end{array}\right) \quad L^T=\left(\begin{array}{cccc}
l_{11} & l_{21} & \cdots & l_{n 1} \\
0 & l_{22} & \cdots & l_{n 2} \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & l_{n n}
\end{array}\right)
$$
So sampling ： $f  \sim \mu + LN(0, 1)$ **this is the NOT mean function**   (n,1) = (n,1) + (n,n)(n,1), we sampling $N(0,1)$ for each $f_i$

kernel function https://peterroelants.github.io/posts/gaussian-process-kernels/

### **Noiseless GP regression**

while give new $f^*$ where there are given $\bar{f} \sim N(0,k)$ http://www.cnblogs.com/hxsyl/p/5229746.ht

<img src="/assets/Machine_LearningNote.assets/image-20230515090059891.png" alt="image-20230515090059891" style="zoom: 50%;" />



合法的协方差矩阵就是 (symmetric) Positive Semi-definite Matrix

　　矩阵A正定是指,对任意的X≠0恒有$X^TAX＞0$。  
　　矩阵A半正定是指,对任意的X≠0恒有$X^TAX≥0$ 

　　判定A是半正定矩阵的充要条件是：A的所有顺序主子式大于或等于零。

　　如果你了解SVM的话，就会接触过一个著名的Mercer Theorem，（当然如果你了解泛函分析的话也会接触过 ），这个M定理是在说：一个矩阵是Positive Semi-definite Matrix当且仅当该矩阵是一个Mercer Kernel .

**所以这是一种贝叶斯方法，和OLS回归不同，这个方法给出了预测值所隶属的整个（后验）概率分布的。再强调一下，我们得到的是f\* 的整个分布！不是一个点估计**

<img src="assets/image-20230510234021480-1683733223487-9.png" alt="image-20230510234021480" style="zoom:60%;" />

***GP posterior general (Bayesian)*** $\mathbf{D = \{}\left( \mathbf{X}_{\mathbf{i}}\mathbf{,\ }\mathbf{f}_{\mathbf{i}} \right)\mathbf{,\ i = 1:N\}}$***,*** where $f_i=f(x_i)$ is the noise-free observation of the function evaluated at $x_i$

***Given a test set*** $\mathbf{X}_{\mathbf{*}}\mathbf{\ of\ size\ }\mathbf{N}_{\mathbf{*}}\mathbf{\times D}$, we want to predict the function output $f_{*}$ , By definition of the GP, the joint distribution has the following form***

$$\begin{pmatrix}
\mathbf{f} \\
\mathbf{f}_{\mathbf{*}} \\
\end{pmatrix}\mathbf{\sim N}\left( \begin{pmatrix}
\mathbf{\mu} \\
\mathbf{\mu}_{\mathbf{*}} \\
\end{pmatrix}\mathbf{,\ }\begin{pmatrix}
\mathbf{K} & \mathbf{K}_{\mathbf{*}} \\
\mathbf{K}_{\mathbf{*}}^{\mathbf{T}} & \mathbf{K}_{\mathbf{**}} \\
\end{pmatrix} \right)$$

***Where*** $\mathbf{K = \kappa}\left( \mathbf{X,X} \right)\mathbf{\ }$***,*** $\mathbf{K}_{\mathbf{*}}\mathbf{=}\mathbf{\kappa}\left( \mathbf{X,}\mathbf{X}_{\mathbf{*}} \right)\mathbf{\ is\ }\mathbf{N \times N}_{\mathbf{*}}$ ***and*** $\mathbf{K}_{\mathbf{**}}\mathbf{=}\mathbf{\kappa}\left( \mathbf{X}_{\mathbf{*}}\mathbf{,}\mathbf{X}_{\mathbf{*}} \right)\mathbf{\ is\ }{\mathbf{N}_{\mathbf{*}}\mathbf{\times N}}_{\mathbf{*}}$,  the posterior has the following form (by utilize the 4.69):
$$
\mathbf{p}\left( \mathbf{f}_{\mathbf{*}} \middle| \mathbf{X}_{\mathbf{*}}\mathbf{,\ X,\ f} \right)\mathbf{= N}\left( \mathbf{f}_{\mathbf{*}} \middle| \mathbf{\mu}_{\mathbf{*}}\mathbf{,\ }\mathbf{\Sigma}_{\mathbf{*}} \right) \tag{15.7}
$$

$$
\mathbf{\mu}_{\mathbf{*}}\mathbf{= \ \mu}\left( \mathbf{X}_{\mathbf{*}} \right)\mathbf{+}\mathbf{K}_{\mathbf{*}}^{\mathbf{T}}\mathbf{K}^{\mathbf{- 1}}\mathbf{(f - \mu}\left( \mathbf{X} \right)\mathbf{)}\tag{15.8}
$$

$$
\mathbf{\Sigma}_{\mathbf{*}}\mathbf{=}\mathbf{K}_{\mathbf{**\ }}\mathbf{-}\mathbf{K}_{\mathbf{*}}^{\mathbf{T}}\mathbf{K}^{\mathbf{- 1}}\mathbf{K}_{\mathbf{*}}\tag{15.9}
$$

here we could use a squared exponential kernel, aka Gaussian kernel or RBF kernel. In 1d, this is given by

$$\mathbf{\kappa}\left( \mathbf{X,}\mathbf{X}^{\mathbf{'}}\mathbf{\ } \right)\mathbf{=}\mathbf{\sigma}^{\mathbf{2}}\mathbf{exp( -}\frac{\mathbf{1}}{\mathbf{2}\mathcal{l}^{\mathbf{2}}}\left( \mathbf{x -}\mathbf{x}^{\mathbf{'}} \right)^{\mathbf{2}}\mathbf{) }$$

**the real algorithm for this computing**
$$
\begin{aligned}
& \overline{f_*}=\mathbf{k}_*^T \mathbf{K}_y^{-1} \mathbf{y}\\
& \mathbf{K}_y=\mathbf{L L}^T \\
& \boldsymbol{\alpha}=\mathbf{K}_y^{-1} \mathbf{y}=\mathbf{L}^{-T} \mathbf{L}^{-1} \mathbf{y}
\end{aligned}
$$

$$
\begin{array}{ll}
\hline \text { Algorithm 15.l: GP regression } \\
\hline 1 & \mathbf{L}=\text { cholesky }\left(\mathbf{K}+\sigma_y^2 \mathbf{I}\right) ; \\
2 & \boldsymbol{\alpha}=\mathbf{L}^T \backslash(\mathbf{L} \backslash \mathbf{y}) ; \\
3 & \mathbb{E}\left[f_*\right]=\mathbf{k}_*^T \boldsymbol{\alpha} ; \\
4 & \mathbf{v}=\mathbf{L} \backslash \mathbf{k}_* ; \\
5 & \operatorname{var}\left[f_*\right]=\kappa\left(\mathbf{x}_* \cdot \mathbf{x}_*\right)-\mathbf{v}^T \mathbf{v} ; \\
6 & \log p(\mathbf{y} \mid \mathbf{X})=-\frac{1}{2} \mathbf{y}^T \boldsymbol{\alpha}-\sum_i \log L_{i i}-\frac{N}{2} \log (2 \pi) \\
\hline
\end{array}
$$

### **Noise GP regression**

<img src="/assets/Machine_LearningNote.assets/image-20230515090941017.png" alt="image-20230515090941017" style="zoom:50%;" />
$$
\kappa_y\left(x_p, x_q\right)=\sigma_f^2 \exp \left(-\frac{1}{2 \ell^2}\left(x_p-x_q\right)^2\right)+\sigma_y^2 \delta_{p q}
$$
We can extend the SE kernel to multiple dimensions as follows:
$$
\kappa_y\left(\mathbf{x}_p, \mathbf{x}_q\right)=\sigma_f^2 \exp \left(-\frac{1}{2}\left(\mathbf{x}_p-\mathbf{x}_q\right)^T \mathbf{M}\left(\mathbf{x}_p-\mathbf{x}_q\right)\right)+\sigma_y^2 \delta_{p q}
$$

#### Learning Covariance Parameters (also called hyper-parameters) via maximum likelihood

Refer to KPM, chapter 15.2.4 Estimating the kernel parameters 

Refer to [Maximum Likelihood Estimation of Gaussian Parameters (jrmeyer.github.io)](http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html)

> *Learning the kernel parameters (maximum (marginal) likelihood, Bayesian, cross validation)* 
>
> include the Noisy parameters $\mathbf{\sigma}_{\mathbf{y}}$ 
> $$
> \begin{aligned}
> \log p(\mathbf{y} \mid \mathbf{X})=\log \mathcal{N}\left(\mathbf{y} \mid \mathbf{0}, \mathbf{K}_y\right) &= \color{red} -\frac{1}{2} \mathbf{y} \mathbf{K}_y^{-1} \mathbf{y}-\frac{1}{2} \log \left|\mathbf{K}_y\right|-\frac{N}{2} \log (2 \pi) \text{ "--the equation used in code for gradient"}\\
> \frac{\partial}{\partial \theta_j} \log p(\mathbf{y} \mid \mathbf{X}) &=\frac{1}{2} \mathbf{y}^T \mathbf{K}_y^{-1} \frac{\partial \mathbf{K}_y}{\partial \theta_j} \mathbf{K}_y^{-1} \mathbf{y}-\frac{1}{2} \operatorname{tr}\left(\mathbf{K}_y^{-1} \frac{\partial \mathbf{K}_y}{\partial \theta_j}\right) \\
> & = \frac{1}{2} \operatorname{tr}\left(\left(\boldsymbol{\alpha} \boldsymbol{\alpha}^T-\mathbf{K}_y^{-1}\right) \frac{\partial \mathbf{K}_y}{\partial \theta_j}\right)
> \end{aligned}
> $$
> where $\alpha = K_y^{-1}y$.  It takes $O(N^3)$ time to compute $K^−1 y$ , and then $O(N^2)$ time per hyperparameter to compute the gradient.
>
> $$\mathbf{\kappa}_{\mathbf{y}}\left( \mathbf{X,}\mathbf{X}^{\mathbf{'}}\mathbf{\ } \right)\mathbf{= E(f}\left( \mathbf{x} \right)\mathbf{- m}\left( \mathbf{x} \right)\left( \mathbf{f}\left( \mathbf{x}^{\mathbf{'}} \right)\mathbf{- m}\left( \mathbf{x}^{\mathbf{'}} \right)^{\mathbf{T}} \right)\mathbf{=}\mathbf{\sigma}_{\mathbf{f}}^{\mathbf{2}}\mathbf{\exp}\left( \mathbf{-}\frac{\mathbf{1}}{\mathbf{2}\mathcal{l}^{\mathbf{2}}}\left( \mathbf{x -}\mathbf{x}^{\mathbf{'}} \right)^{\mathbf{2}} \right)\mathbf{+}\mathbf{\sigma}_{\mathbf{y}}^{\mathbf{2}}\mathbf{\delta =}\mathbf{\sigma}_{\mathbf{f}}^{\mathbf{2}}\mathbf{\exp}\left( \mathbf{-}\frac{\mathbf{1}}{\mathbf{2}}\left( \mathbf{x -}\mathbf{x}^{\mathbf{'}} \right)^{\mathbf{T}}\mathbf{M(x -}\mathbf{x}^{\mathbf{'}} \right)\mathbf{+}\mathbf{\sigma}_{\mathbf{y}}^{\mathbf{2}}\mathbf{\delta,\ }$$
>
> here $$\mathbf{M}_{\mathbf{1}}\mathbf{=}\mathcal{l}^{\mathbf{- 2}}\mathbf{I,\ }\mathbf{M}_{\mathbf{2}}\mathbf{= diag}\left( \mathcal{l} \right)^{\mathbf{- 2}}\mathbf{,\ }\mathbf{M}_{\mathbf{3}}\mathbf{= \ }\mathbf{\Lambda}\mathbf{\Lambda}^{\mathbf{T}}\mathbf{+ \ diag}\left( \mathcal{l} \right)^{\mathbf{- 2}}$$
>
> ***Optimize it Using GD to get the*** $\mathbf{\theta = }\mathcal{l\ }\mathbf{or\ }\mathbf{\sigma}_{\mathbf{y}}\mathbf{or\ }\mathbf{\sigma}_{\mathbf{f}}$
>
> The form of $\frac{∂K_y}{∂θj}$ depends on the form of the kernel, and which parameter we are taking derivatives with respect to. Often we have constraints on the hyper-parameters, such as $σ^2_y ≥ 0.$ In this case, we can define $θ = log(σ^2_y)$, and then use the chain rule. Given an expression for the log marginal likelihood and its derivative, we can estimate the kernel parameters using any standard gradient-based optimizer. **However, since the objective is not convex, local minima can be a problem**.
>
> <img src="/assets/Machine_LearningNote.assets/image-20230513165946657.png" alt="image-20230513165946657" style="zoom:50%;" />
>
> $$\frac{\partial K}{\partial l} = \sigma^{2}\exp\left( \left. \ \frac{\left( - \left( x - x^{'} \right)^{T}\left( x - x^{'} \right) \right)}{2l^{2}} \right.\  \right)\left( \frac{\left( x - x^{'} \right)^{T}\left( x - x^{'} \right)}{l^{3}} \right)$$
>
> $$\frac{\partial\ K}{\partial\sigma_f}\  = 2\sigma\exp\left. \ \left( \frac{- \left( x - x^{'} \right)^{T}\left( x - x^{'} \right)}{2l^{2}} \right.\  \right)$$
>
> $$\frac{\partial\ K}{\partial{\sigma_y}}$$
>
> **In another webpage**([Optimizing Parameters (mlatcl.github.io)](https://mlatcl.github.io/gpss/lectures/04-optimizing-parameters.html))(I also don't see no any further steps based on those complex equations, **in GPy code??**) :
>
> $E(\boldsymbol{\theta})=\frac{1}{2} \log \operatorname{det} \mathbf{K}+\frac{\mathbf{y}^{\top} \mathbf{K}^{-1} \mathbf{y}}{2}$
>
> The parameters are *inside* the covariance function (matrix)
>
> $\begin{aligned} k_{i, j} & =k\left(\mathbf{x}_i, \mathbf{x}_j ; \boldsymbol{\theta}\right) \\ \mathbf{K} & =\mathbf{R} \boldsymbol{\Lambda}^2 \mathbf{R}^{\top}\end{aligned}$
>
> ![img](/assets/Machine_LearningNote.assets/gp-optimize-eigen-1683977224894-28.png)Λ represents distance on axes. R gives rotation.
>
> - $\boldsymbol{\Lambda}$ is diagonal, $\mathbf{R}^{\top} \mathbf{R}=\mathbf{I}$.
> - Useful representation since $\operatorname{det} \mathbf{K}=\operatorname{det} \boldsymbol{\Lambda}^2=\operatorname{det} \boldsymbol{\Lambda}^2$.
>
> <img src="/assets/Machine_LearningNote.assets/image-20230513192941389.png" alt="image-20230513192941389" style="zoom:50%;" />



#### Scalability of the Gradient (no any further steps)

In the previous section, we assumed that we can compute the gradient exactly. However, if the dimension of the vector y, n increases, it might not be possible to compute the above gradient in a reasonable time and cost. Let’s analyze the computational complexity of each term. [Learning Gaussian Process Covariances (chrischoy.github.io)](https://chrischoy.github.io/research/learning-gaussian-process-covariances/) 

make the Gradient of the Posterior Probability easier 
$$
\begin{aligned}
\operatorname{Tr}\left(K^{-1} \frac{\partial K}{\partial \theta_i}\right) & =\operatorname{Tr}\left(K^{-1} \frac{\partial K}{\partial \theta_i} \mathbb{E}\left[\mathbf{r r}^T\right]\right) \\
& =\mathbb{E}\left[\mathbf{r}^T K^{-1} \frac{\partial K}{\partial \theta_i} \mathbf{r}\right]
\end{aligned}
$$

$$
\nabla_{\theta_i} \log p(\mathbf{y} \mid X, \overline{\mathbf{f}} ; \theta) \approx-\frac{1}{2 N} \sum_i^N \mathbf{r}_i^T K^{-1} \frac{\partial K}{\partial \theta_i} \mathbf{r}_i-\frac{1}{2}(\mathbf{y}-\overline{\mathbf{f}})^T K^{-1} \frac{\partial K}{\partial \theta_i} K^{-1}(\mathbf{y}-\overline{\mathbf{f}})
$$

the computing complexity from $O(n^3)$ complexity $O\left(\sqrt{\kappa} n^3 \right)$ or  to $O\left(\sqrt{\kappa} n^2 N_s\right)$ where Ns is the number of samples.

#### ***GPR vs RBF***

<img src="C:\Users\zphil\AppData\Roaming\Typora\typora-user-images\image-20230505232157114.png" alt="image-20230505232157114" style="zoom:80%;" />

Ridge regression : $\left(\mathbf{X}^T \mathbf{X}+\delta^2 \mathbf{I}_d\right) \boldsymbol{\theta}=\mathbf{X}^T \mathbf{y}$ , also the solution can be written as $\boldsymbol{\theta}=\mathbf{X}^T \boldsymbol{\alpha}$, where $\boldsymbol{\alpha}=\delta^{-2}(\mathbf{y}-\mathbf{X} \boldsymbol{\theta})$, 

 $\boldsymbol{\alpha}$ can also be written as follows: $\boldsymbol{\alpha}=\left(\mathbf{X X}^T+\delta^2 \mathbf{I}_n\right)^{-1} \mathbf{y}$, proof like following

$$\begin{aligned}
\delta^2 \alpha & =y-x \theta \\
\delta^2 \alpha & =y-x X^{\top} \alpha \\
X X^{\top} \alpha+\delta^2 I_n \alpha & =y \\
\alpha & =\left(X X^{\top}+\delta^2 I_n\right)^{-1} y
\end{aligned}$$

so the computation can be done with $\theta$ d (feature number) dimension or with $\alpha$ n (data number) dimension 
$$
\begin{aligned}
y^{\star} & =x^{\star} \theta \\
& =x^{\star} X^{\top} \alpha \\
& =x^{\star} X^{\top}\left[X X^{\top}+\delta^2 I_n\right]^{-1} y \\
& =k^{\top} K_y^{-1} y
\end{aligned}
$$

$$
k_n^{\top}=\left[\begin{array}{llll}
x^* x_1^{\top} & x^{*} x_2^{\top} & \ldots & x^* x_n^{\top}
\end{array}\right]\  
K_y = \left[\begin{array}{c}
x_1 \\
\vdots \\
x_n
\end{array}\right]\left[\begin{array}{lll}
x_1^{\top} & \cdots x_4^{\top}
\end{array}\right] 

=\left[\begin{matrix} x_1x_1^T,\cdots,x_1x_n^T\\
  \vdots \\
  x_nx_1^T, \cdots, x_nx_n^T
\end{matrix}\right] ， \delta^2 \text{ is same parameters as } \sigma_y
$$

each $x_ix_j^T$ is dxd matrix. 



### Likelihood for a Gaussian （lots of algebra）

[Maximum Likelihood Estimation of Gaussian Parameters (jrmeyer.github.io)](http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html)
$$
\begin{align}  \mu_{MLE} = \underset{\mu}{\operatorname{argmax}} \mathcal{N}(\mathbf{X}|\mu, \Sigma)\\  \Sigma_{MLE} = \underset{\Sigma}{\operatorname{argmax}} \mathcal{N}(\mathbf{X}|\mu, \Sigma) \end{align}
$$

$$
\begin{align}  \mathcal{LL} &= \sum\limits_{n=1}^{N}  \Big( - \frac{1}{2} \log ( 2\pi\sigma^{2} )  -\frac{1}{2} \Big(\frac{(x_{n} - \mu)^{2}}{\sigma^{2}} \Big) \Big)\\  &= - \frac{N}{2} \log ( 2\pi\sigma^{2} )  + \sum\limits_{n=1}^{N} -\frac{1}{2} \Big(\frac{(x_{n} - \mu)^{2}}{\sigma^{2}} \Big) \\  &= - \frac{N}{2} \log ( 2\pi\sigma^{2} )  -\frac{1}{2\sigma^{2}} \sum\limits_{n=1}^{N} (x_{n} - \mu)^{2} \\ \end{align}
$$

## **Jensen‘s inquality **

Theorem (Jensen's Inequality)
If $f: \mathrm{R} \rightarrow \mathrm{R}$ is a convex function, and $x$ is a random variable, then
$$
\mathbb{E} f(x) \geqslant f(\mathbb{E} x)
$$
Moreover, if $f$ is strictly convex(凸), then equality implies that $x=\mathbb{E} x$ with probability 1 (i.e. $x$ is a constant).
$$
\begin{gathered}
g(x)=-\log x\\ \quad-\log (E[x]) \leq E[-\log x] \\
\log (E[x]) \geqslant E[\log x]
\end{gathered}
$$

## **Lower Bound of marginal log-likelihood**

- Let $q(z)$ be any PMF on $z$, the support of $z$ :

$$
\begin{aligned}
\log p(x \mid \theta) & =\log \left[\sum_z p(x, z \mid \theta)\right] \\
& =\log \left[\sum_z q(z)\left(\frac{p(x, z \mid \theta)}{q(z)}\right)\right] \quad \text { (log of an expectation) } \\
& \geqslant \underbrace{\sum_z q(z) \log \left(\frac{p(x, z \mid \theta)}{q(z)}\right)}_{\mathcal{L}(q, \theta)} \text { (expectation of } \log \text { ) }
\end{aligned}
$$

## **Variational lower bound** and ELBO

For any PMF $q(z)$, we have a lower bound on the marginal log-likelihood
$$
\log p(x \mid \theta) \geqslant \underbrace{\sum_z q(z) \log \left(\frac{p(x, z \mid \theta)}{q(z)}\right)}_{\mathcal{L}(q, \theta)}
$$
**marginal log likelihood $log p(x|\theta)$ also called evidence. so the $\mathcal{L (q, \theta)}$ is the evidence lower bound or ELBO**

To select the inducing input $X_{m}$, apply variational inference in an augmented probability space that involve both the tanning latent function value f and the pseudo-input inducing variable $f_{m}$

The initial joint model $p(y,f)$ is augmented with the variable $f_{m}\ $to form the model  

$$
p\left( y,f,\ f_{m} \right) = p\left( y \middle| f \right)p\left( f \middle| f_{m} \right)p\left( f_{m} \right)
$$
where the conditional prior is given by

$$
p\left( f \middle| f_{m} \right) = N\left( f \middle| K_{nm}K_{mm}^{- 1}f_{m},\ K_{nn} - K_{nm}K_{mm}^{- 1}K_{mn} \right)
$$
Posterior distribution

$p(f|y)$ $p\left( f,\ f_{m} \middle| y \right) = p\left( f \middle| f_{m},y \right)p(f_{m}|y)$

Log Marginal likelihood

$$logp(y) = log\int_{}^{}{p\left( y \middle| f \right)p\left( f \middle| f_{m} \right)p\left( f_{m} \right)dfdf_{m}}$$

Approximate the true posterior distribution by introducing a variational distribution $q(f,f_{m})$ and minimize the KL divergence:

$$KL(q(f,f_{m})|\left| p\left( f,\ f_{m} \middle| y \right) \right) = \int_{}^{}{q\left( f,f_{m} \right)\log\frac{q(f,f_{m})}{p(f,f_{m}|y)}dfdf_{m}}$$

*To determine the variational quantities* $\left( X_{m},\ \phi \right)$*, we **minimize** the KL divergence, which is equivalently expressed as the **maximum** of the following variational low bounder on the true log marginal likelihood*

$$logp(y)\mathcal{\geq L\ \ }or\ Fv(X_{m},\phi ） = \ \int_{}^{}{q(z)\log\left( \frac{p(z,x)}{q(z)} \right)dz} = \ \int_{}^{}{q\left( f,f_{m} \right)\log\left( \frac{p\left( f,f_{m},y \right)}{q\left( f,f_{m} \right)} \right)dfdf_{m}} = \ \int_{}^{}{p\left( f|f_{m} \right)\phi\left( f_{m} \right)\log{\frac{p\left( y \middle| f \right)p\left( f \middle| f_{m} \right)p\left( f_{m} \right)}{p\left( f|f_{m} \right)\phi\left( f_{m} \right)}dfdf_{m}}} = \ \int_{}^{}{p\left( f|f_{m} \right)\phi\left( f_{m} \right)\log{\frac{p\left( y \middle| f \right)p\left( f_{m} \right)}{\phi\left( f_{m} \right)}dfdf_{m}}}\  = \ \int_{}^{}{\phi\left( f_{m} \right)\left\{ \int_{}^{}{p\left( f \middle| f_{m} \right)logp\left( y \middle| f \right)df + \left. \ \log\frac{p(f_{m})}{\phi\left( f_{m} \right)} \right\} df_{m}} \right.\ }$$

***Condition givens***

$$\mathbf{p}\left( \mathbf{y} \middle| \mathbf{f} \right) = \ \mathcal{N}\left( \mathbf{f},\ \mathbf{\sigma}_{\mathbf{noise}}^{\mathbf{2}}\mathbf{I} \right)$$

$$p\left( f \middle| f_{m} \right) = N\left( f \middle| K_{nm}K_{mm}^{- 1}f_{m},\ K_{nn} - K_{nm}K_{mm}^{- 1}K_{mn} \right)$$

$$\log{G\left( f_{m},y \right) = \ \int_{}^{}{p\left( f \middle| f_{m} \right)logp\left( y \middle| f \right)df}} = \int_{}^{}{p\left( f \middle| f_{m} \right)\left\{ - \frac{n}{2}\log\left( 2\pi\sigma^{2} \right) - \frac{1}{2\sigma^{2}}Tr\left\lbrack yy^{T} - 2yf^{T} + ff^{T}\  \right\rbrack \right\} df\ } = - \frac{n}{2}\log\left( 2\pi\sigma^{2} \right) - \ \frac{1}{2\sigma^{2}}Tr\lbrack yy^{T} - 2y\alpha^{T} + \alpha\alpha^{T} + K_{nn} - Q_{nn}\ \rbrack$$

$$\log{G\left( f_{m},y \right) = \log\left\lbrack N\left( y \middle| \alpha,\ \sigma^{2}I \right) \right\rbrack -}\frac{1}{2\sigma^{2}}Tr\left( K_{nn} - Q_{nn} \right)，\alpha = E\left\lbrack f \middle| f_{m} \right\rbrack = \ K_{nm}K_{mm}^{- 1}f_{m}\ and\ \ Q_{nn} = K_{nm}K_{mm}^{- 1}K_{mn}$$

$$Fv\left( X_{m},\phi \right) = \ \int_{}^{}{\phi\left( f_{m} \right)\left\{ \int_{}^{}{p\left( f \middle| f_{m} \right)logp\left( y \middle| f \right)df + \left. \ \log\frac{p(f_{m})}{\phi\left( f_{m} \right)} \right\} df_{m}\  = \ \int_{}^{}{\phi\left( f_{m} \right)\log\frac{N\left( y \middle| \alpha,\ \sigma^{2}I \right)p\left( f_{m} \right)}{\phi\left( f_{m} \right)}df_{m} - \frac{1}{2\sigma^{2}}Tr\left( K_{nn} - Q_{nn} \right)}} \right.\ }$$

Maximum the bound w.r.t. the distribution $\phi$

1.  *The usual way of doing this is to take the derivative w.r.t.* $\phi\left( f_{m} \right)$*, set to zero and obtain the optimal*

> $$\phi^{*}\left( f_{m} \right) = \ \frac{N\left( y \middle| \alpha,\ \sigma^{2}I \right)p\left( f_{m} \right)}{\int_{}^{}{N\left( y \middle| \alpha,\ \sigma^{2}I \right)p\left( f_{m} \right)df_{m}}}$$

2.  *a faster and by far simpler way to compute the optimal boundis by reversing the Jensen’s inequality, moving the log outside of the integral in eq*

$Fv\left( X_{m} \right) = \log{\int_{}^{}{N\left( y \middle| \alpha,\ \sigma^{2}I \right)p\left( f_{m} \right)}df_{m}} - \frac{1}{2\sigma^{2}}Tr\left( K_{nn} - Q_{nn} \right) = \log{N\left( y \middle| 0,\sigma^{2}I + Q_{nn} \right) -}\frac{1}{2\sigma^{2}}Tr\left( K_{nn} - Q_{nn} \right)$

*The optimal distribution* $\phi$ *that gives rise to this bound is given by*

$$\mathbf{\phi}^{\mathbf{*}}\left( \mathbf{f}_{\mathbf{m}} \right)\mathbf{\  \propto \ }\mathbf{N}\left( \mathbf{y} \middle| \mathbf{\alpha,\ }\mathbf{\sigma}^{\mathbf{2}}\mathbf{I} \right)\mathbf{p}\left( \mathbf{f}_{\mathbf{m}} \right)$$

$$\mathbf{\phi}^{\mathbf{*}}\left( \mathbf{f}_{\mathbf{m}} \right) = \ c\ exp\{ - \frac{1}{2}f_{m}^{T}\left( K_{mm}^{- 1} + {\frac{1}{\sigma^{2}}K}_{nm}^{- 1}K_{mm}K_{nm}K_{mm}^{- 1} \right)f_{m} + \frac{1}{\sigma^{2}}y^{T}K_{nm}K_{mm}^{- 1}f_{m}\}$$

## **KL dirvergenc and Variational inference** 

Definition 8.11 (KL divergence). For two distributions q(x) and p(x)

$$
\begin{gathered}
K L(q \mid p)=D_{K L}(q \mid p)=\langle\log q(x)-\log p(x)\rangle_{q(x)} \geq 0 \\
D_{K L}(q \mid p)=\int q(x) \log \left(\frac{q(x)}{p(x)}\right) d x
\end{gathered}
$$
**using approximate q(z) to estimate p(z\|x) **

翻译一下就是， 在给定数据情况下不知道latent variable 的分布，比如GMM,不知道中间有几个高斯以及各个高斯的参数，采用猜测给定高斯分布$q(z)$（随机分配），然后最大化ELBO的方式逼近$p(z|x)$,逼近的结果就是得到逼近想要的$p(x|\theta)$ 也即是$p(x)$

$$D_{KL}\left( q|p \right) = \ \int_{}^{}{q(z)\log\left( \frac{q(z)}{p\left( z|x \right)} \right)dz} = \  - \int_{}^{}{q(z)\log\left( \frac{p\left( z|x \right)}{q(z)} \right)dz}$$

Conditional distribution:$p\left( z \middle| x \right) = \frac{P(z,x)}{p(x)}$

$$D_{KL}\left( q|p \right) = \ \  - \int_{}^{}{q(z)\log\left( \frac{p\left( z|x \right)}{q(z)} \right)dz}$$

$$= \  - \int_{}^{}{q(z)\log\left( \frac{p(z,x)}{q(z)}\frac{1}{p(x)} \right)dz}$$

$$= \  - \int_{}^{}{q(z)\log\left( \frac{p(z,x)}{q(z)} \right)dz} - \int_{}^{}{q(z)\log\left( \frac{1}{p(x)} \right)dz}$$

$$= \ \ \  - \int_{}^{}{q(z)\log\left( \frac{p(z,x)}{q(z)} \right)dz} + \int_{}^{}{q(z)\log\left( p(x) \right)dz}$$

**so we get**: $\color{red}D_{KL}\left( q(z)|p(z|x) \right)\mathcal{+ L =}\log\left( p(x) \right)$, $\mathcal{L = \ }\int_{}^{}{q(z)\log\left( \frac{p(z,x)}{q(z)} \right)dz} = KL(q(z)|p(z,x))$ . using L to manipulate the KL(q\|p), so maximum $L$ --> minimum $D_{KL}\left( q|p \right)$



<img src="C:\Users\zphil\AppData\Roaming\Typora\typora-user-images\image-20230505233755779.png" alt="image-20230505233755779" style="zoom:80%;" />

## MLE

### **Maximum likelihood estimation properties**

The goal is to maximum the likelihood of given the tanning data $(x_{1:n},y_{1:n})$ choose the value of  the parameters ($\theta,\ \sigma$) has more probability of generate the data
$$
\color{red}\hat{\theta}=\underset{\theta}{\arg \max } p\left(x_{1: n} \mid \theta\right)
$$
$$\color{blue}L(\theta,\sigma) = \  - \frac{n}{2}\log{2\pi\sigma^{2} - \frac{1}{2\sigma^{2}}(y - X\theta)^{T}(y - X\theta)}$$

partial derivate to $\sigma$:
$$
\color{blue}\frac{\partial L(\theta,\sigma)}{\partial\sigma} = 0\\
\color{blue}\sigma^2=\frac{1}{n}(y-x \theta)^{\top}(y-x \theta)=\frac{1}{n} \sum_{i=1}^n\left(y_i-x_i\right)^2
$$
partial derivate to $\theta$:

$$\frac{\partial L(\theta)}{\partial\theta} = \ 0 - \frac{1}{2\sigma^{2}}\left\lbrack 0 - 2X^{T}Y + \ 2X^{T}X\theta \right\rbrack = 0$$

$\widehat{\theta} = {X^{T}X}^{- 1}X^{T}y$ =\> $\widehat{\theta} = {X^{T}X + \delta^{2}Ι_{d}}^{- 1}X^{T}y$

**Entropy ** H is a measure of the uncertainity associated with a random variable. it defined as:

$$H(X) = -\Sigma_x p(x|\theta)logp(x|\theta)$$

Bernoulli distribution Example, the entropy is 
$$
\begin{aligned}
H(x) & =-\sum_{x=0}^1 \theta^x(1-\theta)^{k x} \log \left[\theta^x(1-\theta)^{1-x}\right] \\
& =-[(1-\theta) \log (1-\theta)+\theta \log \theta]
\end{aligned}
$$
**Maximum MLE means minimizes the KL divergence**
$$
\color{red}\begin{aligned}
\hat{\boldsymbol{\theta}} & =\arg \max _{\boldsymbol{\theta}} \prod_{i=1}^{\mathbb{N}} p\left(x_i \mid \boldsymbol{\theta}\right) \\
& =\arg \max _{\boldsymbol{\theta}} \sum_{i=1}^{\mathbb{N}} \log p\left(x_i \mid \boldsymbol{\theta}\right) \\
& =\arg \max _{\boldsymbol{\theta}} \frac{1}{N} \sum_{i=1}^N \log p\left(x_i \mid \boldsymbol{\theta}\right)-\frac{1}{N} \sum_{i=1}^N \log p\left(x_i \mid \boldsymbol{\theta}_0\right) \\
& =\arg \max _{\boldsymbol{\theta}} \frac{1}{N} \sum_{i=1}^N \log \frac{p\left(x_i \mid \boldsymbol{\theta}\right)}{p\left(x_i \mid \boldsymbol{\theta}_0\right)} \\
& =\arg \max _{\boldsymbol{\theta}}\int p(x_i \mid \boldsymbol{\theta}_0) \log \frac{p\left(x_i \mid \boldsymbol{\theta}\right)}{p\left(x_i \mid \boldsymbol{\theta}_0\right)} dx , x_i \sim p(x_i|\theta_0) (the true distribution) \\
& \longrightarrow \arg \min _{\boldsymbol{\theta}} \int \log \frac{p\left(x \mid \boldsymbol{\theta}_0\right)}{p\left(x \mid \boldsymbol{\theta}\right)} p\left(x \mid \boldsymbol{\theta}_0\right) d x \\
&= {KL}\left( p\left(x \mid \boldsymbol{\theta}_0\right) | p\left(x \mid \boldsymbol{\theta}\right )\right) \\
&=\underbrace{\operatorname{argmin}}_\theta \underbrace{\int P\left(x \mid \theta_0\right) \log P\left(x \mid \theta_0\right) d x}_{\text {information world -H(x) }}-\underbrace{\int P\left(x \mid \theta_0\right) \log P(x \mid \theta) d x}_{\text {information in model - cross entropy }} \\
\end{aligned}
\$$
$$

Apply the expection equation: 
$$
E(x)=\int x p(x) d x (N->\infty), \\E(x) = \frac{1}{N}\Sigma_{i=1}^{N}x_i
$$

<img src="C:\Users\zphil\AppData\Roaming\Typora\typora-user-images\image-20230506121035292.png" alt="image-20230506121035292" style="zoom: 33%;" />

<img src="C:\Users\zphil\AppData\Roaming\Typora\typora-user-images\image-20230506121001476.png" alt="image-20230506121001476" style="zoom: 33%;" />

<img src="C:\Users\zphil\AppData\Roaming\Typora\typora-user-images\image-20230506121706176.png" alt="image-20230506121706176" style="zoom: 33%;" />



## Approximate Gaussian Process Regression 

- (A Unifying View of Sparse Approximate Gaussian Process Regression) **Joaquin Qui ˜nonero-Candela, Carl Edward Rasmussen** 2005

Probabilistic regression is usually formulated as follows: given a training set D = {(**x***i*, *yi*), *i* =

1, . . . ,*n*} of *n* pairs of (vectorial) inputs **x***i* and noisy (real, scalar) outputs *yi*, compute the predictive distribution of the function values $f_{*}$ (or noisy $y_{*}$) at test locations $\mathbf{x}_{*}$. In the simplest case (which we deal with here) we assume that the noise is additive, independent and Gaussian, such that the relationship between the (latent) function *f* (**x**) and the observed noisy targets *y* are given by The joint GP prior and the independent likelihood are both Gaussian

$$\mathbf{y}_{\mathbf{i}}\mathbf{= f}\left( \mathbf{X}_{\mathbf{i}} \right)\mathbf{+ \ }\mathbf{\varepsilon}_{\mathbf{i}}\mathbf{\ ,\ where\ }\mathbf{\varepsilon}_{\mathbf{i}}\mathcal{\ \sim\ N}\left( \mathbf{0,\ }\mathbf{\sigma}_{\mathbf{noise}}^{\mathbf{2}} \right)$$

Gaussian process (GP) regression is a Bayesian approach which assumes a GP prior2 over functions, i.e. assumes a priori that function values behave according to

$$p\left( f \middle| X_{1},\ X_{2},\ldots,X_{n} \right)\mathcal{= \ N}(0,k)$$

where **f** = \[ *f*1, *f*2, . . . , *fn*\]\> is a vector of latent function values, *fi* = *f* (**x***i*) and *K* is a covariance matrix, whose entries are given by the *covariance function*, *Ki j* = *k*(**x***i*,**x***j*)

$$\mathbf{p(f,}\mathbf{f}_{\mathbf{*}}\mathbf{|y) =}\frac{\mathbf{P(f,}\mathbf{f}_{\mathbf{*}}\mathbf{)\ p(y|f)}}{\mathbf{p}\left( \mathbf{y} \right)}$$

$$\mathbf{p}\left( \mathbf{f}_{\mathbf{*}} \middle| \mathbf{y} \right)\mathbf{= \ }\int_{}^{}{\mathbf{p}\left( \mathbf{f,}\mathbf{f}_{\mathbf{*}} \middle| \mathbf{y} \right)\mathbf{df = \ }\frac{\mathbf{1}}{\mathbf{p(y)}}\int_{}^{}{\mathbf{P(f,}\mathbf{f}_{\mathbf{*}}\mathbf{)\ p(y|f)df}}\mathbf{\ }}$$

$$\mathbf{p}\left( \mathbf{f,}\mathbf{f}_{\mathbf{*}} \right)\mathcal{= \ N}\left( \mathbf{0,\ }\begin{bmatrix}
\mathbf{K}_{\mathbf{f,f}} & \mathbf{K}_{\mathbf{*,f}} \\
\mathbf{K}_{\mathbf{f,*}} & \mathbf{K}_{\mathbf{*,*}} \\
\end{bmatrix} \right)\mathbf{and\ \ p}\left( \mathbf{y} \middle| \mathbf{f} \right)\mathcal{= \ N}\left( \mathbf{f,\ }\mathbf{\sigma}_{\mathbf{noise}}^{\mathbf{2}}\mathbf{I} \right)$$

$$\mathbf{p}\left( \mathbf{f}_{\mathbf{*}} \middle| \mathbf{y} \right)\mathcal{= \ N(}\mathbf{K}_{\mathbf{*,f}}\left( \mathbf{K}_{\mathbf{f,f}}\mathbf{+ \ }\mathbf{\sigma}_{\mathbf{noise}}^{\mathbf{2}}\mathbf{I} \right)^{\mathbf{- 1}}\mathbf{y,\ }\mathbf{K}_{\mathbf{*,*}}\mathbf{-}\mathbf{K}_{\mathbf{*,f}}\left( \mathbf{K}_{\mathbf{f,f}}\mathbf{+ \ }\mathbf{\sigma}_{\mathbf{noise}}^{\mathbf{2}}\mathbf{I} \right)^{\mathbf{- 1}}\mathbf{K}_{\mathbf{f,*}}\mathbf{)\ }$$

***exact expression***

Due to the *consistency* of Gaussian processes, we know that we can recover *p*(**f**\_, **f**) by simply

integrating (marginalizing) out **u** from the joint GP prior *p*(**f**\_, **f**,**u**)

$$\mathbf{p}\left( \mathbf{f}_{\mathbf{*}}\mathbf{,f} \right)\mathbf{= \ }\int_{}^{}{\mathbf{p}\left( \mathbf{f}_{\mathbf{*}}\mathbf{,\ f,u} \right)\mathbf{du = \ }\int_{}^{}{\mathbf{p}\left( \mathbf{f}_{\mathbf{*}}\mathbf{,f} \middle| \mathbf{u} \right)\mathbf{p}\left( \mathbf{u} \right)\mathbf{du\ ,\ where\ p}\left( \mathbf{u} \right)\mathcal{= \ N(}\mathbf{0,\ }\mathbf{K}_{\mathbf{u,u}}\mathbf{)}}}$$

$$\mathbf{fundametal}\ \mathbf{approximation}:$$

$$\mathbf{p}\left( \mathbf{f}_{*},\ \mathbf{f} \right) \cong \mathbf{q}\left( \mathbf{f}_{*},\ \mathbf{f} \right) = \ \int_{}^{}{\mathbf{q}\left( \mathbf{f}_{*} \right|\mathbf{u})\mathbf{q}\left( \mathbf{f} \middle| \mathbf{u} \right)\mathbf{p}\left( \mathbf{u} \right)\mathbf{du}}$$

The name inducing variable is motivated by the fact that **f** and **f**\* can only communicate though **u**, and **u** therefore induces the dependencies between training and test cases. As we shall detail in the following sections, the different computationally efficient algorithms proposed in the literature correspond to different additional assumptions about the two approximate inducing conditionals q(**f**\|**u**), q(**f**\_\|**u**) of the integral in (8).

$$\mathbf{training}\ \mathbf{conditional}:\mathbf{p}\left( \mathbf{f} \middle| \mathbf{u} \right) = \mathcal{N}(\mathbf{K}_{\mathbf{f},\mathbf{u}}\mathbf{K}_{\mathbf{u},\mathbf{u}}^{- \mathbf{1}}\ \mathbf{u},\ \mathbf{K}_{\mathbf{f},\mathbf{f}} - \mathbf{Q}_{\mathbf{f},\mathbf{f}})$$

$$\mathbf{test}\ \mathbf{conditional}:\mathbf{p}\left( \mathbf{f}_{*} \middle| \mathbf{u} \right) = \mathcal{N}(\mathbf{K}_{*,\mathbf{u}}\mathbf{K}_{\mathbf{u},\mathbf{u}}^{- \mathbf{1}}\ \mathbf{u}\ ,\ \mathbf{K}_{*,*} - \mathbf{Q}_{*,*})$$

as special (noise free) cases of the standard predictive equation (6) with **u** playing the role of (noise free) observations. shorthand notation

$$Q_{a,b} \triangleq K_{a,u}K_{u,u}^{- 1}\ K_{u,b}$$

### **The Subset of Regressors (SoR) Approximation**

For any input **x**\_, the corresponding function value *f*\_ is given by:

$$f_{*} = K_{*,u}W_{u}\ with\ p\left( W_{u} \right)\mathcal{= \ N}(0,\ K_{u,u}^{- 1})$$

$f_{*} = K_{*,u}K_{u,u}^{- 1}\ u\ ,\ with\ u\ \sim\ \mathcal{N}\left( 0,K_{u,u} \right)$

### **The Deterministic Training Conditional (DTC) Approximation**

Projected Process Approximation (PPA) : the method as relying on a *likelihood* approximation, based on the projection $f = K_{f,u}K_{u,u}^{- 1}\ u$

$$\mathbf{p}\left( \mathbf{y} \middle| \mathbf{f} \right)\mathbf{\  \cong q}\left( \mathbf{y} \middle| \mathbf{u} \right)\mathcal{= \ N}\left( \mathbf{f,\ }\mathbf{\sigma}_{\mathbf{noise}}^{\mathbf{2}}\mathbf{I} \right)\mathcal{= \ N}\left( K_{f,u}K_{u,u}^{- 1}\ u\mathbf{,\ }\mathbf{\sigma}_{\mathbf{noise}}^{\mathbf{2}}\mathbf{I} \right)$$

Reformulation to $\mathbf{q}_{\mathbf{DTC}}\left( \mathbf{f} \middle| \mathbf{u} \right)\mathcal{= \ N}\left( \mathbf{K}_{\mathbf{f,u}}\mathbf{K}_{\mathbf{u,u}}^{\mathbf{- 1}}\mathbf{u,\ 0} \right)\mathbf{,\ \ and\ ,\ \ }\mathbf{q}_{\mathbf{DTC}}\left( \mathbf{f}_{\mathbf{*}} \middle| \mathbf{u} \right)\mathbf{= p(}\mathbf{f}_{\mathbf{*}}\mathbf{|u)}$

![image-20230519130152624](/assets/Machine_LearningNote.assets/image-20230519130152624.png)

$$
\begin{gathered}
q\left(\mathbf{y} \mid \mathbf{f}_m\right) \propto \exp \left(-\frac{1}{2 \sigma_n^2}\left(\mathbf{y}-P^{\top} \mathbf{f}_m\right)^{\top}\left(\mathbf{y}-P^{\top} \mathbf{f}_m\right)\right) . \\
P=K_{m m}^{-1} K_{m n} \text { so that } \mathbb{E}\left[\mathbf{f} \mid \mathbf{f}_m\right]=P^{\top} \mathbf{f}_m \\
p\left(\mathbf{f}_m\right) \propto \exp \left(-\mathbf{f}_m^{\top} K_{m m}^{-1} \mathbf{f}_m / 2\right)
\end{gathered}
$$

$$
\begin{aligned}
q\left(\mathbf{f}_m \mid \mathbf{y}\right) & \propto \exp \left(-\frac{1}{2} \mathbf{f}_m^{\top}\left(K_{m m}^{-1}+\frac{1}{\sigma_n^2} P P^{\top}\right) \mathbf{f}_m+\frac{1}{\sigma_n^2} \mathbf{y}^{\top} P^{\top} \mathbf{f}_m\right), \\
\mathbb{E}_q\left[f\left(\mathbf{x}_*\right)\right] & =\mathbf{k}_m\left(\mathbf{x}_*\right)^{\top} K_{m m}^{-1} \boldsymbol{\mu} \\
& =\mathbf{k}_m\left(\mathbf{x}_*\right)^{\top}\left(\sigma_n^2 K_{m m}+K_{m n} K_{n m}\right)^{-1} K_{m n} \mathbf{y},
\end{aligned}
$$

$$
\begin{aligned}
\mathbb{V}_q\left[f\left(\mathrm{x}_*\right)\right]= & k\left(\mathbf{x}_*, \mathbf{x}_*\right)-\mathbf{k}_m\left(\mathbf{x}_*\right)^{\top} K_{m m}^{-1} \mathbf{k}_m\left(\mathbf{x}_*\right) \\
& +\mathbf{k}_m\left(\mathbf{x}_*\right)^{\top} K_{m m}^{-1} \operatorname{cov}\left(\mathbf{f}_m \mid \mathbf{y}\right) K_{m m}^{-1} \mathbf{k}_m\left(\mathbf{x}_*\right) \\
=k\left(\mathbf{x}_*,\right. & \left.\mathbf{x}_*\right)-\mathbf{k}_m\left(\mathbf{x}_*\right)^{\top} K_{m m}^{-1} \mathbf{k}_m\left(\mathbf{x}_*\right) \\
& +\sigma_n^2 \mathbf{k}_m\left(\mathbf{x}_*\right)^{\top}\left(\sigma_n^2 K_{m m}+K_{m n} K_{n m}\right)^{-1} \mathbf{k}_m\left(\mathbf{x}_*\right)
\end{aligned}
$$

- $\mathbf{q}_{\mathbf{DTC}}\left( \mathbf{f,}\mathbf{f}_{\mathbf{*}} \right)\mathcal{= \ N}\left( \mathbf{0,\ }\begin{bmatrix}
  \mathbf{Q}_{\mathbf{f,f}} & \mathbf{Q}_{\mathbf{f,*}} \\
  \mathbf{Q}_{\mathbf{*,f}} & \mathbf{K}_{\mathbf{*,*}} \\
  \end{bmatrix} \right)$

$$
\begin{aligned}
& q_{\mathrm{DTC}}\left(\mathbf{f}_{\mathbf{f}} \mid \mathbf{y}\right)=\mathcal{N}\left(Q_{*, \mathrm{f}}\left(Q_{\mathrm{t}, \mathbf{f}}+\sigma_{\text {noise }}^2 I\right)^{-1} \mathbf{y}, K_{*, *}-Q_{*, \mathbf{f}}\left(Q_{\mathrm{t}, \mathrm{f}}+\sigma_{\text {noise }}^2 I\right)^{-1} Q_{\mathrm{t}, *}\right. \\
&=\mathcal{N}\left(\sigma^{-2} K_{*, \mathrm{u}} \Sigma K_{\mathbf{u}, \mathbf{f}} \mathbf{y}, K_{*, *}-Q_{*, *}+K_{*, \mathbf{u}} \Sigma K_{*, \mathbf{u}}^{\top}\right) \\
& \Sigma=\left(\sigma^{-2} K_{\mathbf{u}, \mathbf{f}} K_{\mathbf{f}, \mathbf{u}}+K_{\mathbf{u}, \mathbf{u}}\right)^{-1}
\end{aligned}
$$

### **The Fully Independent Training Conditional (FITC) Approximation**

While the DTC is based on the likelihood approximation given by (17), the SGPP proposes

a more sophisticated likelihood approximation with a richer covariance

$$\mathbf{p}\left( \mathbf{y} \middle| \mathbf{f} \right)\mathbf{\  \cong p}\left( \mathbf{y} \middle| \mathbf{u} \right)\mathcal{= \ N}\left( \mathbf{f,\ }\mathbf{\sigma}_{\mathbf{noise}}^{\mathbf{2}}\mathbf{I} \right)\mathcal{= \ N}\left( K_{f,u}K_{u,u}^{- 1}\ u\mathbf{,\ diag}\left\lbrack \mathbf{K}_{\mathbf{f,f}}\mathbf{-}\mathbf{Q}_{\mathbf{f,f}} \right\rbrack\mathbf{+ \ \ }\mathbf{\sigma}_{\mathbf{noise}}^{\mathbf{2}}\mathbf{I} \right)$$

where diag\[*A*\] is a diagonal matrix whose elements match the diagonal of *A*.

<img src="/assets/Machine_LearningNote.assets/image-20230507234501358.png" alt="image-20230507234501358" style="zoom:50%;" />

### **Posterior Approximations**

>  
>
>  in 2009, Michalis Titsias published a paper that proposed a different approach: “Variational Learning of Inducing Variables in Sparse Gaussian Processes” (Titsias 2009). **This method does not quite fit into the unifying view proposed by Quiñonero-Candela. The key idea is to construct a variational approximation to the posterior process, and learn the pseudo-points Z alongside the kernel parameters by maximising the evidence lower bound (ELBO),** i.e. a lower bound on the log-marginal likelihood. There was quite a bit of prior art on variational inference in Gaussian processes (e.g. Csató and Opper 2002; Seeger 2003): Titsias’ important contribution was to treat Z as parameters of the variational approximation, rather than model parameters.
>
> $$
>  \text {take and extra M pints on the function， } u=f(z) \\
>  p(y,f,u)=p(y|f)p(f|u)p(u)
> $$
>  Instead Of doing
> $$
>  p(f|y,X) =\frac{p(y|f)p(f|X)}{\int {py|f)p(f|X)df}}
> $$
>   We will do 
> $$
>  P(u|y,Z) = \frac {p(y|u)p(u|Z)}{\int p(y|u)p(u|Z)du} \\
>  p\left( y \middle| u \right) = \frac{p\left( y \middle| f \right)p(f|u)}{p(f|y,u)} \\
>  \ln{p\left( y \middle| u \right) = \ \ln{p\left( y \middle| f \right) + \ln\frac{p(f|u)}{p(f|y,u)}}}
> $$
>
> $$
>  \begin{gathered}
>  \ln p(y \mid u)=\mathbb{E}_{p(f \mid u)}[\ln p(y \mid f)]+\mathbb{E}_{p(f \mid u)}\left[\ln \frac{p(f \mid u)}{p(f \mid y, u)}\right] \\
>  \ln p(y \mid u)=\tilde{p}(y \mid u)+KL[p(f \mid u) \| p(f \mid y, u)] \\
>  p(y \mid f)=\mathcal{N}\left(y|f, \sigma^2 I\right) \\
>  p(f \mid u)=\mathcal{N}\left(f \mid K_{n m} K_{m m} u, \widetilde{K}\right) \\
>  p(u)=\mathcal{N}\left(u \mid 0, K_{m m}\right)
>  \end{gathered}
> $$
>
> $$
>  \tilde{p}(\mathbf{y} \mid \mathbf{u})=\prod_{\mathbf{i}=1} \mathcal{N}\left(\mathbf{y}_{\mathbf{i}} \mid \mathbf{k}_{\mathrm{mn}}^{\top} \mathrm{K}_{\mathrm{mm}}^{-1} \mathrm{u}, \sigma^2\right) \exp \left\{-\frac{1}{2 \sigma^2}\left(\mathrm{k}_{\mathrm{nn}}-\mathrm{k}_{\mathrm{mn}}^{\top} \mathrm{K}_{\mathrm{mm}}^{-1} \mathrm{k}_{\mathrm{mn}}\right)\right\}
> $$
>
>  

## ONLINE SPARSE MATRIX GAUSSIAN PROCESSES

KERNAL FUNCTION:

$$
k\left(\mathbf{x}_i, \mathbf{x}_j\right)=c \exp \left(\frac{\left(\mathbf{x}_i-\mathbf{x}_j\right)^2}{\eta^2}\right) \max \left\{0,\left(1-\frac{\mathbf{x}_i-\mathbf{x}_j}{d} \mid\right)^\nu\right\}
$$
At each step during online learning of the GP, the proposed algorithm is presented with a training pair that is used to update the existing model. Assuming at time, the model is given by $P_{t}(f)$ , it is updated upon receiving the training output $y_{t + 1}$ using Bayes law

$$p_{t + 1}(f) \propto p\left( y_{t + 1} \middle| f \right)p_{t}(f)$$

$p\left( y_{t + 1} \middle| f \right)p_{t}$ is the measurement model

### Streaming Sparse Gaussian Process Approximations

This paper provides a new principled framework for deploying Gaussian process probabilistic models

in the streaming setting.

- The framework subsumes Csató and Opper’s two seminal approaches to online regression \[8, 9\] that were based upon *the variational free energy (VFE)* and **expectation propagation (EP)** approaches to approximate inference respectively. In the new framework, these algorithms are recovered as special cases.

- also provide principled methods for learning hyperparameters (learning was not treated in the original work and the extension is non-trivial) and **optimizing pseudo-input locations (previously handled via hand-crafted heuristics)**.

- The approach also relates to the streaming variational Bayes framework \[10

Given N input and real-valued output pairs $\left\{ X_{n},\ y_{n} \right\}_{n\  = 1}^{N}$ a standard GP regression model assumes $y_{n}\  = \ f(x_{n})\  + \ \epsilon_{n}$, where f is an unknown function that is corrupted by Gaussian observation noise $\epsilon_{n}\ \sim\ N(0,\ \sigma_{y}^{2})$. Typically, f is assumed to be drawn from a zero-mean GP prior $f\ \sim\ GP(0;\ k( \bullet \ ;\  \bullet |\theta))$ whose covariance function depends on hyperparameters $\theta$. In this simple model, the posterior over f, $p\left( f \middle| y,\ \theta \right)$ and the marginal likelihood $p\left( y \middle| \theta \right)$ can be computed analytically (here we have collected the observations into a vector $y\  = \left\{ y_{n} \right\}_{n = 1}^{N}\ $. However, these quantities present a computational challenge resulting in an O(N3) complexity for maximum likelihood training and O(N2) per test point for prediction

variational free energy approximation scheme which lower bounds the marginal likelihood of the data using a variational distribution q(f) over the latent function:

$$
\log p(\mathbf{y} \mid \theta)=\log \int \mathrm{d} f p(\mathbf{y}, f \mid \theta) \geq \int \mathrm{d} f q(f) \log \frac{p(\mathbf{y}, f \mid \theta)}{q(f)}=\mathcal{F}_{\mathrm{vfe}}(q, \theta)
$$
Since $F_{vfe}(q,\theta) = \log{p\left( y \middle| \theta \right)} - KL\lbrack q(f)|\left| p\left( f \middle| y,\theta \right) \right\rbrack\ $where $KL\lbrack \bullet ||\  \bullet \rbrack$denotes the Kullback–Leibler

divergence, maximising this lower bound with respect to q(f) guarantees the approximate posterior

gets closer to the exact posterior $p\left( f \middle| y,\ \theta \right)$. Moreover, the variational bound $F_{vfe}(q,\theta)$ approximates

the marginal likelihood and can be used for learning the hyperparameters$\ \theta$.

<img src="/assets/Machine_LearningNote.assets/image-20230522002224288.png" alt="image-20230522002224288" style="zoom:50%;" />

> ![image-20230522002244977](/assets/Machine_LearningNote.assets/image-20230522002244977.png)
>
> $$\mathcal{F\_}\text{vfe\ }\left( q(u),\theta \right) = \int_{}^{}df\ q(f)\log\frac{p\left( \mathbf{y} \middle| f,\theta \right)p\left( \mathbf{u} \middle| \theta \right)p\left( f_{\neq \mathbf{u}} \middle| \mathbf{u},\theta \right)}{p\left( f_{\neq \mathbf{u}} \middle| \mathbf{u},\theta \right)q\left( \mathbf{u} \right)}$$

$$= - KL\lbrack q(\mathbf{u})\| p(\mathbf{u}|\theta)\rbrack + \sum_{n}^{}{}\int d\mathbf{u}q(\mathbf{u})p\left( f_{n}|\mathbf{u},\theta \right)\log p\left( y_{n}|f_{n},\theta \right)$$

$$p(f|\mathbf{y},\theta) \approx q_{vfe}(f) \propto p\left( f_{\neq \mathbf{u}}|\mathbf{u},\theta \right)p(\mathbf{u}|\theta)\mathcal{N}\left( \mathbf{y};\mathbf{K}_{fu}\mathbf{K}_{\mathbf{uu}}^{- 1}\mathbf{u},\sigma_{y}^{2}I \right)$$

$$\log p(\mathbf{y}|\theta) \approx \mathcal{F}_{vfe}(\theta) = \log\mathcal{N}\left( \mathbf{y};\mathbf{0},\mathbf{K}_{fu}\mathbf{K}_{\mathbf{uu}}^{- 1}\mathbf{K}_{\mathbf{uf}} + \sigma_{y}^{2}\mathbf{I} \right) - \frac{1}{2\sigma_{y}^{2}}\sum_{n}^{}{}\left( k_{nn} - \mathbf{K}_{n\mathbf{u}}\mathbf{K}_{\mathbf{uu}}^{- 1}\mathbf{K}_{\mathbf{u}n} \right)$$

$$\mathcal{F}_{vfe}(q(\mathbf{u}),\theta) \approx - KL\lbrack q(\mathbf{u})\| p(\mathbf{u}|\theta)\rbrack + \frac{N}{|B|}\sum_{y_{n} \in B}^{}{}\int d\mathbf{u}q(\mathbf{u})p\left( f_{n}|\mathbf{u},\theta \right)\log p\left( y_{n}|f_{n},\theta \right)$$

Consider an approximation to the true posterior at the previous step, qold(f), which must be updated

to form the new approximation qnew(f),

$$
\begin{eqnarray*}
&& q_{\text {old }}(f) \approx p\left(f \mid \mathbf{y}_{\text {old }}\right)=\frac{1}{\mathcal{Z}_1\left(\theta_{\text {old }}\right)} p\left(f \mid \theta_{\text {old }}\right) p\left(\mathbf{y}_{\text {old }} \mid f\right), \tag{2} \\
&& q_{\text {new }}(f) \approx p\left(f \mid \mathbf{y}_{\text {old }}, \mathbf{y}_{\text {new }}\right)=\frac{1}{\mathcal{Z}_2\left(\theta_{\text {new }}\right)} p\left(f \mid \theta_{\text {new }}\right) p\left(\mathbf{y}_{\text {old }} \mid f\right) p\left(\mathbf{y}_{\text {new }} \mid f\right) . \tag{3}
\end{eqnarray*}
$$
the new approximation cannot access $p(y_{old}|f)$ directly. Instead, we can find an approximation of $p(y_{old}|f)$ by inverting eq. (2), ![](media/image45.emf)

$$
\hat{p}\left(f \mid \mathbf{y}_{\text {old }}, \mathbf{y}_{\text {new }}\right)=\frac{\mathcal{Z}_1\left(\theta_{\text {old }}\right)}{\mathcal{Z}_2\left(\theta_{\text {new }}\right)} p\left(f \mid \theta_{\text {new }}\right) p\left(\mathbf{y}_{\text {new }} \mid f\right) \frac{q_{\text {old }}(f)}{p\left(f \mid \theta_{\text {old }}\right) \tag{4}}
$$
The form of the approximate posterior mirrors that in the batch case, that is, the previous approximate posterior, $\mathbf{q}_{\mathbf{old}}\mathbf{(f)\  = \ p}\left( \mathbf{f}_{\mathbf{\neq a}}\mathbf{;}\mathbf{\theta}_{\mathbf{old}} \right)\mathbf{q}_{\mathbf{old}}\mathbf{(a}\mathbf{)}$ where we assume $q_{old}(a) = \ N(a;m_{a}\ ;\ S_{a}).$ The new posterior approximation takes the same form, but with the new pseudo-points and new hyperparameters: $\mathbf{q}_{\mathbf{new}}\mathbf{(f)\  = \ p}\left( \mathbf{f}_{\mathbf{\neq b}}\mathbf{;}\mathbf{\theta}_{\mathbf{new}} \right)\mathbf{q}_{\mathbf{new}}\mathbf{(b)}$. Similar to the batch case, this approximate inference problem can be turned into an optimization problem using variational inference. Specifically, consider

$$
\begin{eqnarray*}
\mathrm{KL}\left[q_{\text {new }}(f) \| \hat{p}\left(f \mid \mathbf{y}_{\text {old }}, \mathbf{y}_{\text {new }}\right)\right] & =\int \mathrm{d} f q_{\text {new }}(f) \log \frac{p\left(f_{\neq \mathbf{b}} \mid \mathbf{b}, \theta_{\text {new }}\right) q_{\text {new }}(\mathbf{b})}{\frac{Z_1\left(\theta_{\text {old }}\right)}{\mathcal{Z}_2\left(\theta_{\text {new }}\right.} p\left(f \mid \theta_{\text {new }}\right) p\left(\mathbf{y}_{\text {new }} \mid f\right) \frac{q_{\text {old }}(f)}{p\left(f \mid \theta_{\text {old }}\right)}} \tag{5}\\
& =\log \frac{\mathcal{Z}_2\left(\theta_{\text {new }}\right)}{\mathcal{Z}_1\left(\theta_{\text {old }}\right)}+\int \mathrm{d} f q_{\text {new }}(f)\left[\log \frac{p\left(\mathbf{a} \mid \theta_{\text {old }}\right) q_{\text {new }}(\mathbf{b})}{p\left(\mathbf{b} \mid \theta_{\text {new }}\right) q_{\text {old }}(\mathbf{a}) p\left(\mathbf{y}_{\text {new }} \mid f\right)}\right] \\
\end{eqnarray*}
$$

$$
\begin{eqnarray*}
q_{\mathrm{nfe}}(\mathbf{b}) && \propto p(\mathbf{b}) \exp \left(\int \mathrm{da} p(\mathbf{a} \mid \mathbf{b}) \log \frac{q_{\text {old }}(\mathbf{a})}{p\left(\mathbf{a} \mid \theta_{\text {old }}\right)}+\int \mathrm{df} p(\mathbf{f} \mid \mathbf{b}) \log p\left(\mathbf{y}_{\text {new }} \mid \mathbf{f}\right)\right) \tag{6}\\
&& \propto p(\mathbf{b}) \mathcal{N}\left(\hat{\mathbf{y}} ; \mathbf{K}_{\mathbf{f b}} \mathbf{K}_{\mathbf{b b}}^{-1} \mathbf{b}, \Sigma_{\hat{\mathbf{y}}, \mathrm{vfo}}\right) \tag{7}
\end{eqnarray*}
$$



where $\mathrm{f}$ is the latent function values at the new training points,
$$
\hat{\mathbf{y}}=\left[\begin{array}{c}
\mathbf{y}_{\mathrm{ncw}} \\
\mathbf{D}_{\mathrm{a}} S_{\mathrm{a}}^{-1} \mathbf{m}_{\mathrm{a}}
\end{array}\right], \mathbf{K}_{\mathbf{f b}}=\left[\begin{array}{c}
\mathbf{K}_{\mathrm{fb}} \\
\mathbf{K}_{\mathrm{ab}}
\end{array}\right], \Sigma_{\hat{y}, v f e}=\left[\begin{array}{cc}
\sigma_y^2 \mathrm{I} & 0 \\
0 & \mathbf{D}_{\mathrm{a}}
\end{array}\right], \mathbf{D}_{\mathrm{a}}=\left(\mathbf{S}_{\mathrm{a}}^{-1}-\mathbf{K}_{\mathrm{aa}}^{\prime-1}\right)^{-1} .
$$
The negative variational free energy is also analytically available,
$$
\begin{eqnarray*}
&& \mathcal{F}(\theta)=\log \mathcal{N}\left(\hat{\mathbf{y}} ; \mathbf{0}, \mathbf{K}_{\hat{\mathbf{f b}}} \mathbf{K}_{\mathrm{bb}}^{-1} \mathbf{K}_{\mathrm{b} \hat{\mathrm{f}}}+\Sigma_{\hat{\mathbf{y}}, \mathrm{vfe}}\right)-\frac{1}{2 \sigma_{\hat{y}}^2} \operatorname{tr}\left(\mathbf{K}_{\mathrm{ff}}-\mathbf{K}_{\mathbf{f b}} \mathbf{K}_{\mathbf{b b}}^{-1} \mathbf{K}_{\mathrm{bf}}\right)+\Delta_{\mathbf{a}} ; \text { where } \tag{8}\\
&& 2 \Delta_{\mathbf{a}}=-\log \left|\mathbf{S}_{\mathbf{a}}\right|+\log \left|\mathbf{K}_{\mathrm{aa}}^{\prime}\right|+\log \left|\mathbf{D}_{\mathbf{a}}\right|+\mathbf{m}_{\mathrm{a}}^{\top}\left(\mathbf{S}_{\mathrm{a}}^{-1} \mathbf{D}_{\mathrm{a}} \mathbf{S}_{\mathrm{a}}^{-1}-\mathbf{S}_{\mathrm{a}}^{-1}\right) \mathbf{m}_{\mathrm{a}}-\operatorname{tr}\left[\mathbf{D}_{\mathrm{a}}^{-1} \mathbf{Q}_{\mathrm{a}}\right]+\text { const. } \\
&&
\end{eqnarray*}
$$

### A Bayesian Approach to Online Learning

**In the Bayesian approach to statistical inference**, the degrees of prior belief or plausibility of parameters are expressed within probability distributions, the so called prior distributions (or priors) p(θ). Once this idea is accepted, subsequent inferences can be based on Bayes rule of probability. Formally, we may think that data are generated by a two step process:

First, the true parameter θ is drawn at random from the prior distribution p(θ).

Second, the data are drawn at random from $P(D_{t}\ |\theta)$. Bayes rule yields the conditional probability density (posterior) of the unknown parameter θ, given the data:

<img src="/assets/Machine_LearningNote.assets/image-20230522003339315.png" alt="image-20230522003339315" style="zoom:50%;" />

In order to construct an online algorithm within the Bayesian frameweork, we have to find out how the posterior distribution changes when a new datapoint $y_{t + 1}$ is observed. It can be easily shown that the new posterior corresponding to the new dataset $D_{t + 1}$ is given in terms of the old posterior and the likelihood of the new example by *(How is unknow yet… to check the ‘A Tutorial on Learning With Bayesian Networks’ )*

![](media/image50.emf)

does not have the form of an online algorithm, because it requires the knowledge of the entire old dataset Dt.

- Update: Use the **old approximative posterior** $\mathbf{p}\left( \mathbf{\theta} \middle| \mathbf{par}\left( \mathbf{t} \right) \right)$ to perform an update of the form (6.1)

> ![](media/image51.emf)

- Project: The new posterior $p\left( \theta \middle| y_{t + 1},\ par(t) \right)$ will usually not belong to the parametric family $p\left( \theta \middle| par \right)$. Hence, in the next step, it need to be projected into this family in order to obtain $p\left( \theta \middle| par(t\  + \ 1) \right)$. The parameter $par(t\  + \ 1)$ must be chosen such that $p\left( \theta \middle| par(t\  + \ 1) \right)$ is as close as possible to $p\left( \theta \middle| y_{t + 1},\ par(t) \right)$. It is a not clear a priori, which measure of dissimilarity between distributions should be used. Different choices may lead to different algorithms. I have chosen the KL-divergence

![](media/image52.emf)

Minimizing (6.3) can be thought of as minimizing the loss of information in the projection step. For the important case, where the parametric family is an exponential family, i.e. if the densities are of the form

![](media/image53.emf)

If we use a general multivariate Gaussian distribution for $p(\theta|par)$, then

$par\  = \ (mean,\ covariance)\  = \ ({\widehat{\theta}}_{i}\ ,C_{ij}\ )$ Matching the moments results in

![](media/image54.emf)

Using a simple property of centered Gaussian random variables z, namely the fact that for well behaved functions f, we have $E(z\ f(z))\  = \ E(f'(z))・\ E(z^{2})$*,* we can get the explicit update:

![](media/image55.emf)

$\int_{}^{}{P\left( y_{t + 1} \middle| \theta \right)p\left( \theta \middle| part(t) \right)d\theta} = \ E_{u}\left\lbrack P\left( y_{t} + 1 \middle| \widehat{\theta}(t) + \ u \right) \right\rbrack$ where u is a zero mean Gaussian random vector with covariance $C(t)$

Online Learning

Often, learning algorithms for estimating the unknown parameter θ∗ are based on the principle of Maximum Likelihood (ML). It states that we should choose a parameter θ which maximizes the likelihood P(Dt\|θ) of the observed data. Under weak assumptions, ML estimators are asymptotically efficient. As a learning algorithm, one can use e.g. a gradient descent algorithm and iterate

$\theta_{k + 1} = \theta_{k} - \eta_{k}\mathcal{g}_{k} = \ \theta_{k} - \eta_{k}\nabla f(\theta),\ \ g(\theta) = \nabla_{\theta}f(\theta) = \ \frac{1}{n}\sum_{i = 1}^{n}{\nabla_{\theta}f\left( \theta,x_{i} \right)}$

![](media/image56.emf)

![](media/image57.emf)

Here,$\ E_{T}\ (y_{k}|\theta)$ defines the training energy of the examples to be minimzed by the algorithm. When a new example yt+1 is received, the ML procedure requires that the learner has to update her estimate for θ using all previous data. Hence Dt has to be stored in a memory. **The goal of online learning is to calculate a new estimate** $\widehat{\mathbf{\theta}}\mathbf{(}\mathbf{t\ }\mathbf{+ \ 1)}$ **which is only based on the new data point yt+1, the old estimate** $\widehat{\mathbf{\theta}}\mathbf{(}\mathbf{t}\mathbf{)}$ (and possibly a set of other auxiliary quantities which have to be updated at each time step, but are much smaller in number than the entire set of previous training data). A popular idea is to use a procedure similar to (4.1), but to replace the training energy of all examples $\ \sum_{K = 1}^{t}{E_{T}\ (y_{k}|\theta)}\ $by the training energy of the most recent one. Hence, we get

![](media/image58.emf)

### Sparse Online Gaussian Processes

#### at each step of the algorithm we combine the Kullback-Leibler divergence between distributions 

<img src="C:\Users\zphil\AppData\Roaming\Typora\typora-user-images\image-20230505233859783.png" alt="image-20230505233859783" style="zoom:50%;" />

where θ denotes the set of arguments of the densities. If $\widehat{p}$ denotes the approximating Gaussian distribution, one usually tries to minimise $KL(\widehat{p}||p_{post})$, with respect to $\widehat{p}$ which in contrast to $KL(p_{post}||\widehat{p})$requires only the computation of expectations over tractable distributions.

Definition 8.11 (KL divergence). For two distributions q(x) and p(x)

$$KL\left( q \middle| p\  \right) = D_{KL}\left( q|p \right) = \ \left\langle \log{q(x)} \right.\  - \left. \ \log{p(x)} \right\rangle_{q(x)} \geq 0$$

$$D_{KL}\left( q|p \right) = \ \int_{}^{}{q(x)\log\left( \frac{q(x)}{p(x)} \right)dx}$$

likelihood of a single new data point and the (GP) prior from the result of the previous approximation step. Let ${\widehat{p}}_{t}$ denote the Gaussian approximation after processing t examples, we use Byes rule

> $$p_{post}\left( f|y \right) = \frac{p\left( y_{t + 1} \middle| f \right)\widehat{p_{t}}(f)}{\left\langle p\left( y_{t + 1} \middle| f_{D} \right) \right\rangle t}$$

To derive the updated posterior. Since post is no longer Gaussian, we use a variational technique in order to project it to the closest Gaussian process ${\widehat{p}}_{t + 1}$ (see Fig. 1). Unlike the usual variational method, we will now minimize the divergence $KL(p_{post}|{\widehat{p}}_{p}).$ This is possible, because in our on-line method, the posterior contains only the likelihood for a single example and the corresponding non Gaussian integral is one-dimensional, which can, for many relevant cases be performed analytically. It is a simple exercise to show (Opper 1998) that the projection results in the matching of the first two moments (mean and covariance) of $p_{post}$ and the new Gaussian posterior ${\widehat{p}}_{t + 1}$

![](media/image60.emf)

In order to compute the on-line approximations of the mean and covariance kernel Kt we apply Lemma 1 sequentially with only one likelihood term $P(y_{t}|x_{t})$ at an iteration step. Proceeding recursively, we arrive at

![](media/image61.emf)

The averages in (7) are with respect to the Gaussian process at time t and the derivatives taken with respect to $\left\langle f_{t + 1} \right\rangle_{t}\  = \ \left\langle {f(X}_{t + 1}) \right\rangle_{t}$ Note again, that these averages only require a one dimensional integration over the process at the input xt+1. Unfolding the recursion steps in the update rules (6) we arrive at the parametrization for the approximate posterior GP at time t as a function of the initial kernel and the likelihoods (“natural parametrisation”):

![](media/image62.emf)

![](media/image63.emf)

The Sparse GP Algorithm

For each data element (yt+1, xt+1) we will iterate the following steps:

1\. Compute $q^{t + 1},\ r^{t + 1},\ k_{t + 1}^{*},\ k_{t + 1},\ {\widehat{e}}_{t + 1},\ and\ \gamma_{t + 1}.$

2\. If $\ \gamma_{t + 1} < \epsilon_{tol}$ then perform a reduced update with ${\widehat{s}}_{t + 1}$ from eq. (11) without extending the size of the parameters α and C. Advance to the next data.

3\. (else) Perform the update eq. (9) using the unit vector $e_{t + 1}$. Add the current input to the BV set and compute the inverse of the extended Gram matrix using eq. (24).

4\. If the size of the BV set is larger than d, then compute the scores $\varepsilon_{i}$ for all BV s from eq. (27), find the basis vector with the minimum score and delete it using eqs. (25).

#### Variational free energy (VFE) – Variational learning of inducing variables in spare gaussian processes

$$logp(y) \geq \ \int_{}^{}{q(u,f)log\frac{p\left( y \middle| f \right)p\left( f \middle| u \right)p(u)}{p\left( f \middle| u \right)q(u)}dudf}$$

Suppose we have a training dataset $\left\{ \left( x_{i},\ y_{i} \right) \right\}_{i = 1}^{n}\ $of n noisy realizations of some unobserved or latent function so that each scalar yi is obtained by adding Gaussian noise to f(x) at input xi, i.e. yi = fi + ǫi, where ǫi ∼ N(0, σ2) and fi = f(xi). Let X denote all training inputs, y all outputs and f the corresponding training latent function values. The joint probability model is $\ p(y,\ f\ )\  = \ p(y|f\ )p(f\ )$

***Conditional Probabilities***

<img src="/assets/Machine_LearningNote.assets/image-20230522003541092.png" alt="image-20230522003541092" style="zoom:50%;" />

where p(y\|f ) is the likelihood and p(f ) the GP prior. The data induce a posterior GP which is specified by a posterior mean function and a posterior covariance function:

![](media/image65.emf)

To define a sparse method that directly approximates the posterior GP mean and covariance functions in eq. (1). This posterior GP can be also described by the predictive Gaussian $p(z|y)\  = \int_{}^{}{p(z|f\ )p(f\ |y)}df$ , **where** $\mathbf{p(z|f\ )}$ **denotes the conditional prior over any finite set of function points z**. Suppose that we wish to approximate the above Bayesian integral by using a small set of m auxiliary inducing variables $f_{m}$ evaluated at the pseudo-inputs $X_{m}$, which are independent from the training inputs. fm are just function points drawn from the same GP prior as the training function values f . **By using the augmented joint model** $\mathbf{p(y|f\ )p(z,\ }\mathbf{f}_{\mathbf{m}}\mathbf{,\ f\ )}$,

where $\mathbf{p(z,\ }\mathbf{f}_{\mathbf{m}}\mathbf{,\ f\ )}$ is the GP prior jointly expressed over the function values z, f and fm

we equivalently write p(z\|y) as

$$\mathbf{p}\left( \mathbf{z} \middle| \mathbf{y} \right)\mathbf{= \ }\int_{}^{}{\mathbf{p}\left( \mathbf{z} \middle| \mathbf{f}_{\mathbf{m}}\mathbf{,\ f} \right)\mathbf{p}\left( \mathbf{f} \middle| \mathbf{f}_{\mathbf{m}}\mathbf{,\ y} \right)\mathbf{p}\left( \mathbf{f}_{\mathbf{m}} \middle| \mathbf{y} \right)\mathbf{dfd}\mathbf{f}_{\mathbf{m}}}\mathbf{\ \ \ \ \ \ }\left( \mathbf{4} \right)$$

Suppose now that $f_{m}$ is a sufficient statistic for the parameter f in the sense that z and f are independent given fm, i.e. it holds$\ p(z|f_{m},\ f\ )\  = \ p(z|f_{m}).$ The above can be written as

![](media/image66.emf)

$$where\ q(z)\  = \ p(z|y)\ and\ \phi(f_{m})\  = \ p(f_{m}|y).$$

$p(f\ |f_{m})\  = \ p(f\ |f_{m},\ y)$ is true since y is a noisy version of f and because of the assumption we made **that any z is conditionally independent from f given fm**

$\mathbf{p}\left( \mathbf{z} \middle| \mathbf{f}_{\mathbf{m}}\mathbf{,y} \right)\mathbf{= \ }\frac{\int_{}^{}{\mathbf{p}\left( \mathbf{y} \middle| \mathbf{f} \right)\mathbf{p}\left( \mathbf{z,}\mathbf{f}_{\mathbf{m}}\mathbf{,f} \right)\mathbf{df}}}{\int_{}^{}{\mathbf{p}\left( \mathbf{y} \middle| \mathbf{f} \right)\mathbf{p}\left( \mathbf{z,}\mathbf{f}_{\mathbf{m}}\mathbf{,f} \right)\mathbf{dfdz}}}$ ***and by using the*** $\mathbf{p(z|}\mathbf{f}_{\mathbf{m}}\mathbf{,\ f\ )\  = \ p(z|}\mathbf{f}_{\mathbf{m}}\mathbf{).}$

Thus, we expect q(z) to be only an approximation to p(z\|y). In such case, we can choose φ(fm) to be a “free” variational Gaussian distribution, where in general φ(fm) != p(fm\|y), **that depends on a mean vector μ and a covariance matrix A.**

the approximate posterior GP mean and covariance functions as follows

![](media/image67.emf)

$$B\  = \ K_{mm}^{- 1}{AK}_{mm}^{- 1}$$

The question that now arises is how **do we select the φ distribution, i.e. (μ, A), and the inducing inputs Xm.** ***a variational method that allows to jointly specify these quantities and treat Xm as a variational parameter which is rigorously selected by minimizing the KL divergence.***

*使用* $q(z)$*去近似*$p(z|y)$*,这里*$p(z|y)$*是predictive Gaussian*

![](media/image68.png)

Next we describe a variational method that allows to jointly specify these quantities and treat Xm as a variational parameter which is rigorously selected by minimizing the KL divergence.

{X}

## Logistic regression - Optimization

Sigmoid/logistic function

$$P\left( y_{i} = 1 \middle| {\underline{x}}_{i},\ \underline{\theta} \right) = \Pi_{i} = \ sigm(\eta) = \frac{1}{1 + e^{- \eta}} = \frac{1}{1 + e^{- x_{i}\underline{\theta}}}$$

<img src="/assets/Machine_LearningNote.assets/image-20230519163637630.png" alt="image-20230519163637630" style="zoom:80%;" />

<img src="/assets/Machine_LearningNote.assets/image-20230519163647877.png" alt="image-20230519163647877" style="zoom:80%;" />

Logistic regression

$${\pi_{i}\ is\ success\ rate,\ \pi_{i}}^{y_{i}}{({1 - \pi}_{i})}^{1 - y_{i}} = \ \left\{ \begin{array}{r}
\pi_{i}{\ \ \ \ \ \ \ y}_{i} = 1 \\
1 - \pi_{i\ }\ \ \ y_{i} = 0 \\
\end{array} \right.\ $$

$$\mathbf{likelihood\ function\ }$$

$$p\left( y \middle| X,\theta \right) = \ \prod_{i = 1}^{n}{Ber\left( y_{i} \middle| sigm\left( x_{i}\theta \right) \right) = \prod_{i = 1}^{n}{{\pi_{i}}^{y_{i}}{({1 - \pi}_{i})}^{1 - y_{i}}} = \ \prod_{i = 1}^{n}{\left\lbrack \frac{1}{1 + e^{- x_{i}\theta}} \right\rbrack^{y_{i}}\left\lbrack \frac{1}{1 + e^{- x_{i}\theta}} \right\rbrack^{1 - y_{i}}}}$$

*, where* $X_{i}\theta\  = \ \theta_{0} + \sum_{j = 1}^{d}{\theta_{j}\mathcal{x}_{ij}}$

***The cost function (NLL the likelihood function) – cross entropy error function***

$$NLL(W)\, = \, - \log{P\left( y \middle| x,\,\theta \right)\, = \, J(\theta)\, = - \sum_{i = 1}^{n}{y_{i}\log{\pi_{i} + \left( 1 - y_{i} \right)\log\left( 1 - \pi_{i} \right)}}}$$

$$g(\theta)\, = \,\frac{d}{d\theta}J(\theta) = \,\sum_{i = 1}^{n}{{x_{i}}^{T}\left( \pi_{i} - y_{i} \right) = X^{T}(\pi - y)}$$

$H = \frac{d}{d\theta}g(\theta)^{T} = \sum_{i}^{}{\pi_{i}\left( 1 - \pi_{i} \right)x_{i}x_{i}^{T} = X^{T}diag\left( \pi_{i}\left( 1 - \pi_{i} \right) \right)X} = X^{T}SX$*,*

*where* $S = \ diag\left( \pi_{i}\left( 1 - \pi_{i} \right) \right)$*,* $\pi_{i} = sigmoid\left( x_{i}\theta \right)$

*one can show that H is a positive definite , hence the NLL is convex (凸函数) and has a unique global minimum. To find the minimum …*

1.  ***Gradient descent***

2.  ***Newton’s method***

*In the WLS, the unknowns V and β arise together. And the basic idea of Iteratively Re-weighted Least Square is that you are going to update one given the other. The key is finding out the right formula for*

1.  *V that depends on β*

2.  *β that depends on V*

*Then either one would be initialized and updated iteratively.*

<img src="/assets/Machine_LearningNote.assets/image-20230519163609003.png" alt="image-20230519163609003" style="zoom:50%;" />

*Note: Found the likelihood take the negative log found the derivative to get the **gradient** follow the gradient or follow the gradients weighted by the **Hessian***

## Bayesian logistic regression and MCMC

### Bayesian logistic regression

$p\left( y \middle| X,\theta \right) = \ \prod_{i = 1}^{n}{Ber\left( y_{i} \middle| sigm\left( x_{i}\theta \right) \right)} = \ \prod_{i = 1}^{n}{\left\lbrack \frac{1}{1 + e^{- x_{i}\theta}} \right\rbrack^{y_{i}}\left\lbrack \frac{1}{1 + e^{- x_{i}\theta}} \right\rbrack^{1 - y_{i}}}$ 

Posterior in Bayesian:

> $p\left( \theta \middle| D \right) = \ \frac{p\left( y \middle| x,\theta,\  \right)p(\theta)}{p\left( y \middle| x \right)}$ *,* $z=p\left( y \middle| x \right) = \ \int_{}^{}{p\left( y \middle| x,\theta \right)p(\theta)d\theta}$ *this integrate cannot be done by hand .   y is binary… so we have* $p\left( \theta \middle| D \right)\  \propto \ p\left( y \middle| x,\theta,\  \right)p(\theta)$
>
> And *with gaussian prior* $p(\theta) = (2\pi\Sigma)^{- 1\text{/}2}e^{- \frac{1}{2}(\theta - \mu)^{T}\Sigma^{- 1}(\theta - \mu)}$

$$
\begin{aligned}
& z=\int p(y |\theta) p(\theta) d \theta \\ 
& \text {introduce the } q(\theta)=N(0,1000) \\
& z=\int \frac{p(y|\theta) p(\theta) q(\theta)}{q(\theta)} d \theta \\
& \text {introduce the } w(\theta) = \frac{p(y|\theta) p(\theta) }{q(\theta)} \\
& z=\int \omega(\theta) q(\theta) d \theta \\
& \Theta^{(i)} \sim q(\theta), \quad i=1: N \\
& z \approx \frac{1}{N} \sum_{i=1}^N \omega\left(\theta^{(i)}\right)\\
&
\end{aligned}
$$

<img src="/assets/Machine_LearningNote.assets/image-20230519165455741.png" alt="image-20230519165455741" style="zoom:80%;" />
$$
\begin{aligned}

\text {we have } P(d \theta | data) &=P(\theta |data) d \theta \\

P(d \theta \mid \text { data }) &=\frac{1}{N} \sum_{i=1}^N \omega\left(\theta^{(i)}\right) \delta_{\theta^{(i)}}(d \theta) \\

\delta_{\theta^{(i)}}(d \theta) &=\text { Number of samples } \theta^{(i)} \text { in the interval } d \theta 
\end{aligned}
$$
the bayesian prediction function: 

**with normalized weight (w)**
$$
\begin{aligned}
& P\left(y_{t+1} \mid x_{t+1}, y_{1: t}, x_{1: t}\right)=\int P\left(y_{t+1} \mid x_{t+1}, \theta\right) P\left(d \theta \mid x_{1: t}, y_{1: t}\right) \\
& =\int P\left(y_{t+1} \mid x_{t+1}, \theta\right) P\left(\theta \mid x_{1:t}, y_{1: t}\right) d \theta \\
& \approx \int P\left(y_{t+1} \mid x_{t+1}, \theta\right) \frac{1}{N} \sum_{i=1}^N \omega\left(\theta^{(i)}\right) \delta_{\theta^{(i)}}(d \theta) \\
& \approx \frac{1}{N} \sum_{i=1}^N \int P\left(y_{i+1} \mid x_{+1+1}, \theta\right) \omega\left(\theta^{(i)}\right) \delta_{\theta^{(i)}}(d \theta) \\
& \approx \frac{1}{N} \sum_{i=1}^{N} \underbrace{P\left(y_{t+1} \mid x_{t+1}, \theta^{(i)}\right.}_{\text {likelihood }}) \omega\left(\theta^{(i)}\right) \\
\end{aligned}
$$
**with un-normalized weight (w)**
$$
\begin{aligned}
P(\theta \mid D) &=\frac{1}{z} P(D \mid \theta) P(\theta)=\frac{P(D \mid \theta) P(\theta)}{\int P(D \mid \theta) P(\theta) d \theta} \\

P(y_{t+1}|x_{t+1},D) &= P\left(y_{t+1} \mid x_{t+1}, y_{1: t}, x_{1: t}\right) \\
&= \int P\left(y_{t+1} \mid x_{t+1}, \theta\right) P (\theta | D) d\theta \\
& =\frac{1}{z} \int P\left(y_{t+1} \mid x_{t+1}, \theta\right) P (D|\theta)P(\theta) d\theta \\
& =\frac{\int P\left(y_{t+1} \mid x_{t+1}, \theta\right) P(D \mid \theta) P(\theta) \frac{q(\theta)}{q(\theta)} d \theta}{\int P(D \mid \theta) P(\theta) \frac{q(\theta)}{q(\theta)} d \theta} \\
& =\frac{\int P\left(y_{t+1} \mid x_{t+1}, \theta\right) \omega(\theta) q(\theta) d \theta}{\int \omega(\theta) q(\theta) d \theta} \\

& = \frac{\frac{1}{N} \sum_{i=1}^N \omega\left(\theta^{(i)}\right) P\left(y_{t+1} \mid x_{t+1}, \theta^{(i)}\right)}{\frac{1}{N} \sum_{i=1}^N \omega\left(\theta^{(i)}\right)} \\

&= \sum_{i=1}^N \widetilde{\omega}^i\left(\theta^{(i)}\right) P\left(y_{t+1} \mid x_{t+1}, \theta^{(i)}\right) \\

& \text {given the }w^i=\frac{\omega^i}{\sum_j \omega^j} \\
& \text {here } q(\theta) \sim N(\mu, \sigma^2 I) , \\
& \text{select q hyperparameters could be selected via cross validation or bayesian optimization?}
\end{aligned}
$$

$$
\begin{aligned}
\Pi(x) = P\left(\theta \mid x_{1::}, y_{i: 1}\right) &=\frac{1}{z} \prod_{i=1}^{t}\left[\pi_i y_i\left(1-\pi_i\right)^{1-y_i}\right] e^{-\frac{1}{2 \delta^2} \theta^{\top} \theta} \\

\tilde{\Pi}(x) &=\frac{1}{z} \Pi(x) \\
z &= \int \Pi(x) d x
\end{aligned}
$$
want $\theta^{(i)} \sim P\left(\theta \mid Y_{1: t}, X_{1: t}\right)$

Predictive distribution,

we have $D\  = \ \left( X_{1:n},\ Y_{1:n} \right)$, new input $X_{n + 1}$, we want to predict $Y_{n + 1}$, *Bayesian belive it should integerate out of the effect of the parameters done by maraginalization*

$$P\left( Y_{n + 1} \middle| X_{n + 1},D \right) = \int_{}^{}{P\left( Y_{n + 1},\theta \middle| X_{n + 1},D \right)d\theta}
$$$= \ \int_{}^{}{P\left( Y_{n + 1} \middle| \theta,\ X_{n + 1},D \right)P\left( \theta \middle| X_{n + 1},D \right)d\theta\ }$*,*

$P\left( \theta \middle| X_{n + 1},D \right)$ *is the posterior* $\theta$ *which the* $X_{n + 1}$ *will do nothing to it , D is droped due to the* $\theta$ *told the information about the Data…* 
$$
= \ \int_{}^{}{P\left( Y_{n + 1} \middle| \theta,\ X_{n + 1} \right)P\left( \theta \middle| D \right)d\theta\ }
$$
*logistic function :* $P\left( Y_{n + 1} \middle| \theta,\ X_{n + 1} \right) = \ {\pi_{n + 1}}^{y_{n + 1}}{({1 - \pi}_{n + 1})}^{1 - y_{n + 1}}$

***\[ Bayesian say the prediction will be given by the likelihood , each prediction will be weigthed by how probablity it is, and the probablity is measured accoridng to the posterior distriubution which is the quantity that take into account prior information and the training data \]***

***With* Monte Carlo method**

$$= \ \int_{}^{}{P\left( Y_{n + 1} \middle| \theta,\ X_{n + 1} \right)P\left( \theta \middle| D \right)d\theta\ } = \ \frac{1}{n}\sum_{i = 1}^{n}{P\left( Y_{n + 1} \middle| \theta^{(i)},\ X_{n + 1} \right)}$$

### Markov Chain Basic

<img src="/assets/Machine_LearningNote.assets/image-20230519213253623.png" alt="image-20230519213253623" style="zoom:80%;" />

$X_n$ state after $n$ transitions
- belongs to a finite set
- initial state $X_0$ either given or random
- transition probabilities:

$$
\begin{aligned}
p_{i j} & =\mathbf{P}\left(X_1=j \mid X_0=i\right) \\
& =\mathbf{P}\left(X_{n+1}=j \mid X_n=i\right)
\end{aligned}
$$

Markov property/assumption:
"given current state, the past doesn't matter"
$$
\begin{aligned}
p_{i j} & =\mathbf{P}\left(X_{n+1}=j \mid X_n=i\right) \\
& =\mathbf{P}\left(X_{n+1}=j \mid X_n=i, X_{n-1}, \ldots, X_0\right)
\end{aligned}
$$
$X_{n+1} \text{ conditionally independent to } X_{n-1}, \ldots, X_0 \text{with given }X_n$

<img src="/assets/Machine_LearningNote.assets/image-20230520203000570.png" alt="image-20230520203000570" style="zoom: 50%;" />

<img src="/assets/Machine_LearningNote.assets/image-20230520221040191.png" alt="image-20230520221040191" style="zoom: 50%;" />

<img src="/assets/Machine_LearningNote.assets/image-20230520232437236.png" alt="image-20230520232437236" style="zoom:50%;" />



```python
#eactly show how frequency of the bebing in j is calcauted in code
state_space = ("sunny", "cloudy", "rainy")
...
n_steps = 20000
states = [0]
for i in range(n_steps):
    states.append(np.random.choice((0, 1, 2), p=transition_matrix[states[-1]]))
states = np.array(states)
...
offsets = range(1, n_steps, 5)
for i, label in enumerate(state_space):
    #here cacluate the sum of the frequence of the certain states and normalize it is the eactly \pi(x)
    ax.plot(offsets, [np.sum(states[:offset] == i) / offset 
            for offset in offsets], label=label)
```

<img src="/assets/Machine_LearningNote.assets/image-20230521004115245.png" alt="image-20230521004115245" style="zoom:50%;" />

Above gives one distribution answer to the question " what's the system state after certain time t"

### **Monte Carlo method**

<img src="/assets/Machine_LearningNote.assets/image-20230519162005088.png" alt="image-20230519162005088" style="zoom:67%;" />A geometric 

### MCMC: Metropolise-Hastings

**(refer to frequncey be in state j)** In order to sample from a distribution $π(x)$, a MCMC algorithm constructs and simulates a Markov chain whose stationary distribution is $π(x)$, meaning that, after an initial “burn-in” phase, the states of that Markov chain are distributed according to $π(x)$. <font color=red>**We thus just have to store the states to obtain samples from *π*(*x*).**</font>

For didactic purposes, let’s for now consider both a discrete state space and discrete “time”. The key quantity characterizing a Markov chain is the transition operator $T(x_{i+1}∣x_i)$ which gives you the probability of being in state $x_{i+1}$ at time $i+1$ given that the chain is in state $x_i$ at time $i$.

 a transition matrix *T*, or be continuous, in which case *T* would be a transition *kernel*.  while considering continuous distributions, but all concepts presented here transfer to the discrete case.

If we could design the transition kernel in such a way that the next state is already drawn from *π*, we would be done, as our Markov chain would… well… immediately sample from *π*. Unfortunately, to do this, we need to be able to sample from *π*, which we can’t.



1. Initialise $x^{(0)}$
2. For $i=0$ to $N-1$
   - $\Rightarrow$ Sample $u \sim U_{[0,1]}$.
   - $\Rightarrow$ Sample $x^{\star} \sim q\left(x^{\star} \mid x^{(i)}\right)$. e,g. $x^{*}=x^{(i)}+N\left(0,6^2\right)$
   - If $u<A\left(x^{(i)}, x^{\star}\right)=\min \left\{1, \frac{p\left(x^{\star}\right) q\left(x^{(i)} \mid x^{\star}\right)}{p\left(x^{(i)}\right) q\left(x^{\star} \mid x^{(i)}\right)}\right\}$ 

$$
x^{(i+1)}=x^{\star}
$$

​			else
$$
x^{(i+1)}=x^{(i)}
$$

$$
\begin{aligned}
r\left(\theta_{\text {new }}, \theta_{t-1}\right) & =\frac{\text { Posterior probability of } \theta_{\text {new }}}{\text { Posterior probability of } \theta_{t-1}} \\
& = \frac{p(D|\theta_{new})p(\theta_{new})}{p(D|\theta_{t-1})p(\theta_{t-1}))}\\
& =\frac{\operatorname{Beta}\left(1,1, \theta_{\text {new }}\right) \times \operatorname{Binomial}\left(10,4, \theta_{\text {new }}\right)}{\operatorname{Beta}\left(1,1, \theta_{t-1}\right) \times \operatorname{Binomial}\left(10,4, \theta_{t-1}\right)}
\end{aligned}
$$


### Geometric Series

[Geometric Series -- from Wolfram MathWorld](https://mathworld.wolfram.com/GeometricSeries.html)

series $\sum_k a_k$ is a series for which the ratio of each two consecutive terms $a_{k+1} / a_k$ is a constant function of the summation index $k$. The more general case of the ratio a rational function of the summation index $k$ produces a series called a hypergeometric series.
For the simplest case of the ratio $a_{k+1} / a_k=r$ equal to a constant $r$, the terms $a_k$ are of the form $a_k=a_0 r^k$. Letting $a_0=1$, the geometric sequence $\left\{a_k\right\}_{k=0}^n$ with constant $|r|<1$ is given by
$$
S_n=\sum_{k=0}^n a_k=\sum_{k=0}^n r^k
$$
is given by
$$
S_n \equiv \sum_{k=0}^n r^k=1+r+r^2+\ldots+r^n
$$
Multiplying both sides by $r$ gives
$$
r S_n=r+r^2+r^3+\ldots+r^{n+1}
$$
and subtracting (3) from (2) then gives
$$
\begin{aligned}
(1-r) S_n & =\left(1+r+r^2+\ldots+r^n\right)-\left(r+r^2+r^3+\ldots+r^{n+1}\right) \\
& =1-r^{n+1}
\end{aligned}
$$
so
$$
S_n \equiv \sum_{k=0}^n r^k=\frac{1-r^{n+1}}{1-r}
$$
For $-1<r<1$, the sum converges as $n \rightarrow \infty$, in which case
$$
S \equiv S_{\infty}=\sum_{k=0}^{\infty} r^k=\frac{1}{1-r}
$$
Similarly, if the sums are taken starting at $k=1$ instead of $k=0$
$$
\begin{aligned}
\sum_{k=1}^n r^k & =\frac{r\left(1-r^n\right)}{1-r} \\
\sum_{k=1}^{\infty} r^k & =\frac{r}{1-r}
\end{aligned}
$$
the latter of which is valid for $|r|<1$.



## Support Vector Machines (SVMs)

### ***Support Vector Machine (SVM) - Optimization objective***

- So far, we've seen a range of different algorithms

  - With supervised learning algorithms - performance is pretty similar

    - What matters more often is;

      - The amount of training data

      - Skill of applying algorithms

- One final supervised learning algorithm that is widely used - **support vector machine (SVM)**

- - Compared to both logistic regression and neural networks, a SVM sometimes gives a cleaner way of learning non-linear functions

  - Later in the course we'll do a survey of different supervised learning algorithms

**An alternative view of logistic regression**

- Start with logistic regression, see how we can modify it to get the SVM

  - As before, the logistic regression hypothesis is as follows  
    <img src="/assets/Machine_LearningNote.assets/image-20230519125842097.png" alt="image-20230519125842097" style="zoom:50%;" />

  - And the sigmoid activation function looks like this  
    <img src="/assets/Machine_LearningNote.assets/image-20230519125857873.png" alt="image-20230519125857873" style="zoom:50%;" />

  - In order to explain the math, we use z as defined above

- What do we want logistic regression to do?

  - We have an example where y = 1

    - Then we hope $h_θ(x)$ is close to 1

    - With $h_(θ)(x)$ close to 1, $(θ^T* x)$ must be **much larger** than 0  
      <img src="/assets/Machine_LearningNote.assets/image-20230519184128085.png" alt="image-20230519184128085" style="zoom:67%;" />

  - Similarly, when y = 0

    - Then we hope h_(θ)(x) is close to 0

    - With h_(θ)(x) close to 0, (θ*^(T)* x) must be **much less** than 0

  - This is our classic view of logistic regression

    - Let's consider another way of thinking about the problem

- Alternative view of logistic regression

  - If you look at cost function, each example contributes a term like the one below to the overall cost function  
    <img src="/assets/Machine_LearningNote.assets/image-20230519183649572.png" alt="image-20230519183649572" style="zoom:50%;" />
    - For the overall cost function, we sum over all the training examples using the above function, and have a 1/m term
  
- If you then plug in the hypothesis definition (h_(θ)(x)), you get an expanded cost function equation;  
  $$
  =-y \log \frac{1}{1+e^{-\theta^T x}}-(1-y) \log \left(1-\frac{1}{1+e^{-\theta^T x}}\right)
  $$
  
- - So each training example contributes that term to the cost function for logistic regression

&nbsp;

- If y = 1 then only the first term in the objective matters

  - If we plot the functions vs. z we get the following graph  
    <img src="/assets/Machine_LearningNote.assets/image-20230519184257735.png" alt="image-20230519184257735" style="zoom:50%;" />
    - This plot shows the cost contribution of an example when y = 1 given z
    
      - So if z is big, the cost is low - this is good!
    
      - But if z is 0 or negative the cost contribution is high
    
      - This is why, when logistic regression sees a positive example, it tries to set θ*^(T)* x to be a very large term
  
- If y = 0 then only the second term matters

- - We can again plot it and get a similar graph  
    <img src="/assets/Machine_LearningNote.assets/image-20230519184318627.png" alt="image-20230519184318627" style="zoom:50%;" />
    - Same deal, if z is small then the cost is low
    
      - But if s is large then the cost is massive

**SVM cost functions from logistic regression cost functions**

- To build a SVM we must redefine our cost functions

  - When y = 1

    - Take the y = 1 function and create a new cost function

    - Instead of a curved line create two straight lines (magenta) which acts as an approximation to the logistic regression y = 1 function  
      <img src="/assets/Machine_LearningNote.assets/image-20230519184335130.png" alt="image-20230519184335130" style="zoom:50%;" />
  - Take point (1) on the z axis
      
    - Flat from 1 onwards
      
    - Grows when we reach 1 or a lower number
      
  - This means we have two straight lines
      
    - Flat when cost is 0
      
    - Straight growing line after 1
      
- So this is the new y=1 cost function
    
  - Gives the SVM a computational advantage and an easier optimization problem
    
  - We call this function **cost₁(z)** 

&nbsp;

- Similarly

  - When y = 0

    - Do the equivalent with the y=0 function plot  
      <img src="/assets/Machine_LearningNote.assets/image-20230519184350925.png" alt="image-20230519184350925" style="zoom:50%;" />

    - We call this function **cost₀(z)**

- So here we define the two cost function terms for our SVM graphically

  - How do we implement this?

**The complete SVM cost function**

- As a comparison/reminder we have logistic regression below  
  $$
  \min _\theta \frac{1}{m}\left[\sum_{i=1}^m y^{(i)}\left(-\log h_\theta\left(x^{(i)}\right)\right)+\left(1-y^{(i)}\right)\left(\left(-\log \left(1-h_\theta\left(x^{(i)}\right)\right)\right)\right]+\frac{\lambda}{2 m} \sum_{j=1}^n \theta_j^2\right.
  $$

  - If this looks unfamiliar its because we previously had the - sign outside the expression
  
- For the SVM we take our two logistic regression y=1 and y=0 terms described previously and replace with

  - cost₁(θ*^(T)* x)

  - cost₀(θ*^(T)* x)

- So we get  
  $$
  \min _\theta \frac{1}{m} \sum_{i=1}^m\left[y^{(i)} \operatorname{cost}_1\left(\theta^T x^{(i)}\right)+\left(1-y^{(i)}\right) \operatorname{cost}_0\left(\theta^T x^{(i)}\right)\right]+\frac{\lambda}{2 m} \sum_{i=1}^n \theta_j^2
  $$

**SVM notation is slightly different**

- In convention with SVM notation we rename a few things here

- *1) Get rid of the 1/m terms*

  - This is just a slightly different convention

  - By removing 1/m we should get the same optimal values for 

    - 1/m is a constant, so should get same optimization

    - e.g. say you have a minimization problem which minimizes to u = 5

      - If your cost function \* by a constant, you still generates the minimal value

      - That minimal value is different, but that's irrelevant

- *2) For logistic regression we had two terms;*

  - Training data set term (i.e. that we sum over m) = **A**

  - Regularization term (i.e. that we sum over n) = **B**

    - So we could describe it as A + λB

    - Need some way to deal with the trade-off between regularization and data set terms

    - Set different values for λ to parametrize this trade-off

  - Instead of parameterization this as A + λB

    - For SVMs the convention is to use a different parameter called C

    - So do CA + B

    - If C were equal to 1/λ then the two functions (CA + B and A + λB) would give the same value

- So, our overall equation is  
  
- $$
  \min _\theta C \sum_{i=1}^m\left[y^{(i)} \operatorname{cost}_1\left(\theta^T x^{(i)}\right)+\left(1-y^{(i)}\right) \operatorname{cost}_0\left(\theta^T x^{(i)}\right)\right]+\frac{1}{2} \sum_{i=1}^n \theta_j^2
  $$

&nbsp;

- Unlike logistic, h_(θ)(x) doesn't give us a probability, but instead we get a direct prediction of 1 or 0

- - So if θ*^(T)* x is equal to or greater than 0 --\> h_(θ)(x) = 1

  - Else --\> h_(θ)(x) = 0

### ***Large margin intuition***

- Sometimes people refer to SVM as **large margin classifiers**

  - We'll consider what that means and what an SVM hypothesis looks like

  - The SVM cost function is as above, and we've drawn out the cost terms below  
    <img src="/assets/Machine_LearningNote.assets/image-20230519184544995.png" alt="image-20230519184544995" style="zoom:50%;" />

  - Left is cost₁ and right is cost₀

  - What does it take to make terms small

    - If y =1

      - cost₁(z) = 0 only when z \>= 1

    - If y = 0

      - cost₀(z) = 0 only when z \<= -1

  - Interesting property of SVM

    - If you have a positive example, you only really *need* z to be greater or equal to 0 

      - If this is the case then you predict 1

    - SVM wants a bit more than that - doesn't want to \*just\* get it right, but have the value be quite a bit bigger than zero

      - Throws in an extra safety margin factor

- Logistic regression does something similar

- What are the consequences of this?

  - Consider a case where we set C to be huge

    - C = 100,000

    - So considering we're minimizing CA + B

      - If C is huge we're going to pick an A value so that A is equal to zero

      - What is the optimization problem here - how do we make A = 0?

    - Making A = 0

      - If y = 1

        - Then to make our "A" term 0 need to find a value of θ so (θ*^(T)* x) is greater than or equal to 1

      - Similarly, if y = 0

        - Then we want to make "A" = 0 then we need to find a value of θ so (θ*^(T)* x) is equal to or less than -1

    - So - if we think of our optimization problem a way to ensure that this first "A" term is equal to 0, we re-factor our optimization problem into just minimizing the "B" (regularization) term, because 

      - When A = 0 --\> A\*C = 0 

    - So we're minimizing B, under the constraints shown below  
      <img src="/assets/Machine_LearningNote.assets/image-20230519184604818.png" alt="image-20230519184604818" style="zoom:67%;" />

  - Turns out when you solve this problem you get interesting decision boundaries  
    <img src="/assets/Machine_LearningNote.assets/image-20230519184622787.png" alt="image-20230519184622787" style="zoom:50%;" />

  - The green and magenta lines are functional decision boundaries which could be chosen by logistic regression

    - But they probably don't generalize too well

  - The black line, by contrast is the the chosen by the SVM because of this safety net imposed by the optimization graph

    - More robust separator

  - Mathematically, that black line has a larger minimum distance (margin) from any of the training examples  
    <img src="/assets/Machine_LearningNote.assets/image-20230519184639911.png" alt="image-20230519184639911" style="zoom:50%;" />

  -  By separating with the largest margin you incorporate robustness into your decision making process

- We looked at this at when C is very large

- - SVM is more sophisticated than the large margin might look

    - If you were just using large margin then SVM would be very sensitive to outliers   
      ![](media/image106.png)

    - You would risk making a ridiculous hugely impact your classification boundary

      - A single example might not represent a good reason to change an algorithm

      - If C is very large then we *do* use this quite naive maximize the margin approach  
        ![](media/image107.png)

      - So we'd change the black to the magenta

    - But if C is reasonably small, or a not too large, then you stick with the black decision boundary

  - What about non-linearly separable data?

    - Then SVM still does the right thing if you use a normal size C

    - So the idea of SVM being a large margin classifier is only really relevant when you have no outliers and you can easily linearly separable data

  - Means we ignore a few outliers 

### ***Large margin classification mathematics (optional)***

**Vector inner products **

- Have two (2D) vectors u and v - what is the inner product (*u^(T)* *v*)?  
  ![](media/image108.png)

  - Plot *u* on graph

    - i.e *u*₁ vs. *u*₂  
      ![](media/image109.png)

  - One property which is good to have is the **norm** of a vector

    - Written as \|\|u\|\|

      - This is the euclidean length of vector u

    - So \|\|u\|\| = SQRT(*u*₁*²* + *u*₂*²*) = real number

      - i.e. length of the arrow above

      - Can show via Pythagoras 

  - For the inner product, take *v* and orthogonally project down onto u

    - First we can plot v on the same axis in the same way (*v*₁ vs *v*₁)

    - Measure the length/magnitude of the projection  
      ![](media/image110.png)

    - So here, the green line is the projection

      - p = length along u to the intersection

      - p is the magnitude of the projection of vector *v* onto vector *u*

  - Possible to show that

    - *u^(T)* *v* = p \* \|\|u\|\|

      - So this is one way to compute the inner product

    - *u^(T)* *v = u*₁*v*₁+ *u*₂*v*₂

    - So therefore

      - **p \* \|\|u\|\| = *u*₁*v*₁+ *u*₂*v*₂**

      - This is an important rule in linear algebra

    - We can reverse this too

      - So we could do 

        - *v^(T)* *u = v*₁*u*₁+ v₂*u*₂

        - Which would obviously give you the same number

  - p can be negative if the angle between them is 90 degrees or more  
    ![](media/image111.png)

    - So here p is negative

- Use the vector inner product theory to try and understand SVMs a little better

**SVM decision boundary**  
![](media/image112.png)

- For the following explanation - two simplification

  - Set θ₀= 0 (i.e. ignore intercept terms)

  - Set n = 2 - (x₁, x₂)

    - i.e. each example has only 2 features

- Given we only have two parameters we can simplify our function to  
  ![](media/image113.png)

- And, can be re-written as   
  ![](media/image114.png)

  - Should give same thing 

- We may notice that  
  ![](media/image115.png)

  - The term in red is the norm of θ

    - If we take θ as a 2x1 vector

    - If we assume θ₀ = 0 its still true

- So, finally, this means our optimization function can be re-defined as   
  ![](media/image116.png)

- So the SVM is minimizing the squared norm

&nbsp;

- Given this, what are the (θ*^(T)* x) parameters doing?

  - Given θ and given example x what is this equal to

    - We can look at this in a comparable manner to how we just looked at u and v

  - Say we have a single positive training example (red cross below)  
    ![](media/image117.png)

  - Although we haven't been thinking about examples as vectors it can be described as such  
    ![](media/image118.png)

  - Now, say we have our parameter vector θ and we plot that on the same axis  
    ![](media/image119.png)

  - The next question is what is the inner product of these two vectors  
    ![](media/image120.png)

    - p, is in fact p^(i), because it's the length of p for example i

      - Given our previous discussion we know   
        (θ*^(T)* x^(i )) = p^(i )\* \|\|θ\|\|  
                   = θ₁x^(i)₁ + θ₂x^(i)₂ 

      - So these are both equally valid ways of computing θ*^(T)* x^(i)

- What does this mean?

  - The constraints we defined earlier 

    - (θ*^(T)* x) \>= 1 if y = 1

    - (θ*^(T)* x) \<= -1 if y = 0

  - Can be replaced/substituted with the constraints

    - p^(i )\* \|\|θ\|\| \>= 1 if y = 1

    - p^(i )\* \|\|θ\|\| \<= -1 if y = 0

  - Writing that into our optimization objective   
    ![](media/image121.png)

&nbsp;

- So, given we've redefined these functions let us now consider the training example below  
  ![](media/image122.png)

  - Given this data, what boundary will the SVM choose? Note that we're still assuming θ₀ = 0, which means the boundary has to pass through the origin (0,0)

    - Green line - small margins  
      ![](media/image123.png)

      - SVM would not chose this line

        - Decision boundary comes very close to examples

        - Lets discuss *why* the SVM would **not** chose this decision boundary

  - Looking at this line 

    - We can show that θ is at 90 degrees to the decision boundary  
      ![](media/image124.png)

      - **θ is always at 90 degrees to the decision boundary** (can show with linear algebra, although we're not going to!)

- So now lets look at what this implies for the optimization objective

  - Look at first example (x¹)  
    ![](media/image125.png)

  - Project a line from x¹ on to to the θ vector (so it hits at 90 degrees)

    - The distance between the intersection and the origin is (**p¹**)

  - Similarly, look at second example (x²)

  - - Project a line from x² into to the θ vector

    - This is the magenta line, which will be **negative **(**p²**)

  - If we overview these two lines below we see a graphical representation of what's going on;  
    ![](media/image126.png)

  - We find that both these p values are going to be pretty small

  - If we look back at our optimization objective 

    - We know we need p¹ \* \|\|θ\|\| to be bigger than or equal to 1 for positive examples

      - If p is small

        - Means that \|\|θ\|\| must be pretty large

    - Similarly, for negative examples we need p² \* \|\|θ\|\| to be smaller than or equal to -1

      - We saw in this example p² is a small negative number

        - So \|\|θ\|\| must be a large number

  - Why is this a problem?

    - The optimization objective is trying to find a set of parameters where the norm of theta is small

      - So this doesn't seem like a good direction for the parameter vector (because as p values get smaller \|\|θ\|\| must get larger to compensate)

        - So we should make p values larger which allows \|\|θ\|\| to become smaller

- So lets chose a different boundary  
  ![](media/image127.png)

  - Now if you look at the projection of the examples to θ we find that p¹ becomes large and \|\|θ\|\| can become small

  - So with some values drawn in  
    ![](media/image128.png)

  - This means that by choosing this second decision boundary we can make \|\|θ\|\| smaller

    - Which is why the SVM choses this hypothesis as better 

    - This is how we generate the large margin effect  
      ![](media/image129.png)

    - The magnitude of this margin is a function of the p values

      - So by maximizing these p values we minimize \|\|θ\|\| 

- Finally, we did this derivation assuming θ₀ = 0,

  - If this is the case we're entertaining only decision boundaries which pass through (0,0)

  - If you allow θ₀ to be other values then this simply means you can have decision boundaries which cross through the x and y values at points other than (0,0)

  - Can show with basically same logic that this works, and even when θ₀ is non-zero when you have optimization objective described above (when C is very large) that the SVM is looking for a large margin separator between the classes

### ***Kernels - 1: Adapting SVM to non-linear classifiers***

-  What are kernels and how do we use them

  - We have a training set

  - We want to find a non-linear boundary  
    ![](media/image130.png)

  - Come up with a complex set of polynomial features to fit the data

    - Have h_(θ)(x) which 

      - Returns 1 if the combined weighted sum of vectors (weighted by the parameter vector) is less than or equal to 0

      - Else return 0

    - Another way of writing this (new notation) is

      - That a hypothesis computes a decision boundary by taking the sum of the parameter vector multiplied by a **new feature vector f**, which simply contains the various high order x terms

      - e.g.

        - h_(θ)(x) = θ₀+ θ₁f₁+ θ₂f₂ + θ₃f₃

        - Where

          - f₁= x₁

          - f₂ = x₁x₂

          - f₃ = ...

          - i.e. not specific values, but each of the terms from your complex polynomial function

    - Is there a better choice of feature f than the high order polynomials?

      - As we saw with computer imaging, high order polynomials become computationally expensive

- New features 

- - Define three features in this example (ignore x₀)

  - Have a graph of x₁ vs. x₂ (don't plot the values, just define the space)

  - Pick three points in that space  
    ![](media/image131.png)

  - These points l¹, l², and l³, were chosen manually and are called **landmarks**

    - Given x, define f1 as the similarity between (x, l¹)

      - = exp(- (\|\| x - l¹ \|\|² ) / 2σ²)  
        = ![](media/image132.png)

      - **\|\| x - l¹ \|\|** is the euclidean distance between the point x and the landmark l¹ squared

        - Disussed more later

      - If we remember our statistics, we know that 

        - σ is the **standard deviation**

        - σ² is commonly called the **variance**

    - Remember, that as discussed  
      ![](media/image133.png)

  - So, f2 is defined as

    - f2 = similarity(x, l¹) = exp(- (\|\| x - l² \|\|² ) / 2σ²)

  - And similarly

    - f3 = similarity(x, l²) = exp(- (\|\| x - l¹ \|\|² ) / 2σ²)

  - This similarity function is called a **kernel**

    - This function is a **Gaussian Kernel**

  - So, instead of writing similarity between x and l we might write

  - - f1 = k(x, l¹)

**Diving deeper into the kernel**

- So lets see what these kernels do and why the functions defined make sense

  - Say x is close to a landmark

    - Then the squared distance will be ~0

      - So  
        ![](media/image134.png)

        - Which is basically e⁻⁰

          -  Which is close to 1

    - Say x is far from a landmark

      - Then the squared distance is big

        - Gives e^(-large number)

          - Which is close to zero

    - Each landmark defines a new features

- If we plot f1 vs the kernel function we get a plot like this

  - Notice that when x = \[3,5\] then f1 = 1

  - As x moves away from \[3,5\] then the feature takes on values close to zero

  - So this measures how close x is to this landmark  
    ![](media/image135.png)

**What does σ do?**

- **σ² **is a parameter of the Gaussian kernel

  - Defines the steepness of the rise around the landmark

- Above example σ² = 1

- Below σ² = 0.5  
  ![](media/image136.png)

  - We see here that as you move away from 3,5 the feature f1 falls to zero much more rapidly

- The inverse can be seen if σ² = 3  
  ![](media/image137.png)

&nbsp;

- Given this definition, what kinds of hypotheses can we learn?

  - With training examples x we predict "1" when

  - θ₀+ θ₁f₁+ θ₂f₂ + θ₃f₃ \>= 0

    - For our example, lets say we've already run an algorithm and got the

      - θ₀ = -0.5

      - θ₁ = 1

      - θ₂ = 1

      - θ₃ = 0

    - Given our placement of three examples, what happens if we evaluate an example at the **magenta dot** below?  
      ![](media/image138.png)

    - Looking at our formula, we know f1 will be close to 1, but f2 and f3 will be close to 0

      - So if we look at the formula we have

        - θ₀+ θ₁f₁+ θ₂f₂ + θ₃f₃ \>= 0

        - -0.5 + 1 + 0 + 0 = 0.5

          - 0.5 is greater than 1

    - If we had **another point** far away from all three  
      ![](media/image139.png)

      - This equates to -0.5

        - So we predict 0

  - Considering our parameter, for points near l¹ and l² you predict 1, but for points near l³ you predict 0

  - Which means we create a non-linear decision boundary that goes a lil' something like this;  
    ![](media/image140.png)

    - Inside we predict y = 1

    - Outside we predict y = 0

- So this show how we can create a non-linear boundary with landmarks and the kernel function in the support vector machine

- - But

  - - How do we get/chose the landmarks

    - What other kernels can we use (other than the Gaussian kernel)

### ***Kernels II***

- Filling in missing detail and practical implications regarding kernels

- Spoke about picking landmarks manually, defining the kernel, and building a hypothesis function

  - Where do we get the landmarks from?

  - For complex problems we probably want lots of them

**Choosing the landmarks**

- Take the training data

- For each example place a landmark at exactly the same location

- So end up with m landmarks

  - One landmark per location per training example

  - Means our features measure how close to a training set example something is

- Given a new example, compute all the f values

  - Gives you a feature vector f (f₀ to f_(m))

    - f₀ = 1 always

- A more detailed look at generating the f vector

  - If we had a training example - features we compute would be using (x^(i), y^(i))

    - So we just cycle through each landmark, calculating how close to that landmark actually x^(i) is

      - f₁^(i), = k(x^(i), l¹)

      - f₂^(i), = k(x^(i), l²)

      - ...

      - f_(m)^(i), = k(x^(i), l^(m))

    - Somewhere in the list we compare x to itself... (i.e. when we're at f_(i)^(i))

      - So because we're using the Gaussian Kernel this evalues to 1

    - Take these m features (f₁, f₂ ... f_(m)) group them into an \[m +1 x 1\] dimensional vector called f

      - f^(i) is the f feature vector for the ith example

      - And add a 0th term = 1

- Given these kernels, how do we use a support vector machine

**SVM hypothesis prediction with kernels**

- Predict y = 1 if (θ*^(T)* f) \>= 0

  - Because θ = \[m+1 x 1\] 

  - And f = \[m +1 x 1\] 

- So, this is how you make a prediction assuming you already have θ

  - How do you get θ?

**SVM training with kernels**

- Use the SVM learning algorithm  
  ![](media/image141.png)

  - Now, we minimize using f as the feature vector instead of x

  - By solving this minimization problem you get the parameters for your SVM

- In this setup, m = n

  - Because number of features is the number of training data examples we have 

- One final mathematic detail (not crucial to understand)

  - If we ignore θ₀ then the following is true  
    ![](media/image142.png)

  - What many implementations do is   
    ![](media/image143.png)

    - Where the matrix M depends on the kernel you use

    - Gives a slightly different minimization - means we determine a rescaled version of θ

    - Allows more efficient computation, and scale to much bigger training sets

    - If you have a training set with 10 000 values, means you get 10 000 features

      - Solving for all these parameters can become expensive

      - So by adding this in we avoid a for loop and use a matrix multiplication algorithm instead 

- You can apply kernels to other algorithms

  - But they tend to be very computationally expensive

  - But the SVM is far more efficient - so more practical

- Lots of good off the shelf software to minimize this function

&nbsp;

- **SVM parameters (C)**

  - Bias and variance trade off

  - Must chose C

    - C plays a role similar to 1/LAMBDA (where LAMBDA is the regularization parameter)

  - Large C gives a hypothesis of **low bias high variance** --\> overfitting

  - Small C gives a hypothesis of **high bias low variance** --\> underfitting

- **SVM parameters (σ²)**

  - Parameter for calculating f values

    - Large σ² - f features vary more smoothly - higher bias, lower variance

    - Small σ² - f features vary abruptly - low bias, high variance

### ***SVM - implementation and use***

- So far spoken about SVM in a very abstract manner

- What do you need to do this

  - Use SVM software packages (e.g. liblinear, libsvm) to solve parameters θ

  - Need to specify

    - Choice of parameter C

    - Choice of kernel

**Choosing a kernel**

- We've looked at the **Gaussian kernel**

  - Need to define σ (σ²)

    - Discussed σ²

  - When would you chose a Gaussian?

    - If n is small and/or m is large

      - e.g. 2D training set that's large

  - If you're using a Gaussian kernel then you may need to implement the kernel function

    - e.g. a function  
      fi = kernel(x1,x2)

      - Returns a real number

    - Some SVM packages will expect you to define kernel

    - Although, some SVM implementations include the Gaussian and a few others

      - Gaussian is probably most popular kernel

  - NB - make sure you perform **feature scaling** before using a Gaussian kernel 

    - If you don't features with a large value will dominate the f value

- Could use no kernel - **linear kernel**

  - Predict y = 1 if (θ*^(T)* x) \>= 0

    - So no f vector

    - Get a standard linear classifier

  - Why do this?

    - If n is large and m is small then

      - Lots of features, few examples

      - Not enough data - risk overfitting in a high dimensional feature-space

- Other choice of kernel

  - Linear and Gaussian are most common

  - Not all similarity functions you develop are valid kernels

    - Must satisfy **Merecer's Theorem**

    - SVM use numerical optimization tricks

      - Mean certain optimizations can be made, but they must follow the theorem

  - **Polynomial Kernel**

    - We measure the similarity of x and l by doing one of

      - (x*^(T)* l)² 

      - (x*^(T)* l)³ 

      - (x*^(T)* l+1)³ 

    - General form is

      - (x*^(T)* l+Con)^(D) 

    - If they're similar then the inner product tends to be large

    - Not used that often

    - Two parameters

    - - Degree of polynomial (D)

      - Number you add to l (Con)

    - Usually performs worse than the Gaussian kernel

    - Used when x and l are both non-negative

  - **String kernel**

    - Used if input is text strings

    - Use for text classification

  - **Chi-squared kernel**

  - **Histogram intersection kernel**

**Multi-class classification for SVM**

- Many packages have built in multi-class classification packages

- Otherwise use one-vs all method

- Not a big issue

**Logistic regression vs. SVM**

- When should you use SVM and when is logistic regression more applicable

- If n (features) is large vs. m (training set)

- - e.g. text classification problem

  - - Feature vector dimension is 10 000

    - Training set is 10 - 1000

    - Then use logistic regression or SVM with a linear kernel

- If n is small and m is intermediate

  - n = 1 - 1000

  - m = 10 - 10 000

  - Gaussian kernel is good

- If n is small and m is large

  - n = 1 - 1000

  - m = 50 000+

    - SVM will be slow to run with Gaussian kernel

  - In that case

    - Manually create or add more features

    - Use logistic regression of SVM with a linear kernel

- Logistic regression and SVM with a linear kernel are pretty similar

  - Do similar things

  - Get similar performance

- A lot of SVM's power is using diferent kernels to learn complex non-linear functions

- For all these regimes a well designed NN should work

  - But, for some of these problems a NN might be slower - SVM well implemented would be faster

- SVM has a convex optimization problem - so you get a global minimum

- It's not always clear how to chose an algorithm

  - Often more important to get enough data

  - Designing new features

  - Debugging the algorithm

- SVM is widely perceived a very powerful learning algorithm

### *θ is always at 90 degrees to the decision boundary *

<https://stackoverflow.com/questions/10177330/why-is-weight-vector-orthogonal-to-decision-plane-in-neural-networks>

- The weights are simply the coefficients that define a separating plane. For the moment, forget about neurons and just consider the geometric definition of a plane in N dimensions:

- w1\*x1 + w2\*x2 + ... + wN\*xN - w0 = 0

- You can also think of this as being a dot product:

- w\*x - w0 = 0

- where w and x are both length-N vectors. This equation holds for all points on the plane. Recall that we can multiply the above equation by a constant and it still holds so we can define the constants such that the vector w has unit length. Now, take out a piece of paper and draw your x-y axes (x1 and x2 in the above equations). Next, draw a line (a plane in 2D) somewhere near the origin. w0 is simply the perpendicular distance from the origin to the plane and w is the unit vector that points from the origin along that perpendicular. If you now draw a vector from the origin to any point on the plane, the dot product of that vector with the unit vector w will always be equal to w0 so the equation above holds, right? This is simply the geometric definition of a plane: a unit vector defining the perpendicular to the plane (w) and the distance (w0) from the origin to the plane.

- Now our neuron is simply representing the same plane as described above but we just describe the variables a little differently. We'll call the components of x our "inputs", the components of w our "weights", and we'll call the distance w0 a bias. That's all there is to it.

- Getting a little beyond your actual question, we don't really care about points on the plane. We really want to know which side of the plane a point falls on. While w\*x - w0 is exactly zero on the plane, it will have positive values for points on one side of the plane and negative values for points on the other side. That's where the neuron's activation function comes in but that's beyond your actual question.







