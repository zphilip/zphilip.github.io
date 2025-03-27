---
layout: post
title: Understanding SVD and PCA in Mathematics
date: 2024-03-25
categories: LEARNING
tags: AI SVD PCA
---
# Mathematics in ML/SVD



## 1. Mathematics in ML/SVD (Singular value decomposition) 奇异值分解

https://www.youtube.com/watch?v=yLdOS6xyM_Q&list=PLLssT5z_DsK9JDLcT8T62VtzwyW9LNepV&index=46

[Material store in : Learning Material\AIML Course\Mining of Massive Datasets] 

奇异值分解(1)——奇异值分解的证明
识，准备写三简唡客，分别为奇异值分解的证明，奇异值分解的性质，奇异值分解的应用，这一筒是有关奇异值分解的证明，话不会说，进入正题。

首先来者定义

### 奇异值分解(Singular Value Decomposition)

$$
A=U\left[\begin{array}{llll}
\sigma_1 & & & \\
& \ddots & \\
& & \sigma_r & \\
& & & 0
\end{array}\right] V^T=U \sum V^T
$$

其中 $\mathrm{r}=\operatorname{rank} \mathrm{A}, \sum \in R^{m \times n}$. 习蕾上，设 $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_{\mathrm{r}}>0$ ，称 $\sigma_1, \ldots, \sigma_r$ 为奇吕值(singular value). 称 $U$ 和 $V$ 的前 $r$ 列向量为奇异向量(singular vector), 这个分解为奇昔值分解，简称SVD。

这里要注息的一点是 $\sum$ 右下角的 0 是 0 拒阵的思思，下面用两种方法给出这个走理的证明。

奇异值分解的证明一
在证明之前需要两个引理

引理1
对任意矩阵 $A \in R^{m \times n} ， A^T A \in R^{n \times n}$ 为半正定对称追阵，从而 $A^T A$ 的持征值非负

证明 :
$\left(A^T A\right)^T=A^T A$ ，所以 $A^T A$ 为则称迎阵，接着证明 $A^T A$ 为半正走对称迎阵。任取 $x \in R^n$ ，考虑二次型
$$
\begin{gathered}
x^T A^T A x=(A x)^T(A x)=\sum_{i=1}^n x_i^{\prime 2} \geq 0 \\
\text { 其中 } A x=\left(x_1^{\prime}, \ldots, x_n^{\prime}\right)
\end{gathered}
$$

所以 $A^T A$ 为半正定矩阵, 所以 $A^T A$ 的特征值非负。

引理2
$$
\text { 对任意矩阵 } A \in R^{n \times n}, \operatorname{rank}(A)=\operatorname{rank}\left(A^T A\right)
$$

如果我们可以证明 $A x=0, A^T A x=0$ 同解，梛么由解空间的维数公式可得
$$
n-\operatorname{rank}(A)=n-\operatorname{rank}\left(A^T A\right)
$$

从而可以证明该诘论。接下来证明 $A x=0, A^T A x=0$ 同解。
证明 :
如果 $A x=0$, 那 $\angle A^T A x=A^T 0=0$, 所以 $A x=0$ 的解一走是 $A^T A x=0$ 的解,
反之, 如果 $A^T A x=0$, 两边左晒 $x^T$ 可得 $x^T A^T A x=x^T 0=0$, 所以 $(A x)^T A x=0$, 从而 $A x=0$, 所以 $A^T A x=0$ 的船一定是 $A x=0$ 的解.

结合以上两点 $A x=0, A^T A x=0$ 同解，所以
$$
\begin{aligned}
n-\operatorname{rank}(A) & =n-\operatorname{rank}\left(A^T A\right) \\
\operatorname{rank}(A) & =\operatorname{rank}\left(A^T A\right)
\end{aligned}
$$

以上两个引理会在证明中使用到, 下面开始正讯证明.

证明
由于结论中有正交矩阵，而对称迎阵可以正交相似于对角阵，所以我们自然联相到抣造对称起阵。命 $A_1=A^T A \in R^{n \times n}$ ，设 $\operatorname{rank} A=r$ ，由引|理2可得 $\operatorname{rank} A_1=\operatorname{rank}\left(A^T A\right)=\operatorname{rank} A=r$ ，而可对角化的拒阵的泆等于矩阵的非零待征值数里，对称矩阵显然可逆，所以 $A_1$ 有 $r$ 个非零特佂值，由引理 1 我们知造 $A_1$ 为半正定矩阵，其特征值均为非负数，所以 $A_1$ 有 $r$ 个正特征值，设 $A_1$ 的持征伹为
$$
\sigma_1^2, \sigma_2^2, \ldots, \sigma_\tau^2 \text { ，其中 } \sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_\tau>0 ，
$$

由对称矩阵的性质可知, $A_1$ 可以正交相似于对角龪阵
$$
A_2=\operatorname{diag}\{\sigma_1^2, \sigma_2^2, \ldots, \sigma_r^2, \underbrace{0, \ldots, 0}_{n-m 0}\} \in R^{n \times n}
$$

如果记
$$
S=\left[\begin{array}{lll}
\sigma_1 & & \\
& \ddots & \\
& & \sigma_r
\end{array}\right] \in R^{\times r}
$$

那么
$$
A_2=\left[\begin{array}{cc}
S^2 & 0 \\
0 & 0
\end{array}\right]
$$

我们 $A_2$ 设对应的正交矩阵为 $V$ ，从而
$$
\begin{aligned}
& A_1=V A_2 V^T \\
& V^T A_1 V=A_2
\end{aligned}
$$

对 $V$ 进行分块，记其前 $r$ 列构成的矩阵为 $V_1$ ，后 $n-r$ 列沟成的矩阵为 $V_2 ， V=\left(V_1, V_2\right)$ ，带入计算可得
$$
\begin{gathered}
\left(V_1, V_2\right)^T A_1\left(V_1, V_2\right)=\left[\begin{array}{cc}
S^2 & 0 \\
0 & 0
\end{array}\right] \\
V_1^T A_1 V_1=S^2, V_2^T A_1 V_1=0, V_1^T A_1 V_2=0, V_2^T A_1 V_2=0
\end{gathered}
$$

将 $A_1=A^T A$ 带入第 $1 ， 4$ 个等式，分剈对其处理，先处理第 1 个等式
$$
\begin{gathered}
V_1^T A^T A V_1=S^2 \\
\text { 先左皐 } S^{-1} \text {, 再右承 } S^{-1} \\
S^{-1} V_1^T A^T A V_1 S^{-1}=I_r
\end{gathered}
$$

因为 $S$ 为对角知阵，所以 $S^{-1}$ 地为对角矩阵，从而 $\left(S^{-1}\right)^T=S^{-1}$ ，因此
$$
\begin{gathered}
S^{-1} V_1^T A^T A V_1 S^{-1}=\left(S^{-1}\right)^T V_1^T A^T A V_1 S^{-1}=\left(A V_1 S^{-1}\right)^T A V_1 S^{-1}=I_r \\
\text { 记 } U_1=A V_1 S^{-1} \in R^{m \times r} \text {, 那么 } U_1^T U_1=0
\end{gathered}
$$

所以 $U_1$ 是 $R^m$ 中正交向量祖成的茂轫，我们可以将其补充成一个 $R^{m \times m}$​ 的正交矩阵，下面严格证明一下，
首先由引|理2可知
$$
\operatorname{rank}\left(U_1\right)=\operatorname{rank}\left(U_1^T U_1\right)=\operatorname{rank}\left(I_r\right)=r
$$

接下来我们考虚 $U_1^T x=0$ 的解空间，由于 $\operatorname{rank}\left(U_1\right)=r, U_1^T \in R^{r \times m}$ ，所以 $U_1^T x=0$ 有 $m-r$ 个线性无关的解 $u_1, \ldots, u_{m-r}$ ，对这些解利用站密特正交化，并对其本位化，可以得到 $\$ U_{-} 2=\left(u_{-} 1^{\wedge}, \ldots, u_{-} / m-\right.$
$$
U_1^T U_2=0, U_2^T U_2=I_{m-r}
$$

结合 $U_1^T U_1=I_r$ ，那 $\angle U=\left(U_1, U_2\right)$ 为正交矩阵。
再来粕处理第 4 个等式，将 $A_1=A^T A$ 带入可得
$$
\begin{gathered}
V_2^T A^T A V_2=\left(A V_2\right)^T A V_2=0 \\
A V_2=0
\end{gathered}
$$

这个惟出关系在引遇 2 中有说明
现在饿们找到了 $U, V$, 就可以来计算 $U^T A V$ ，捋其写成分块炮阵的形式。
$$
\begin{aligned}
U^T A V & =\left(U_1, U_2\right)^T A\left(V_1, V_2\right) \\
& =\left[\begin{array}{ll}
U_1^T A V_1 & U_1^T A V_2 \\
U_2^T A V_1 & U_2^T A V_2
\end{array}\right]
\end{aligned}
$$

分㫜分析这四个棓分，先香 $U_1^T A V_1$ ，将 $U_1=A V_1 S^{-1} ， V_1^T A_1 V_1=S^2, A_1=A^T A$ 带入
$$
\begin{aligned}
U_1^T A V_1 & =\left(A V_1 S^{-1}\right)^T A V_1 \\
& =S^{-1} V_1^T A^T A V_1 \\
& =S^{-1} V_1^T A_1 V_1 \\
& =S^{-1} S^2 \\
& =S
\end{aligned}
$$

再曋 $U_1^T A V_2$ ，将 $A V_2=0$ 带入可得
$$
U_1^T A V_2=0
$$

演着看 $U_2^T A V_1$ ，考虑 $U_1=A V_1 S^{-1}, U_2^T U_1=0$
$$
U_2^T U_1=U_2^T A V_1 S^{-1}=0
$$

因为 $S$ 可逆，所以
$$
U_2^T A V_1=0
$$

最后考壉 $U_2^T A V_2$ ，将 $A V_2=0$ 带入可得
$$
U_2^T A V_2=0
$$

综上所述
$$
\begin{aligned}
& U^T A V=\left(U_1, U_2\right)^T A\left(V_1, V_2\right) \\
& =\left[\begin{array}{ll}
S & 0 \\
0 & 0
\end{array}\right] \\
& =\left[\begin{array}{llll}
\sigma_1 & & & \\
& \ddots & & \\
& & \sigma_r & \\
& & & 0
\end{array}\right] \\
& \stackrel{\text { i2축 }}{=} \sum \\
&
\end{aligned}
$$

注思 $U, V$ 为正交避车，所以上或也等价于
$$
A=U\left[\begin{array}{cccc}
\sigma_1 & & & \\
& \ddots & & \\
& & \sigma_\tau & \\
& & & 0
\end{array}\right] V^T=U \sum V^T
$$
这就是奇异值分解的形式，证毕。

### 奇异值分解的证明二

前一部分和证明一相同，这里简单釷迷下
$$
\begin{gathered}
A \in R^{n \times n}, \operatorname{rank} A=r, \\
A_1=A^T A \in R^{n \times n}, \operatorname{rank} A_1=r, \\
A_1 \text { 正交相似于对角阵 } A_2, A_2=\operatorname{diag}\{\sigma_1^2, \sigma_2^2, \ldots, \sigma_r^2, \underbrace{0, \ldots, 0}_{n-m 0}\} \in R^{n \times n} \\
\text { 其中 } \sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_r>0
\end{gathered}
$$

设 $\left(v_1, \ldots, v_n\right), v_i \in R^{n_3} A_1=A^T A$ 的单位待证正交向量，那么
$$
A^T A v_i= \begin{cases}\sigma_i^2 v_i, & 1 \leq i \leq r \\ 0, & r+1 \leq i \leq k\end{cases}
$$

注意对于 $1 \leq i \leq r$,
$$
\begin{gathered}
v_i^T A^T A v_i=\sigma_i^2 v_i^T v_i=\sigma_i^2 \\
\left\|A v_i\right\|^2=\sigma_i^2 \\
\left\|A v_i\right\|=\sigma_i
\end{gathered}
$$

记 $u_i=\frac{A v_i}{\sigma_i}, u_i \in R^m, 1 \leq i \leq r$ ，我们来证明 $\left(u_1, \ldots, u_r\right)$ 为一组率位正交向量，注童 $A^T A v_i=\sigma_i^2 v_i$
$$
\begin{gathered}
u_i^T u_i=\frac{v_i^T A^T A v_i}{\sigma_i^2}=\frac{\sigma_i^2}{\sigma_i^2}=1 \\
u_i^T u_j=\frac{v_i^T A^T A v_j}{\sigma_i \sigma_j}=\frac{\sigma_i^2 v_i^T v_j}{\sigma_i \sigma_j} \\
\text { 因为 } v_i, v_j(i \neq j) \text { 正交, 所以 } \\
u_i^T u_j=\frac{\sigma_i^2 v_i^T v_j}{\sigma_i \sigma_j}=0
\end{gathered}
$$

从而 $\left(u_1, \ldots, u_r\right)$ 为单位正交向量, 现在将 $u_i=\frac{A v_i}{\sigma_i}$ 变形为 $A v_i=\sigma_i u_i, 1 \leq i \leq r$, 棈这 $r$ 个式子写成矩倩的形式
$$
A\left(v_1, \ldots, v_r\right)=\left(u_1, \ldots, u_r\right) \operatorname{diag}\left\{\sigma_1^2, \sigma_2^2, \ldots, \sigma_r^2\right\}
$$
基。

由于 $v_i \in R^m$ ，舫以由基扩张定理，可以将 $\left(u_1, \ldots, u_r\right)$ 扩张为 $\left(u_1, \ldots, u_r, u_{r+1}, \ldots u_m\right)$ ，使得 $\left(u_1, \ldots, u_r, u_{r+1}, \ldots u_m\right)$ 为 $R^m$ 的一组单位正交基，现在将 $\left(v_1, \ldots, v_n\right),\left(u_1, \ldots, u_m\right)$ 带入上式可得
$$
A\left(v_1, \ldots, v_n\right)=\left(u_1, \ldots, u_n\right)\left[\begin{array}{llll}
\sigma_1 & & & \\
& \ddots & & \\
& & \sigma_r & \\
& & & 0
\end{array}\right]
$$

记 $V=\left(v_1, \ldots, v_n\right), U=\left(u_1, \ldots, u_n\right)$ ，注息 $U, V$ 均为正交矩阵，那么上式可以表达为
$$
\begin{gathered}
A V=U\left[\begin{array}{llll}
\sigma_1 & & & \\
& \ddots & & \\
& & \sigma_r & \\
& & & 0
\end{array}\right] \\
A=U\left[\begin{array}{cccc}
\sigma_1 & & & \\
& \ddots & & \\
& & \sigma_r & \\
& & & 0
\end{array}\right] V^T=U \sum V^T
\end{gathered}
$$

定理得证.

总结块迎阵的形式找到 $U$ 的一部分 $U_1$ ，将其补充为 $U$ ，层后利用之前的一些得到的一些性质证明了奇吕值分解；证明二定证明一的筒化版，主要利用了 $\frac{A e_i}{\sigma_i}$ 的正交性，本愿上和证明一定相化的.

这部分的内容是比较理论的，但个人认为还是艇有必要的，后续还会介绍奇异值分解的性同以及应用。

婧考资科:

起阵素咞值分解
絨性代数公开课

### Rank of a matrix

If we consider a square matrix, the columns (rows) are linearly independent only if the matrix is nonsingular. In other words, the rank of any nonsingular matrix of order m is m. The rank of a matrix A is denoted by ρ(A).

![image-20240311131651158](/assets/Mathematics%20in%20MLSVD%20(Singular%20value%20decomposition)%20and%20PCA.assets/image-20240311131651158.png)

### SVD (Singular value decomposition)

![image-20240311133742547](/assets/Mathematics%20in%20MLSVD%20(Singular%20value%20decomposition)%20and%20PCA.assets/image-20240311133742547.png)

### SVD Concept similarity

![image-20240311140603163](/assets/Mathematics%20in%20MLSVD%20(Singular%20value%20decomposition)%20and%20PCA.assets/image-20240311140603163.png)

 ### SVD How Dimensionality reduction by SVD and Forbenius Distance

![image-20240311142423368](/assets/Mathematics%20in%20MLSVD%20(Singular%20value%20decomposition)%20and%20PCA.assets/image-20240311142423368.png)

 

![image-20240311144511549](/assets/Mathematics%20in%20MLSVD%20(Singular%20value%20decomposition)%20and%20PCA.assets/image-20240311144511549.png)

### SVD in [ [Data-Driven Science and Engineering\]](https://www.youtube.com/playlist?list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv), Notice something before the Dim-Reduc

![image-20240314135506521](/assets/Mathematics%20in%20MLSVD%20(Singular%20value%20decomposition)%20and%20PCA.assets/image-20240314135506521.png)

### SVD How to compute

![image-20240311153628512](/assets/Mathematics%20in%20MLSVD%20(Singular%20value%20decomposition)%20and%20PCA.assets/image-20240311153628512.png)

### SVD Decompose example

![image-20240311152715141](/assets/Mathematics%20in%20MLSVD%20(Singular%20value%20decomposition)%20and%20PCA.assets/image-20240311152715141.png)

 ![image-20240314142051486](/assets/Mathematics%20in%20MLSVD%20(Singular%20value%20decomposition)%20and%20PCA.assets/image-20240314142051486.png)

### SVD Snapshot method (not recommanded)

If the compute can load the two vector of the data , then it is possible to compute the $X^TX$  which has mxm size which is smaller than nxn. 

Using Eigen decomposition to get the right sigluar V matrix and the eigen value $\Sigma$ matrix.  then it lead to compute the left matrix. 

![image-20240314160813959](/assets/Mathematics%20in%20MLSVD%20(Singular%20value%20decomposition)%20and%20PCA.assets/image-20240314160813959.png)

### SVD compress example

D:\Mylearning\zphilip48.ai\Data-DrivenScienceandEngineering\databook_python\CH01\CH01_SEC02.ipynb	"Python SVD compress example "

```python
# only take part of the SVD --- different rank of the SVD matrix
U, S, VT = np.linalg.svd(X,full_matrices=False)
S = np.diag(S)

j = 0
for r in (5, 20, 100):
    # Construct approximate image
    Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    plt.figure(j+1)
    j += 1
    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r = ' + str(r))
    plt.show()
```



## 2. CUR Decomposition

![image-20240314165829854](/assets/Mathematics%20in%20MLSVD%20(Singular%20value%20decomposition)%20and%20PCA.assets/image-20240314165829854.png)

![image-20240318140547974](/assets/Mathematics%20in%20MLSVD%20(Singular%20value%20decomposition)%20and%20PCA.assets/image-20240318140547974.png)

## 3. Unitray Transformations/Least Square, Regression and pseudoinverse , the SVD

![image-20240319144249742](/assets/Mathematics%20in%20MLSVD%20(Singular%20value%20decomposition)%20and%20PCA.assets/image-20240319144249742.png)

## 4. PCA (Principal Component Analysis)

### How principal component analysis works

Two major components are calculated in PCA: the first principal component (PC1) and the second principal component (PC2).

- First principal component

The first principal component (PC1) is the direction in space along which the data points have the highest or most variance. It is the line that best represents the shape of the projected points. The larger the variability captured in the first component, the larger the information retained from the original dataset. No other principal component can have a higher variability.

- Second principal component

We calculate the second principal component (PC2) in the same way as PC1. PC2 accounts for the next highest variance in the dataset and must be uncorrelated with PC1. That is, PC2 must be orthogonal, i.e. perpendicular, to PC1. This relationship can also be expressed as the correlation between PC1 and PC2 equals zero. 

A scatterplot is typically used to show the relationship between PC1 and PC2 when PCA is applied to a dataset. PC1 and PC2 axis will be perpendicular to each other.

![Lines on a graph with an orthogonal relationship](/assets/Mathematics%20in%20MLSVD%20(Singular%20value%20decomposition)%20and%20PCA.assets/image.png)

### SVD and PCA relationship

- SVD Explaination

Let's try to understand using a data matrix $X$ of dimension $n \times d$, where $d \gg n$ and $\operatorname{rank}(X)=n$​.

  1. $$\underset{n \times d}{X}=\underset{n \times n } {U}\underset { n \times n } { \Sigma } \underset{ n \times d}{V^T}$$ (reduced SVD),   with $X^T X=V \Sigma^T U^T U \Sigma V^T=V \Sigma^2 V^T$ (since unitary / orthonormal $U, V$ and diagonal $\Sigma$ )	
  2. perform singular value decomposition of $\mathbf{X}$, we obtain a decomposition

$$
\mathbf{X}=\mathbf{U S V}^{\top},
$$
where $\mathbf{U}$ is a unitary matrix (with columns called left singular vectors), $\mathbf{S}$ is the diagonal matrix of singular values $s_i$ and $\mathbf{V}$ columns are called right singular vectors. From here one can easily see that
$$
\mathbf{C}=\mathbf{V S U}^{\top} \mathbf{U} \mathbf{S V}^{\top} /(n-1)=\mathbf{V} \frac{\mathbf{S}^2}{n-1} \mathbf{V}^{\top}
$$
meaning that right singular vectors $\mathbf{V}$ are principal directions (eigenvectors) and that singular values are related to the eigenvalues of covariance matrix via $\lambda_i=s_i^2 /(n-1)$. ==Principal components are given by $\mathbf{X V}=\mathbf{U S V}^{\top} \mathbf{V}=\mathbf{U S}$.==

- PCA Explaination:  

1. and the covariance matrix (assuming $X$ is already mean centered, i.e., the columns of $X$ have 0 means)
   $C=E\left[X^T X\right]-E[X]^T E[X]=\frac{X^T X}{n-1}-0=V \frac{\Sigma^2}{n-1} V^T=\bar{V} \Lambda \bar{V}^T$ (PCA, by spectral decomposition)
   $\Longrightarrow \Lambda=\frac{\Sigma^2}{n-1}$ and $V=\bar{V}$​​ upto sign flip.
2. Let the real values data matrix $\mathbf{X}$ be of $n \times p$ size, where $n$ is the number of samples and $p$ is the number of variables. Let us assume that it is centered, i.e. column means have been subtracted and are now equal to zero.

Then the $p \times p$ covariance matrix $\mathbf{C}$ is given by $\mathbf{C}=\mathbf{X}^{\top} \mathbf{X} /(n-1)$. It is a symmetric matrix and so it can be diagonalized:
$$
\mathbf{C}=\mathbf{V} \mathbf{L} \mathbf{V}^{\top},
$$
where $\mathbf{V}$ is a matrix of eigenvectors (each column is an eigenvector) and $\mathbf{L}$ is a diagonal matrix with eigenvalues $\lambda_i$ in the decreasing order on the diagonal. The eigenvectors are called principal axes or principal directions of the data. Projections of the data on the principal axes are called principal components, also known as PC scores; these can be seen as new, transformed, variables. ==The $j$-th principal component is given by $j$-th column of $\mathbf{X V}$. The coordinates of the $i$-th data point in the new PC space are given by the $i$-th row of $\mathbf{X V}$.==

Let's validate the above with eigenfaces (i.e., the principal components / eigenvectors of the covariance matrix for such a face dataset) using the following face dataset:

```
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces

X = fetch_olivetti_faces().data
X.shape # 400 face images of size 64×64 flattened
# (400,4096)
n = len(X)

# z-score normalize
X = X - np.mean(X, axis=0) # mean-centering
# X = X / np.std(X, axis=0)  # scaling to have sd=1

# choose first k eigenvalues / eigenvectors for dimensionality reduction
k = 25

# SVD
U, Σ, Vt = np.linalg.svd(X, full_matrices=False) 

# PCA
pca = PCA(k).fit(X)
PC = pca.components_.T
#Vt.shape, PC.shape
```

![enter image description here](/assets/Mathematics%20in%20MLSVD%20(Singular%20value%20decomposition)%20and%20PCA.assets/WUWE6.png)

Here the differences in the eigenvectors are due to sign ambiguity (refer to https://www.osti.gov/servlets/purl/920802)

3. Let me start with PCA. Suppose that you have $n$ data points comprised of $d$ numbers (or dimensions) each. If you center this data (subtract the mean data point $\mu$ from each data vector $x_i$ ) you can stack the data to make a matrix
   $$
   X=\left(\frac{\frac{x_1^T-\mu^T}{x_2^T-\mu^T}}{\frac{\vdots}{x_n^T-\mu^T}}\right) .
   $$

   The covariance matrix
   $$
   S=\frac{1}{n-1} \sum_{i=1}^n\left(x_i-\mu\right)\left(x_i-\mu\right)^T=\frac{1}{n-1} X^T X
   $$
   measures to which degree the different coordinates in which your data is given vary together. So, it's maybe not surprising that PCA -- which is designed to capture the variation of your data -- can be given in terms of the covariance matrix. In particular, the eigenvalue decomposition of $S$ turns out to be
   $$
   S=V \Lambda V^T=\sum_{i=1}^r \lambda_i v_i v_i^T,
   $$
   where $v_i$ is the $i$-th Principal Component, or PC, and $\lambda_i$ is the $i$-th eigenvalue of $S$ and is also equal to the variance of the data along the $i$​-th PC. This decomposition comes from a general theorem in linear algebra, and some work does have to be done to motivate the relation to PCA.

   ![PCA of a Randomly Generated Gaussian Dataset](/assets/Mathematics%20in%20MLSVD%20(Singular%20value%20decomposition)%20and%20PCA.assets/basic-pca.png)

   SVD is a general way to understand a matrix in terms of its column-space and row-space. (It's a way to rewrite any matrix in terms of other matrices with an intuitive relation to the row and column space.) For example, for the matrix $A=\left(\begin{array}{ll}1 & 2 \\ 0 & 1\end{array}\right)$ we can find directions $u_i$ and $v_i$ in the domain and range so that

   ![SVD for a 2x2 example](/assets/Mathematics%20in%20MLSVD%20(Singular%20value%20decomposition)%20and%20PCA.assets/basic-svd.png)



在参考[1]中从奇异值分解（SVD）和主成分分析（PCA）的关系的角度来描述奇异值（以及SVD得到的另外两个矩阵）的物理意义。


 设X是一个数据矩阵（在此不把它理解成变换），每一列表示一个数据点，每一行表示一维特征。

 对X做主成分分析（PCA）的时候，需要求出各维特征的协方差，这个协方差矩阵是$XX^T$。

其实需要先把数据平移使得数据的均值为0，不过在此忽略这些细节）
 PCA做的事情，是对这个协方差矩阵做对角化：$XX^T= P_AP_T$ 

可以这样理解上式右边各项的物理意义：用一个均值为0的多维正态分布来拟合数据，则正交矩阵P的每一列是正态分布的概率密度函数的等高线（椭圆）的各个轴的方向，而对角矩阵A的对角线元素是数据在这些方向上的方差，它们的平方根跟椭圆各个轴的长度成正比。


现在来看数据矩阵X的奇异值分解：$X = USV^T$，其中U、V各列是单位正交的，S是对角阵，对角元非零。

由此式可以得到$XX^T = USV^TVSU^T = US^2U^T$

也就是说，SVD中的矩阵U相当于PCA中的矩阵P，不过仅保留了A的非零特征值对应的那些特征向量，而S=A1/2（也只保留了非零特征值）。


所以，SVD中的U代表了X中数据形成的正态分布的轴的方向（一组单位正交基），S代表了这些轴的长度（分布的标准差）。

那么V呢？可以把US放在一起看成一个由伸缩和旋转组成的坐标变换（不包括平移），数据矩阵X是由数据矩阵$V^T$经此变换得来的，而$V^T$的各列（V的各行）则服从标准正态分布。这也就是说，$V^T$的各维特征（$V^T$的各行，V的各列）是互不相关的且各自的方差均为1，也就是说V的各列是单位正交的。


也就是说，奇异值和特征值是有联系的，奇异值是特征值的算数平方根，而因为求特征值必须是方阵，奇异值则是每一个矩阵都可以求，因此求奇异值比起特征值的适用范围更广。

> 之前一篇博客证明了奇异值分解，这篇博客将证明一些比较重要的性质，
> 首先回顾下奇开值分解
> 设 $A$ 是一个 $m \times n$ 矩阵, 则存在 $m$ 阶正交矩阵 $U$ 和 $n$ 阶正交矩阵 $V$. 满足
> $$
> A=U\left[\begin{array}{cccc}
> \sigma_1 & & & \\
> & \ddots & & \\
> & & \sigma_r & \\
> & & & 0
> \end{array}\right] V^T=U \sum V^T
> $$
> $\propto U, V$ 与四个基本子空间的关系
> 这里来讨论 $U, V$ 与 $A$ 的四个基本子空问的关系。
> 对 $A=U \sum V^T$ 两边右乘 $V$ 可得 $A V=U \sum V^T V=U \sum$ ，考虑左右两个矩阵的每一列可得
> $$
> A v_i=\sigma_i u_i(i=1, \ldots r) A v_i=0(i=r+1, \ldots, n) \text { 其中 } v_i \in R^n, u_i \in R^m
> $$
>
> 任取 $y \in C(A)$ ，存在 $x \in R^n$ ，使得 $y=A x$ ，因为 $x \in R^n, V$ 为 $n$ 阶正交知哖，所以存在 $z \in R^n$ ，使得 $x=V z$ ，所以
> $$
> y=A x=A V z=(\sigma_1 u_1, \ldots, \sigma_r u_r, \underbrace{0, \ldots, 0}_{n \rightarrow r \text { 同軎 }}) z=\sum_{i=1}^r \sigma_i z_i u_i
> $$
>
> 从而任意 $y \in C(A)$ 可以由 $\left(u_1, \ldots, u_r\right)$ 线性表出，由于 $\left(u_1, \ldots, u_r\right)$ 线性无关，所以 $\left(u_1, \ldots, u_r\right)$ 为 $C(A)$ 的一组基,
> 接言对 $A=U \sum V^T$ 取转置可得 $A^T=V \sum^T U^T$ ，两边右乘 $U$ 可得 $A^T U=V \sum^T$ ，考虑左右两个矩㳔的每一列可得
> $$
> A^T u_i=\sigma_i v_i(i=1, \ldots, r) A^T u_i=0(i=r+1, \ldots, m)
> $$
>
> 任取 $x \in N\left(A^T\right), x \in R^m, A^T x=0$ ，因为 $U$ 为 $m$ 阶正交知阵，所以存在 $z \in R^m$ ，使得 $x=U z$ ，带入 $A^T x=0$ 可得
> $$
> A^T x=A^T U_z=(\sigma_1 v_1, \ldots, \sigma_r v_r, \underbrace{0, \ldots, 0}_{m-r \uparrow 0 \text { 可角 }}) z=\sum_{i=1}^r \sigma_i z_i v_i=0
> $$
>
> 因为 $\left(v_1, \ldots, v_r\right)$ 线性无关，且 $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_r>0$ ，所以 $z_i=0,(i=1, \ldots, r)$ ，从而 $x$ 可以表示为
> $$
> x=\sum_{i=r+1}^m z_i u_i
> $$
>
> 从而任竞 $x \in N\left(A^T\right)$ 可以由 $\left(u_{r+1}, \ldots, u_m\right)$ 线性表出，由于 $\left(u_{r+1}, \ldots, u_m\right)$ 线珄无关，所以 $\left(u_{r+1}, \ldots, u_m\right)$ 为 $N\left(A^T\right)$ 的一组基。由对称性可知 $\left(v_1, \ldots, v_r\right)$ 为 $C\left(A^T\right)$ 的一组基， $\left(v_{r+1}, \ldots, v_n\right)$ 为 $N(A)$ 的一组基,
> 奇异值分解的约化形式
>
> 有时候会百到奇异值分解的另一种形式:
> $$
> A=U \sum V^T, U \in R^{m \times r}, \sum \in R^{r \times r}, V \in R^{n \times r} V^T V=I_r, U^T U=I_r
> $$
>
> 这种形式为奇异值分解的约化形式。
> $$
> U \sum=(\sigma_1 u_1, \ldots, \sigma_r u_r, \underbrace{0, \ldots, 0}_{n-r \uparrow 0 \text { 砉 }}) A=U \sum V^T=(\sigma_1 u_1, \ldots, \sigma_r u_r, \underbrace{0, \ldots, 0}_{n-r \uparrow 0 \text { 可晋 }})\left(v_1, \ldots, v_m\right)^T=\sum_{i=1}^r \sigma_i u_i v_i^T
> $$
>
> 现在做一个新的记号
> $$
> U_1=\left(u_1, \ldots, u_r\right) \in R^{m \times r}, \sum_1=\operatorname{diag}\left\{\sigma_1, \ldots, \sigma_r\right\} \in R^{r \times r}, V_1=\left(v_1, \ldots v_r\right) \in R^{n \times r} \text { 由正交性可得 } U_1^T U_1=I_r, V_1^T V_1=I_r
> $$
>
> 那么
> $$
> A=U_1 \sum_1 V_1^T, U_1 \in R^{m \times r}, \sum_1 \in R^{r \times r}, V_1 \in R^{n \times r} V_1^T V_1=I_r, U_1^T U_1=I_r
> $$



> 1. **Compute Mean Row**:
>
>    - Calculate the mean row vector $\bar{X}$ by averaging the rows of the data matrix (X):
>      $$
>      \bar{x} = \frac{1}{n} \sum x_j
>      \bar{X} = [\bar{x}_1, \bar{x}_2, \ldots, \bar{x}_j]
>      $$
>
>
> 2. **Subtract Mean**:
>
>    - Form a new matrix (B) by subtracting the mean row $\bar{X}$ from each row of the original data matrix (X):
>      $$
>      B = X - \bar{X}
>      $$
>
> 3. **Covariance Matrix of Rows of B**:
>
>    - Compute the covariance matrix (C) of the rows of matrix (B):
>      $$
>      C = B^T B
>      $$
>
>
> 4. **Compute Eigenvectors of C**:
>
>    - $B = U \Sigma V^T $ in SVD , $C = B^TB  = V \Sigma \Sigma^T V^T $ so $CV = V D$ (D is eigen value, $ \lambda = \sigma^2$)
>
>    - $T = BV  = U\Sigma$​ , T is principle components
>    
>    - The first principal component $\mathbf{u}_1$ is given as
>      $$
>      \mathbf{u}_1=\underset{\left\|\mathbf{u}_1\right\|=1}{\operatorname{argmax}} \mathbf{u}_1^* \mathbf{B}^* \mathbf{B} \mathbf{u}_1,
>      $$
>      which is the eigenvector of $\mathrm{B}^* \mathrm{~B}$ corresponding to ==the largest eigenvalue==. Now it is clear that $\mathbf{u}_1$ is the left singular vector of $\mathbf{B}$ corresponding to the largest singular value.
>    
>      It is possible to obtain the principal components by computing the eigendecomposition of $\mathbf{C}$ :
>      $$
>      \mathrm{CV}=\mathrm{VD},
>      $$
>      which is guaranteed to exist, since $\mathbf{C}$ is Hermitian.
>    
>    - Find the eigenvalues and eigenvectors of the covariance matrix (C).
>    
>    - Solve the characteristic equation:
>      $$
>      \det(C - \lambda I) = 0
>      $$
>      where $\lambda$ represents the eigenvalue.
>    
>    - For each eigenvalue $\lambda$, solve the system of equations:
>      $$
>      (C - \lambda I)X = 0
>      $$
>      ​    to find the corresponding eigenvector $X$.
>    
> 5. **Principal Components**:
>
>    - The eigenvectors obtained represent the principal components of the data.
>    - These eigenvectors provide directions in the original feature space along which the data varies the most.
>

> The covariance matrix being expressed as **B<sup>T</sup>B** is a characteristic of certain linear transformations and their relationship to data. Let’s delve into this:
>
> 1. **Linear Transformations**:
>    - Suppose we have a matrix **B** that represents a linear transformation from one vector space to another.
>    - This transformation can be thought of as mapping points from an original space (input space) to a new space (output space).
> 2. **Covariance Matrix**:
>    - The covariance matrix, denoted as **Σ**, captures the covariances between variables in a random vector.
>    - For a random vector X with n components, the covariance matrix is given by: $\Sigma = \frac{1}{N} \sum_{i=1}^{N} (X_i - \mu)(X_i - \mu)^T $ where:
>      - $X_i$ represents the **i-th** observation (vector) in the dataset.
>      - $\mu$ is the mean vector of the observations.
>      - The sum is taken over all **N** observations.
> 3. **Linear Transformation and Covariance**:
>    - Consider a linear transformation of the form: $ Y = BX $ where:
>      - **Y** is the transformed vector.
>      - **B** is the transformation matrix.
>      - **X** is the original vector.
> 4. **Covariance of Transformed Variables**:
>    - The covariance matrix of the transformed variables Y can be expressed as: $ \Sigma_Y = \frac{1}{N} \sum_{i=1}^{N} (Y_i - \mu_Y)(Y_i - \mu_Y)^T $
>      - Substituting $Y = BX$: $ \Sigma_Y = \frac{1}{N} \sum_{i=1}^{N} (BX_i - B*\mu)(BX_i - B*\mu)^T $
>      - Distributing the terms: $ \Sigma_Y = B \left(\frac{1}{N} \sum_{i=1}^{N} (X_i - \mu)(X_i - \mu)^T\right) B^T $
>      - Recognize that the expression inside the parentheses is the covariance matrix of the original variables **X**: $ \Sigma_Y = B \Sigma_X B^T $
> 5. **Conclusion**:
>    - The covariance matrix of the transformed variables **Y** is related to the covariance matrix of the original variables **X** by the transformation matrix **B**.
>    - Hence, we can express it as **B<sup>T</sup>Σ<sub>X</sub>B**.
>
> In summary, the use of **B<sup>T</sup>B** arises from the relationship between linear transformations and covariance matrices, providing a convenient way to compute the covariance of transformed variable

## 3 **Mahalanobis Distance**

[马氏距离(Mahalanobis distance)](https://www.cnblogs.com/hust-yingjie/p/5954861.html)

**马氏距离(Mahalanobis distance)**是由印度统计学家马哈拉诺比斯（[P. C. Mahalanobis](http://en.wikipedia.org/wiki/P._C._Mahalanobis)）提出的，表示数据的[协方差](http://scau200630760309.blog.163.com/wiki/协方差)距离。它是一种有效的计算两个未知[样本集](http://scau200630760309.blog.163.com/w/index.php?title=样本集&action=edit&redlink=1)的相似度的方法。与[欧氏距离](http://scau200630760309.blog.163.com/wiki/欧氏距离)不同的是它考虑到各种特性之间的联系（例如：一条关于身高的信息会带来一条关于体重的信息，因为两者是有关联的）并且是尺度无关的（scale-invariant），即独立于测量尺度。 对于一个均值为

![\mu = ( \mu_1, \mu_2, \mu_3, \dots , \mu_p )^T](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMsAAAAYBAMAAACmfxSBAAAAMFBMVEX////MzMwwMDBQUFB0dHSenp4MDAxiYmIWFha2trZAQEDm5uYEBAQiIiKKiooAAADoe+fAAAAAAXRSTlMAQObYZgAAAnlJREFUSA21lT9oU1EUxr/EpMnLS98LorMdKkUECbiJw1scjEt0sEvFIDqIIFFLHUTMVu0U6NJBSECHKgiZ6lKJgoPgEnHLIA8tTv4pdBAttJ5z7r157yV5lVhyIPee33fud8/lvdwEGHcs3dm6XRl3E1i+U0Fx7G1SyDXhD2+TKQzXR1XPsmHCj7P9iCuMqCdbZGjzMCwSU8PU/9Bko69xxmQ9rjKqfpQMz+NM2bjCyPoFctyMc5XiCiPrLlD647HN6SDf95I+DcjRRVHiTaLxHXivlGzZVCab+Mn56kOODqeP6GNk5r1JrQiPXTibipOe0V0fL0yuZz5KVN6L+syEO8jJcYEDRVNttLBocj3zUZQ8WVCSkHX3WIjSG5q0KZisbWSrCjNNIy/A/m1yPW/TLLJzWrcReonHLVkhlK/P9dkMZipo1xUEbbpIb7IWejdvCJWc122EGnDL4hZK4LLA4ECv1ZRyVVP+hVzNM6Dm6zQp2bQRstBWXbXladTVI7eK1ZR9f+MZkPS1am1hpdnrqcRpQMvUZs3rEZ6Eafk47KuQj8g9aBStpYwz4a/Qz2Zdt8lcmT8/ow6pFbpRgJapzXqtR6lmmDDXst9Rm7fkW/dUwgIWZs4cvIU8LlIbQgl6kAOxJheFZX5oXo9m0QroG1wPqX4vC10RL+EU8MXUXd9kwZwr87Xh4DaUKbLuHaoHtMxfJ0dWhQYWdoRP2jeAD6ZEN2Ig7A5fG4r0g2kkynKJiI7s7obImT8BHO73kkDXhuNjqQCrKSkNn00SnmdDMp8vWBSlwX9fn34kp2Sr1zQmJYsdHHpW+wyrRhuc2+cm/7a/uhZe8xecybL0zobbYgAAAABJRU5ErkJggg==)

，[协方差矩阵](http://scau200630760309.blog.163.com/wiki/协方差矩阵)为Σ的多变量矢量

![x = ( x_1, x_2, x_3, \dots, x_p )^T](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAAAYBAMAAABNSK+CAAAAMFBMVEX///8MDAwEBAS2traenp5QUFAWFhaKiooiIiLm5uYwMDBAQEBiYmJ0dHTMzMwAAADdHjBWAAAAAXRSTlMAQObYZgAAAltJREFUSA21lT1oFFEUhY/Z2Z2dZGcSsEglWAkWhkVILCwcUi9kG60UFzQ2NqsgVuJaRRuZtUlhJCsWIUVgGm3MalqJgWljDAxopQRT2GhhvO933v48yQj7CPPu+c65c+dvCTDaVdq8fvXlaEcAF3AHxVEPSfAW7vAhlXQ4z0f9KstP25ru24x8fJ3i3i9LT9CxGDnxN8q7U5amcX6fFjMHHg+BidjSULLwvNjpACealq49C8+ND+G/77Kug+3Kp77upX4cfI53OjrUqzRWhWHflMxLyk8eRiTe3GKrxfAN+igUZhov8Hg5+0p6FQ+YB8PekLyCUj0wM1SfBXrxFhaR6lCv0lgVhn1ZMg+TobLVfkh3IvBTgSL8BAofXxnKeSCVatI7Dwu1reGCrlTBnwzD92YkcmjuCs6zx0qLqbHqvBCDR2aLlQ25IoDxTvipOVZDig3gkr5npgIMXpw4EZgt1qbcA9zGsoJyP8nOwbEckpZir1nAasoDXAH82+zrZJLbK0t7VWBN2gtfFrEla7XRRyHxDLxT9Hj+zNUrZL421ew6V8xHO6G3KAsRbk4XGgBdLV/f95/vR7JWG/0YJaYhZ+i2urvdXcCPTYX5iHneOep6xobwoqHCp50pgP6sq60dely+FjVEmbqLySRTOkPT+HJaxRhuXfPBYiJUjIbofzmFDwfVTM1i1VAqj6asivV2hHKq+WDhtSSr/U7wVfk/jo6QKXeH3qr2VCYIZVW+9g64qPDQvWZQdW0C/UtRQt/2GIt32MG63NRqHdOYo5wfHTP8nzHvUWh0/gXveqFN+/KdvQAAAABJRU5ErkJggg==)

，其马氏距离为

 

![D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x-\mu)}](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQ4AAAAfBAMAAADkYj4cAAAAMFBMVEX///90dHQwMDAEBATm5uZQUFDMzMyenp62traKiooWFhZAQEAMDAwiIiJiYmIAAAAFYLcWAAAAAXRSTlMAQObYZgAAA/9JREFUWAnFVkuIHFUUPf2Z/tV0TScIRoRMJQtBkJiAuMki5QQFMSS9iYIIKUWziANp14KWBBeSBEsQF6KmUcQ4C2lhRLGdOAQXfkB6pYiKFT8oRJMoySQhJvHceq+qX/WMU+mGZi7Uveee+86tV7c+3cDQNn19DDb0LgBnBM0YJNVgDE1HaJl3RxCNQZIfQ89RWj41imgkjatUOgy2mCdRHCTj/H8LI4gmPNV1Q9w8HXtMH0xT/SznA7UT+8480qcE3SZuVRHrf/3bXXhXFmr7XUfLjRkzVkLAbplMCnNaL+A12JNXu92FHU1Ve+NuxtVFXFC5CMxqBdOSQydWdKIw4GRajw1wRjoDNNBFrnyVpOXoiuxjdREX2OeB8pJWAHUvhh/GwIx1Jt+YRBpbMqoveHwa0Dk8xGQfWSLUF7nqAg9lB2KAowkywBOcWNvIB2Clx0HywlC4iy6eMveRJaKiQ8XJpN9vCSokyAB/cIBxe4NO4Fagusis/A9dy+52u+9H88gSAQcDKuR2KvuO4bnTu9fzDrVizog98hRUNs6fcgwayPWw0wW+5WPRkcKdhprzyBIB01QXr4g0Mplho96b9mCFJD6eE+PZlZV4joIHPIofc0uK0r7ewTHCE8BkINROuTHa2LPgrS4C7uDi6mU6Zby5NmrNCjDRjDkVS556+6ZcIMTb8BWt/VSABcK9yO3jnWDLz3QBeOZKgAwR127nkT8Hm7F8zcM5+VxOtUjanaRTBHaFXEg0xcMF33bafrHHBb3o4j2G04LF4s+hyrJFJRnFniaKvJrdbcg+gGfFDe7j5zbwAHnZB8rRutJmsS1C/IniJYZkHzczCaSgLFOUk7fsE87xEHC4CUiKh+UnJLovxvNRZekWFmXEmGhzJqZ9z62R2au5ckjAdrFlivKL3AMfJLvgViZbwOd8GXjz7+Fz34ib6MjJOYSTPuDXmsV0+W9YPTLRs8FFT/Io9ui0ZYoKDi/f5T5q/n17KLodOLb/az6IqPlMTXtT7SMfoLo0G9pmCZVrONIJgeOKjX4UDjj9JVki66evupsDrr/farReZbyVPwXrZja6fNX6XRT6BdGfwlwHpU03bfo1VbZPnnpliw9sVWz9y7m5t643+0tuRBStftl+CB8Q7Uq0TydIg4N+3iUsLtINGj8fYmUVBqtDiMLycbxDveXHTaJ/DXEisRDmo/Qjk9SYnw+xvBeFFdwNiu7dVnKev+By445uUll2aVZb/SmcWeE0/HyIHY78Sm5o0Uu6i7p2s2X1/HyUVhsmq/DrKjjLK5oZWpTzlTL9HEbcZUeVNqiw3Nvuci5mRhLF4nT8IUzna5Wd9dbqzOnzzqbTNcv4ozMm+w8FkE9ZXumUvAAAAABJRU5ErkJggg==)

马氏距离也可以定义为两个服从同一分布并且其协方差矩阵为Σ的随机变量

![\vec{x}](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAPBAMAAADNDVhEAAAALVBMVEX///8MDAxQUFAwMDBAQEB0dHSenp7m5uYWFhYiIiK2traKiopiYmLMzMwAAACCJjQWAAAAAXRSTlMAQObYZgAAAFpJREFUCB1jYACBC2CSgeE0kOaaOXNmXwFYgL0NIs67AEKDyaVHaoKBDN4AzglxQH08DBwKXEA+OwNfAlieIQ9CMWgAxYDGMTQxlALpfXdlGA4A6cuhxTFA0wHCXRE0j8s9AAAAAABJRU5ErkJggg==)

与

![\vec{y}](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAATBAMAAAC5GZqkAAAAMFBMVEX///8EBARAQEAMDAxQUFAwMDB0dHSenp7m5uYWFhYiIiK2traKiopiYmLMzMwAAAA1hG37AAAAAXRSTlMAQObYZgAAAHFJREFUCB1jYACBB2CSgeEOkOZetWrV/AawAMd0iDjfBggNIvlSCxhigTQvSwDDHyBdx3mB+weQbnjfwGsApBk8GbgUQPRcBv4DIFqcYX8DiJ7DEAyiGLbmTgLTDAzfQTSnA9sEML3h8QMQzXPXA0QBABfzGMBT/7Q6AAAAAElFTkSuQmCC)

的差异程度：

![d(\vec{x},\vec{y})=\sqrt{(\vec{x}-\vec{y})^T\Sigma^{-1} (\vec{x}-\vec{y})}](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQgAAAAfCAMAAAAsjKNaAAAAM1BMVEX////Q0NCKioowMDAMDAzm5ua2trbMzMwiIiJiYmJ0dHSenp4WFhZQUFAEBARAQEAAAACuaqKgAAAAAXRSTlMAQObYZgAABApJREFUaAXtWImu4yoMZZISQjb4/699NthADNmmt9Kr7qAqZTHnYGMbEqU+XSb/FeXTZlCq/zzFVzB03Vcs8/OL7PXnOb6C4V9kxG2ap6/YrsYirzz5alxAOpbnfzF+2Hwqn4CeTjyQX01CVEU1dZppSPU7lZGErnAlVuJZnF29tetLSuzafVrrQyKjhEJu84hs5wLfNhK+7dIeF5KH1cHGoUtcicA8A6yhQyMA0MuPDsu4+XUn39mNDfGUCHQUCq0LYJuY2oYxlMUXbpP6/jxJf7Q/J7g7jaAheDDFOAcPqMye/cvJNbAh7hMlHq33YAsaeWTDQl3ppXQP7Il9ru6Og40nrfwSV05F7sTzYs90nmv7tSvFhnhKFHRMRGEVHtQz6BWpmEY2wL6B9yVJHlZMdOFrXIlQ8ujsmNtGProPjWSIx0RBx51CM6YIK/Hl+mJ7a3c3em205U1cCUA8IUXEsdmTK8Cu2ZAvnAvnM3nEe0RqdpMNKWIkzxsm67SdCv8zU48B0YfcN96ODY6MQ1ype4snpAgS7NFxG4UMcayAnNMimmC5JuRhgjNg5HFUk88Zw6oVzRJj1NFRAB1uyaWOGINZDsoxbhzPzxZPShEgplNw5DlYI4a3iIagb7A0BSPqibkz54jZqAV6hyiwXkWQIQvyWXaIu9cGzoWaZ1p99P4o272Kk4ynz9a74HTvEOHe48mEoOEBZsfqzgU1uAz02nirsbTROKVV9ItchmPrANfkQtpd85T3pZr7LSKPqw4pgg0B7WiYginmkGiBK0PYPp4+mg1G6xO4ps+FLEcLGQ95JtofPkaLFWL1HSLjMQIgEmBTyLOgHfSmbUKGEBmUIlQRGu0c0UWgiZd7ghvAiwdGYIOHJGY22EFwvkMUfF5DJECGfMWsAE9METocTLSEDbaCUoQqY7bQoajGpMqRcYZbTArVUx5OvkrXiTnMPlHgmggnr14P8NeHLey8MRjl8DMbb8ECQz29+CzRXhK6aC/BBskQFW4hKqpnPOkubCK+mArNt4gGuEXodUWNpxCbpp8m3UMnnqr80jdAx0Ixf32hsiiSP9JVuLUK3HPGM77iWb158dLFk2sF0khVOSNSmtXO08rwoJNk5hyYpWQtJJ78ka6BK2eU7fs85Sys/xiRq5ye8h3mjIneRPK7v1xHbmNIpciAe1eFm0V3tac8u8nQ+CmiFIVMoOm4grRgXvGaNFw7BN44VfmRrsJlfPn/lEfO/zGirgwFYOFc2cE3IjpNi42Wy8htOEB33zskbpbc157y7GdD68eIxIerikhhDr1RfLc32CWuxLzJI6fJL2/VeNXxt0QVULtj5Gt2e/j39Nri1fX3aN3Q1NQncUPqN3TdSyX/H0v8B7UQLnqUpfipAAAAAElFTkSuQmCC)

 

如果协方差矩阵为单位矩阵，马氏距离就简化为欧式距离；如果协方差矩阵为对角阵，其也可称为正规化的马氏距离。

 

![d(\vec{x},\vec{y})=\sqrt{\sum_{i=1}^p  {(x_i - y_i)^2 \over \sigma_i^2}}](data:image/png;base64,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)

 

其中σi是xi的[标准差](http://scau200630760309.blog.163.com/wiki/标准差)。

 

**基础知识：**

假设空间中两点x，y，定义：

欧几里得距离：

![img](data:image/png;base64,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)

Mahalanobis距离：

![img](data:image/png;base64,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)

不难发现，如果去掉马氏距离中的协方差矩阵，就退化为欧氏距离。那么我们就需要探究这个多出来的因子究竟有什么含义。

 

**第一个例子****基础知识**

从下往上的一段50米长的坡道路，下面定一个A点，上面定B一个点。假设有两种情况从A到B：

a)坐手扶电梯上去。

b)从手扶电梯旁边的楼梯爬上去。

两种情况下我们分别会产生两种不同的主观感受，坐电梯轻松愉快，感觉很快就从A到了B——“A与B真近”；走楼梯爬的气喘吁吁很累，感觉走了好久才走到B——“A与B真远”。

 

**第二个例子**

观看落日之时，由于大气的折射效应，太阳形状产生形变并且视觉位置也比真实位置高。

![img](data:image/png;base64,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)

 

**解释**

以上两个例子看似和模式识别没有关系，实际上都引入了“相对论”的问题。回到问题本身，欧式距离就好比一个参照值，它表征的是当所有类别等概率出现的情况下，类别之间的距离。此时决策面中心点的位置就是两个类别中心的连线的中点。如图1所示。而当类别先验概率并不相等时，显然，如果仍然用中垂线作为决策线是不合理的，将出现判别错误（绿色类的点被判别为红色类），假设图1中绿色类别的先验概率变大，那么决策线将左移，如图2黄线。左移的具体位置，就是通过马氏距离来获得的。马氏距离中引入的协方差参数，表征的是点的稀密程度。

![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAqcAAAFyCAYAAADI/nMFAAAgAElEQVR4AeydB5gUxdaGvwk7s3nZJUnOQSSDAooiQUkGjCAY0WvAjFflKvpzUVGUIEg2YUJEr6IEQUERUFERUBTJOSPssnl2J/xPNS69CxtmerpnOnz1PONWdVedOuetcTlbfU61bc2aNQEASEpKQp06dbBp0ybRZJs8+H3g/w/8fcDfh/z3gP8e0h+gPxBxf8AWCAQk51T6DcT/kAAJkAAJKCbw008/oVOnTorHcyAJkAAJkABA55TfAhIgARJQicDJkyeRkpKikjSKIQESIAFrErBb02xaTQIkQALqEzh06JD6QimRBEiABCxGgM6pxRac5pIACWhHQOycspAACZAACYRHgM5pePw4mgRIgARIgARIgARIQEUCjDlVESZFkQAJWJsAY06tvf60ngRIQB0C3DlVhyOlkAAJkAAYc8ovAQmQAAmET4DOafgMKYEESIAEJAKMOeUXgQRIgATCJ0DnNHyGlEACJEACJEACJEACJKASAcacqgSSYkiABEiAMaf8DpAACZBA+AS4cxo+Q0ogARIgAYkAY075RSABEiCB8AnQOQ2fISWQAAmQgESAMaf8IpAACZBA+ATonIbPkBJIgARIgARIgARIgARUIsCYU5VAUgwJkAAJMOaU3wESIAESCJ8Ad07DZ0gJJEACJCARYMwpvwgkQAIkED4BOqfhM6QEEiABEpAIMOaUXwQSIAESCJ8AndPwGVICCZAACZAACZAACZCASgQYc6oSSIohARIgAcac8jtAAiRAAuET4M5p+AwpgQRIgAQkAow55ReBBEiABMInQOc0fIaUQAIkQAISAcac8otAAiRAAuEToHMaPkNKIAESIAESIAESIAESUIkAY05VAkkxJEACJMCYU34HSIAESCB8Atw5DZ8hJZAACZCARIAxp/wikAAJkED4BOichs+QEkiABEhAIsCYU34RSIAESCB8AnROw2dICSRAAiRAAiRAAiRAAioRYMypSiAphgRIgAQYc8rvAAmQAAmET4A7p+EzpAQSIAESkAgw5pRfBBIgARIInwCd0/AZUgIJkAAJSAQYc8ovAgmQAAmET4DOafgMKYEESIAESIAESIAESEAlAow5VQkkxZAACZAAY075HSABEiCB8Alw5zR8hpRAAiRAAhIBxpzyi0ACJEAC4ROgcxo+Q0ogARIgAYkAY075RSABEiCB8AnQOQ2fISWQAAmQAAmQAAmQAAmoRIAxpyqBpBgSIAESYMwpvwMkQAIkED4B7pyGz5ASSIAESEAiwJhTfhFIQCMCuVcBWbZTH9+fGk1CsXohQOdULytBPUiABAxPgDGnhl9CGkACJKADAnROdbAIVIEESIAESIAESKAMAnk3Ab4FZdzkZTMSYMypGVeVNpEACUSFAGNOo4Kdk5qZQP5dQOGbsoXxawBHJ7nNmikJcOfUlMtKo0iABKJBgDGn0aDOOU1LIP+Rko5p3HI6pqZd7JKG0TktyYMtEiABElBMgDGnitFxIAmUJOB5GiicJF+L+xxw9pDbrJmaAJ1TUy8vjSMBEiABEiABgxHwvAgUjJGVjp0DOK+S26yZngCdU9MvMQ0kARKIFIHmzZtHairOQwIkQAKmJUDn1LRLS8NIgAQiTYAxp5EmzvlMR6BgMlDwlGyW+3Ug5ia5raA2bdo0PPHEEwpGcki0CDijNTHnJQESIAGzEWDMqdlWlPZElEDhW4DnYXlK90TAdZfcVlB755138Mcff0A4qCzGIUDn1DhrRU1JgARIgARIwJwECucB+XfKtrmeB1yPyG0Ftf/973/45ptvIBxUFmMR4GN9Y60XtSUBEtAxAcac6nhxqJp+CXgXAfkDZf1cIwD303KbNcsRoHNquSWnwSRAAloRYMypVmQp1zIEYh4E3C+Gbe6SJUswZ84c7pqGTTI6AvhYPzrcOSsJkIAJCTDm1ISLSpO0JeD9Dsi7QtU5Vq9ejVdffRXCQWUxJgE6p8ZcN2pNAiRAAiRAAsYm4FsL5F0q2+C8A4idLLcV1NavX48RI0ZAOKgsxiVgCwQCAeOqT81JgARIQD8ExM5pSkqKfhSiJiSgVwK+zUDuubJ2zhuAuHlyW0Ft69atGDhwIISDymJsAow5Nfb6UXsSIAEdEWDMqY4Wg6rol4B/X0nH1NEvbMf04MGDuOKKK+iY6nfVQ9KMzmlIuNiZBEiABMomwJjTstnwDglIBAIngJy6MgzHJUD8IrnNGgkAYMwpvwYkQAIkQAIkQAKRJ2DvAMR/F/a84o/Cjh07QuyespiDAJ1Tc6wjrSABEtABAZ5zqoNFoAr6JRAoALIrq6qfz+dD3bp1wacWqmKNujA+1o/6ElABEiABsxBgzKlZVpJ2aEIgO1EWa28GJKyV2wprIgGRjqlCeDoeRudUx4tD1UiABIxFgP9IGmu9qG0ECWSlASg8NaGtNpCwOezJq1SpgmPHjoUthwL0R4DOqf7WhBqRAAmQAAmQgHkIZNcBkP6PPalA4j7z2EZLNCFA51QTrBRKAuYjMH78eHz77bfmM0xFixhzqiJMijIHgZzmQGD/P7bEAEknVLGrfv362Lx5M+Li4lSRRyH6IkDnVF/rQW1IQHcE8vLyMGvWLOnxmdPJHMryFogxp+XR4T0SUIfAeeedhxUrVkA81mcxJwE6p+ZcV1pFAqoQSE9Px9dff41FixZh0KBBuPjii1WRa1YhjDk168rSLkUEcjoC/i3y0MRsua6w1qlTJ8ybNw9i55TFvAS4DWLetaVlJBAWgezsbOn91E8//bT0OJ+7FGHh5GASsBaB3G6A/1fZ5sTjgM0ltxXUevTogcmTJ0PsnLKYmwCdU3OvL60jAcUEPvroI2mHYvXq1UhKSlIsx0oDGXNqpdWmrWUSyO0P+FbKtxP2AjaRra+8XHXVVRg5ciTEzimL+QnQOTX/GtNCEgiZgEh+EvGT4qc4R5AlOAKCGXkFx4q9TEog70bAt1g2Lv4vwC6y9VlIIHgCdE6DZ8WeJGAJAuPGjUN+fj6uueYatGzZ0hI2q2UkY07VIhk5OVm33onCb4vt8p0xtb1mDVT6qez7Z3RnsziB+F8AR/PiVxTVb7rpJtx1110Qj/VZrEGAzqk11plWkkCFBHJzc/Hee+/h+PHj6N+/Py666KIKx7ADCRiJQN6kKcgbNykklf0HD+FEnSYlxriu7IfEaaHJKSHArI28oYD3Y1WtE06peKQvPizWIWALBAIB65hLS0mABEojcOLECaxcuRKzZ8/GqFGj0LZt29K68VoFBMTOKR/rVwApwrc9n32BnIce02zW2AfuRfyT2snXTHG1Bec/CBROkaXGrQCc3eS2gtojjzwiPb0RDiqLtQjQObXWetNaEjiLgMjKF4fri2SD5cuX8+zAswgFf0EcCs6kqOB5admzcNX3yBp8u5ZTlJAdP2okYu+8rcQ1yzQ8I4CCsbK5cQsBZ3+5raAmTgmpWrUqhIPKYj0CPOfUemtOi0mgBIG5c+diypQpWLVqFdLSwsuoLSHYgg3GnEZ/0b2bt0iP4SPpmAqrc0c9L83r+WJh9CFEUgPP8yUd09iPwnZMI6k+59InATqn+lwXakUCESEgkp/Ebt+ECROQnJwMu52/EiICnpOoTsD/93HJOcy87ArVZYciMOf+RyU9Cn9YE8owY/YtmAgUPCPrHvsmEHOj3FZYe/HFFxEfH89dU4X8zDCM/xKZYRVpAwkoICAcU4/HI2Xl81BrBQBLGcJH+qVAicClQF4eMtp1jsBMwU+RNfAWyUn1bd0e/CAj93RPBmKGhm2BOGQ/MzMT4rE+i3UJ0Dm17trTcosSEFn5M2bMkLLyu3Xrxqx8Fb8H4pxTlsgSONG8LdKbto7spCHMdrJnX/j2HwhhhEG6FrwOeIarquxbb72F7du3Q+ycslibAJ1Ta68/rbcYAZGVv3TpUukzaNAgdO3a1WIEtDWXMafa8j1TenqHC4GcnDMv6659ssul8Ken604vxQoVzgE8d8vDXS8CrgfltoLavHnzpNcli51TFhLgOaf8DpCARQiIrHzxKtL//ve/WLZsGbPyLbLuZjUz4+JeCBw9ZhjzMlpfgNTtf8DmdhtG51IV9X4O5A+Rb7meBtwj5LaC2qJFi/DJJ59Ir0tWMJxDTEiAO6cmXFSaRAKlEWBWfmlU1L3GmFN1eZYl7WTvK+Hfvaes27q9nt6sjW51C0ox73Igb4DcNeYRwP283GaNBFQiQOdUJZAUQwJ6JlCUlT9x4kQkJSUxK1+jxWLMqUZgi4nNvGYgfJs2F7tioKrPh/QW7Q2kcDFVfWuAvF7yhZi7gNiJclth7bvvvsO0adO4a6qQn1mH0Tk168rSLhL4h8Arr7wiZeVfe+21YFa+tl8LxpxqyzdzyB3wrl2n7SQaSw9kZSG9I18NLDCvXbsWzz77LMRjfRYSKE6AzmlxGqyTgIkIiKz86dOnIz09HZdeeikuvPBCE1lHU6xGIOtf98O7crUpzA4cOYqMSy4zji2+P4DcLrK+zpuA2NfltoKaOF/53nvvhdg5ZSGBMwnQOT2TCNskYAICRVn5X3/9NQYOHMjjoiK0pow5jRBoE0zj37UbJ/terX9L/LuA3Fayno6rgLg5cltBbd++fRgwYIC0c6pgOIdYgACz9S2wyDTRWgSysrKkrPzRo0dDOKdVqlSxFoAoWitiTlNSUqKogTmnzn74cRQu+cqcxunZKv8xIKehrKGjBxD/udxmjQQ0IsCdU43AUiwJRIsAs/KjRR5gzKn67HOeehYFn85XX7AOJPr+2ITM627SgSalqBDIA3KqyTfsnYD45XJbYU081REhRuKxPgsJlEWAO6dlkeF1EjAgAZH8dOTIEbz66qtITEw0oAVUmQRIQHcE7OcBCWvCVqugoACNGzeGcFBZSKA8Atw5LY8O75GAgQi8/PLLEL/8RVZ+ixYtDKS5eVRlzKm6a5n7/EvwvPehukJ1Js3781pk3XKnzrQCkK3+H7dpaWl0TPW30rrUiM6pLpeFSpFA8AREVr44JzAjI4NZ+cFj06QnzznVBCuFRppAloib9p+a1VYfSPgjbA3omIaN0FIC6JxaarlprNkIiMdjS5YswfLly5mVr4PFZcypeouQO2Ey8me+qZ5AHUsqXLESWfc8oA8Ns2sCyDyli60KkLgrbL3q1KmD7du3w+VyhS2LAqxBgDGn1lhnWmlCAkVZ+c899xyz8k24vjSJBCJOILspEDj0z7SxQOKxiKvACUlAEODOKb8HJGBQAiIr/7XXXpOOjRKPzFiiT4Axp+qsQd6MN5A/8TV1hBlESuHipch+9InoaZvTDghsk+dPzJbrYdTE/xM//PAD+DsqDIgWHMqdU4Mset6M11GwaGlQ2ia8MArO1i2D6stOxiRQlJU/adIkJCQkGNMIE2rNc05NuKhWNCkxA7A5wra8Y8eOmD9/PsRjfRYSCIUAndNQaEWgr+d/81H4w49nzeT9dQP8O3aedb20C3mvTICtWtWzbsXddzccjRuddZ0XjEVg7Nix8Hq9uO6665iVr7OlY8ypzhaE6gRHILcr4N8QXN8ge3Xr1g0zZswAnyYECYzdShCgc1oCR+QbvoOHULBg0emJC7/8Gt5f151uK6kUrlhV6jCbyw17/brSPXuNGnBf1b/UfryoTwIiK3/27NnSQe/9+/dHly7F3nWtT5WpFQmETCD/3TnIe2FsyOPMMKDgk8+QEx8P8fQrYiW3D+D7Xp4u4QBgC+8tZ+L3k3hDndg5ZSEBJQTonCqhpsIY75Zt8B84AN/mrch78RUVJFYswvO+fF6go1lT2BJPPQ52tmsDe2pqxQLYI2oEjh8/jhUrVkhZ+c8++yzatGkTNV04cdkEuEtUNhve0SGBvOsAX7FwsYQtgF1k67OQQHQJ0DmNIP+Az4fAkaMI+P3Inz4LBf+L3iv5fFu2Ivu2f0nWJ0yZCGeHdrAlJMCeWimCRDhVMASKsvJfeOEFfPXVV6hSpUoww9gnCgQYcxoF6JxSGYG82wDvp/LY+HWAvancVli78cYbMWzYMIjH+iwkoJQAnVOl5BSMC5w8Kb1H2X/8BFBYoECCNkNyHhsB2O1wDx6IhFFPazMJpSom8OGHH+Ljjz/GqlWrEB8fr1gOB2pPgDGn2jPmDCoQyB8GeN+VBcWtAhzt5LbC2tChQ3H99ddDPNZnIYFwCNA5DYdekGMLFi1B3vTXAa8X/sOHAa8vyJER6ubxSBMVzF8A79p1sKUkI3Hqq7BXCi/uKELam3oa8UrSI0eOYPLkyczKN/VK0zgSMDaBBx98EF27doXYOWUhgXAJ2AKBQCBcIRxfNgGRfZ//zgfwrVc3E7LsGVW443LB1fsyxD36IBxNmN2vAlFFIoqy8nv06MHkJ0UEIz9I7JympPCPOqXkPSIhKJpnfSpVXOVxsff9C/FPaXTmaf4TQGGxPIe4LwFnn7AsGDFiBGrVqgXhoLKQgBoEeAi/GhTLkOH5YhE8780xlmMqbCkokE4QyH9zNrx/bCrDOl7WioDIyp86dSoyMzPRvXt3OqZagdZArog5ZSEB3RLw/LekYxr7SdiO6fPPP49KlSrRMdXtohtTMT7W12DdAgUF8P70C/Jfmw7f5i0azBAZkZ4P5kqxqP6el8JeqZKUNBWZma07i8jK//bbb6WPyMpv3bq1dWEY0HLGnBpw0ayicsE4oKDYEVWxs4GY66xiPe00GAE6pyouWCA7G/70DAQyMpB9/yMIpGeoKD06osTOr/g42rZB4ozJsJ9THTZH+G8OiY41+p5V7JSuXr0aL774IpYuXcqsfH0vF7UjAeMQKJgBeB6X9XVPBWJuk9sKaxMnTkR+fj7E7ikLCahJgI/1VaTp+XwhTvboi8xrB5nCMS2Oxvf7RmQJuzKM73AXt0tP9blz5+K1116TsvIrV66sJ9WoS5AEeM5pkKB03K3loV3I8OssaTUcXoXvAZ77ZAnusYBrmNxWWHv99dexd+9eOqYK+XFY+QTonJbPJ+i7edNnSY/xkZ8P5J/Kfg96sBE6+v3wHzmKrMF3wPv7RiNobCgdRVb+li1bMGXKFOm4KJvNZij9qewpAow5Nf434Y8aDdD+0B4UMle4zMWcM2cOfv75Z4idUxYS0IIAnVOVqPr3H4T/wEGVpOlUjM8H36a/kDt2AgpX/6BTJY2n1ksvvYTCwkLccMMNfA+18ZavhMaMOS2Bw7CNnbUaovaBHWh0YKdhbZAUL/wMyL9VtsH1LOAK7xSAzz//HAsXLoTYOWUhAa0I0DlVgax4nO/70zpZ7d6Vq+HbYfBf2iqse7giRFa+2CkVb4ASWfmdO3cOVyTHkwAJqETgSO3GWFujHlof2q2SxAiL8X4F5F8rTxrzGOD+r9xWUFu+fDneeustiJ1TFhLQkgCd0zDoiqz8wpWrkT9lBry/rg9DkvGG+jZvhdfAJxFEm7jIyl+0aBG+++47DBw4EBdeeGG0VeL8KhBgzKkKEHUkItXuwOKqtdDl8B4daRWEKr7vgbzecseYe4DYcXKbNRLQOQFm6ytcoEC+B76t25B9/6OWTBLyvP8hAjk5iH/6SdirV1NI0ZrDRFa+eBWpOGR/yZIlzMo30ddAxJzyEH7lC+q+/hoEcvOQ+/T/KRei8sjazhjMrlwDPY/sw/LqdVSWroE4329AbldZsPNmIHaG3FZYW7NmDcaMGQOxe8pCAloToHOqkLBICsoafDvwz6s/FYox9LCCLxbB//dxJM+ZbWg7Iq28yMqfN2+e5KDGxsZGenrOpyEBxpxqCDeKopvFuDAutSoq79+Oy2LjMbdKTc20cd86RL23QzkHAHHvha3rH3/8gUceeQTCQWUhgUgQ4GN9BZQLFn2JnMefsrRjKmHz+eBdtx6ZN90Gf8ZJBSStN0TslhZl5cfFxYFZ+db7DtBiYxJo54rF8dqNMTQhBXceP6xPI/zbgdy2quq2a9cuDBo0iI6pqlQprCICdE4rIlTKff/xE/Dv3FXKHQteysmVMvjh9VrQ+NBMFofr+3w+3HjjjczKDw2dYXoz5tQwS6VY0cvjEnBFXAIeOnFEsQxNBvoPAzlNZNGOy4C4z+S2gtqxY8fQo0cPiJ1TFhKIJAE6pyHSFglQ4sMiEwjk5SN/9nvwmf0oLdnkkGoiK18crp+Tk4NLL70UnTp1Cmk8OxuHAM85Nc5ahaPpNfFJuMAdh/9kHAtHjHpjA1lATg1Znv1CIP4ruc0aCRiMAJ3TEBesYOkyFC5dFuIok3fPy0P+pKnw79lrckNDN09k5YszAUUClNgxZVZ+6AyNNIIxp+GvVuytgxH39JPhC9JYws0JyajviJHiUCdknlBlNtf11yDhhVGhyRIvC8hOlsfYWwMJ38tthbW8vDy0aNEC4rE+CwlEmgCd0xCIizckBbKzQxhhra4Sn6wsaxldjrXCURFO6SuvvIJp06ahdevW5fTmLRIgAaMRuCepkhSHmhsIYGZWlF7tnJ0oY7M1AhJ+k9th1KpWrQrxWJ+FBKJBgM5pCNRzHhuBgs8XhjDCWl0FH/FCApZTBD766CPpkP2VK1eicuXKxGIBAow5tcAil2LiyJTK2O0rxPs5maXcNd4lcRwanwIYb93MpDGPkgphNcWh+/D5Qhhhsa6FheTzz5KLrPyjR49i6tSpEFn5LNYgwHNO1VnnuHvvQiA3F/kTX1NHYASkvFipqpQkFW+z4dr4pJBnjOnXG4kTXw5tXLY4Yzr31BhbdSBxe2jjS+lds2ZN7N27Fw6Ho5S7ZV8atXEm3tulbHPi0mod8WZn/ZxtW7aVvBMpAtw5DYK0eJSfM/K/8G3fEURva3fxfPIZPB9/amkIxbPymzVrZmkWVjOeu01WW/GS9k5Oq47FeTlSHOqq/H+cxpJd1GtlNwQCRY/dE4HE8I+3atq0KdauXcsXSai3SpSkkAB3ToMAF/AUoGDhYgSOqxP0HsSUhu3i2/A7vK1awn1DsXc6G9aa0BQXWflvvPGGlJXfv39/ZuWHho+9SaAEgfjhDwFi93TmmyWu673xRuVz8AaAq48dQILdjvauil+0EXPpJUiaOSV403JaAYFiiUpJ2sf6b0jfgutW/Tt4HUPoueLoWjT64spyR8y6YCR6nsOTTsqFZKKb3Dk10WLSlOgRKMrKX716tZSV36VLl+gpw5mjRoAxp1FDr7uJP69aC0+kH8OWwgJ1dcvpAviLnTuaqE6Sbrt27aSTRcRjfRYSiDYB7pxWsAKB/Hz49+0HfP4KevJ2EQERBuE/fAT2c6oXXTL1T/EoVyQ9iaz8JUuWMPnJ1KtdvnGMOS2fT6h340eOkGJPPe99GOpQXfRfVr0Ouhzeg3lVaqKOM6ZUnZwXdETSe0HuDuf2AvzFXiGacASwJZQqN5SLXbt2xVtvvQXxWF+UNosHIturcVhCKAoCuPvn588a8Uvv95HmTjnrOi8YnwCd0wrW0PvbRmQNvh0QyVAsQREQJxr4/z6O5Dmzg+pv9E5z587FvHnzpGOj3G630c2h/mEQYMxpGPBMOvTHc+qhzaHd2O/z4kCthoi16euBZZ8+ffDSSy/h2ZOzsemLZw21Cucvvfm0vtXcafix9zun26wYm4C+/i/RI0txwDEd09BWxu+3DDPxS33r1q2YPn06YmNjYbPZQmPF3iRAAuUSSBgzGq5rB5TbR+83f6tRXzoPtcnBYnGiABwtWyD5f0HuCucNAHzLZVMTdgB2ka2vrPx73UQkXlgDv3VOx20nxmJT5k5lgnQy6qjnhBS3KmJXb1yt/5c46ASbbtWgc6rbpaFieicwZswY+P1+Kca06HGY3nWmftoSYMyptnyNLn1XzYZocECBE5h3M+D9XDY//nfA3lBus0YCJiNA59RkC0pztCcgsvInTZoE8Xq/7t27Mytfe+SGmUHEnLKoTyBx0iuI6XO5+oIjLNFps2FDjXo47+Au2BvUR8qXxRzOsnTJvwfwfiDfjf8BcLSS2wprxyZtQMIltRDfXvnuq8KpOYwEKiRA57RCROyghEAgPQMFK1ZCenGBEgE6HfP3339LGa0//PADbrjhBjArX6cLFSW1GHOqHfik16fCeUlX7SaIkOQUuwMiUSp11TK0b9++/FnzHwMKZ8l94r4GHMpPApm85UPp0XdKv/r4MvU36bG+LNw8tV9PbDr9iF885n/29+nmMc4iltA5LWeh/ZlZ8B8O/2DjcqYw7S3f1m3IefjfCGRqf/5epCAKx2PVqlUYN24cpk2bhtatW0dqas5DAiQAIPmDt+HsWIFDp3NStqQkaec0EAhI5yJffPHFwWkc+xng7BVc3zN6zdn9peSsTdoyB8dnb0JM3SQk9ahzRi/zNj/Yvfi0s/rG9s/Ma6iJLLMFxP8hLKUSyH//Q+Q+Mxrweku9z4vlE7ClpSJl+ZewVzHHe+VnzpwpZeUvXrwYLpeLyU/lL78l74o/YMR7yVm0JXCy95Xwbdqs7SRaSHc4kLa7pN7iD14Rv/7ll1+WnNHzLFDwnHxNOKcxwSeGrTq6HrevKZl9nz53K2wuBypd20iWa9Ha+HbDMaBOd4tar3+zuXNa3hqJrHM6puURssw98UrSLVu2YMaMGRDHRTEr3zJLH5KhjDkNCZfizilLF8Bev57i8dEamLrlt7OmFjunDz30EK6//nr5nmfsGY7peyE5prIguZYxfwcC/gAdUxkJazomQOdUx4tD1fRBQOxqiAcMgwYNQpMmTfShFLXQJQHGnEZuWSqtWgZbtaqRmzDMmSr9/jNsZZyD3LdvX+n3i/ij9/67bUDBCHk29wwgRj7PU75Reu2vk7ukR9hn7pqW3tu6Vx9bP0HitPrYeutC0LHlPIRfx4tD1aJLQGTlz5o1S8rK79+/Py644ILoKsTZSYAEShBI/fUHnGjeFsjJKXFdb42UH1fAnpparlpi5zRQ8DaQf4fczz0OcLaQ2gIAACAASURBVN0jtxXWMr/cA98JDyoPbaFQAoeRQGQJcOc0srxVne0nTx72eAtVlUlhpwiIrPwFCxZgzZo10jmmnTt3JhoSqJAAzzmtEJHqHdI2bwB0/PILEXfvqF2rYrsLPynpmLr+C7geq3jcPz2O5adLO4FXfPdQiTFZ3+6HZ/dJOqYlqMiN2358FutO/CVfYE0XBOic6mIZlCnxdk4mFufl4JiPCVvKCJY+KiMjAytXrsSECRMwdepUtGoV/pmCpc/Eq2YjwJjT6Kxo2t6tqLS+2Dvno6NGiVmTPnoPafu2wdG0cYnrQTVcT+DZF7wYP358UN3L6pTz4yHkbTiGqvfxZJGyGPG6PgnQOdXnugSl1Wup1bDLW4jxmelB9Wen4AjMnTsXU6ZMkY6NSktLC24Qe5EAAMacRu9rIE4FEc5g8tcLo6cEgISpEyU9Yi4M4WmL90sg/4YSeo8ePRoHDx6Uki/ff//9EveCaeSuP4asFQdQ7dF2wXRnHxLQFQE6p7pajtCUibHZ8EhSKuo6nXjoxJHQBrN3qQREVv7WrVshjo3icVGlIuJFEtA1AWfzZpJzmDRndkT1jB81UprXfdUVoc3rXQXk9ZPHxAwD3GOlttg5FcmYS5Yswfz58+U+xWrivjhovvNXtxa7ymooBG5Y/YTEcG8O3/AWCjct+9I51ZJuBGTXdDrROzYBLV1uPJZ+FJni+CsWRQReeOEFZuUrIsdBRQQYc1pEIvo/Yy6+SHIWEyaH92i8IktiH7hXmif2ztsq6nr2fd86IO8S+brzNiB2qtz+pyZ2TmfPno1ly5adda/V4pI7rqJD/l8ncHL+Dpzzn45n9eeFsgl0X3635KTmevPK7sQ7ESHAbP2IYNZ2kkYxLoh3Nh/z+fB6dgZsAHrExqOtK1bbiU0iPScnB6+//jo8Hg/69evHrHyTrGs0zBAxpzyEPxrky57Tfc1VEB9R8iZNQd64SWV3DvKO68p+SJwWphz/FiC3gzyj8zogruzdXrFz2qtXLyQkJEivTe627C7szz37iZlnVyaOz/4LtcZeJMtmLSQCrRbfiDhHLP7o/3FI49hZPQLcOS2Hpb1ObTjPL/bLo5y+0b5VzxmDB5MqYae3EOsLPFiRn4c/CjzRVkv38xfPyr/hhhvArHzdL5muFWTMqa6XB3EPPyDtcorY1KJPTPdiO5dlqG9v3Oh0fzEufMf0AJDTXJ7N0QeI+0Rul1ETO6fDhw+X4lB37dx1Vq/CQzk49up6OqZnkeEFoxHgzmk5K+bq3g22uDhk3TCknF76uZVsd2BqWnVJoZczT+DD3EwMs1dCLWdM5JWMdcNery7g0O/fP0VZ+a+++ioWLVqEypXN8ZrVyC82ZyQB4xJIevdNQylf8+WL0DC9CvbevRy1Xu4KRyW3pL83PR+HRv2EujN7GMoevSqb58vHBUtuwc993tOriqbWS7+eg6mxa2/cY0mpaOiMwf3pR7WfrJQZnK1bIfl/c2GrVKmUu/q4xKx8fayDmbRgzKmZVlMjWwIZQE5tWbijKxD/pdwOslZ3Vk/se+g7+PO98Od5ceCRVXRMg2QXbLfjBRlSDGq/bx8Mdgj7qUSAO6cqgdSbGIfNhgFxSUizO3D1sQOYXfkcpNodkVPTZoMtRr9fL5GVf+zYMSkrPyYmCjvLkVsJzhRBAow5jSBsI04V8ALZxd4UZW8HxK9SbEn9dy/Hrhu/RMDjO72DqlgYB5KAjghw51RHi6G2KpUdDnR1x2FgfBKeyfgbmwsL1J7CkPKef/55KSv/pptuQpMmTQxpA5XWJwHGnOpzXXSjVXairIqtKZCwTm5XULv35xekXbwN6VtK9Kz/UZ8SbTZIwAwE6JyaYRXLsaGqw4mr4xLRyBmDBXnZ2FCQX05vc98SWfkivrSgoAA9evTA+eefb26DaR0JkIB+CGRVAfBPkqqtJpBY0slUqujugUvQ8PMrUOvVi7H3nm+UiuE4EtAVATqnFSyHLTUVMd27AS7jPvpNsNvxaHIavIFARLL4pVMOOunL8RNZ+V988QV++uknMCu/gi89bysmwJhTxegsNDAFSDwQkr3Dfx2Prw+f/XrW3bd+hXrvXibJcqbGosaoTtj/8HchyWbniglsydot7Vrf+sMzFXdmD1UI0DmtAKOzWRMkTHwZtqSkCnrq//Z/UiqjAAEpi397YYH06lPhsKpdYi69BPGPP6q2WMXyirLyJ02aJL2WtFWrVoplcSAJlEdAxJyykMBZBLLrATh+1uVwLohs/TqTu8EeK8f2x9RIQNVH2mHn1Qtx6P/OdmbDmY9jSSCSBOicRpK2DuYqyuK/8Mhe9Du6H+l+nw600lYFkZX/2muvYfXq1UhLS9N2Mkq3NAHGnFp6+Us3PqcFENj7zz0HkJRRer9Srr7wx5vSjt3nB1aUcrf0S+4GydJj/pQBjXD4xbWld+JVRQS+/3uDtB5iJ5tFWwJ0TrXlqzvpRVn831Srg7crn4Pbjx/GOhPHoY4ZMwZbt26V3gDldDqlw6t1tyhUiARIwJwEci4A/H/JtiVmy/UwauIIqRqju5SboR/friqSLq2FoxPXhzETh5JAdAjQOQ2Cuy0xAQkvPgd7k8ZB9NZ/F5HF39LlRjuXG/clVcJHOVn4Nj9XFcVdN14H98DrVZEVrpDnnntOEjF48GA0bmyOtQuXCcdrS4Axp9ryNZT03O6A/xdZ5cRjgC38V0ofeHw1qg9vh5hz4mXZZdQSutRAXNuqODb99zJ68DIJ6JMAndMg1sXmdsPV93LYq5jrDUJumx1XxCWilcuNXwrysSw/Jwga5Xdxtm4JZ5voxnSKrPyJEyeisLAQPXv2RMeOHctXmndJQCUCjDlVCaTRxeReCfiKPYpP2A3YRLZ+eOXQM2tQ+a7z4KqfHLSgpO614a6fIsWhnpijzgkBQU/OjiSgkACd0xDAOS/oCHvdOiGMMEbXmxOScY7dgR88edJHqdZ64CMO1v/888/xyy+/SFn5nTp1UmoOx5FAyAQYcxoyMvMPiP8TsIuEqODLzG3/k2Ib39o5//Sgw2N+Qcr1jRDbrNgh/qfvll9J7ltPikO12W3I+HRH+Z15t0ICIgZ45G9TK+zHDsoJ0DkNgV38vx+ByEQ3Y7k1MQUtYtyYmZ2Bnd4C6dipUO2Me+xhuMSxW1Eq6enpWLlypZT8JBKgmJUfpYXgtCRgZQJ5gwDfQlUJHJ2wHkk96iC+TdWw5KYOagpfpgcnF+8OSw4Hk4DWBOicak3YQPKvjUvEvxJS0O/oAUNm8Yus/ClTpjAr30DfObOpyphTs61oiPbk3wl4P5IHxf8EOFrIbR3UKt/eAoV7s5D1zT4daEMVSKB0AnROS+dS5tW4B+9D7IP3lXnfyDfsNhvauWIxu/I5uC2ELH5bSjKS5r0PZ5vWUTNfZOVv27YNb7zxBhwOB7Pyo7YS1p6YMacWXv/8h4HCt2QAcd8AjgvkdhC1/+1dLj3Of/mv2ad7H5v2O+LaV0VC53NOXwu3UuXeVsj7/bgUh5q77mi44iw5/sM9S6S1mrj5A0var7XRdE5DJGw/pzrEx6xFvE1KZPHf/08W/zfBZPE7HHA0aQxbQsXZo1pwE1n5NpsNIiu/UaNGWkxBmSQQFAHGnAaFyXydPE8BhZNlu+K+AJzd5bbC2vG3NsHdMAVJl9ZWKKHsYdUeaSvFoZ78YhfyN50ouyPvkEAUCNA5VQDd2aolXNcNUDDSGENEFn//uES0/ieLf1zmCczNySxVeXutmoi9+07Y4uNKva/lRZGVP2HCBHi9XvTo0YNZ+VrCpmwSIIHSCXjGAAUvyvdiPwScV8pthTWRWe9IcyO5T2jJVKFOJ155evzdv+DZeTLUoexPApoRoHOqAK2zXRvEDbsbMT27Ay6XAgnGGDIkIRk17A5sLPRgbUE+lublwBPwn1ZenFzguv4axN1/D2zxkd01FVn58+fPx6+//sqs/NMrwkq0CTDmNNorYI75Mz7dDpFZX2lAZJ4E1XrpIhyb/BsKD6rzkgBzrAKtiCYBOqcK6TuaNkHCqy9D7KIiLvyDlRWqofkwkcX/TuUaGJyQDLGDurGgALn+Uw5qTLeLIU4wiHQpysqfOnUqJk+ejJYtW0ZaBc5HAqUSYMxpqVjMe7FgElDwtGyf+w0gZpDcNlCt9quX4NDoX6Q4VH9uoYE0p6pmJEDnNIxVtaWkIOnTD+Fs2yYMKcYY2i7GjTlVauDm44ei/rpTkZUvjopavXo10tLSjAGQWlqCAGNOLbHMp4wseBPwFPvj3P0q4LozbADimCdfZiHEsU+RLnVndJfiUPcMXY6APxDp6TkfCZwmQOf0NIrQKyIJx2a3BkJha6rdgc+r1sLbOZlYfOfNiHtoWOjQwhzxwgsvSFn5b731Fux2O7Pyw+TJ4SRAAgoIFH4EeO6SB7peAFwPy20FtS8Pfo/7xj8mHfNU+fZzFUhQb0iDuX2we9AS9QSaWNKUrXOlrP33di0ysZWRN80anpXGXOMeeQDOS7pqPEv0xTttNjSLceG+p5/Cn558vL1I3YOmK7Jw9OjRkkM6ZMgQNGzYsKLuvE8CESfAmNOII4/8hN6FQH6xR/eu/wDup8LWI+eHQ8j74zjEMU96KPXevxy7b/lKD6pQBwsSoHOqwqLHXNgZsXfcipgovh1JBTMqFhEXi9i7h6LnLTfjoosuwr59+/DOO+9UPC7MHiIrf/z48fD5fFJWfocOHcKUyOEkoA0Bxpxqw1W3UmMeAtxjwlbvyy+/xO0TH0S1h9uGLUstAXaXA3WmdJNiUPc9+J1aYimHBIIiQOc0KEwVd3L16g730Nvg7BzaocsVS9ZHD1taKlxX9kfsww/AUbcOrrjiCnTp0gU//fQTFixYgPz8fE0ULcrKX79+PbPyNSFMoWoSYMypmjR1KMu7AsgL/5io4patWrVKSuz8+ONPil/WRd2R4pZiUKv/ux0OPL5aFzpRCWsQcFrDzMhY6br0YthTU5D9yBPShP4DB4A8bZy2yFh0ahbxBiixK5w4/qUS0/bu3VtKSHrwwQdRtWpVpKSkoHr16qolKYms/O+++w7Tpk3DF198gcqVK5eYnw0SIAESiBgB3y9AXrGD9WOGArGTIjZ9NCdy1UtG5bvOw8FnfkTN57pEUxXObREC3DlVeaEdrVsh5ZsvpY9ZsvhdV12BhIkvl0qqY8eOkuN49dVXo0WLFvjwww9L7afkopA1ZcoUZuUrgccxUSHAmNOoYNd+Ut9fQG6xp2LOG4HYN8Oed926dXjqqacgHuvrvcQ2S0Xq9U1w+IVf9K4q9TMBATqnKi+ilMEvsvhtNiSMexGuq/qrPEPkxAn9k79ehLiHh5WZFS/sFMc5rVixAhs3bsSOHTvw/PPPh62kyMrfvn07RFZ+EdOwhVIACWhMgDGnGgOOhnj/XiC3hTyzoz8Q95HcVljbsmUL7rrrLojH+kYpcW2qIKlXHSkO9djU342iNvU0IAE6pxoumojNjL33X3APukHDWTQQnZiA+LHPS2/AcjZvCnv16uVO4nQ6ce6550qH4d98881wOBwYNWpUuWPKu/nf//6XWfnlAeI93RJgzKlul0aZYoHjQE6x14c6ugHx4Z9ScuDAAVx11VUQO6dGKwmdzpHiUN2NU/D3G38aTX3qaxACtkAgwJN2NV4s74bfUbD8WyAvD/nvzpF+ajylIvGO1i0R0+NS2OLiEHv7zYpfSfrzzz9j0aJFSExMxLBhw5CQkBCUPiIrf/r06cjOzka/fv1wwQXFHqMFJYGdSCC6BESCYKdOnaKrBGdXj4BwTrOrnJJn7wgkhP9IOyMjA+eddx6EgyrKL8f/xKDvR6incwQlnfxiJ3yZBUi7uXkEZ9X3VFM6jkDfmhfpW0kDaMeEqAgskrNta4hPICsL/oyT0k/fn3/Bv2dvBGaveApnp/Nhq5wG8TrS2MEDKx5QQQ/hVAqHdMyYMfjss89w+eWXo1q1auWOEln5X331FTZs2IARI0bwlaTl0uJNvRJgzKleV0aBXgGP7JgqGG6FISlXNUT6vG3I+GQ7Kl3f2Aom08YIEaBzGiHQYhpbUhISx70ozZg3aSo8ny+Qsvn9+0/9BR1BVYC4WNhr15KmjH/6STjbqfsKVrEzIF4xKhKlXC4XevXqVWYW/4kTJ6SsfLFr+vnnnzMrP6JfBE6mJgERcypOrWAxAYHsRNkIe3NVdk29Xi/q168PsXtqlpJ6YxMcf/cvKQ616oNtpJhUs9hGO6JHgM5plNjHPjQM4uNd8zOybrw54lo4W7dC0scfaDpvamoqVq5ciZ49e0LsjN5///2lzjd37lzMmzfPUIkBpRrCi5YnwJhTk3wFslIBeE8ZY6sDJPylimGVKlWSwpZUEaYjIZVvPRfic3TyBthiHUjsWlNH2lEVIxJgQlSUVq0oA93ZphWSly0+/RHHNmlVpOz7f+YSR0MV6SB+alGK5IuMe5HF/9xzz5WY5o477pBir7Zt24a33377tD4lOrFBAiRAApEkkF0bQNHOZhqQqE74VZUqVfD3339H0pKIz1XtobYQr2HNXXsk4nNzQnMRYEKUztbTu/FP+HbvKaFV/quvwbd1e4lrFTVcA6+XYkiL94vp0gn2KtE5yF5kpS5duhQejwfDhw/HY489hrZt20qH9zdq1Ah8JWnxlWLdqATEzikf6xt19QBkNwcCW/4xwAUkeVQxpl69evj1118hHNQzi5ETos60pah9aPTPqHRtI8S1jM6/N0V6ROMnE6LUoc7H+upwVE2Ks9V5EJ8SJT8f/n37S1yqqCGy7kUSll5K+/bt4fP5pCx+cQ7qnDlzMHjwYHTvXuyNK3pRlnqQgEICjDlVCM7Ew8TLSUR4U2mOqVnNrvHsBTj41A849OcJ1J56KVy1i8XvmtVo2qUqATqnquLURpj7hmu1ERxhqSIGVWTtf/vtt9JRUZs2bUKdOnXQuLE+szw9vkKsOrYOBf7CCkmluVLQuUqrCvuxg7kJMObUwOub06HYrimAxGwDGxN91WuOuVBSYt+wb1FjdGc4q8RFXylqYBgCdE4Ns1TGVlTsKC1YsAC//fYbPv74Y8kYEXMqHvMPGjQINWtGL4De6/dhb+4hiJ/Fy8nCbDyx/lWInxWVVimN8XK7R87qFu+MRe348l9icNYgXiABEogsgdxLAH+xA/ETTwC2mLB1EMfqffLJJxCP9a1a6kzrjj1Dl6H25EvgSHRZFQPtDpEAY05DBMbuygiITH3xvodp06aVEFDW9RKdNG4c95xEn2/vx4mCk6rPdEHllvjwolPHh6kunAJ1R4Axp7pbkooVyu0H+Iq92z5hH2AXSVHhFRGyNHbs2ApfJmLGmNPwyBl7NGNO1Vk/7pyqw5FSyiFw++23o3Xr1rjpppvO6jVy5EiIo6Ruu+02vPPOO2fd1+rClK1zsWD/Skm8L+DDycIsTab6PWMben8z7LTs+5sOxFW1u51us2IuAow5Ndh65t1whmP6lyqO6ZVXXolnn322QsfUYLSoLglEjACd04ihtt5EmZmZUmb++eefj969e6NGjRpnQRDXxC9yt9uNO++8ExMmTNAs2zmrMBdj/nwD4uefJ3dgb+7hs/RR+0K+z4Pt2ftOi319x6f46tCPaJxUB480H3L6OivmIMCYUwOtY94dgPcTWeH4tYA4bD/MIsKU7r77biZ7hsmRw61NgM6ptddfM+t3794tnV0qEp769euHBg0alDmXSIhyOp3SQf3CORU7reX1L1NQGTd+S9+Kb4/8gjyfBwsOrJR+ltFV88ubTu6E+NTMqHp6ruvq9kIdxqWe5sEKCZAACZCAtQnQObX2+mti/fbt26UzTQ8cOIDx48cHtRMqXun36KOPSuefLl68WNppDTeLf83fG3Hck4FVx9bj471fa2KrUqEH847hta1zTw9vklQX9RJqoGUlfZ5ccFpRVsol0Lx5+Dtv5U7Am+oQyH8A8M6WZcV9Bzg6yG2FNfH0Z8CAAdLTIIUiOIwESEC87j0gslRYSEAlAgcPHpRiSH///XfMnl3sl38I8ocOHYqWLVsqzuIXWfd7cg7i3+snQsR8GqX0r3kxHj/3VtRJOMcoKlPPMwhs3rwZdFDPgKK3pudJoOBlWau4RYCzn9xWWHv44Yel2HrhoCopXx78Hg+sfUnJUI7REQEmRKmzGHx9qTocKQWQsvFfeOEF6VWlSh1TAVK87nTnzp0Qh/WLv52KPsFAFn1FctOg7/9jKMdU2Lbo4Co8seHVkOwNhgn7RI4AY04jx1rRTJ7nSjqmsfNUcUyfeuopNGzYUIqbV6QXB5EACZQgQOe0BA42wiEgzi0VryJ95plnwhEjjRVZ/E2aNIF4u0rXrl1x4sSJoGRuzNiOG1c/qVn2fVBKhNHpd0n/J4I6WzWMaTiUBKxHoGACUPCsbHfsW0DMDXJbYW3MmDFITEyE2DllsS6BUa3uxY6rFqBvzYusC0FFyxlzqiJMq4oSu0XDhw+Xjk0RWfnnnBP+Y2kh46qrrpLeICUO6he/+EeMGCE97i+L88qj6zB928fYnXOwrC66vy6y+zee3I7H10+Eyx6DG+tdjm7Vwo+F073hJlGQj/QNspDu14CYO8JWdtKkScjOzoZwUFlIgATUI0DnVD2WlpQksvLFY3jxBhSRlS8Sm9QqYhdWfHJycnD48GHpzVKi3qlTpxJTfLJ3GfblHsHGjG34+fgfJe4ZsVHo9+KbI79IqosTBnx+H3qcc4ERTbGczjznVKdLXjAL8DymU+WoFgmQwJkE6JyeSYTtoAkUZeWLf5DHjRsXVFZ+0MKLdUxISJCy+EePHo1vvvlGOnaqQ4cO8PgKsfLor3hjx2fYlrW32AjzVL87+ivESwLinLHoUqW1eQwzqSWMOdXhwhZ+AHjukRVzvQS4HpDbCmtvvvmmFBsvdk9ZSIAE1CXAmFN1eVpGmjgm6osvvsDatWvx+uuva+aYFgcq3rji9/vxwQcfSP8oZHtz8dRvU0zrmBbZvvrYBrz059vYnrUP4iQCFhIggSAJFM4H8m+WO7tGAu4n5bbC2kcffYQffvgBdEwVAuQwEqiAAJ3TCgDx9tkEhIMoYqx27NghHbR/dg/trjz99NNSopQ4birg92s3kc4k/3FyO276foRhE710hlMzdRhzqhna0AV7lwH518jjYh4F3M/JbYW1hQsX4tNPP4XYOWUhARLQhgAf62vD1dRSRVZ+mzZtMHjw4KjYKV4PWK1aNXS9uCsKh9cFEvk1jspCcNKzCDDm9Cwk0bng+xHIu0yeO+ZfQOwEua2wtmLFCsyYMQPCQWUhARLQjgB3TrVjazrJIp5O7FiKhKSrr75alax8JZA2Fu7ER3Hf40Q3N47O/B0FezKViDHcmCxvLp7cMBlbM/cYTnerKMyYU6ustPp2iiOIXm77iPqCKVFTAg80HSQdIXVLg/6azmM14dxystqKK7R3165d0iN8kY2vdlZ+KCp9c/hnvL97MX71bkNCp+rwZXiQ/f0hYPUhuJtUQsIF1UMRZ6i+Iov/2yO/ICUmEeIXYdvUZobSn8qSQEQI+DYCuRfKUzkHA7Gz5LbC2i+//IJRo0ZB7J6ykAAJaEuAzqm2fE0hfdu2bVi6dKl0nNMrr7wSkeSn0sD9eOx3vLNrIVYfWy/dtsc6Uenqhkj/aCsK9mYDPj/ssQ7Eta5S2nDTXJu//1s4bQ44bHa0qtTENHaZwRDGnEZ5Ff07gdxip1o4rwbiPghbqb/++gvDhg2DcFBZSIAEtCdA51R7xoaeQWTlL1iwABs3box48tOZ4KZsnYs1xzeeeRmpA5tK17JXHUTmkj1wpLoRUzMBNod5o1Y+2bcMboeLzulZ34boXmDMaRT5+48COY1kBRw9gbj5clthbe/evbj22mshHFQWEiCByBAw77/ekeFn6ll8Ph9efPFF6dimt99+O2q2BgIB+AJ+BBAoV4fEi2siuV99HBr5I3yZBdI76ssdYPCbgos/YJ0TC4ywXIw5jdIqBXKBnGIhPfbOQPyysJU5fvy49PrkSDmm19XtKcUvPnHu7WHrTgHaEripXh9prR5tPkTbiSwqnc6pRRc+GLNF8lPDhg0h3nMfzXKyMBuDVj+J3zK2VaiGiDut/vT5ODJmLTzbMirsb+QOCw+uxOPrXzWyCdSdBNQhkJ0oy7G3BBJ+lNsKa+K1yU2bNoXYOWUhARKILAE6p5HlbYjZxO6POC6qc+fOUc3KL4Ildk135xyCeO98RcXudsDdIAWpA5sga/l+5P56tKIhhr2fWZiDg3nHDKu/GRVnzKkZV5U2kQAJRJoAndNIE9f5fCIrf/z48WjQoAH69u0rvds+mirvzz2Cmds+QZ4vP2g1bDF2xHesjtjmqcjfmo6cn48EPdZoHQWfGds+Rq43eD5Gs9FI+oqYU5YIE8hKBopCfmwNgISz49KVaFS5cmWIx/osJEACkSdA5zTyzHU7o8jKX7RoEY4cOYKHH34Y4tioaJcDuUfx5s75yAti1/RMXZO610ZM1Xh4Nqcj77e/z7xtirbYOX1zh+BD51QPC8qY0wivQnYNAFmnJrVVBRJ3qqJAamoq0tPTVZGlRMg9Ta6T4hmHNhygZDjHaEzg6lqX4vk292s8i7XF0zm19vqftr4oK3/dunWYOXNm1I6LOq0QgPSCTOzNPVz8Usj1pF514GqQjMyle1CwNwsBn/kSiLwBP7Zl7UOONy9kPhxAAoYlkN0ECBT9fogDEtUJ4aldu7aUBBoTE2NYNFScBIxOgM6p0VdQBf2LZ+W/9dZbKkhUR8SiA6swYsPksIVJWfz96+PQM2vgO1kgOagBf/mZ/2FPGkEBmYXZGPLDU/g9iISxCKplyakYcxqhZc9pCwS2y5MlZsv1MGpi/dasWQOxc8pCAiQQPQJ0TqPHXjczi+SniMdaPQAAIABJREFURo0a4ZlnntGNTmor4m4sZ/Hvf+A7iDNRWUhAbQKMOVWbaBDyEk8CNnP+U/Z0yzulx/viMTJL9AlcVKWttB4TOjwWfWVMrgEP4Tf5ApdnnoiPE7GlXbp0Qe/evVG9erFzAssbaMB7p7L4k5E6qCkChT7kb8tA5lIvknvXM6A1VFmvBBhzGoGVyb0I8P+m+kQdOnTA/PnzIR7rs5AACUSXAJ3T6PKP2uwiK188whc7piIrXw/JT8VhLD/8M5Yf+bn4pbDrp7L4q0lybG4HPFsykPn1XiRfVjds2XoQ8PHerxHncKNtajM9qEMdSEB9Arm9Ad8PstyEg4BNZOuHVy655BLMmjULDMsIjyNHk4BaBOicqkXSQHK2bt2KpUuX4tixYxg7dqwukp/OxLfy6K9YeXTdmZdVa8e3rwbYbcj9+QhyfjiEuI7VYHc5VJMfDUGf71+BdqnN6ZxGA/4/c9K50RB+3rWA7yt5goStgF1k64dX+vXrh+effx5i51SvRTxGzvXl4+vDa/Sqoqn1apZUH4u7v2ZqG/VmHJ1Tva2Ixvrs378fCxcuxJ9//ok333xT49n0LT6+bVU4EmNw/I0/YU9xwd0wBfY4/i+h71XTt3Yi5jQlJUXfShpRu7xbAe9nsubx6wF7E7mtsHbDDTfggQcegNg5ZSEBEtAPAXNGkeuHr6408Xq9eOmllyAe6VvdMS1aGClR6j8dcXTcOni2psNMWfxFNvJn5Agw5lQD1vn3Ad73ZMFxqwFHW7mtsCYSQW+88UaInVMWEiABfRHgNpG+1kNTbcQv4/bt22Pw4MGazmM04fZEF2qOuRAn3vkL3gwPkroxIcJoa0h9LUIgbgngvMgixpY0c8YFT0sXrlv1b2xI31LyJluaEKjsqoSf+xT7w0iTWSi0NALcOS2Nismuid2c2267DRdddBGuvvpqU2flK1k6m8OGmBoJSLmqIQr2ZCFzyR4lYnQxZu6eJZiz+0td6GJFJRhzqvKq5z8OFM5QWShOP8oXj/WNVv538TjpOKPa8eY9XUUPaxLniKVjGsWF4M5pFOFHYuqiR/hNmjRBnz59dJeVXxqDeXu+isrOQGyLNPjzvfBszcCJ9zdDZPSnXNkA9ljj/G+yOXM3xIclOgQYc6oid88ooHCcLDD2f4Czt9xWWHvyySelrHzxJImFBEhAnwS4c6rPdVFFK5GVL5Kfjh8/jgcffNAQjqkwXGSd/3FyhyoMQhUisvjdzVPhPZKHwkM5yFl9SHrUH6oc9rcmAcacqrTuBa8ABf+VhcW+A8RcK7cV1p577jmkpaVJO6cKRXAYCZBABAgYZ0soAjDMNAWz8pWvpsjiFx9fdgGOvPQrbC474kRmf7JLuVCOJAESCI5AwXTA84Tc1z0NiLlVbiusTZgwAQUFBaZ5E953vd6QSLRcdD3yfB6FVDisNAIb+81DvDOutFu8FiEC3DmNEOhITlNYWChl5e/evZtZ+WGAd4hEqee7IGv5PmSvOoCAzx+GNA61AgHGnIa5yoXvAp5hshD3y4DrPrnNGgmQgCUI0Dk14TIPHTpUevPTyJEjTWhd5E2q+mAbeA/nIn3etshPzhkNRUDEnLLoj4B4+5N4miQe65utbOz3sdlMipo93/acJSWbcdc0aktwemI6p6dRGL8i4t1uvfVWKSt/wIABqFbt1Ks6jW9ZdC1wVolDYs860hukTry3ObrKcHZdE2DMaRjLU/gpkH+bLMD1f4DrcbmtsPbBBx9g7dq1EI/1WUiABIxBgDGnxlinCrUUWflvvPEGmjVrhr59+6JevXoVjmGH4Am46ycDvgAK9mUFP4g9SYAEgiPgXQrkXyf3jfk34B4ltxXW5s+fj8WLF0M4qGYtNptN2u07lp+Ozl+FH5trVk7l2fVx15fRPu3c8rrwXoQJ0DmNMHAtphNZ+UuWLEF6ejqeeOIJvj5RC8gA3I1SpI9G4inWBAQYc6pgEb2rgbw+8sCYe4HYV+S2wtqyZcswe/ZsCAfVCqVqbKrkpP51cheu+O4hK5isio3vdBlNx1QVkuoKoXOqLs+IS9u3b590XNSmTZukndOIK8AJSYAEThPgOaenUQRX8W0A8i6W+zpvAWKny22FtR9//FFKChUOqtXKuSkNJCd11dH1uH3Ns1YzP2h7x7cbjgF1ugfdnx0jS4DOaWR5qzqbyMofO3Ys7HY7HVNVyVIYCSgjwJhTZdykUc5rgLh3wxDAoSRAAmYhQOfUwCsp3nDSoUMHDBkyxMBWUHUSIAFLEvBvA3LbqW76xo0bMXz4cIjdUxYSIAFjEmC2vgHXrSgrv2vXrjBjVv7wc29Gt2odDLgyVNnqBBhzGuQ3wH8IyGkqd3ZcDsR9KrcV1nbu3InBgwfTMQVwcbV2mHb+UwpJmnfYf1oMlcIe+Ehf32vMnVN9r89Z2olfviIrX/wjaNas/A5pLVAnvvpZtvNCxQR6Vr8Avc7pVHFH9tCEAGNOg8AayARyasodHRcB8UvltsLa0aNH0atXL4jfkSynCPSu0UVyxIp4TN7yISZtmVPUtMzPIfX7YXRrvszBSAtO59RAq1WUlZ+RkYEnn3ySWfkGWrtIqXpxtfa4pFr7SE3Hec4gwJjTM4Cc2Qz4gewU+aq9DRC/Wm4rrOXm5qJly5YQDipL2QQeanYTxEeUf6+biM/2f1N2ZwPfERsc87qONbAFVJ3OqUG+AyIrf8GCBdi8eTNef/11g2hNNUmABEigGIHsRLlhawwkbJDbYdTEC0eys7PDkMChJEACeiJA51RPq1GGLgUFBaez8umYlgGJl0lABwQYcxrkItjOARL5OuAgaWnSbVz7RyE+ReXKFQ9jU6ZxQyKqudPwY+93iszhT4MToHNqgAUUWfkdO3bEzTffbABtqSIJWJcAY07LWfvsqgDyyumg7FZycjIyMzOVDeYoEiABXRKgc6rLZTmllIhfe+CBB3DJJZfg8ssvR9Wq4pe7NcpN9fvCH/Bjzp4l1jBYBSuHN78Zl1bvqIIkilBKgDGnZZDLbgAE/v7nZhKQeKiMjqFdrlGjBvbv3y+d9RzaSPY+k8CCSyedvtRm8UBke3NPt/Va+aX3+0hzF4th1qui1CtkAnROQ0YWmQEi41Q8wm/RogX69OmDevXqRWZinczSPLk+miXX14k2xlBDvBuapxwYY60spWVOSyCw+x+TbUCSOrucTZo0wbp16yB2TlnUJfBbv4/OErghfQuuW/Xvs65H6sKsC0aiJ08iiRTuqM9D5zTqS3C2Alu2bMHSpUshdmH+85//8Jfv2Yh4hQR0SYAxp2csS05nwP+nfDFRnaSltm3bYvHixRA7pywkQALmI0DnVGdrunfvXixcuBDCQZ01a5bOtIusOuJxTYOEmtiVczCyExtsNofNjiZJdZHojDOY5uZTlzGnxdY0tyfg/0m+kHAUsMXLbdYMRaBtarMSZ6aeqfyojTPx3q6FZ14OqX19nV4Y2+7hkMawszkJ2AKBQMCcphnPKpGVL16753A4MGmSHP9jPEvU0/invzdi8A98y0l5RNNcyVjSfSoquyuV1433IkDgp59+QqdOfAmChFo4p75i52gK59Qeftz8RRddhKlTp0LsnrKQAAmYkwB3TnW0rkVZ+bfccouOtKIqJEACJBAigdyrz3BMd6rimPbu3Rsvv/wyHdMQl4PdScBoBOxGU9iM+orYUuGQiqz8AQMGoEqVKmY0U5FNTZPrYXz74Uhy8nFgaQCbJtXF+PaPIcmZUNptXoswAcacitOihgC+L2Ty8RsBewO5rbB27bXXSk+WxM4pCwmQgLkJcOc0yutr9az8ivCnupJxcdX2iLHHVNTVkvcruZL5ulIdrbzlY07z7wa8xd7dHv8j4GgZ9grdeuut0jnPYueUhQRIwPwEuHMaxTUWSU/ilaTiAOn777/fcsdFBYvebY9B/5pdUYUxlSWQNUiohYurtitxjY3oErD0Oaf5w4HCYq9WjlsGODqHvSD33XcfevXqBbFzykICJGANAtw5jdI679mzh1n5QbJPjInHqNb3YlvWXvztyQhylPm7XVi1DYY1vdH8htJC4xGImw84expPb2pMAiSgCwJ0TqOwDB6PB+PGjZPeamL146JCwS8e7Ytjk3wBfyjDTNnXaXMgxuYwpW1GNsqyMaeeZ4DCiaov3eOPP47WrVtDPNZnIQESsA4BPtaPwlqLrPxGjRrhmWeeicLsxp3y5XYP48pa3YxrgIqaP9B0IB5oNkhFiRSlBgERc2q54hkLFDwvmx37PuC8Wm4rrI0aNQrVqlWDeKzPQgIkYC0CdE4juN4iHu3mm29Gt27dmJWvgHu12DTc3fg6DKnfV8Fo8wx5rPktuKZOD4hkMRZ9EbBczGnBFKBghLwI7plAzBC5rbD2yiuvSCPFzikLCZCA9QjwsX6E1lxk5YtH+C1btkTfvn1Rt27dCM1srmmaJdfDwHq94Q8E8OGeJeYyLkhr2qU1R+346kH2ZjcS0IhA4duA50FZuHs84LpbbiusTZ8+HUePHkWRg6pQDIeRAAkYmAB3TiOweJs3b5ay8rOzszFs2DA6pmEyPy+lEa6sdUmYUow33GWPQb+aXVHVnWo85S2isWViTgs/BvKHyqvqGg24hstt1kiABEggDALcOQ0DXjBDRVb+okWLsHXrVsycOTOYIewTBIEEZzxaJDfElqw98AV8QYwwdpcERxzapTXDc62HoZIrydjGmFh7S55z6noCcKsTP//uu+/i999/h9g9ZSEBErAuAe6carj2+fn5Ulb+vn376JiqzLllpUZ4p8toVI9Ngx02laXrS5w4oUA8yn+ny3N0TPW1NGdpY4mYU+9iIF/9I8w+/fRTLFu2jI7pWd8qXiAB6xGgc6rhmjMrX0O4AFJcSfio61i0SW2q7URRli5CGMa1ezTKWnB6EgDgXQnk9ZdRxNwPuMfKbYW1pUuX4v3334fYOWUhARIgATqnGnwHxO7JkCFD0L17dykrv3LlyhrMQpFiR7FmXFU8fd5d6F69oymBDK7fF/c0vh5VYxlnaoQFNnXMqe9XIK/YUW7O24HYKWEvy+rVqzFhwgSInVMWEiABEhAEbIFAIEAU6hEoysqvVKkSBg8ezOQn9dCWK2n1sfXYdHIn9uYcNk0W/6B6vTG4Xl+cV6lRubbzpn4IiORHUzqo/s1AzrkyaOf1QNzHclthbcOGDXjggQcgHFQWEiABEigiwISoIhIq/BT/MC1ZsgQ5OTl46qmnkJzMcyhVwBqUiK5V20F8dmbvR1ZhDr4+vAYef2FQY/XUqYq7Ei6p1kFS6Y6GV6NxUh09qUddKiBgyphT//6SjqmjryqOaQUoeZsESMDCBOicqrT4zMpXCWSYYhom1sbo1sOQUZiN9embkePNC1Ni5IanuVLQ65xOeKHNA5GblDORQJQIbNu2DbfffjvE7ikLCZAACRQnwMf6xWkorOfl5eGJJ56A0+nExInqv19aoVqWHzZ0zSj8fPwPeP0+FAa8uuXhtsfAbrNDPMYf2fJfutWTilVMQOycpqSkVNzRKD0C6UB2mqyt42IgfqXcVlgTR25dcsklEA4qCwmQAAmcSYDO6ZlEFLRvuukmdOrUCbfccguY/KQAoEZDjuWnw+MvwPx932Lilg80miV8sa+2/7d0VFSCM46vJA0fZ1QlmCrmNFAIZLtknvb2QMKvclthLTMzE82aNYNwUFlIgARIoDQCfKxfGpUgr4ldkvvuu0/Kyu/duzcd0yC5RapbUYa7eA99y0qNpVjUZ36fhixvbqRUKHOepkn18GSL26X74iisVBfjk8uEZaAbpoo5zU6UyduaqeKY+v1+1K5dG8JBZSEBEiCBsgjQOS2LTAXXd+zYgVmzZqFNmzbo06cPs/Ir4BXN27Xiq0F88nweKRZV/Pz60I9Yl7454mpddk5ntE87FzXiquBSkx5/FXGonFB9Alni+LuCU3JttYDEyP+/or5RlEgCJGAUAnROFazUX3/9BXFotIg1FTunzMpXADEKQ+IcbtzS4NQB4tXcqWiQWEvSQmT1C2dVi+x+kX3f7Z/sezHZgNrdcWHVNlGwnlNGgoApjpHKrgvgxD+4KgGJ+1VDJ35XZmdnqyaPgkiABMxJgM5piOu6e/duLF68WArknzFjRoij2V0vBAbU6Q7xEeVkQTZOFmThZKH8j6bI8t+RHdo/yg6bA82S68Fpc5w287yURni+zf2n26yYm4CIozRVQpSKy1W1alUcPXpURYkURQIkYFYCdE5DWFmxUzp+/HgpK5+OaQjgdN41xZWI2V1Gl9By7fFNuGPN/5W4VlFDxI3O7jwald0mytauyGjeL0HA8DGn4qD9wL5/bHICSekl7FPaaNCgATZt2oT4+HilIjiOBEjAQgTonIaw2HfccQc6d+4sZeWHMIxdDUigVaXGWNJ9akiai+OgKrmSQhrDziSgGwI55wPiTVBFJVF+klB0ScnPli1b4ptvvoHYOWUhARIggWAI0DkNgpLYDbn33nvRs2dPXH755czKD4KZ0bu4HS4picrodlD/yBIwbMxp7qWAf60MK/FvwOaW2wpr4o/5uXPnQuycspAACZBAsATonFZASmTlz5w5E+3atYM4LqpuXZEswEICJEACZxMwZMxp7hWA7zvZmIQ9gE1k67OQAAmQQHQI0Dkth7vIyl+yZAk8Ho+0c8qs/HJg8RYJkAAMH3Mavwmwq/MHuHjS9Oqrr0I81mchARIggVAI0Dktg5bIyl+0aBHEzun06dPL6MXLJEACJGBgAnkDAd8i1Q24+uqr8dRTT0kx+qoLp0ASIAHTE6BzWsoS5+bmSln5MTExdExL4cNLJEACpRMwVMxp3lDAO082JP5nwHGu3FZYGzJkCIYOHSrF6CsUwWEkQAIWJ2C3uP2lmi9+sTZq1AgjR44s9T4vkgAJkEBpBAzzvvj8hwDv27IJcd8CjvPltsLa3XffjX79+kHsnLKQAAmQgFICdE6LkcvIyMCgQYOkv/ivvfZapKWlFbvLKgmQAAmUT8AQMaee/wCFr8mGxC0AnJfKbdZIgARIIMoE+Fj/nwUQsaXiYP0OHTowKz/KX0pOTwIkoBEBzwtAwUuy8Ni5gPMKuR1Gbfjw4ejYsSPEY30WEiABEgiHAJ1TAEVZ+YWFhbjnnnvArPxwvlIcSwLWJWComFMVl+mZZ55B7dq1IR7rs5AACZBAuAQs75zu2rWLWfnhfos4ngRIQCJgqHNOY98EYgaGvXJjx46Fy+WC2DllIQESIAE1CFjaOc3JycGECRPArHw1vkqUQQIkoPuYU/fTgPioVKZMmYITJ05AOKgsJEACJKAWAUs7pyIrv0uXLrjtttvU4kk5JEACJGAJAm+//TY2b94M4aCykAAJkICaBCyZrS+y8gcOHIjLLrsMIis/NTVVTaaURQIkYFECVo05tehy02wSIAGNCFhu57TojU/nn3++lJVfp04djdBSLAmQgNUIGCrmNIzF+fjjj7Fy5UqI3VMWEiABElCbgKWc06KsfK/XK2WVMitf7a8T5ZGAtQnoPuZUheVZvHgx5s2bB+GgspAACZCAFgRsgUAgoIVgvcncuXMnPv30U4if06ZN05t61IcESMAEBH766Sd06tTJBJbQBBIgARKIHgFL7JyKrPyJEydKWfl0TKP3ZePMJGB2Aow5NfsK0z4SIIFIELBEQtQdd9yBRo0aQRwUzUICJEACWhEQMacsJEACJEAC4REwtXNalJV/+eWXMys/vO8JR5MACQRBwAoxp0FgYBcSIAESCIuAaR/rb9++HTNmzACz8sP6fnAwCZAACZAACZAACUSUgCmd002bNmHJkiXw+XzMyo/o14mTkYC1CTDm1NrrT+tJgATUIWA651Rk44ujTnbt2oWpU6eqQ4lSSIAESCAIAlY55zQIFOxCAiRAAooJmM45Fdn44nQsOqaKvxMcSAIkoJAAY04VguMwEiABEihGwHTO6ciRI4uZxyoJkAAJkAAJkAAJkICRCFjmEH4jLQp1JQESMCYBsXOakpJiTOWpNQmQAAnohICpj5LSCWOqQQIkYBECPOfUIgtNM0mABDQlQOdUU7wUTgIkYCUCjDm10mrTVhIgAa0I0DnViizlkgAJkAAJkAAJkAAJhEyAMachI+MAEiABEiidAGNOS+fCqyRAAiQQCgHunIZCi31JgARIoBwCjDktBw5vkQAJkECQBOicBgmK3UiABEigIgKMOa2IEO+TAAmQQMUE6JxWzIg9SIAESIAESIAESIAEIkSAMacRAs1pSIAEzE+AMafmX2NaSAIkoD0B7pxqz5gzkAAJWIQAY04tstA0kwRIQFMCdE41xUvhJEACViLAmFMrrTZtJQES0IoAnVOtyFIuCZAACZAACZAACZBAyAQYcxoyMg4gARIggdIJMOa0dC68SgIkQAKhEODOaSi02JcESIAEyiHAmNNy4PAWCZAACQRJgM5pkKDYjQRIgAQqIsCY04oI8T4JkAAJVEyAzmnFjNiDBEiABEiABEiABEggQgQYcxoh0JyGBEjA/AQYc2r+NaaFJEAC2hPgzqn2jDkDCZCARQgw5tQiC00zSYAENCVA51RTvBROAiRgJQKMObXSatNWEiABrQjQOdWKLOWSAAmQAAmQAAmQAAmETIAxpyEj4wASIAESKJ0AY05L58KrJEACJBAKAe6chkKLfUmABEigHAKMOS0HDm+RAAmQQJAE6JwGCYrdSIAESKAiAow5rYgQ75MACZBAxQTonFbMiD1IgARIgARIgARIgAQiRIAxpxECzWlIgATMT4Axp+ZfY1pIAiSgPQHunGrPmDOQAAlYhABjTi2y0DSTBEhAUwJ0TjXFS+EkQAJWIsCYUyutNm0lARLQigCdU63IUi4JkAAJkAAJkAAJkEDIBBhzGjIyDiABEiCB0gkw5rR0LrxKAiRAAqEQ4M5pKLTYlwRIgATKIcCY03Lg8BYJkAAJBEmAzmmQoNiNBEiABCoiwJjTigjxPgmQAAlUTIDOacWM2IMESIAESIAESIAESCBCBBhzGiHQnIYESMD8BBhzav41poUkQALaE+DOqfaMOQMJkIBFCDDm1CILTTNJgAQ0JUDnVFO8FE4CJGAlAow5tdJq01YSIAGtCNA51Yos5ZIACZAACZAACZAACYRMgDGnISPjABIgARIonQBjTkvnwqskQAIkEAoB7pyGQot9SYAESKAcAow5LQcOb5EACZBAkATonAYJit1IgARIoCICjDmtiBDvkwAJkEDFBOicVsyIPUiABEiABEiABEiABCJEgDGnEQLNaUiABMxPgDGn5l9jWkgCJKA9Ae6cas+YM5AACViEAGNOLbLQNJMESEBTAnRONcVL4SRAAlYiwJhTK602bSUBEtCKAJ1TrchSLgmQAAmQAAmQwP+3czc5cRxRAIBrpGwRJ2BnIhRlj4RPgFe5QXawhENklw0sYZdt9jYnsAUXQEhwl4lqoOwWngzTVPVMdddnyZqhu7p+vmo/P3oeECDQW0DNaW8yFxAgQGC5gJrT5S6OEiBAoI+AJ6d9tLQlQIDACgE1pytwnCJAgMCaApLTNaE0I0CAwFsCak7fEnKeAAECbwtITt820oIAAQIECBAgQGBDAmpONwRtGAIEpi+g5nT6e2yFBAgML+DJ6fDGRiBAoBEBNaeNbLRlEiAwqMDs9vZ2HkfY2dkJe3t74f7+fjGgr3m4H/x7EA/EQ/8/+P9RPiAf2HQ+4GP9QXN/nRMg0JLA3d1dODw8bGnJ1kqAAIHiApLT4qQ6JECgVQE1p63uvHUTIFBSQM1pSU19ESDQtICa06a33+IJECgkIDktBKkbAgQI+D2n7gECBAjkC0hO8w31QIAAAQIECBAgUEhAzWkhSN0QIEBAzal7gAABAvkCnpzmG+qBAAECCwE1p24EAgQI5AtITvMN9UCAAIGFgJpTNwIBAgTyBSSn+YZ6IECAAAECBAgQKCSg5rQQpG4IECCg5tQ9QIAAgXwBT07zDfVAgACBhYCaUzcCAQIE8gUkp/mGeiBAgMBCQM2pG4EAAQL5ApLTfEM9ECBAgAABAgQIFBJQc1oIUjcECBBQc+oeIECAQL6AJ6f5hnogQIDAQkDNqRuBAAEC+QKS03xDPRAgQGAhoObUjUCAAIF8gV/yu9ADgfUFZrPZ+o1fWs7n897XuIAAAQItCIipLexye2tUc9renm91xTGQvk42lx1Lk1x1LrXxSqAWATWntexEO/NYFiOXHUsiq86lNl4JbFvAx/rb3gHjFxR4CpcfZ+Hj5VPBPnVFYH0BNafrW2lZscDNaYhJbPevuFrxfk1wapLTCW7qmJYUg1/80w2C6VjfdTxd/hnOv/W9SnsC5QTUnJaz1NP7BFL8fHdMjYnpp+twdPG4+JQrftI1f7wI4Xw/zE5v3jcpVxHoKSA57QmmeTmBFEQXwS8GwJfa0vS67kg3p8/f4e/LTNcl044AgQkKlImpv4WLx3n4evbhh9CHs/DPxVEI138FH0z9YPFuOAHJ6XC2el4hEINoSkpTQE3HVly29NTx1XNiG7+7P1rawkECmxE4ODjYzEBGIfBKIMXPGFezYurxWejmpa+G8SWBjQhITjfCbJDXAt2noymYdo+9bu9rAmMQUHM6hl2a5hy78bN8TH0Kn/+NNVO/h187D1SnKWlVNQj4VVI17ELDc0jf4adgGim6QbZhGksfoYCa0xFu2sSmPEhMvfn7uZ7/5I9wPDEvy6lTQHJa575MelYpeMZFdhPR9L57ftIQFkeAAIECAt2YmeJo7Da9757vPdzTZfj46TqEcBK+XElNe/u54F0CktN3sbkoRyAFzNRHCpzpeHqN59O51NYrgZoF1JzWvDvTnVs3ZsZVpriZjqfX7rm1NGJiun8evsXEdH7lqelaaBqVEFBzWkJRH1l8X8y4AAACFklEQVQCMXDGvymgZnXmYgJbFFBzukV8Q38XKBJTJabfPb3ZvIAnp5s3N2JHICWkKZh2TnlLYHQCak5Ht2WTm3CRmPryu07D0UV4/HoW/AzU5G6T6hckOa1+i6Y9wZiUxj/dgDrtFVsdAQIEhhPIjqkpMT35EuZqTIfbKD2vFJjN0528spmTBMoIpCS0T29v3aLxl/Av6vV/6lSd1E8kDgwqEJ+c7u7uDjqGzgl0BUrH1P+Ppy+jSlq7/N4PJCA5HQhWt8sFYiB9K9nsXtm3ffda7wlsWuDh4SH4oahNq7c9Xt8Y2bd927pWvy0BPxC1LflGx+2TmEaivu0bZbXsSgTUnFayEQ1No2+M7Nu+IUpLrUhAclrRZpgKAQIECBAgQKB1AR/rt34HWD8BAsUE1JwWo9QRAQINC3hy2vDmWzoBAmUF/J7Tsp56I0CgTQHJaZv7btUECAwgoOZ0AFRdEiDQnIDktLktt2ACBAgQIECAQL0Cak7r3RszI0BgZAJqTke2YaZLgECVAp6cVrktJkWAwBgF1JyOcdfMmQCB2gQkp7XtiPkQIDBaATWno906EydAoCIByWlFm2EqBAgQIECAAIHWBdSctn4HWD8BAsUE1JwWo9QRAQINC3hy2vDmWzoBAmUF1JyW9dQbAQJtCkhO29x3qyZAYAABNacDoOqSAIHmBCSnzW25BRMgQIAAAQIE6hX4D8Wq/RirwZEkAAAAAElFTkSuQmCC)

 

从哲学上来说，用马氏距离处理数据时，不再把数据单纯的看作是冷冰冰的数字——那个引入的协方差，承认了客观上的差异性，就好像是有了人类的感情倾向，使得模式识别更加“人性化”也更加“视觉直观”。