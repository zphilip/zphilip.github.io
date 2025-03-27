---
layout: post
title: Markov Chain Monte Carlo
date: 2021-01-26
categories: LEARNING
tags: AI MCMC
---

### Markov Chain Basic

<img src="/assets/Gradient%20boosting/image-20230519213253623.png" alt="image-20230519213253623" style="zoom:80%;" />

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

<img src="/assets/Gradient%20boosting/image-20230524160307826.png" alt="image-20230524160307826" style="zoom:67%;" />

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



Above gives one distribution answer to the question " what's the system state after certain time t"

### **Monte Carlo method**

<img src="/assets/Gradient%20boosting/image-20230519162005088.png" alt="image-20230519162005088" style="zoom:67%;" />A geometric 
$$
\begin{array}{cl}
F:= & \int f(x) p(x) d x \approx \frac{1}{N} \sum_{i=1}^N f\left(x_i\right)=: \hat{F} \quad \text { if } x_i \sim p \\
\mathbb{E}_p(\hat{F})=F & \operatorname{var}_p(\hat{F})=\frac{\operatorname{var}_p(f)}{N}
\end{array}
$$

### MCMC: Metropolise-Hastings 

- Let the sequence {$X^{t}$ } denote a Markov chain for t = 0, 1, 2, . . ., where $X^{t} = (X_1^{(t)} , . . . , X_p^{(t)}  ) $and the state space is either continuous or discrete.
- The MCMC sampling strategy is to construct a Markov chain that converges to a stationary distribution equal to the target distribution f .
- For sufficiently large t, a realization $X^{t}$ from this chain will have approximate marginal distribution f .

- The art of MCMC lies in the construction of a suitable chain

- Metropolis-Hastings is a specific implementation of MCMC. It works well in high dimensional spaces as opposed to Gibbs sampling and rejection sampling.

How MCMC could sample from a distribution $π(x)$

> **(refer to frequency be in state j)** In order to sample from a distribution $π(x)$, **a MCMC algorithm constructs and simulates a Markov chain whose stationary distribution is $π(x)$**, meaning that, after an initial “burn-in” phase, the states of that Markov chain are distributed according to $π(x)$. <font color=red>**We thus just have to store the states to obtain samples from *π*(*x*).**</font>
>
> For didactic purposes, let’s for now consider both a discrete state space and discrete “time”. The key quantity characterizing a Markov chain is the transition operator $T(x_{i+1}∣x_i)$ which gives you the probability of being in state $x_{i+1}$ at time $i+1$ given that the chain is in state $x_i$ at time $i$.

transition *kernel*

> **a transition matrix *T*, or be continuous, in which case *T* would be a transition *kernel***.  while considering continuous distributions, but all concepts presented here transfer to the discrete case.
>
> If we could design the transition kernel in such a way that the next state is already drawn from *π*, we would be done, as our Markov chain would… well… immediately sample from *π*. Unfortunately, to do this, we need to be able to sample from *π*, which we can’t.
>
> A way around this is to split the transition kernel $T(x_{i+1}∣x_i)$ into two parts: **a proposal step and an acceptance/rejection step (not real reject , using old value).** 
>
> - The proposal step features a proposal distribution $q(x_{i+1}∣x_i)$, from which we can sample possible next states of the chain. In addition to being able to sample from it, we can choose this distribution arbitrarily. But, one should strive to design it such that samples from it are both as little correlated with the current state as possible and have a good chance of being accepted in the acceptance step. 
>
> acceptance/rejection step is the second part of the transition kernel and corrects for the error introduced by proposal states drawn from $q \neq π$. It involves calculating an acceptance probability $p_{acc}(x_{i+1}∣x_i)$ and accepting the proposal $x_{i+1}$ with that probability as the next state in the chain. 
>
> Drawing the next state $x_{i+1}$ from $T(x_{i+1}∣x_i)$  is then done as follows: 
>
> - first, a proposal state $x_{i+1}$  is drawn from $q(x_{i+1}∣x_i)$, 
> - It is then accepted as the next state with probability $p_{acc}(x_{i+1}∣x_i)$ or rejected with probability $ 1- p_{acc}(x_{i+1}∣x_i)$ , in which case the current state is copied as the next state. another interpret here is 
>
> > > if $a = p_{acc}(x_{i+1}∣x_i) \geq 1$, accept: $x_{t+1} = x^{\prime}$ 
> > >
> > > else
> > > 	accept with probability a: $x_{t+1} = x^{\prime}$
> > > 	stay with probability $1-a: x_{t+1} = x_t$
> > >
> > > In practice will generate $u \sim U_{[0,1]}$ ， if u < a (sample under the ratio) accept the new proposal sample points , otherwise old sample point will be used. 
>
> We thus have
> $$
> T\left(x_{i+1} \mid x_i\right)=q\left(x_{i+1} \mid x_i\right) \times p_{\text {acc }}\left(x_{i+1} \mid x_i\right)
> $$
> A sufficient condition for a Markov chain to have *π* as its stationary distribution is the transition kernel obeying *detailed balance* or, in the physics literature, *microscopic reversibility*:
> $$
> \pi\left(x_i\right) T\left(x_{i+1} \mid x_i\right)=\pi\left(x_{i+1}\right) T\left(x_i \mid x_{i+1}\right)
> $$
> This means that the probability of being in a state $x_i$ and transitioning to $x_{i+1}$ must be equal to the probability of the reverse process, namely, being in state $x_{i+1}$ and transitioning to $x_i$. Transition kernels of most MCMC algorithms satisfy this condition.
>
> For the two-part transition kernel to obey detailed balance, we need to choose $p_{acc}$ correctly, meaning that is has to correct for any asymmetries in probability flow from / to $x_{i+1}$ or $x_i$. Metropolis-Hastings uses the Metropolis acceptance criterion:
> $$
> p_{\text {acc }}\left(x_{i+1} \mid x_i\right)=\min \left\{1, \frac{\pi\left(x_{i+1}\right) \times q\left(x_i \mid x_{i+1}\right)}{\pi\left(x_i\right) \times q\left(x_{i+1} \mid x_i\right)}\right\}
> $$
>
> $$
> \begin{aligned}
> p(x) T\left(x \rightarrow x^{\prime}\right) & =p(x) \cdot q\left(x^{\prime} \mid x\right) \min \left[1, \frac{p\left(x^{\prime}\right) q\left(x \mid x^{\prime}\right)}{p(x) q\left(x^{\prime} \mid x\right)}\right] \\
> & =\min \left[p(x) q\left(x^{\prime} \mid x\right), p\left(x^{\prime}\right) q\left(x \mid x^{\prime}\right)\right] \\
> & =p\left(x^{\prime}\right) \cdot q\left(x \mid x^{\prime}\right) \min \left[\frac{p(x) q\left(x^{\prime} \mid x\right)}{p\left(x^{\prime}\right) q\left(x \mid x^{\prime}\right)}, 1\right] \\
> & =p\left(x^{\prime}\right) T\left(x^{\prime} \rightarrow x\right)
> \end{aligned}
> $$
>
> so that Markov Chains satisfying detailed balance have at least one stationary distribution
> $$
> \int p(x) T\left(x \rightarrow x^{\prime}\right) d x=\int p\left(x^{\prime}\right) T\left(x^{\prime} \rightarrow x\right) d x=p\left(x^{\prime}\right) \int T\left(x^{\prime} \rightarrow x\right) d x=p\left(x^{\prime}\right)
> $$
> Now here’s where the magic happens: we know *π* only up to a constant, but it doesn’t matter, because that unknown constant cancels out in the expression for $p_{acc}$! It is this property of  $p_{acc}$ which makes algorithms based on Metropolis-Hastings work for unnormalized distributions. Often, symmetric proposal distributions with $q(x_i∣x_{i+1})= q(x_{i+1}∣x_i)$ are used, in which case the Metropolis-Hastings algorithm reduces to the original, but less general Metropolis algorithm developed in 1953 and for which
> $$
> p_{\text {acc }}\left(x_{i+1} \mid x_i\right)=\min \left\{1, \frac{\pi\left(x_{i+1}\right)}{\pi\left(x_i\right)}\right\}
> $$
> We can then write the complete Metropolis-Hastings transition kernel as
> $$
> T(x_{i+1}|x_i) = \begin{cases}                   q(x_{i+1}|x_i) \times p_\mathrm{acc}(x_{i+1}|x_i) &: x_{i+1} \neq x_i \mbox ; \\                   1 - \int \mathrm{d}x_{i+1} \ q(x_{i+1}|x_i) \times p_\mathrm{acc}(x_{i+1}|x_i) &: x_{i+1} = x_i\mbox .                 \end{cases}
> $$
> and <font color=red> we record the state $x_0, x_1, ... , x_i, ...x_n$ will present the real posterior distribution $p(x)$ </font>

1. Initialise $x^{(0)}$
2. For $i=0$ to $N-1$
   - $\Rightarrow$ Sample $u \sim U_{[0,1]}$.
   - $\Rightarrow$ Sample $x^{\star} \sim q\left(x^{\star} \mid x^{(i)}\right)$. e,g. $x^{*}=x^{(i)}+N\left(0,6^2\right)$
   - If $u<A\left(x^{(i)}, x^{\star}\right)=\min \left\{1, r\left(\theta_{\text {new }}, \theta_{t-1}\right) = \frac{p\left(x^{\star}\right) q\left(x^{(i)} \mid x^{\star}\right)}{p\left(x^{(i)}\right) q\left(x^{\star} \mid x^{(i)}\right)}\right  \}$

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
& = \frac{\prod_i^n f\left(d_i | \Theta=\theta^{\prime}\right) P\left(\theta^{\prime}\right)}{\prod_i^n f\left(d_i | \Theta=\theta\right) P(\theta)} \\
& =\frac{\operatorname{Beta}\left(1,1, \theta_{\text {new }}\right) \times \operatorname{Binomial}\left(10,4, \theta_{\text {new }}\right)}{\operatorname{Beta}\left(1,1, \theta_{t-1}\right) \times \operatorname{Binomial}\left(10,4, \theta_{t-1}\right)}
\end{aligned}
$$

The “likelihood” of each new sample is decided by a function *f* . That’s why *f* must be proportional to the posterior we want to sample from. *f* is commonly chosen to be a probability density function that expresses this proportionality.

#### Find the <font color=red> maximum of $\tilde{p}(x)$ </font> (reference)

1. given current estimate $x_t$
2. draw proposal $x^{\prime} \sim q\left(x^{\prime} \mid x_t\right)$
3. evaluate

$$
a=\frac{\tilde{p}\left(x^{\prime}\right)}{\tilde{p}\left(x_t\right)}
$$

- if $a \geq 1$, accept: $x_{t+1} = x^{\prime}$
- else stay: $x_{t+1} = x_t$

#### Meptropolis-Hastings performs a random walk

Rule of Thumb: [MacKay, (29.32)]

- typical use-case: high-dimensional $D$ problem of largest length-scale $L$, smallest $\varepsilon$, isotropic proposal distribution

- have to set width of $q$ to $\approx \varepsilon$, otherwise acceptance rate $r$ will be very low.

- then Metropolis-Hastings does a random walk in D dimensions, moving a distance of $\sqrt{\mathbb{E}\left[\left\|x_t-x_0\right\|^2\right]} \sim \epsilon \sqrt{r t}$
- so, to create one independent draw at distance $L$, MCMC has to run for at least

$$
t \sim \frac{1}{r}\left(\frac{L}{\epsilon}\right)^2
$$

iterations. In practice (e.g. if the distribution has islands), the situation can be much worse.

1. Producing exact samples is just as hard as high-dimensional integration. Thus, practical MC methods sample from a unnormalized density ˜p(x) = Z · p(x) 

2. even this, however, is hard. Because it is hard to build a globally useful approximation to the integrand 
3. Markov Chain Monte Carlo circumvents this problem by using local operations. It only converges well on the scale in which the local models cover the global problem. Thus the local behaviour has to be tuned.

### Gibbs Sampling

$$
x_t \leftarrow x_{t-1} ; x_{t i} \sim p\left(x_{t i} \mid x_{t 1}, x_{t 2}, \ldots, x_{t(i-1)}, x_{t(i+1)}, \ldots\right)
$$

a special case of Metropolis-Hastings:
$$
\begin{aligned}
& \rightarrow q\left(x^{\prime} \mid x_t\right)=\delta\left(x^{\prime},-x_{t, \backslash i}\right) p\left(x_i^{\prime} \mid x_{t, \backslash i}\right) \\
& >p\left(x^{\prime}\right)=p\left(x_i^{\prime} \mid x_{\backslash}^{\prime}\right) p\left(x_{\backslash i}^{\prime}\right)=p\left(x_i^{\prime} \mid x_{t, \backslash i}\right) p\left(x_{t, \backslash i}\right) \\
& >\text { acceptance rate: } \\
& a=\frac{p\left(x^{\prime}\right)}{p\left(x_t\right)} \cdot \frac{q\left(x_t \mid x^{\prime}\right)}{q\left(x^{\prime} \mid x_t\right)} \quad=\frac{p\left(x_i^{\prime} \mid x_{t, \backslash i}\right) p\left(x_{t, \backslash i}\right)}{p\left(x_{t i} \mid x_{t, \backslash i}\right) p\left(x_{t, \backslash i}\right)} \cdot \frac{q\left(x_t \mid x^{\prime}\right)}{\delta\left(x_{\backslash i}^{\prime}-x_{t, \backslash i}\right) p\left(x_i^{\prime} \mid x_{t, \backslash i}\right)} \\
& =\frac{q\left(x_t \mid x^{\prime}\right)}{p\left(x_{t i} \mid x_{t, \backslash i}\right) \delta\left(x_{\backslash i}^{\prime}-x_{t, \backslash i}\right)} \quad=1 \\
&
\end{aligned}
$$

### Hamiltonian Monte Carlo

consider Boltzmann distributions $P(x)=z^{-1} \exp (-E(x)$ )
augment the state-space by auxiliary momentum variables $p=\dot{x}$. Define Hamiltonian ("potential and kinetic energy")
$$
H(x, p)=E(x)+K(p) \quad \text { with, e.g. } K(p)=\frac{1}{2} p^{\top} p
$$
do Metropolis-Hastings with $p, x$ coupled by to Hamiltonian dynamics
$$
\dot{x}:=\frac{\partial x}{\partial t}=\frac{\partial H}{\partial p} \quad \dot{p}:=\frac{\partial p}{\partial t}=-\frac{\partial H}{\partial x}
$$
nb: need to solve an ODE
note that, due to additive structure of Hamiltonian, this (asymptotically) samples from the factorizing joint
$$
P_H(x, p)=\frac{1}{Z_H} \exp (-H(x, p))=\frac{1}{Z_H} \exp (-E(x)) \cdot \exp (-K(p)) \quad \text { with } P_H(x)=\int P_H(x, p) d p=P(x)
$$

### Rejection Sampling (reference)

(a simple method [Georges-Louis Leclerc, Comte de Buffon, 1707-1788)

![image-20230522202804698](/assets/Gradient%20boosting/image-20230522202804698.png)

$\rightarrow$ for any $p(x)=\tilde{p}(x) / Z$ (normalizer $Z$ not required)

- choose $q(x)$ s.t. $c q(x) \geq \tilde{p}(x)$

- draw $s \sim q(x) \text{ the proposal distribution}, u \sim \operatorname{Uniform}[0, c q(s)]$

- reject if $u>\tilde{p}(s)$

- The rejection ration is exponentially increased with dimension D

  Example:

  - $p(x)=\mathcal{N}\left(x ; 0, \sigma_p^2\right)$
  - $q(x)=\mathcal{N}\left(x ; 0, \sigma_q^2\right)$
  - $\sigma_q>\sigma_p$

  - optimal $c$ is given by

  $$
  c=\frac{\left(2 \pi \sigma_q^2\right)^{D / 2}}{\left(2 \pi \sigma_\rho^2\right)^{D / 2}}=\left(\frac{\sigma_q}{\sigma_p}\right)^D=\exp \left(D \ln \frac{\sigma_q}{\sigma_p}\right)
  $$

  - acceptance rate is ratio of volumes: $1 / c$
  - rejection rate rises exponentially in $D$
    for $\sigma_q / \sigma_p=1.1, D=100,1 / c<10^{-4}$

### Importance Sampling (reference)

computing $\tilde{p}(x), q(x)$, then throwing them away seems wasteful instead, rewrite (assume $q(x)>0$ if $p(x)>0$ )
$$
\begin{aligned}
\phi & =\int f(x) p(x) d x=\int f(x) \frac{p(x)}{q(x)} q(x) d x \\
& \approx \frac{1}{S} \sum_s f\left(x_s\right) \frac{p\left(x_s\right)}{q\left(x_s\right)}=: \frac{1}{S} \sum_s f\left(x_s\right) w_s \quad \text { if } x_s \sim q(x)
\end{aligned}
$$
this is just using a new function $g(x)=f(x) p(x) / q(x)$, so it is an unbiased estimator $w_s$ is known as the importance (weight) of sample $s$ if normalization unknown, can also use $\tilde{p}(x)=Z p(x)$
$$
\begin{aligned}
\int f(x) p(x) d x & =\frac{1}{Z} \frac{1}{S} \sum_s f\left(x_s\right) \frac{\tilde{p}\left(x_s\right)}{q\left(x_s\right)} \\
& =\frac{1}{S} \sum_s f\left(x_s\right) \frac{\tilde{p}\left(x_s\right) / q\left(x_s\right)}{\frac{1}{s} \sum_t 1 \tilde{p}\left(x_t\right) / q\left(x_t\right)}=: \sum_s f\left(x_s\right) \tilde{w}_s
\end{aligned}
$$
this is consistent, but biased

### Geometric Series (reference)

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