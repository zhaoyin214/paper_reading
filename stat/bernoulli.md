# 伯努利分布


## 伯努利分布

*伯努利分布（Bernoulli distribution）*：**单次（only once）随机实验**，结果有两种可能：事件$0$（失败）和事件$1$（成功）。假设$X \in \{ 0, 1 \}$为二值随机变量（binary random variable），$x = 1$的概率为$\theta$、$x = 0$的概率为$1 - \theta$，则$X$服从伯努利分布，记为$X \sim \text{Ber} (\theta)$，其概率累积函数（probability mass function，pmf）定义为：

$$\text{Ber} (x ; \theta) = \theta^{x} (1 - \theta)^{1 - x}$$

即

$$\text{Ber} (x ; \theta) = \begin{cases}
\theta, & \text{ if } x = 1\\
1 - \theta, & \text{ if } x = 0\\
\end{cases}$$


*多项伯努利分布（multinoulli distribution）*：**单次（only once）随机实验**，结果有$K$种可能。假设$\mathbf{x} = (x_{1}, \cdots, x_{K})$为随机向量（random vector），$x_{j} \in \{ 0, 1 \}$为二值随机变量，$x_{j} = 1$的概率为$\theta_{j}$、$x_{j} = 0$的概率为$1 - \theta_{j}$，则当事件$j$发生时，$x_{j} = 1$、$x_{i} = 0$（$i \not = j$）。多项伯努利分布的概率累积函数为：

$$\text{Multinoulli} (x ; \theta) = \prod_{j = 1}^{K} \theta_{j}^{x_{j}}$$

## 二项分布和多项分布

*二项分布（binomial distribution）*：**$n$重伯努利随机实验**，假设$X \in \{ 0, 1, \cdots, n \}$表示事件$1$发生的次数。若事件$1$的发生的概率为$\theta$，则$X$服从二项分布，记为$X \sim \text{Bin} (n, \theta)$，其概率累积函数（probability mass function，pmf）定义为：

$$\text{Bin} (k ; n, \theta) =
\begin{pmatrix}
n \\
k \\
\end{pmatrix} \theta^{k} (1 - \theta)^{n - k}$$

<img src="http://www.stat.yale.edu/Courses/1997-98/101/binpdf.gif" />

*多项分布（multinomial distribution）*：**$n$重多项伯努利随机实验**，假设$\mathbf{x} = (x_{1}, \cdots, x_{K})$为随机向量（random vector），其中$x_{j}$表示事件$j$发生的次数。若事件$j$的发生的概率为$\theta_{j}$，则$\mathbf{x}$服从多项分布，其概率累积函数（probability mass function，pmf）定义为：

$$\text{Mu} (\mathbf{x} ; n, \mathbf{\theta}) =
\begin{pmatrix}
n \\
x_{1} \cdots x_{K} \\
\end{pmatrix} \prod_{j = 1}^{K} \theta^{x_{j}} =
\frac{n!}{x_{1}! x_{2}! \cdots x_{K}!} \prod_{j = 1}^{K} \theta^{x_{j}}, \quad
\sum_{j = 1}^{K} x_{j} = n$$

## 贝塔分布

*贝塔分布（beta distribution）*：紧支撑为$[0, 1]$（support over the interval），

$$\text{Beta} (x ; \alpha, \beta) = \frac{1}{B (\alpha, \beta)} x^{\alpha - 1} (1 - x)^{\beta - 1}$$

其中，$B (\alpha, \beta)$为贝塔函数（beta function），

$$B (\alpha, \beta) = \frac{\Gamma(\alpha) \Gamma(\beta)}{\Gamma(\alpha + \beta)}$$

$\Gamma(x)$为伽马函数（Gamma function）为广义阶乘函数（an extension of the factorial function）

$$\Gamma(x) = \begin{cases}
(x - 1)! & \quad \text{if } x \text{ is a positive integer} \\
\int_{0}^{\infty} u^{x - 1} e^{- u} du & \quad \text{if } x \text{ is complex with a positive real part} \\
\end{cases}$$

伽马函数
<img src="https://upload.wikimedia.org/wikipedia/commons/5/52/Gamma_plot.svg" width="400" />

贝塔分布
<img src="https://gss0.bdstatic.com/-4o3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike80%2C5%2C5%2C80%2C26/sign=eaa4487ca66eddc432eabca958b2dd98/730e0cf3d7ca7bcbee16a1b6b5096b63f624a83e.jpg" width="400" />

### 贝塔分布应用

在$[0, 1]$区间随机抽取$n$个数并降序排列，求第$k$个数为$x$的概率为

$[0, 1]$区间均匀抽取的随机数落在$[0, x]$子区间的概率为$x$、落在$[x, 1]$子区间的概率为$1 - x$，则第$k$个数为$x$的概率为

$$f(x) = n \left( \begin{matrix}
  n - 1 \\
  k - 1 \\
\end{matrix} \right)
x^{k - 1} (x - x)^{n - k}$$

▇

$n$：抽取结果恰为$x$有$n$种可能；

$\left( \begin{matrix}
  n - 1 \\
  k - 1 \\
\end{matrix} \right)
x^{k - 1} (x - x)^{n - k}$：二项分布，抽取结果$\gt x$、$\lt x$

▇

令$\alpha = k$，$\beta = n - k + 1$，则$f(x)$服从贝塔分布

$$f(x) = \frac{1}{B (\alpha, \beta)} x^{\alpha - 1} (1 - x)^{\beta - 1}$$

## 狄利克雷分布

*狄利克雷分布（Dirichlet distribution，多元Beta分布，multivariate Beta distribution）*：在实数域以正单纯形（standard simplex）为支撑集（support）的高维连续概率分布，是贝塔分布在高维情形的推广。

在贝叶斯推断（Bayesian inference）中，狄利克雷分布可作为多项分布的共轭先验、在机器学习（machine learning）可用于构建狄利克雷混合模型（Dirichlet mixture model）。狄利克雷分布在函数空间内对应的随机过程（stochastic process）称为狄利克雷过程（Dirichlet process）。

概率密度函数（）

独立同分布（independent and identically distributed，iid）连续随机变量$\mathbf{X} \in \R_{d}$
