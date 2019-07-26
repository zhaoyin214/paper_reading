# Bayesian Learning for Neural Networks

---
Radford M. Neal, [Bayesian Learning for Neural Networks][bayes_learning], phd, Univ. of Toronto, 1995

[bayes_learning]: http://www.cs.utoronto.ca/~radford/ftp/thesis.pdf "Bayesian Learning for Neural Networks"
---

## 摘要


## 1 引言


### 1.2 贝叶斯神经网络（Bayesian neural networks）

#### 1.2.1 多层感知器（multilayer perceptron networks）

$$f_{k}(x) = b_{k} + \sum_{j} v_{jk} h{j}(x) \tag{1.5}$$

$$h_{j}(x) = \tanh \left(a_{j} + \sum_{i} u_{ij} x_{i} \right) \tag{1.6}$$

非线性激活函数（non-linear activation function）


实值目标回归模型（a regression model with real-valued targets）：给定输入$x$，目标$y_{k}$的条件概率（conditional distribution for the targets）可定义为高斯分布（Gaussian with $y_{k}$ having a mean of $f_{k}(x)$ and a standard deviation of $\sigma_{k}$）$N(f_{k}(x), \sigma_{k})$

$$P(y | x) = \prod_{k} \frac{1}{\sqrt{2 \pi} \sigma_{k}}
\exp \left( - \frac{(f_{k}(x) - y_{k})^{2}}{2 \sigma_{k}^{2}} \right) \tag{1.7}$$

分类问题（a classification task）

$$P(y = k | x) = \frac{\exp(f_{k}(x))}{\sum_{k^{\prime} \exp(f_{k^{\prime}}(x))}} \tag{1.8}$$


预测分布方程(1.3)写为：

$$\begin{aligned}
P \left( y^{(n + 1)} \right. | & \left. x^{(n + 1)}, \left( x^{(1)}, y^{(1)} \right), \cdots, \left( x^{(n)}, y^{(n)} \right) \right) \\
= & \int P\left( y^{(n + 1)} |x^{(n + 1)}, \theta \right)
P \left( \theta | \left( x^{(1)}, y^{(1)} \right), \cdots, \left( x^{(n)}, y^{(n)} \right) \right)
d \theta
\end{aligned} \tag{1.9}$$

似然改写为：

$$L \left( \theta | \left( x^{(1)}, y^{(1)} \right), \cdots, \left( x^{(n)}, y^{(n)} \right) \right) =
\prod_{i = 1}^{n} P \left( y^{(i)} | x^{(i)}, \theta \right) \tag{1.10}$$


回归模型预测分布的均值

$$\hat{y}_{k}^{(n + 1)} =
\int f_{k} \left(x^{(n + 1)}, \theta \right)
P \left( \theta | \left( x^{(1)}, y^{(1)} \right), \cdots, \left( x^{(n)}, y^{(n)} \right) \right)
d \theta \tag{1.11}$$

#### 1.2.2 网络模型和先验的选择（selecting a network model and prior）


#### 1.2.3 自动确定相关性模型（the automatic relevance determination (ARD) model）

#### 1.2.4 贝叶斯学习神经网络（an illustration of bayesian learning for a neural network）

#### 1.2.5 高斯近似应用（implementations based on gaussian approximations）

### 1.3 马尔可夫链蒙特卡罗方法（Markov chain Monte Carlo methods）

#### 1.3.1 马尔可夫链蒙特卡罗积分（Monte Carlo integration using Markov chains）

$$E[a] = \int a(\theta) Q(\theta) d\theta \tag{1.12}$$

采样近似

$$E[a] \approx \frac{1}{N} \sum_{t=1}^{N} a \left( \theta^{(t)} \right) \tag{1.13}$$

$$Q \left( \theta^{\prime} \right) = \int T \left(\theta^{\prime} | \theta \right) Q(\theta) d\theta \tag{1.14}$$

*可逆（reversible）*

$$ T \left(\theta^{\prime} | \theta \right) Q(\theta) =  T \left( \theta | \theta^{\prime} \right) Q \left( \theta^{\prime} \right) \tag{1.15}$$

*各态历经性（ergodic）*


#### 1.3.2 吉布斯采样（Gibbs sampling）

#### 1.3.3 Metropolis算法（The Metropolis algorithm）
